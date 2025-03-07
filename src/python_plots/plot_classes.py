import os
from multiprocessing import Process
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import typing as tp
from pathlib import Path
import contextily as ctx
from pyproj import Transformer
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import substring
from src.misc.globals import *
import pandas as pd
import warnings

FIG_SIZE = (15,10)
# Number of historical points to be displayed on the x-axis
PLOT_LENGTH = 200
# Delay between frames in milliseconds
REALTIME_UPDATE_INTERVAL = 200
VEHICLE_POINT_SIZE = 12
CTX_PROVIDER = ctx.providers.CartoDB.Positron# ctx.providers.Stamen.TonerLite
BOARDER_SIZE = 1000

import matplotlib.colors
tab20 = plt.cm.get_cmap('Oranges', 20)
cl = tab20(np.linspace(0.2, 1, 5))
STATUS_COLOR_LIST= ["lightgrey","red","blue","orange","green","dimgrey","purple","beige"]
OCCUPANCY_COLOR_LIST =  list(cl) + ['dodgerblue'] + ['dimgrey']

# SMALL_SIZE = 8
# MEDIUM_SIZE = 18
# BIGGER_SIZE = 20

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class PyPlot(Process):

    def __init__(self, nw_dir, shared_dict: dict, plot_folder: str = None, plot_extent=None, output_dir=None):
        """ Class for plotting real time information

        :param nw_dir:      network directory, where the background map is/will be saved
        :param shared_dict: a dictionary object for sharing real time information
        :param plot_folder: full path of the folder for saving the animation. If provided, each frame is saved
                            in the location.
        :param plot_extent: Tuple of (lon_1, lon_2, lat_1, lat_2) marking the left bottom and top right boundary of
                            the map
        """

        super().__init__()
        self.bg_map_path = os.path.join(nw_dir, "downloaded_map.tif")
        self.shared_dict: dict = shared_dict
        self.plot_folder: Path = Path(plot_folder) if plot_folder else None

        self.fig, self.grid_spec, self.axes = None, None, None
        self.plot_extent = plot_extent
        x, y = self.convert_lat_lon(plot_extent[2:4], plot_extent[0:2])
        x[0], y[0] = x[0] - BOARDER_SIZE, y[0] - BOARDER_SIZE
        x[1], y[1] = x[1] + BOARDER_SIZE, y[1] + BOARDER_SIZE
        self.plot_extent_3857 = x + y
        # Download the map image using the extent
        _, bbox = ctx.bounds2raster(x[0], y[0], x[1], y[1], path=self.bg_map_path, source=CTX_PROVIDER)
        #self.plot_extent_3857 = bbox
        
        self._times = []
        self._pax_counts = {}
        self._state_counts = {}
        self._avg_occ = []
        self._avg_wait_time = []
        self._avg_ride_time = []
        self._avg_detour_time = []
        self._accepted_requests = []
        self._rejected_requests = []
        
        self._key_to_plot_func = {
            'status_count': self._create_status_count_plot,
            'occupancy_count': self._create_occ_count_plot,
            'occupancy_average': self._create_avg_occ_plot,
            'occupancy_stack_chart': self._create_occ_stack_plot,
            'waiting_time_average': self._create_avg_wait_time_plot,
            'ride_time_average': self._create_avg_ride_time_plot,
            'detour_time_average': self._create_avg_detour_time_plot ,
            'service_rate': self._create_service_rate_stack_plot
        }

        self.line_alignment = None
        # load PT Line alignment parameters
        if output_dir is not None:
            scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(output_dir)
            dir_names = get_directory_dict(scenario_parameters)

            schedules = pd.read_csv(os.path.join(dir_names[G_DIR_PT], scenario_parameters[G_PT_SCHEDULE_F]))
            # TODO: to generalize for more than 1 line
            for key, _ in schedules.groupby(["LINE", "line_vehicle_id", "vehicle_type"]):
                self.line_id = key[0]
                break
            alignment_file = scenario_parameters.get(G_PT_ALIGNMENT_F, 0)
            self.line_alignment: LineString = gpd.read_file(
                os.path.join(dir_names[G_DIR_PT], alignment_file.format(line_id=self.line_id)))['geometry'].iloc[0]

            self.pt_fixed_length_km: float = scenario_parameters.get(G_PT_FIXED_LENGTH, None)  # km
            self.crs_km = "EPSG:32632"

            line_alignment_km: LineString = self.convert_lat_lon_line(self.line_alignment, from_epsg='epsg:4326',
                                                                      to_epsg=self.crs_km, always_xy=True)
            split_point = self.pt_fixed_length_km * 1000 / line_alignment_km.length
            first_part_km = substring(self.line_alignment, 0, split_point, normalized=True)
            second_part_km = substring(self.line_alignment, split_point, 1, normalized=True)

            lon, lat = first_part_km.xy
            self.first_part_x, self.first_part_y = self.convert_lat_lon(lat, lon, from_epsg='epsg:4326')
            lon, lat = second_part_km.xy
            self.second_part_x, self.second_part_y = self.convert_lat_lon(lat, lon, from_epsg='epsg:4326')

    def convert_lat_lon(self, lats: list, lons: list, to_epsg: str = "epsg:3857", from_epsg: str = 'epsg:4326'):
        proj_transformer = Transformer.from_proj(from_epsg, to_epsg)
        x, y = proj_transformer.transform(lats, lons)
        return list(x), list(y)

    def convert_lat_lon_line(self, line: LineString, to_epsg: str = "epsg:3857", from_epsg: str = 'epsg:4326',
                             always_xy:bool = False):
        lats, lons = line.xy
        proj_transformer = Transformer.from_proj(from_epsg, to_epsg, always_xy=always_xy)
        x, y = proj_transformer.transform(lats, lons)
        return LineString(zip(x, y))

    def generate_plot_axes(self):
        fig = plt.figure(1, figsize=FIG_SIZE, tight_layout=True)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        gs.update(wspace=0.025, hspace=0.5)
        axes = [plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2]), plt.subplot(gs[:, 0:2])]
        return fig, gs, axes

    def draw_plots(self):
        #print("draw")
        # color_list = ['blue','orange','green','red','purple','beige']
            
        self._times.append(self.shared_dict["sim_time_float"])
        for k, v in self.shared_dict["status_counts"].items():
            if self._state_counts.get(k) is None:
                self._state_counts[k] = []
            self._state_counts[k].append(v)
        for k, v in self.shared_dict["pax_info"].items():
            if self._pax_counts.get(k) is None:
                self._pax_counts[k] = []
            self._pax_counts[k].append(v)
        self._avg_occ.append(self.shared_dict["avg_pax"])
        self._avg_wait_time.append(self.shared_dict["avg_wait_time"])
        self._avg_ride_time.append(self.shared_dict["avg_ride_time"])
        self._avg_detour_time.append(self.shared_dict["avg_detour_time"])
        self._accepted_requests.append(self.shared_dict["accepted_users"])
        self._rejected_requests.append(self.shared_dict["rejected_users"])
        #print("here")
        if self.shared_dict.get("plot_1") is not None:
            self._key_to_plot_func[self.shared_dict["plot_1"]](0)
        if self.shared_dict.get("plot_2") is not None:
            self._key_to_plot_func[self.shared_dict["plot_2"]](1)
        if self.shared_dict.get("plot_3") is not None:
            self._key_to_plot_func[self.shared_dict["plot_3"]](2)
            
        if self.shared_dict['map_plot'] == "occupancy" and self.shared_dict['parcels']:
            possible_status = ['0 (route)','1','2','3','4','0 (reposition)','idle']
            masks = []
            color_list = OCCUPANCY_COLOR_LIST
            # route
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "route"
            condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            for i in range(1, 5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
                
            # repo
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "reposition"
            condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
        elif self.shared_dict['map_plot'] == "occupancy" and self.shared_dict['passengers']:
            possible_status = ['0 (route)','1','2','3','4','0 (reposition)','idle']
            masks = []
            color_list = OCCUPANCY_COLOR_LIST
            # route
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "route"
            condition_2 = self.shared_dict["veh_coord_status_df"]["passengers"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            for i in range(1, 5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["passengers"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
                
            # repo
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "reposition"
            condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
        elif (self.shared_dict['map_plot'] == "occupancy" 
              and not self.shared_dict['parcels'] 
              and not self.shared_dict['passengers']):
            possible_status = ['0 (route)','1','2','3','4','0 (reposition)','idle']
            masks = []
            color_list = OCCUPANCY_COLOR_LIST
            # route
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "route"
            condition_2 = self.shared_dict["veh_coord_status_df"]["pax"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            for i in range(1, 5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["pax"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
            # repo
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "reposition"
            condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
                
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
        elif self.shared_dict['map_plot'] == "vehicle_status":
            possible_status = self.shared_dict["possible_status"]
            color_list = STATUS_COLOR_LIST
            masks = []
            for status in possible_status:
                masks.append(self.shared_dict["veh_coord_status_df"]["status"] == status)
        elif self.shared_dict['map_plot'] == "zone":
            possible_status = ['-1','0','1','2','3']
            color_list = STATUS_COLOR_LIST
            vid_zone = [0,0,0,0,1,2,3,4,1,2,3,4]
            masks = []
            for status in range(len(possible_status)):
                masks.append([vid_zone[i] == status for i in range(len(vid_zone))])

        axes = self.axes

        # Plot PT line alignment
        if self.line_alignment is not None:
            axes[3].plot(self.first_part_x, self.first_part_y, color="black", linewidth=0.5, zorder=1)
            axes[3].plot(self.second_part_x, self.second_part_y, color="black", linewidth=0.5, linestyle="--", zorder=1)


        # Plot the data on the map
        ### Plot the vehicle status statistics
        ###
        axes[3].axis(self.plot_extent_3857)
        axes[3].set_xlim(self.plot_extent_3857[:2])
        axes[3].set_ylim(self.plot_extent_3857[2:])
        passengers = self.shared_dict["total_passengers"]
        parcels = self.shared_dict["total_parcels"]
        mode = "passengers" if self.shared_dict['passengers'] else "parcels"
        if not self.shared_dict['parcels'] and not self.shared_dict['passengers']:
            mode = "pax"
        # axes[3].text(0.87,0.97, f"Mode: {mode}"
        #              f"\n Number of passengers: {passengers} \n Number of parcels = {parcels} ",
        #              transform=axes[3].transAxes)
        axes[3].text(0.8, 0.97, f"\n Number of passengers: {passengers}",
                     transform=axes[3].transAxes)
        ctx.add_basemap(axes[3], source=self.bg_map_path)

        vehicle_df = self.shared_dict["veh_coord_status_df"]

        for i, mask in enumerate(masks):
            coords = vehicle_df[mask]["coordinates"].to_list()
            x, y = [], []
            if coords:
                lons, lats = list(zip(*coords))
                x, y = self.convert_lat_lon(lats, lons)
            axes[3].scatter(x, y, s=VEHICLE_POINT_SIZE, label=possible_status[i],color = color_list[i])
        axes[3].legend(loc="lower left")
        axes[3].axis('off')
        rounded_simulation_time = self.shared_dict["simulation_time"] - timedelta(microseconds=self.shared_dict["simulation_time"].microsecond)
        str_simulation_time = "Time: " + rounded_simulation_time.strftime("%H:%M:%S")
        axes[3].set_title(str_simulation_time)
        

    def save_single_plot(self, datetime_stamp: tp.Union[str, datetime]):
        if self.fig is None:
            self.fig, self.grid_spec, self.axes = self.generate_plot_axes()
        [ax.clear() for ax in self.axes]
        self.draw_plots()
        if self.plot_folder.exists() is False:
            self.plot_folder.mkdir()
        if type(datetime_stamp) == datetime:
            file_name = "plot_{}.png".format(datetime_stamp.strftime("%d-%b-%Y %H-%M-%S"))
        else:
            file_name = "plot_{}.png".format(datetime_stamp)
        plt.savefig(str(self.plot_folder.joinpath(file_name)), bbox_inches = 'tight')

    def __animate(self, i):
        [ax.clear() for ax in self.axes]
        try:
            self.draw_plots()
            if self.grid_spec is not None:
                self.grid_spec.tight_layout(self.fig)
        except KeyError:
            pass

    def run(self):
        def frame():
            i = 0
            while True:
                if self.shared_dict.get("stop", False) is False:
                    i = i + 1
                    yield i
                else:
                    return

        self.fig, self.grid_spec, self.axes = self.generate_plot_axes()
        ani = animation.FuncAnimation(self.fig, self.__animate, interval=REALTIME_UPDATE_INTERVAL)

        # TODO: This is to ignore plot tight_layout warnings
        # warnings.filterwarnings("ignore", category=UserWarning)

        plt.show()
        
    def _create_occ_stack_plot(self, axis_id):
        self.axes[axis_id ].set_title("Occupancy Stack Chart")
        self.axes[axis_id ].set_ylabel("Number Vehicles")
        list_list_values = [self._pax_counts[k] for k in ['0 (route)','1','2','3','4', '0 (reposition)','idle']]
        self.axes[axis_id ].stackplot(self._times, *list_list_values,
                                            colors=OCCUPANCY_COLOR_LIST,
                                            labels = ['0 (route)','1','2','3','4','0 (reposition)','idle' ])
        self.axes[axis_id ].legend(loc="upper left")
        self.axes[axis_id ].set_xlabel("Simulation Time [h]")
        self.axes[axis_id ].tick_params(axis="x", which="both", labelbottom=True)
        self.axes[axis_id ].tick_params(axis="y", which="both", labelleft=True)
        
        # Get the current tick positions
        xticks = self.axes[axis_id ].get_xticks()

        # Filter out ticks outside the plot limits
        s, e = self.axes[axis_id ].get_xlim()
        max_xlim = s + 0.9 * (e - s)
        filtered_xticks = [tick for tick in xticks if s <= tick <= max_xlim]

        # Set the filtered ticks
        self.axes[axis_id ].set_xticks(filtered_xticks)
        
    def _create_status_count_plot(self, axis_id):
        self.axes[axis_id ].set_title("Route Status Counts")
        self.axes[axis_id ].bar(self.shared_dict["status_counts"].keys(), 
                                    self.shared_dict["status_counts"].values(),
                                    color = STATUS_COLOR_LIST)
        self.axes[axis_id ].set_ylim(0,len(self.shared_dict["veh_coord_status_df"]))
        self.axes[axis_id ].set_xticks(list(self.shared_dict["status_counts"].keys()))
        xtick_labels = [str(x) if len(str(x)) < 5 else "\n" + str(x) for x in self.shared_dict["status_counts"].keys()]
        self.axes[axis_id ].set_xticklabels(xtick_labels, rotation=0)
        
    def _create_occ_count_plot(self, axis_id):
        self.axes[axis_id ].set_title("Occupancy Counts")
        ks = [ '0 (route)','1','2','3','4', '0 (reposition)','idle' ]
        self.axes[axis_id ].bar(ks, 
                                    [self.shared_dict["pax_info"][k] for k in ks],
                                    color = OCCUPANCY_COLOR_LIST)
        self.axes[axis_id ].set_ylim(0,len(self.shared_dict["veh_coord_status_df"]))
        self.axes[axis_id ].set_xticks(ks)
        xtick_labels = [str(x) if len(str(x)) < 5 else "\n" + str(x) for x in ks]
        self.axes[axis_id ].set_xticklabels(xtick_labels, rotation=0)
        self.axes[axis_id ].set_xlabel("Occupancy")
        self.axes[axis_id ].set_ylabel("Number of Vehicles")
        
    def _create_avg_occ_plot(self, axis_id):
        self.axes[axis_id].set_title("Average Occupancy")
        self.axes[axis_id].plot(self._times, self._avg_occ)
        self.axes[axis_id].set_xlabel("Simulation Time [h]")
        self.axes[axis_id].set_ylabel("Average Occupancy")
        #self.axes[axis_id].legend(loc="upper left")
        
    def _create_avg_wait_time_plot(self, axis_id):
        self.axes[axis_id].set_title("Average Waiting Time")
        self.axes[axis_id].set_ylabel("Waiting Time [s]")
        self.axes[axis_id].set_xlabel("Simulation Time [h]")
        self.axes[axis_id].plot(self._times, self._avg_wait_time)
        
    def _create_avg_ride_time_plot(self, axis_id):
        print("ride times", self._avg_ride_time)
        self.axes[axis_id].set_title("Average Ride Time")
        self.axes[axis_id].set_ylabel("Ride Time [s]")
        self.axes[axis_id].set_xlabel("Simulation Time [h]")
        self.axes[axis_id].plot(self._times, self._avg_ride_time)
        
    def _create_avg_detour_time_plot(self, axis_id):
        self.axes[axis_id].set_title("Average Detour Time")
        self.axes[axis_id].set_ylabel("Detour Time [s]")
        self.axes[axis_id].set_xlabel("Simulation Time [h]")
        self.axes[axis_id].plot(self._times, self._avg_detour_time)
        
    def _create_service_rate_stack_plot(self, axis_id):
        self.axes[axis_id].set_title("Requests States")
        self.axes[axis_id].set_ylabel("Number of Requests")
        self.axes[axis_id].stackplot(self._times, self._accepted_requests, self._rejected_requests,
                                            colors=["green","red"],
                                            labels = ["accepted","rejected"])
        self.axes[axis_id].legend(loc="upper left")
        self.axes[axis_id].set_xlabel("Simulation Time [h]")
        # Get the current tick positions
        xticks = self.axes[axis_id ].get_xticks()

        # Filter out ticks outside the plot limits
        s, e = self.axes[axis_id ].get_xlim()
        max_xlim = s + 0.9 * (e - s)
        filtered_xticks = [tick for tick in xticks if s <= tick <= max_xlim]

        # Set the filtered ticks
        self.axes[axis_id ].set_xticks(filtered_xticks)
