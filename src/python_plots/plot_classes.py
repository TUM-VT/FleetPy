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
from datetime import datetime


FIG_SIZE = (15,10)
# Number of historical points to be displayed on the x-axis
PLOT_LENGTH = 200
# Delay between frames in milliseconds
REALTIME_UPDATE_INTERVAL = 200
VEHICLE_POINT_SIZE = 12
CTX_PROVIDER = ctx.providers.CartoDB.Positron# ctx.providers.Stamen.TonerLite
BOARDER_SIZE = 1000


class PyPlot(Process):

    def __init__(self, nw_dir, shared_dict: dict, plot_folder: str = None, plot_extent=None):
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
        self.plot_extent_3857 = bbox
        #exit()

    def convert_lat_lon(self, lats: list, lons: list, to_epsg: str = "epsg:3857"):
        proj_transformer = Transformer.from_proj('epsg:4326', to_epsg)
        x, y = proj_transformer.transform(lats, lons)
        return list(x), list(y)

    def generate_plot_axes(self):
        fig = plt.figure(1, figsize=FIG_SIZE)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        gs.update(wspace=0.025, hspace=0.5)
        axes = [plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2]), plt.subplot(gs[:, 0:2])]
        return fig, gs, axes

    def draw_plots(self):

        #color_list = ['blue','orange','green','red','purple','beige']
        if self.shared_dict['color_list']:
            color_list = stack_chart_color_list = self.shared_dict['color_list']
        else:
            stack_chart_color_list = ["lightgrey","red","blue","orange","green","dimgrey"]
            color_list = stack_chart_color_list
        reveresed_stack_chart_color_list = stack_chart_color_list[1:][::-1]
        stack_chart_color_list.reverse()
        reveresed_stack_chart_color_list.insert(0,"white")
        if self.shared_dict['status_type'] == 2 and self.shared_dict['parcels']:
            possible_status = ['idle','0','1','2','3','4']
            masks = []
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
            color_list = reveresed_stack_chart_color_list
            for i in range(5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
        elif self.shared_dict['status_type'] == 2 and self.shared_dict['passengers']:
            possible_status = ['idle','0','1','2','3','4']
            masks = []
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
            color_list = reveresed_stack_chart_color_list
            for i in range(5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["passengers"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
        elif (self.shared_dict['status_type'] == 2 
              and not self.shared_dict['parcels'] 
              and not self.shared_dict['passengers']):
            possible_status = ['idle','0','1','2','3','4']
            masks = []
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
            color_list = reveresed_stack_chart_color_list
            for i in range(5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["pax"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
        elif self.shared_dict['status_type'] == 1:
            possible_status = self.shared_dict["possible_status"]
            masks = []
            for status in possible_status:
                masks.append(self.shared_dict["veh_coord_status_df"]["status"] == status)

        axes = self.axes
        # Plot the data on the map
        ### Plot the vehicle status statistics
        available_plots = ['status_count','occupancy','occupancy_stack_chart','avg_wait_time','avg_ride_time','avg_detour_time']
        plots = [p for i,p in enumerate(available_plots) if self.shared_dict['plot_args'][i] == 1]
        plot_slots = [0,1,2]
        time_line = self.shared_dict["time_line"]
        if "status_count" in plots:
            status_count_idx = plot_slots.pop(0)
            axes[status_count_idx].set_title("status_count")
            axes[status_count_idx].bar(self.shared_dict["status_counts"].keys(), 
                                        self.shared_dict["status_counts"].values(),
                                        color = color_list)
            axes[status_count_idx].set_ylim(0,len(self.shared_dict["veh_coord_status_df"]))
            axes[status_count_idx].set_xticks(list(self.shared_dict["status_counts"].keys()))
            axes[status_count_idx].set_xticklabels(self.shared_dict["status_counts"].keys(), rotation=45)
        if "occupancy" in plots:
            occupancy_idx = plot_slots.pop(0)
            axes[occupancy_idx].set_title("Occupancy")
            axes[occupancy_idx].plot(time_line, self.shared_dict["pax_list"])
            axes[occupancy_idx].set_ylim(0, 5)
            #axes[occupancy_idx].legend(loc="upper left")
        if "occupancy_stack_chart" in plots:
            occupancy_stack_chart_idx = plot_slots.pop(0)
            axes[occupancy_stack_chart_idx ].set_title("Occupancy Stack Chart")
            axes[occupancy_stack_chart_idx ].set_ylabel("Number Vehicles")
            list_values = []
            list_values = [
                list(self.shared_dict["pax_info"][-1]),
                list(self.shared_dict["pax_info"][0]),
                list(self.shared_dict["pax_info"][1]),
                list(self.shared_dict["pax_info"][2]),
                list(self.shared_dict["pax_info"][3]),
                list(self.shared_dict["pax_info"][4])
            ]
            same_length = True
            le = None
            for l in list_values:
                if le is None:
                    le = len(l)
                elif le != len(l):
                    same_length = False
                    break
            if same_length:
                pass
            else:
                print("Error in plotting occupancy stack chart")
                print([len(l) for l in list_values])
                min_len = min([len(l) for l in list_values])
                for i, l in enumerate(list_values):
                    l = l[:min_len]
                    list_values[i] = l
            #print(len(time_line), le)
            if len(time_line) < le:
                for i, l in enumerate(list_values):
                    l = l[:len(time_line)]
                    list_values[i] = l
            elif le < len(time_line):
                time_line = time_line[:le]
            axes[occupancy_stack_chart_idx ].stackplot(time_line, list_values[1],
                                              list_values[2], list_values[3],
                                              list_values[4], list_values[5],
                                                list_values[0],
                                                colors=stack_chart_color_list,
                                                labels = ["0","1","2","3","4","idle"])
            axes[occupancy_stack_chart_idx ].legend(loc="upper left")
            axes[occupancy_stack_chart_idx ].set_xlabel("Simulation Time [h]")
        if "avg_wait_time" in plots:
            avg_wait_time_idx = plot_slots.pop(0)
            axes[avg_wait_time_idx].set_title("Average Waiting Time")
            axes[avg_wait_time_idx].set_ylabel("Waiting Time [s]")
            axes[avg_wait_time_idx].set_xlabel("Simulation Time [h]")
            axes[avg_wait_time_idx].plot(self.shared_dict["avg_wait_time"])
        if "avg_ride_time" in plots:
            avg_ride_time_idx = plot_slots.pop(0)
            axes[avg_ride_time_idx].set_title("Average Ride Time")
            axes[avg_ride_time_idx].set_ylabel("Ride Time [s]")
            axes[avg_ride_time_idx].set_xlabel("Simulation Time [h]")
            axes[avg_ride_time_idx].plot(self.shared_dict["avg_ride_time"])
        if "avg_detour_time" in plots:
            avg_detour_time_idx = plot_slots.pop(0)
            axes[avg_detour_time_idx].set_title("Average Detour Time")
            axes[avg_detour_time_idx].set_ylabel("Detour Time [s]")
            axes[avg_detour_time_idx].set_xlabel("Simulation Time [h]")
            axes[avg_detour_time_idx].plot(self.shared_dict["avg_detour_time"])
        ###
        axes[3].axis(self.plot_extent_3857)
        axes[3].set_xlim(self.plot_extent_3857[:2])
        axes[3].set_ylim(self.plot_extent_3857[2:])
        passengers = self.shared_dict["total_passengers"]
        parcels = self.shared_dict["total_parcels"]
        mode = "passengers" if self.shared_dict['passengers'] else "parcels"
        if not self.shared_dict['parcels'] and not self.shared_dict['passengers']:
            mode = "pax"
        axes[3].text(0.87,0.97, f"Mode: {mode}"
                     f"\n Number of passengers: {passengers} \n Number of parcels = {parcels} ",
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
        axes[3].legend(loc="upper left")
        axes[3].axis('off')
        axes[3].set_title(str(self.shared_dict["simulation_time"]))

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
        plt.show()
