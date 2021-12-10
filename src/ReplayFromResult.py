import time
import datetime
# import geojson
# import eventlet

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from typing import Optional
from pathlib import Path
from src.python_plots.plot_classes import PyPlot
from multiprocessing import Manager
from abc import ABCMeta, abstractmethod

from src.FleetSimulationBase import build_operator_attribute_dicts
from src.misc.globals import *
PORT = 4200
EPSG_WGS = 4326

""" Eventlet should be put to sleep for around 2s. """  # should be reduced for a smoother visual experience
FRAMERATE_PER_SECOND = 0.5
PYPLOT_FRAMERATE = 30
AVAILABLE_LAYERS = ["vehicles"]  # add layers here when they are implemented


def interpolate_coordinates(row, nodes_gdf):
    """This function approximates the coordinates of a vehicle position.
    For simplicity and computational effort, it is assumed that street sections are small compared to the earth radius
    and linear interpolation has sufficient accuraccy.

    :param pos: (o_node, d_node, rel_pos) of an edge | d_node, rel_pos can be None (if vehicle is idle in o_node)
    :param nodes_gdf: GeoDataFrame with node geometry in WGS84 coordinates
    :return: lon, lat
    """
    pos = row["nw_pos"]
    p0 = nodes_gdf.loc[pos[0], "geometry"]
    p0_lon = p0.x
    p0_lat = p0.y
    if pos[1] == -1:
        return p0_lon, p0_lat
    else:
        p1 = nodes_gdf.loc[pos[1], "geometry"]
        p1_lon = p1.x
        p1_lat = p1.y
        return p0_lon + (p1_lon - p0_lon)*pos[2], p0_lat + (p1_lat - p0_lat)*pos[2]
        #return (p0_lon + p1_lon) / 2, (p0_lat + p1_lat) / 2

def interpolate_coordinates_with_edges(row, nodes_gdf, edges_gdf):
    """This function approximates the coordinates of a vehicle position.
    For simplicity and computational effort, it is assumed that street sections are small compared to the earth radius
    and linear interpolation has sufficient accuraccy.
    This function also interpolates on the edge geometries.

    :param pos: (o_node, d_node, rel_pos) of an edge | d_node, rel_pos can be None (if vehicle is idle in o_node)
    :param nodes_gdf: GeoDataFrame with node geometry in WGS84 coordinates
    :return: lon, lat
    """
    pos = row["nw_pos"]
    p0 = nodes_gdf.loc[pos[0], "geometry"]
    p0_lon = p0.x
    p0_lat = p0.y
    if pos[1] == -1:
        return p0_lon, p0_lat
    else:
        edge_geometry = edges_gdf.loc[(pos[0], pos[1]), "geometry"]
        full_length = edge_geometry.length
        next_length = 0
        current_index = 0
        current_part_len = None
        while current_index < len(edge_geometry.coords) - 1:
            prev_p = edge_geometry.coords[current_index]
            next_p = edge_geometry.coords[current_index]
            current_part_len = LineString([prev_p, next_p]).length
            next_length += current_part_len
            if next_length/full_length > pos[2]:
                break
            current_index += 1
        p0_lat, p0_lon = edge_geometry.coords[current_index]
        p1_lat, p1_lon = edge_geometry.coords[current_index + 1]
        frac_on_part = 1.0 - ( (next_length - pos[2] * full_length) / current_part_len)
        return p0_lon + (p1_lon - p0_lon)*frac_on_part, p1_lat + (p1_lat - p0_lat)*frac_on_part

def point_str_to_pos(p_str):
    p_info = p_str.split(";")
    return int(p_info[0]), int(p_info[1]), float(p_info[2])


def prep_nw_output(nw_row):
    geo = nw_row["geometry"]
    return {"type":"Point", "coordinates": [geo.x, geo.y]}


def prep_output(gdf_row):
    # TODO # change moving to new status classification ("idle","in-service","charging")
    geo = gdf_row["geometry"]
    if gdf_row["moving"]:
        moving = 1
    else:
        moving = 0
    return {"type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [geo.x, geo.y]
            },
            "properties": {
                "id": gdf_row["vid"],
                "occupancy": gdf_row["pax"],
                "soc": gdf_row["soc"],
                "moving": moving
            }}


class State:
    def __init__(self, vid_str, time, pos, end_time, end_pos, soc, pax, moving, status, trajectory_str = None):
        self.vid_str = vid_str
        self.time = time
        if type(pos) == str:
            self.pos = point_str_to_pos(pos)
        else:
            self.pos = pos
        self.soc = soc
        self.pax = pax
        self.moving = moving
        self.status = status
        self.end_time = end_time
        if type(end_pos) == str:
            self.end_pos = point_str_to_pos(end_pos)
        else:
            self.end_pos = end_pos
        self.trajectory = []
        if trajectory_str is not None and not pd.isnull(trajectory_str):
            for tmp_str in trajectory_str.split(";"):
                tmp_str2 = tmp_str.split(":")
                self.trajectory.append((int(tmp_str2[0]), float(tmp_str2[1])))

    def to_dict(self):
        # TODO # make definition of transferred attributes here
        return {"vid": self.vid_str, "nw_pos": self.pos, "soc": self.soc, "pax": self.pax, "moving": self.moving,
                "status": self.status}

    def return_state(self, replay_time):
        """This method allows a standardized way to prepare the output.

        :param replay_time: current replay time
        :return: state-dict or empty dict
        """
        if replay_time > self.end_time:
            self.pos = self.end_pos
            return None
        if not self.moving:
            return self.to_dict()
        else:
            while self.time < replay_time:
                if len(self.trajectory) != 0:
                    if self.trajectory[0][1] <= replay_time:
                        self.time = self.trajectory[0][1]
                        self.pos = (self.trajectory[0][0], -1, -1)
                        self.trajectory = self.trajectory[1:]
                        continue
                    else:
                        target = self.trajectory[0][0]
                        target_time = self.trajectory[0][1]
                        if self.pos[2] < 0:
                            cur_pos = 0.0
                        else:
                            cur_pos = self.pos[2]
                        delta_pos = (1.0 - cur_pos)/(target_time - self.time)*(replay_time - self.time)
                        self.pos = (self.pos[0], target, cur_pos + delta_pos)
                        self.time = replay_time
                else:
                    target = self.end_pos[0]
                    target_time = self.end_time
                    target_pos = self.end_pos[2]
                    if target_pos is None:
                        print("is this possible??")
                    if self.pos[2] < 0:
                        cur_pos = 0.0
                    else:
                        cur_pos = self.pos[2]
                    delta_pos = (target_pos - cur_pos)/(target_time - self.time)*(replay_time - self.time)
                    self.pos = (self.pos[0], target, cur_pos + delta_pos)
                    self.time = replay_time
            return self.to_dict()
                    
        # if replay_time == self.time:
        #     return self.to_dict()
        # else:
        #     return None


class ReplayVehicle:
    def __init__(self, op_id, vid, veh_df, start_time, end_time):
        self.op_id = op_id
        self.vid = vid
        self.vid_df = veh_df.reset_index()
        self.start_time = start_time
        self.end_time = end_time
        #
        self.active_row = 0
        self.init_pax = 0
        self.init_pos = self.vid_df.loc[0, G_VR_LEG_START_POS]
        try:
            self.init_soc = self.vid_df.loc[0, G_VR_LEG_START_SOC]
        except:
            self.init_soc = 1.0
        #
        self.last_state = State(str(self), self.start_time, self.init_pos, self.start_time, self.init_pos, self.init_soc, self.init_pax, False, "idle")

    def __str__(self):
        return f"{self.op_id}-{self.vid}"

    def get_veh_state(self, replay_time):
        """This method returns the current vehicle state.

        :param replay_time: current simulation replay time
        :return: json with state information of this vehicle
        """
        # TODO # adopt for smooth soc
        same_state = self.last_state.return_state(replay_time)
        if same_state is not None:
            return same_state
        # first check active row if simulation time is still within its boundaries
        if replay_time < self.vid_df.loc[self.active_row, G_VR_LEG_START_TIME] or\
                replay_time > self.vid_df.loc[self.active_row, G_VR_LEG_END_TIME]:
            self.vid_df["started"] = self.vid_df[G_VR_LEG_START_TIME] <= replay_time
            self.active_row = self.vid_df["started"].sum() - 1
        if self.active_row == -1:
            self.active_row = 0
            end_time = self.vid_df.loc[self.active_row, G_VR_LEG_START_TIME]
            self.last_state = State(str(self), replay_time, self.init_pos, end_time, self.init_pos, self.init_soc, self.init_pax, False, "idle")
        else:
            pax = self.vid_df.loc[self.active_row, G_VR_NR_PAX]
            route_start_time = self.vid_df.loc[self.active_row, G_VR_LEG_START_TIME]
            route_end_time = self.vid_df.loc[self.active_row, G_VR_LEG_END_TIME]
            # check status
            if route_end_time > replay_time:
                status = self.vid_df.loc[self.active_row, G_VR_STATUS]
                end_pos = self.vid_df.loc[self.active_row, G_VR_LEG_END_POS]
                # TODO # change status to "idle", "in-service", "charging" instead of "moving"
                if status in ["route", "reposition", "to_charge", "to_depot"]:
                    moving = True
                    trajectory_str = self.vid_df.loc[self.active_row, G_VR_REPLAY_ROUTE]
                else:
                    moving = False
                    trajectory_str = None
                # TODO # soc!
                start_soc = self.vid_df.loc[self.active_row, G_VR_LEG_START_SOC]
                self.last_state = State(str(self), replay_time, self.last_state.pos, route_end_time, end_pos, start_soc, pax, moving, status, trajectory_str=trajectory_str)
            else:
                status = "idle"
                moving = False
                end_time = self.end_time
                if self.active_row + 1 < self.vid_df.shape[0]:
                    end_time = self.vid_df.loc[self.active_row + 1, G_VR_LEG_START_TIME]
                self.last_state = State(str(self), replay_time, self.last_state.pos, end_time, self.last_state.pos, self.last_state.soc, 0, moving, status)
        #
        return_state = self.last_state.return_state(replay_time)
        return return_state

    def get_current_vehicle_trajectory(self, replay_time):
        """This method returns the current vehicle trajectory as geojson collection containing
        - polyline with currently assigned route
        - stops for requests

        :param replay_time: current replay time
        :return: geojson feature collection
        """
        # TODO # get_current_vehicle_trajectory()
        # TODO # use request (rq_time, pu/do) information to decide whether
        #  - to include a stop to a route or
        #  - to mark it with 'currently_unknown' flag (next stop)
        #  - to mark whether it is a drop-off/pick-up/pu&do stop
        pass


class Singleton(ABCMeta):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance


# TODO # rename base class (more layers than just vehicles)
class VehicleMovementSimulation(metaclass=Singleton):
    """
    Base class for all the simulations. Must be a singleton.
    """
    @abstractmethod
    def start(self, socket_io):
        pass

    @abstractmethod
    def step(self):
        pass

    @property
    @abstractmethod
    def started(self):
        pass


class Replay(VehicleMovementSimulation):
    def __init__(self):
        self._socket_io = None
        self._started = False
        self._sc_loaded = False
        self._inv_frame_rate = 1 / FRAMERATE_PER_SECOND
        self._time_step = None
        self._paused = False
        self._act_layer = "vehicles"
        self._layer_changed = False
        self._last_veh_state = None
        self._last_replay_time = None
        self._current_kpis = {}
        #
        self.sim_start_time = None
        self.sim_end_time = None
        self.replay_time = None
        self.nw_dir = None
        self.node_gdf = None
        self.edge_gdf = None
        self.n_op = None
        self.list_op_dicts = None
        self.poss_veh_states = []
        #
        self.steps_per_real_sec = 1
        self.replay_vehicles = {}       # (op_id, vid) -> ReplayVehicle
        self.operator_vehicles = {}     # op_id -> list_vehicles

    def load_scenario(self, output_dir, start_time_in_seconds = None, end_time_in_seconds = None):
        """This method has to be called to load the scenario data.

        :param output_dir: scenario output dir to be processed
        :param start_time_in_seconds: determines when replay is started in simulation time (None : starts with simulation start time)
        :return: None
        """
        print(f"Running replay for simulation {output_dir} ...")
        # # connection to server
        # # --------------------
        # print(f"... connecting to MobiVi server")
        # self.s = socket.socket()
        # try:
        #     self.s.connect(("localhost", PORT))
        # except:
        #     raise AssertionError("Please start visualization server by calling 'ng serve' in mobivi-front directory."
        #                          f" Check that the server is running on 'http://localhost:{PORT}/'!")

        # general settings
        # ----------------
        print(f"... loading scenario information")
        scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(output_dir)
        dir_names = get_directory_dict(scenario_parameters)
        replay_mode = scenario_parameters[G_SIM_REPLAY_FLAG]
        if not replay_mode:
            raise AssertionError("Original simulation was not saved in replay mode!")
        if start_time_in_seconds is None:
            self.sim_start_time = scenario_parameters[G_SIM_START_TIME]
        else:
            self.sim_start_time = start_time_in_seconds
        self.sim_end_time = scenario_parameters[G_SIM_END_TIME]
        if end_time_in_seconds is not None and end_time_in_seconds < self.sim_end_time:
            self.sim_end_time = end_time_in_seconds
        #
        self.replay_time = self.sim_start_time
        self._time_step = self.steps_per_real_sec * self._inv_frame_rate
        print(f"Time step: {self._time_step}")

        # load network, compute border node positions, emit network information
        # --------------------------------------------------------------
        print(f"... loading network information")
        self.nw_dir = dir_names[G_DIR_NETWORK]
        nw_base_dir = os.path.join(dir_names[G_DIR_NETWORK], "base")
        crs_f = os.path.join(nw_base_dir, "crs.info")
        node_all_info_f = os.path.join(nw_base_dir, "nodes_all_infos.geojson")
        if os.path.isfile(crs_f):
            with open(crs_f) as fh_in:
                n_crs = {"init":fh_in.read().strip()}
                n_epsg = int(n_crs["init"][5:])
            if not os.path.isfile(node_all_info_f):
                node_f = os.path.join(nw_base_dir, "nodes.csv")
                node_df = pd.read_csv(node_f, index_col=0)
                self.node_gdf = gpd.GeoDataFrame(node_df, geometry=gpd.points_from_xy(node_df["pos_x"], node_df["pos_y"]),
                                                 crs=n_crs)
            else:
                self.node_gdf = gpd.read_file(node_all_info_f)
                self.node_gdf.crs = n_crs
            if n_epsg != EPSG_WGS:
                self.node_gdf = self.node_gdf.to_crs({"init":f"epsg:{EPSG_WGS}"})
        elif os.path.isfile(node_all_info_f):
            self.node_gdf = gpd.read_file(node_all_info_f)
            if self.node_gdf.crs != f"epsg:{EPSG_WGS}":
                self.node_gdf = self.node_gdf.to_crs({"init":f"epsg:{EPSG_WGS}"})
            # check that units are correct
            if self.node_gdf["geometry"].x.max() > 180 or self.node_gdf["geometry"].x.max() < -180:
                raise AssertionError("GeoJSON format assumes WGS input format!")
            if self.node_gdf["geometry"].y.max() > 90 or self.node_gdf["geometry"].y.max() < -90:
                raise AssertionError("GeoJSON format assumes WGS input format!")
        else:
            raise AssertionError(f"Neither {crs_f} or {node_all_info_f} were found! -> Insufficient GIS information.")

        # load zone system if available
        # -----------------------------
        # TODO # get spatial information of zone system if available (and transform to WGS coordinates)

        # load vehicle trajectories, prepare replay mode
        # ----------------------------------------------
        print("... processing vehicle data")
        self.n_op = scenario_parameters[G_NR_OPERATORS]
        self.list_op_dicts = build_operator_attribute_dicts(scenario_parameters, self.n_op, prefix="op_")
        for op_id in range(self.n_op):
            fleet_stat_f = os.path.join(output_dir, f"2-{op_id}_op-stats.csv")
            fleet_stat_df = pd.read_csv(fleet_stat_f)
            fleet_stat_df = fleet_stat_df[fleet_stat_df["start_time"] >= self.sim_start_time]
            possible_status = fleet_stat_df["status"].unique()
            self.poss_veh_states = list(set(self.poss_veh_states).union(possible_status)) + ["idle"]
            for vid, veh_stat_f in fleet_stat_df.groupby(G_RQ_VID):
                self.replay_vehicles[(op_id, vid)] = ReplayVehicle(op_id, vid, veh_stat_f, self.sim_start_time,
                                                                   self.sim_end_time)
        states_codes = {status.display_name: status.value for status in VRL_STATES}
        self.poss_veh_states = sorted(self.poss_veh_states, key=lambda x: states_codes[x])
        print("... processing user data")
        # TODO # load vehicle data

        print(" ... initiation successful")
        self._sc_loaded = True

    @property
    def started(self):
        return self._started

    def start(self, socket_io):
        self._socket_io = socket_io
        if not self._started:
            if self._sc_loaded:
                self._started = True
                self._socket_io.start_background_task(self.run())
            else:
                raise IOError("No scenario was loaded!")

    def run(self):
        """This is the typical run method to replay a video."""
        print("#"*40)
        print(" ... running replay simulation")
        self.replay_time = self.sim_start_time
        while self.replay_time <= self.sim_end_time:
            print(self.replay_time)
            if self.replay_time % 600 == 0:
                print(f"simulation time: {self.replay_time}/{self.sim_end_time}")
            self.step()
            eventlet.sleep(self._inv_frame_rate)

    def step(self):
        self._emit_current_information()
        self._last_replay_time = self.replay_time
        self.replay_time += self._time_step

    def pause(self):
        """
        Notify the main thread so it could close the websocket.
        :return:
        """
        # TODO !
        pass

    def set_sim_time(self, new_time):
        """This method directly sets the simulation time to a given time and sends the vehicle information to the
        socket.

        :param new_time: new simulation time
        """
        self._last_replay_time = self.replay_time
        self.replay_time = new_time
        self._emit_current_information()

    def set_time_step(self, sim_seconds_per_real_second):
        """This method could be called by an UI functionality to increase or reduce the speed of the simulation.

        :param sim_seconds_per_real_second: new time step.
        """
        self._time_step = sim_seconds_per_real_second * self._inv_frame_rate

    def reset(self):
        """This method resets the vehicle positions to its initial state."""
        self.set_sim_time(self.sim_start_time)

    def set_active_layer(self, layer_name):
        if layer_name in AVAILABLE_LAYERS:
            self._act_layer = layer_name
            self._layer_changed = True
        else:
            raise AssertionError(f"Layer {layer_name} not implemented!")

    def get_gj_network_info(self):
        """This method prepares the GeoDataFrame that will be sent to MobiVi in the initialization."""
        # tmp_df = self.node_gdf.copy()
        # tmp_df = self.node_gdf[self.node_gdf[G_NODE_STOP_ONLY]]
        list_indices = []
        list_indices.append(self.node_gdf["geometry"].x.idxmin())
        list_indices.append(self.node_gdf["geometry"].x.idxmax())
        list_indices.append(self.node_gdf["geometry"].y.idxmin())
        list_indices.append(self.node_gdf["geometry"].y.idxmax())
        tmp_df = self.node_gdf[self.node_gdf.index.isin(list_indices)]
        tmp = tmp_df.apply(prep_nw_output, axis=1)
        tmp_list = tmp.to_list()
        network_feature_collection = geojson.dumps(geojson.FeatureCollection(tmp_list))
        return network_feature_collection

    def _emit_current_information(self):
        """This method prepares and sends information to MobiVi-front."""
        if self.replay_time != self._last_replay_time:
            # TODO # update and send self._current_kpis for current replay time with keyword 'KPI'
            pass
        if self._act_layer == "vehicles":
            list_pos_df = pd.DataFrame([veh.get_veh_state(self.replay_time) for veh in self.replay_vehicles.values()])
            # TODO # compare with self._last_veh_state to filter for vehicles with changes (unless self._layer_changed)
            list_pos_df[["lon", "lat"]] = list_pos_df.apply(interpolate_coordinates, args=(self.node_gdf, ), axis=1,
                                                            result_type="expand")
            gdf = gpd.GeoDataFrame(list_pos_df, geometry=gpd.points_from_xy(list_pos_df["lon"], list_pos_df["lat"]),
                                   crs=self.node_gdf.crs)
            output_list = [prep_output(row) for _, row in gdf.iterrows()]
            msg = geojson.dumps(geojson.FeatureCollection(output_list))
            print(msg)
            self._socket_io.emit("vehicles", msg, namespace="/dd")
        if self._act_layer == "act_rq":
            # TODO # return list of currently active requests (rq_time <= replay_time <= drop-off time)
            # divide into in-vehicle and waiting requests
            pass
        if self._act_layer == "unserved_rq":
            # TODO # collect and return list of requests that have not been served in replay_time - X
            # aggregate on node-level
            pass


class ReplayPyPlot(Replay):
    def __init__(self, live_plot: bool = True, create_images: bool = True):
        """ Class for python based visualization of the simulation results

        :param live_plot: If True, the plots are displayed in real time using a separate CPU process. If False,
                            the each of the frame is saved in a folder called "plots" inside the results folders.
        :param create_images: only significant if live_plot is False. If both False, only csv-files of the vehicle locations
                            are created and no images
        """
        super().__init__()
        self._inv_frame_rate = 1 / PYPLOT_FRAMERATE
        # TODO # read date from scenario name if available
        today = datetime.datetime.today()
        self.dtuple = (today.year, today.month, today.day)
        self.live_plot = live_plot
        self.create_images = create_images
        self._shared_dict: dict = {}
        self._manager: Optional[Manager] = None
        self.plots_dir: Optional[Path] = None
        self._plot_class_instance: Optional[PyPlot] = None
        self._map_extent = None

    def load_scenario(self, output_dir, start_time_in_seconds = None, end_time_in_seconds = None, plot_extend=None):
        super().load_scenario(output_dir, start_time_in_seconds=start_time_in_seconds, end_time_in_seconds=end_time_in_seconds)
        self.plots_dir = Path(output_dir).joinpath("plots")
        if plot_extend is None:
            bounds = self.node_gdf.bounds
            self._map_extent = (bounds.minx.min(), bounds.maxx.max(), bounds.miny.min(), bounds.maxy.max())
        else:
            self._map_extent = plot_extend

    def start(self, socket_io=None):
        self._socket_io = None
        if self.live_plot is True:
            self._manager = Manager()
            self._shared_dict = self._manager.dict()
            self._plot_class_instance = PyPlot(self.nw_dir, self._shared_dict, plot_extent=self._map_extent)
            self._plot_class_instance.start()
        else:
            self._plot_class_instance = PyPlot(self.nw_dir, self._shared_dict, str(self.plots_dir), self._map_extent)
        if not self._started:
            if self._sc_loaded:
                self._started = True
                self.run()
            else:
                raise IOError("No scenario was loaded!")

    def run(self):
        """This is the typical run method to replay a video."""
        print("#"*40)
        print(" ... running replay simulation")
        self.replay_time = self.sim_start_time
        while self.replay_time <= self.sim_end_time:
            print(self.replay_time)
            if self.replay_time % 600 == 0:
                print(f"simulation time: {self.replay_time}/{self.sim_end_time}")
            self.step()
            time.sleep(self._inv_frame_rate)
        self._end_plotting()

    def _end_plotting(self):
        self._shared_dict["stop"] = True
        if self.live_plot is True:
            self._plot_class_instance.join()
            self._manager.shutdown()

    def _emit_current_information(self):
        list_pos_df = pd.DataFrame([veh.get_veh_state(self.replay_time) for veh in self.replay_vehicles.values()])
        list_pos_df["coordinates"] = list_pos_df.apply(interpolate_coordinates, args=(self.node_gdf,), axis=1,
                                                       result_type="reduce")
        list_pos_df.set_index("vid", inplace=True)
        #
        sim_time = datetime.datetime(self.dtuple[0], self.dtuple[1], self.dtuple[2], 0,0,0) + \
                   datetime.timedelta(seconds=self.replay_time)
        # TODO # add dictionary for additional geographic information: key -> list_of_coordinates
        dict_add_coord = {}
        # TODO # add dictionary for additional scalar information: key -> value
        dict_add_values = {}
        info_dict = {"simulation_time": sim_time,
                     "veh_coord_status_df": list_pos_df,
                     "possible_status": self.poss_veh_states,
                     "additional_coordinates_dict": dict_add_coord,
                     "additional_values_dict": dict_add_values,
                     "stop": False}
        # Share the updated information dictionary with the plot class
        self._shared_dict.update(info_dict)
        if self.live_plot is False:
            if self.create_images:
                self._plot_class_instance.save_single_plot(sim_time)
            if not os.path.isdir(self.plots_dir):
                os.mkdir(self.plots_dir)
            list_pos_df.to_csv(os.path.join(self.plots_dir, "data_points_{}.csv".format(self.replay_time)))
