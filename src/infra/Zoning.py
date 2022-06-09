# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

# src imports
# -----------

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
# from src.fleetctrl.FleetControlBase import PlanRequest # TODO # circular dependency!
# set log level to logging.DEBUG or logging.INFO for single simulations
LOG_LEVEL = logging.WARNING
LOG = logging.getLogger(__name__)
NOON = 12*3600


# -------------------------------------------------------------------------------------------------------------------- #
# main class
# ----------
class ZoneSystem:
    # TODO # scenario par now input! add in fleetsimulations
    def __init__(self, zone_network_dir, scenario_parameters, dir_names):
        # general information
        self.zone_general_dir = os.path.dirname(zone_network_dir)
        self.zone_system_name = os.path.basename(self.zone_general_dir)
        self.zone_network_dir = zone_network_dir
        general_info_f = os.path.join(self.zone_general_dir, "general_information.csv")
        try:
            self.general_info_df = pd.read_csv(general_info_f, index_col=0)
        except:
            self.general_info_df = None
        # network specific information
        node_zone_f = os.path.join(zone_network_dir, "node_zone_info.csv")
        self.node_zone_df = pd.read_csv(node_zone_f)
        # pre-process some data
        self.zones = sorted(self.node_zone_df[G_ZONE_ZID].unique().tolist()) # TODO
        if self.general_info_df is not None:
            self.all_zones = self.general_info_df.index.to_list()
        else:
            self.all_zones = list(self.zones)
        if self.general_info_df is not None:
            self.node_zone_df = pd.merge(self.node_zone_df, self.general_info_df, left_on=G_ZONE_ZID, right_on=G_ZONE_ZID)
        self.node_zone_df.set_index(G_ZONE_NID, inplace=True)
        self.zone_centroids = None # zone_id -> list node_indices (centroid not unique!)
        if G_ZONE_CEN in self.node_zone_df.columns:
            self.zone_centroids = {}
            for node_id, zone_id in self.node_zone_df[self.node_zone_df[G_ZONE_CEN] == 1][G_ZONE_ZID].items():
                try:
                    self.zone_centroids[zone_id].append(node_id)
                except KeyError:
                    self.zone_centroids[zone_id] = [node_id]
        # # edge specific information -> not necessary at the moment
        # edge_zone_f = os.path.join(zone_network_dir, "edge_zone_info.csv")
        # self.edge_zone_df = pd.read_csv(edge_zone_f)
        self.current_toll_cost_scale = 0
        self.current_toll_coefficients = {}
        self.current_park_costs = {}
        self.current_park_search_durations = {}
        # reading zone-correlation matrix if available
        if scenario_parameters.get(G_ZONE_CORR_M_F):
            # load correlation matrix files; these are saved as sparse matrices by scipy module
            # important: name of squared matrix depends on linear correlation matrix
            tmp_k_f = os.path.join(self.zone_general_dir, scenario_parameters[G_ZONE_CORR_M_F])
            tmp_k2_f = tmp_k_f.replace("zone_to_zone_correlations", "zone_to_zone_squared_correlations")
            if not os.path.isfile(tmp_k_f) or not os.path.isfile(tmp_k2_f):
                raise IOError(f"Could not find zone-to-zone correlation files {tmp_k_f} or {tmp_k2_f}!")
            self.zone_corr_matrix = load_npz(tmp_k_f).todense()
            self.zone_sq_corr_matrix = load_npz(tmp_k2_f).todense()
        else:
            self.zone_corr_matrix = np.eye(len(self.zones))
            self.zone_sq_corr_matrix = np.eye(len(self.zones))
        # read forecast files
        if scenario_parameters.get(G_FC_FNAME) and scenario_parameters.get(G_FC_TYPE):
            fc_dir = dir_names.get(G_DIR_FC)
            self.fc_temp_resolution = int(os.path.basename(fc_dir))
            forecast_f = os.path.join(fc_dir, scenario_parameters.get(G_FC_FNAME))
            if os.path.isfile(forecast_f):
                fc_type = scenario_parameters.get(G_FC_TYPE)
                self.forecast_df = pd.read_csv(forecast_f)
                self.fc_times = sorted(self.forecast_df[G_ZONE_FC_T].unique())
                self.forecast_df.set_index([G_ZONE_FC_T, G_ZONE_ZID], inplace=True)
                self.in_fc_type = f"in {fc_type}"
                self.out_fc_type = f"out {fc_type}"
                if self.in_fc_type not in self.forecast_df.columns or self.out_fc_type not in self.forecast_df.columns:
                    raise IOError(f"Could not find forecast data for {fc_type} in {forecast_f}")
                drop_columns = []
                for col in self.forecast_df.columns:
                    if col != self.in_fc_type and col != self.out_fc_type:
                        drop_columns.append(col)
                self.forecast_df.drop(drop_columns, axis=1, inplace=True)
            else:
                raise IOError(f"Could not find forecast file {forecast_f}")
        else:
            self.forecast_df = None
            self.in_fc_type = None
            self.out_fc_type = None
            self.fc_times = []
            self.fc_temp_resolution = None
        self.demand = None

    def register_demand_ref(self, demand_ref):
        self.demand = demand_ref

    def get_zone_system_name(self):
        return self.zone_system_name

    def get_zone_dirs(self):
        return self.zone_general_dir, self.zone_network_dir

    def get_all_zones(self):
        """This method returns a list of zone_ids that have a node in them

        :return: zone_ids
        :rtype: list
        """
        return self.zones

    def get_complete_zone_list(self):
        """This method returns a list of all zone_ids.

        :return: zone_ids
        :rtype: list
        """
        return self.all_zones

    def get_all_nodes_in_zone(self, zone_id):
        """This method returns all nodes within a zone. This can be used as a search criterion.

        :param zone_id: id of the zone in question
        :type zone_id: int
        :return: list of node_ids
        :rtype: list
        """
        tmp_df = self.node_zone_df[self.node_zone_df[G_ZONE_ZID] == zone_id]
        return tmp_df.index.values.tolist()

    def get_random_node(self, zone_id):
        """This method returns a random node_id for a given zone_id.

        :param zone_id: id of the zone in question
        :type zone_id: int
        :return: node_id of a node within the zone in question; return -1 if invalid zone_id is given
        :rtype: int
        """
        tmp_df = self.node_zone_df[self.node_zone_df[G_ZONE_ZID] == zone_id]
        if len(tmp_df) > 0:
            return np.random.choice(tmp_df.index.values.tolist())
        else:
            return -1

    def get_random_centroid_node(self, zone_id):
        if self.zone_centroids is not None:
            nodes = self.zone_centroids.get(zone_id, [])
            if len(nodes) > 0:
                return np.random.choice(nodes)
            else:
                return -1
        else:
            raise EnvironmentError("No zone centroid nodes defined! ({} parameter not in"
                                   " node_zone_info.csv!)".format(G_ZONE_CEN))

    def get_zone_from_node(self, node_id):
        """This method returns the zone_id of a given node_id.

        :param node_id: id of node in question
        :type node_id: int
        :return: zone_id of the node in question; return -1 if no zone is found
        :rtype: int
        """
        return self.node_zone_df[G_ZONE_ZID].get(node_id, -1)

    def get_zone_from_pos(self, pos):
        """This method returns the zone_id of a given position by returning the zone of the origin node.

        :param pos: position-tuple (edge origin node, edge destination node, edge relative position)
        :type pos: list
        :return: zone_id of the node in question; return -1 if no zone is found
        :rtype: int
        """
        return self.get_zone_from_node(pos[0])

    def check_first_last_mile_option(self, o_node, d_node):
        """This method checks whether first/last mile service should be offered in a given zone.

        :param o_node: node_id of a trip's start location
        :type o_node: int
        :param d_node: node_id of a trip's end location
        :type d_node: int
        :return: True/False
        :rtype: bool
        """
        if G_ZONE_FLM in self.node_zone_df.columns:
            mod_access = self.node_zone_df[G_ZONE_FLM].get(o_node, True)
            mod_egress = self.node_zone_df[G_ZONE_FLM].get(d_node, True)
        else:
            mod_access = True
            mod_egress = True
        return mod_access, mod_egress

    def set_current_park_costs(self, general_park_cost=0, park_cost_dict={}):
        """This method sets the current park costs in cent per region per hour.

        :param general_park_cost: this is a scale factor that is multiplied by each zones park_cost_scale_factor.
        :type general_park_cost: float
        :param park_cost_dict: sets the park costs per zone directly. Code prioritizes input over general_park_cost.
        :type park_cost_dict: dict
        """
        if park_cost_dict:
            for k,v in park_cost_dict.items():
                if k in self.general_info_df.index:
                    self.current_park_costs[k] = v
        else:
            for k, zone_scale_factor in self.general_info_df[G_ZONE_PC].items():
                self.current_park_costs[k] = general_park_cost * zone_scale_factor

    def set_current_toll_cost_scale_factor(self, general_toll_cost):
        self.current_toll_cost_scale = general_toll_cost

    def set_current_toll_costs(self, use_pre_defined_zone_scales=False, rel_toll_cost_dict={}):
        """This method sets the current toll costs in cent per meter.

        :param use_pre_defined_zone_scales: use each zones toll_cost_scale_factor of zone definition.
        :type use_pre_defined_zone_scales: bool
        :param rel_toll_cost_dict: sets the toll costs per zone directly. Code prioritizes input over general_toll_cost.
        :type rel_toll_cost_dict: dict
        """
        if rel_toll_cost_dict and self.current_toll_cost_scale > 0:
            for k,v in rel_toll_cost_dict.items():
                if k in self.general_info_df.index:
                    self.current_toll_coefficients[k] = self.current_toll_cost_scale * v
        elif use_pre_defined_zone_scales and self.current_toll_cost_scale > 0:
            for k, zone_scale_factor in self.general_info_df[G_ZONE_TC].items():
                self.current_toll_coefficients[k] = self.current_toll_cost_scale * zone_scale_factor

    def get_external_route_costs(self, routing_engine, sim_time, route, park_origin=True, park_destination=True):
        """This method returns the external costs of a route, namely toll and park costs. Model simplifications:
        1) Model assumes a trip-based model, in which duration of activity is unknown. For this reason, park costs
        are assigned to a trip depending on their destination (trip start in the morning) or the origin (trip starts
        in the afternoon).
        2) Toll costs are computed for the current point in time. No extrapolation for the actual route time is
        performed.

        :param routing_engine: network and routing class
        :type routing_engine: Network
        :param sim_time: relevant for park costs - am: destination relevant; pm: origin relevant
        :type sim_time: float
        :param route: list of node ids that a vehicle drives along
        :type route: list
        :param park_origin: flag showing whether vehicle could generate parking costs at origin
        :type park_origin: bool
        :param park_destination: flag showing whether vehicle could generate parking costs at destination
        :type park_destination: bool
        :return: tuple of total external costs, toll costs, parking costs in cent
        :rtype: list
        """
        park_costs = 0
        toll_costs = 0
        if route:
            # 1) park cost model
            if sim_time < NOON:
                if park_destination:
                    # assume 1 hour of parking in order to return the set park cost values (current value!)
                    d_zone = self.get_zone_from_node(route[-1])
                    park_costs += self.current_park_costs.get(d_zone, 0)
            else:
                if park_origin:
                    # assume 1 hour of parking in order to return the set park cost values (current value!)
                    o_zone = self.get_zone_from_node(route[0])
                    park_costs += self.current_park_costs.get(o_zone, 0)
            # 2) toll model
            for i in range(len(route)-1):
                o_node = route[i]
                d_node = route[i+1]
                zone = self.get_zone_from_node(o_node)
                length = routing_engine.get_section_infos(o_node, d_node)[1]
                toll_costs += np.rint(self.current_toll_coefficients.get(zone, 0) * length)
        external_pv_costs = park_costs + toll_costs
        return external_pv_costs, toll_costs, park_costs

    def _get_trip_forecasts(self, trip_type, t0, t1, aggregation_level, scale = None):
        """This method returns the number of expected trip arrivals or departures inside a zone in the
        time interval [t0, t1]. The return value is created by interpolation of the forecasts in the data frame
        if necessary. The default if no values can be found for a zone should be 0.

        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :param scale: scales forecast distributen by this value if given
        :type scale: float
        :return: {}: zone -> forecast of arrivals
        :rtype: dict
        """

        if trip_type == "in":
            col = self.in_fc_type
        elif trip_type == "out":
            col = self.out_fc_type
        else:
            raise AssertionError("Invalid forecast column chosen!")
        #
        if aggregation_level is not None:
            tmp_forecast_df = self.forecast_df.reset_index().groubpy([G_ZONE_FC_T,
                                                                      aggregation_level]).aggregate({col: sum})
        else:
            tmp_forecast_df = self.forecast_df
        #
        def _create_forecast_dict(tmp_col, row_index, tmp_return_dict, tmp_scale_factor=1.0):
            # LOG.info(f"{self.forecast_df}")
            # LOG.info(f"{tmp_forecast_df}")
            # LOG.info(f"{row_index} | {G_ZONE_FC_T}")
            try:
                tmp_df = tmp_forecast_df.xs(row_index, level=G_ZONE_FC_T)
            except:
                LOG.info("couldnt find forecast for t {}".format(row_index))
                return {}
            tmp_dict = tmp_df[tmp_col].to_dict()
            for k, v in tmp_dict.items():
                try:
                    tmp_return_dict[k] += (v * tmp_scale_factor)
                except KeyError:
                    tmp_return_dict[k] = (v * tmp_scale_factor)
            return tmp_return_dict
        #
        return_dict = {}
        # get forecast of initial time interval
        last_t0 = t0
        if t0 not in self.fc_times:
            # check whether t0 and t1 are valid times
            if t0 > self.fc_times[-1] or t1 < self.fc_times[0]:
                # use first/last forecast and scale
                if t1 > self.fc_times[0]:
                    last_t0 = self.fc_times[0]
                else:
                    last_t0 = self.fc_times[-1]
                scale_factor = (t1 - t0) / self.fc_temp_resolution
                return_dict = _create_forecast_dict(col, last_t0, return_dict, scale_factor)
                # if scale is not None:
                #     for key, val in return_dict.items():
                #         return_dict[key] = val * scale
                return return_dict
            else:
                # get forecast from t0 to next value in self.fc_times
                for i in range(len(self.fc_times)):
                    # last_t0 = self.fc_times[i]
                    # next_t0 = self.fc_times[i+1]
                    next_t0 = self.fc_times[i]
                    if next_t0 > t1:
                        if last_t0 == t0:
                            scale_factor = (t1 - t0) / self.fc_temp_resolution
                            return_dict = _create_forecast_dict(col, self.fc_times[i-1], return_dict, scale_factor)
                            return return_dict
                        break
                    if last_t0 <= t0 and t0 < next_t0:
                        scale_factor = (next_t0 - last_t0) / self.fc_temp_resolution
                        # scale down the values
                        return_dict = _create_forecast_dict(col, next_t0, return_dict, scale_factor)
                        last_t0 = next_t0
                        break
        # add forecasts of next intervals as well
        while t1 - last_t0 > self.fc_temp_resolution:
            return_dict = _create_forecast_dict(col, last_t0, return_dict)
            last_t0 += self.fc_temp_resolution
            if last_t0 not in self.fc_times:
                break
        # append rest of last interval
        if t1 != last_t0:
            scale_factor = (t1 - last_t0) / self.fc_temp_resolution
            return_dict = _create_forecast_dict(col, last_t0, return_dict, scale_factor)
            if scale is not None:
                for key, val in return_dict.items():
                    return_dict[key] = val * scale
        return return_dict

    def get_trip_arrival_forecasts(self, t0, t1, aggregation_level=None, scale = None):
        """This method returns the number of expected trip arrivals inside a zone in the time interval [t0, t1].
        The return value is created by interpolation of the forecasts in the data frame if necessary.
        The default if no values can be found for a zone should be 0.

        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :param scale: scales forecast distributen by this value if given
        :type scale: float
        :return: {}: zone -> forecast of arrivals
        :rtype: dict
        """
        if self.in_fc_type is None:
            raise AssertionError("get_trip_arrival_forecasts() called even though no forecasts are available!")
        return self._get_trip_forecasts("in", t0, t1, aggregation_level, scale = scale)

    def get_trip_departure_forecasts(self, t0, t1, aggregation_level=None, scale = None):
        """This method returns the number of expected trip departures inside a zone in the time interval [t0, t1].

        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :type aggregation_level: int
        :param scale: scales forecast distributen by this value if given
        :type scale: float
        :return: {}: zone -> forecast of departures
        :rtype: dict
        """
        if self.out_fc_type is None:
            raise AssertionError("get_trip_departure_forecasts() called even though no forecasts are available!")
        return self._get_trip_forecasts("out", t0, t1, aggregation_level, scale = scale)

    def get_zone_correlation_matrix(self):
        """This method returns the zone correlation matrix for a given bandwidth (see PhD thesis of Flo) for further
        details.

        :return: N_z x N_z numpy matrix, where N_z is the number of forecast zones
        """
        return self.zone_corr_matrix

    def get_squared_correlation_matrix(self):
        """This method returns the squared zone correlation matrix for a given bandwidth (see RFFR Frontiers paper of
        Arslan and Flo or PhD thesis of Flo) for further details.

        :return: N_z x N_z numpy matrix, where N_z is the number of forecast zones
        """
        return self.zone_sq_corr_matrix

    def get_centroid_node(self, zone_id):
        # TODO # after ISTTT: get_centroid_node()
        pass

    def get_parking_average_access_egress_times(self, o_node, d_node):
        # TODO # after ISTTT: get_parking_average_access_egress_times()
        t_access = 0
        t_egress = 0
        return t_access, t_egress

    def get_cordon_sections(self):
        # TODO # after ISTTT: get_cordon_sections()
        pass

    def get_aggregation_levels(self):
        """This method returns a dictionary of

        :return:
        """
        # TODO # after ISTTT: get_aggregation_levels()
        # is this necessary?
        pass

    def draw_future_request_sample(self, t0, t1, request_attribute = None, attribute_value = None, scale = None): #request_type=PlanRequest # TODO # cant import PlanRequest because of circular dependency of files!
        """ this function returns future request attributes drawn probabilistically from the forecast method for the intervall [t0, t1]
        currently origin is drawn from get_trip_departure_forecasts an destination is drawn form get_trip_arrival_forecast (independently! # TODO #)
        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param request_attribute: (not for this class) name of the attribute of the request class. if given, only returns requests with this attribute
        :type request_attribute: str
        :param attribute_value: (not for this class) if and request_attribute given: only returns future requests with this attribute value
        :type attribute_value: type(request_attribute)
        :param scale: scales forecast distribution by this values
        :type scale: float
        :return: list of (time, origin_node, destination_node) of future requests
        :rtype: list of 3-tuples
        """ 
        dep_fc = self.get_trip_departure_forecasts(t0, t1, scale = scale)
        arr_fc = self.get_trip_arrival_forecasts(t0, t1, scale = scale)

        N_dep = sum(dep_fc.values())
        N_arr = sum(arr_fc.values())

        if N_dep == 0 or N_arr == 0:
            return []

        dep_zones = [dep_z for dep_z, dep_val in dep_fc.items() if dep_val > 0]
        dep_prob = [dep_val/N_dep for dep_val in dep_fc.values() if dep_val > 0]
        arr_zones = [arr_z for arr_z, arr_val in arr_fc.items() if arr_val > 0]
        arr_prob = [arr_val/N_arr for arr_val in arr_fc.values() if arr_val > 0]

        future_list = []
        tc = t0
        #LOG.warning(f"draw future: dep {N_dep} arr {N_arr} from {t0} - {t1} with scale {scale}")
        while True:
            tc += np.random.exponential(scale=float(t1-t0)/N_dep)
            if tc > t1:
                break
            o_zone = np.random.choice(dep_zones, p=dep_prob)
            d_zone = np.random.choice(arr_zones, p=arr_prob)
            o_node = self.get_random_centroid_node(o_zone)
            d_node = self.get_random_centroid_node(d_zone)
            future_list.append( (int(tc), o_node, d_node) )
        #LOG.warning(f"future set: {len(future_list)} | {future_list}")

        return future_list