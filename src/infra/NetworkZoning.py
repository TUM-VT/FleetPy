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
from src.infra.Zoning import ZoneSystem
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

NOON = 12*3600

class NetworkZoneSystem(ZoneSystem):
    def __init__(self, zone_network_dir, scenario_parameters, dir_names):
        super().__init__(zone_network_dir, scenario_parameters, dir_names)
        # # edge specific information -> not necessary at the moment
        # edge_zone_f = os.path.join(zone_network_dir, "edge_zone_info.csv")
        # self.edge_zone_df = pd.read_csv(edge_zone_f)
        self.current_toll_cost_scale = 0
        self.current_toll_coefficients = {}
        self.current_park_costs = {}
        self.current_park_search_durations = {}


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