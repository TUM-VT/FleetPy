"""
Authors: Roman Engelhardt, Florian Dandl
TUM, 2020
In order to guarantee transferability of models, Network models should follow the following conventions.
Classes should be called
Node
Edge
Network
in order to guarantee correct import in other modules.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np

# src imports
# -----------
from src.routing.NetworkBasic import NetworkBasic
from src.routing.routing_imports.Router import Router

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_NetworkBasicWithStore = {
    "doc" : """
        This routing class does all routing computations based on Dijkstra's algorithm. 
        Compared to NetworkBasic.py, this class stores already computed travel infos in a dictionary and returns the values from this dictionary if queried again.
        """,
    "inherit" : "NetworkBasic",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class NetworkBasicWithStore(NetworkBasic):
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        """
        The network will be initialized.
        This network stores routing results from return_travel_costs_1to1 in a database to retrieve them in case they are queried again
        additionally, if calling the function return_travel_costs_1to1, internally a dijkstra to all boarding nodes is called in case origin and destination is a boarding node
            these results are not returned, but directly stored in the database in case they are needed again

        :param network_name_dir: name of the network_directory to be loaded
        :param type: determining whether the base or a pre-processed network will be used
        :param scenario_time: applying travel times for a certain scenario at a given time in the scenario
        :param network_dynamics_file_name: file-name of the network dynamics file
        :type network_dynamics_file_name: str
        """
        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)

        self.travel_time_infos = {} #(o,d) -> (tt, dis)

    def update_network(self, simulation_time, update_state = True):
        """This method can be called during simulations to update travel times (dynamic networks).

        :param simulation_time: time of simulation
        :type simulation_time: float
        :return: new_tt_flag True, if new travel times found; False if not
        :rtype: bool
        """
        return super().update_network(simulation_time, update_state=update_state)

    def return_travel_costs_1to1(self, origin_position, destination_position, customized_section_cost_function = None):
        """
        This method will return the travel costs of the fastest route between two nodes.
        :param origin_position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param destination_position: (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return: (cost_function_value, travel time, travel_distance) between the two nodes
        """
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[1]
        origin_node = origin_position[0]
        origin_overhead = (0.0, 0.0, 0.0)
        if origin_position[1] is not None:
            origin_node = origin_position[1]
            origin_overhead = self.get_section_overhead(origin_position, from_start=False)
        destination_node = destination_position[0]
        destination_overhead = (0.0, 0.0, 0.0)
        if destination_position[1] is not None:
            destination_overhead = self.get_section_overhead(destination_position, from_start=True)
        if self.travel_time_infos.get( (origin_node, destination_node) ) is not None:
            s = self.travel_time_infos[(origin_node, destination_node)]
            if s[0] == np.inf: # TODO # seems to be a bug. Do you know what's the problem here?
                LOG.warning(f"in return_travel_costs_1to1, travel_time_infos from nodes {origin_node} to {destination_node}"
                         f"yields s={s}")
                R = Router(self, origin_node, destination_nodes=[destination_node], mode='bidirectional',
                           customized_section_cost_function=customized_section_cost_function)
                s = R.compute(return_route=False)[0][1]
                LOG.warning(f"after recalculation of the routes, s={s}")
        else:
            R = Router(self, origin_node, destination_nodes=[destination_node], mode='bidirectional', customized_section_cost_function=customized_section_cost_function)
            s = R.compute(return_route=False)[0][1]
            if customized_section_cost_function is None:
                self._add_to_database(origin_node, destination_node, s[0], s[1], s[2])
        return (s[0] + origin_overhead[0] + destination_overhead[0], s[1] + origin_overhead[1] + destination_overhead[1], s[2] + origin_overhead[2] + destination_overhead[2])

    def _reset_internal_attributes_after_travel_time_update(self):
        self.travel_time_infos = {}

    def add_travel_infos_to_database(self, travel_info_dict):
        """ this function can be used to include externally computed (e.g. multiprocessing) route travel times
        into the database if present

        it adds all infos from travel_info_dict to its database self.travel_time_infos
        its database is from node to node, therefore overheads have to be removed from routing results

        :param travel_info_dict: dictionary with keys (origin_position, target_positions) -> values (cost_function_value, travel_time, travel_distance)
        """
        for od_pos, s in travel_info_dict.items():
            # since only travel infos between nodes are stored the inverse function from get_travel_costs_1to1 has to be called here
            origin_position, destination_position = od_pos
            origin_node = origin_position[0]
            origin_overhead = (0.0, 0.0, 0.0)
            if origin_position[1] is not None:
                origin_node = origin_position[1]
                origin_overhead = self.get_section_overhead(origin_position, from_start=False)
            destination_node = destination_position[0]
            if self.travel_time_infos.get( (origin_node, destination_node) ) is not None:
                continue
            destination_overhead = (0.0, 0.0, 0.0)
            if destination_position[1] is not None:
                destination_overhead = self.get_section_overhead(destination_position, from_start=True)
            s_adopted = (s[0] - origin_overhead[0] - destination_overhead[0], s[1] - origin_overhead[1] - destination_overhead[1], s[2] - origin_overhead[2] - destination_overhead[2])
            self.travel_time_infos[(origin_node, destination_node)] = s_adopted

    def _add_to_database(self, o_node, d_node, cfv, tt, dis):
        """ this function is call when new routing results have been computed
        depending on the class the function can be overwritten to store certain results in the database
        """
        if self.travel_time_infos.get( (o_node, d_node) ) is None:
            self.travel_time_infos[ (o_node, d_node) ] = (cfv, tt, dis)