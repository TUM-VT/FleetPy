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
from src.routing.NetworkBasicWithStore import NetworkBasicWithStore


# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

class NetworkImmediatePreproc(NetworkBasicWithStore):
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        """
        The network will be initialized.
        This network immdiatly computes the travel times between all nodes
        -> only useful for small networks
        :param network_name_dir: name of the network_directory to be loaded
        :param type: determining whether the base or a pre-processed network will be used
        :param scenario_time: applying travel times for a certain scenario at a given time in the scenario
        :param network_dynamics_file_name: file-name of the network dynamics file
        :type network_dynamics_file_name: str
        """
        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        self._preprocess_globally()

    def update_network(self, simulation_time, update_state = True):
        """This method can be called during simulations to update travel times (dynamic networks).

        :param simulation_time: time of simulation
        :type simulation_time: float
        :return: new_tt_flag True, if new travel times found; False if not
        :rtype: bool
        """
        if super().update_network(simulation_time, update_state=update_state):
            self._preprocess_globally()
            return True
        else:
            return False

    def _preprocess_globally(self):
        self.travel_time_infos = {}
        targets = [self.return_node_position(node.node_index) for node in self.nodes]
        for o_pos in targets:
            res = self.return_travel_costs_1toX(o_pos, targets)
            for d_pos, _, tt, dis in res:
                self.travel_time_infos[(o_pos[0], d_pos[0])] = (tt, dis)
    