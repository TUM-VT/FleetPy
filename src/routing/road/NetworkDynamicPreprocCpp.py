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
import numpy as np
import time
from src.routing.road.NetworkBasicWithStoreCpp import NetworkBasicWithStoreCpp


# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_NetworkDynamicPreprocCpp = {
    "doc" : """This network tracks the most frequently requested destination nodes ("hot nodes") and preprocesses
        a travel time matrix between them after each travel time update""",
    "inherit" : "NetworkBasicWithStore",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class NetworkDynamicPreprocCpp(NetworkBasicWithStoreCpp):
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        """
        The network will be initialized.
        This network tracks the most frequently requested destination nodes ("hot nodes") and preprocesses
        a travel time matrix between them after each travel time update
        :param network_name_dir: name of the network_directory to be loaded
        :param type: determining whether the base or a pre-processed network will be used
        :param scenario_time: applying travel times for a certain scenario at a given time in the scenario
        :param network_dynamics_file_name: file-name of the network dynamics file
        :type network_dynamics_file_name: str
        """
        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        self._max_preproc_nodes = 1800
        self._coverage_preproc_nodes = 0.9
        self._destination_count = {}
        # hot node travel time matrix: node_index -> matrix index (-1: not hot); matrix[i, j] = (tt, dis)
        self._hot_idx = np.full(len(self.nodes), -1, dtype=np.int32)
        self._hot_mat = np.empty((0, 0, 2), dtype=np.float32)
        self._n_queries = 0
        self._n_hot_hits = 0
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

    def load_tt_file(self, scenario_time, ext_path=None):
        """
        loads new travel time files for scenario_time
        """
        x = super().load_tt_file(scenario_time, ext_path=ext_path)
        self._preprocess_globally()
        return x

    def _select_hot_destinations(self, k_max, coverage):
        if not self._destination_count:
            return []
        nodes = np.fromiter(self._destination_count.keys(), dtype=np.int64)
        counts = np.fromiter(self._destination_count.values(), dtype=np.float64)
        k = min(k_max, len(nodes))
        idx = np.argpartition(counts, -k)[-k:]          # Top-k ohne Gesamtsortierung, O(n)
        idx = idx[np.argsort(counts[idx])[::-1]]        # nur die k Gewinner sortieren
        cum = np.cumsum(counts[idx]) / counts.sum()
        n_cov = int(np.searchsorted(cum, coverage)) + 1 # wird coverage nie erreicht, bleiben alle k
        return nodes[idx[:n_cov]].tolist()

    def _preprocess_globally(self):
        """ computes a dense travel time / distance matrix between all hot nodes (most frequently requested
        destinations) with one backward Dijkstra per hot node """
        if self._n_queries > 0:
            LOG.info(f"hot node matrix hit rate since last preprocessing: {self._n_hot_hits / self._n_queries:.3f} "
                     f"({self._n_hot_hits}/{self._n_queries})")
        t = time.time()
        self._n_queries = 0
        self._n_hot_hits = 0
        hot = self._select_hot_destinations(self._max_preproc_nodes, self._coverage_preproc_nodes)
        k = len(hot)
        self._hot_idx.fill(-1)
        self._hot_idx[np.asarray(hot, dtype=np.int64)] = np.arange(k, dtype=np.int32)
        mat = np.full((k, k, 2), np.inf, dtype=np.float32)
        tt_factor = self._current_tt_factor if self._current_tt_factor is not None else 1.0
        for j, d in enumerate(hot):
            res = self.cpp_router.computeTravelCostsXto1(d, hot)  # [(origin_node, tt, dis), ...]
            if not res:
                continue
            arr = np.asarray(res, dtype=np.float64)
            arr = arr[arr[:, 1] >= -0.0001]  # negative tt -> not reachable
            rows = self._hot_idx[arr[:, 0].astype(np.int64)]
            mat[rows, j, 0] = arr[:, 1] * tt_factor
            mat[rows, j, 1] = arr[:, 2]
        self._hot_mat = mat
        LOG.info(f"preprocessed hot node matrix with {k} nodes in {time.time() - t:.1f}s")

    def return_travel_costs_1to1(self, origin_position, destination_position, customized_section_cost_function=None):
        """
        This method will return the travel costs of the fastest route between two nodes.
        Routes between hot nodes are looked up in the preprocessed matrix, all others are computed by the parent class.
        :param origin_position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param destination_position: (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return: (cost_function_value, travel time, travel_distance) between the two nodes
        """
        d_node = destination_position[0]
        self._destination_count[d_node] = self._destination_count.get(d_node, 0) + 1
        if customized_section_cost_function is None:
            self._n_queries += 1
            o_node = origin_position[1] if origin_position[1] is not None else origin_position[0]
            i = self._hot_idx[o_node]
            j = self._hot_idx[d_node]
            if i >= 0 and j >= 0:
                trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
                if trivial_test is not None:
                    return trivial_test[1]
                self._n_hot_hits += 1
                tt, dis = self._hot_mat[i, j].tolist()
                if origin_position[1] is not None:
                    o_oh = self.get_section_overhead(origin_position, from_start=False)
                    tt += o_oh[1]
                    dis += o_oh[2]
                if destination_position[1] is not None:
                    d_oh = self.get_section_overhead(destination_position, from_start=True)
                    tt += d_oh[1]
                    dis += d_oh[2]
                return (tt, tt, dis)
        return super().return_travel_costs_1to1(origin_position, destination_position,
                                                customized_section_cost_function=customized_section_cost_function)