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

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
# from src.fleetctrl.FleetControlBase import PlanRequest # TODO # circular dependency!
# set log level to logging.DEBUG or logging.INFO for single simulations
LOG_LEVEL = logging.WARNING
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_ZoneSystem = {
    "doc": """this is the basic class describe a zone systen.
    it requires input file to match nodes to specific zones.
    The zones can be either defined on a operator level or a global level.
    Therefore, different parameters can be seleced to define the zones used.""",
    "inherit": None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_RA_OP_ZONE_SYSTEM, G_RA_OP_REPO_ZONE_SYSTEM, G_ZONE_SYSTEM_NAME],
    "mandatory_modules": [],
    "optional_modules": []
}

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
        node_zone_f = os.path.join(self.zone_network_dir, "node_zone_info.csv")
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
        self.zone_boarding = None
        if G_ZONE_BOARD in self.node_zone_df.columns:
            self.zone_boarding = {}
            for node_id, zone_id in self.node_zone_df[self.node_zone_df[G_ZONE_BOARD] == 1][G_ZONE_ZID].items():
                try:
                    self.zone_boarding[zone_id].append(node_id)
                except KeyError:
                    self.zone_boarding[zone_id] = [node_id]
                    
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

    def get_random_node(self, zone_id, only_boarding_nodes=False):
        """This method returns a random node_id for a given zone_id.

        :param zone_id: id of the zone in question
        :type zone_id: int
        :param only_boarding_nodes: if True, only nodes with boarding attribute are considered (if G_ZONE_BOARD provide in node_zone_info.csv, else all nodes are considered)
        :type only_boarding_nodes: bool
        :return: node_id of a node within the zone in question; return -1 if invalid zone_id is given
        :rtype: int
        """
        if only_boarding_nodes and self.zone_boarding is not None:
            boarding_nodes = self.zone_boarding.get(zone_id, [])
            if len(boarding_nodes) > 0:
                return np.random.choice(boarding_nodes)
            else:
                return -1
        else:
            tmp_df = self.node_zone_df[self.node_zone_df[G_ZONE_ZID] == zone_id]
            if len(tmp_df) > 0:
                return np.random.choice(tmp_df.index.values.tolist())
            else:
                return -1

    def get_random_centroid_node(self, zone_id):
        if self.zone_centroids is not None:
            nodes = self.zone_centroids.get(zone_id, [])
            if len(nodes) == 1:
                return nodes[0]
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

    def get_centroid_node(self, zone_id):
        # TODO # after ISTTT: get_centroid_node()
        pass
