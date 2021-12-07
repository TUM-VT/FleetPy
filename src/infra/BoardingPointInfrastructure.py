# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import enum
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
LOG = logging.getLogger(__name__)

def routing_min_distance_cost_function(travel_time, travel_distance, current_node_index):
    """computes the customized section cost for routing (input for routing functions)

    :param travel_time: travel_time of a section
    :type travel time: float
    :param travel_distance: travel_distance of a section
    :type travel_distance: float
    :param current_node_index: index of current_node_obj in dijkstra computation to be settled
    :type current_node_index: int
    :return: travel_cost_value of section
    :rtype: float
    """
    return travel_distance

class BoardingPointInfrastructure():
    def __init__(self, infrastructure_dir, routing_engine, load_preprocessed = True):
        """ this class handles active boarding points in the network 
        :param infrastructure: directory of boarding point data (\data\infra\{infra_name}\{network_name})
        :param routing_engine: Network object
        :param load_preprocessed: try loading the file for preprocessed node to closest bp in walking range
                see preprocess_boarding_point_distances.py for preprocessing
        """
        self.routing_engine = routing_engine
        self.boarding_point_node_positions = {}
        bp_df = pd.read_csv(os.path.join(infrastructure_dir, "boarding_points.csv"))
        for node_index in bp_df["node_index"].values:
            self.boarding_point_node_positions[self.routing_engine.return_node_position(node_index)] = 1

        self.computed_possible_boarding_points = {} # origin_node_pos -> walking_range -> list (boarding_node_pos, distance)
        self.preprocessed_nearest_boarding_points = {}  # origin_node_pos -> nearest bp_os, walking distance 
        if load_preprocessed:
            preprocessed_file = os.path.join(infrastructure_dir, "closest_bps_preprocessed.csv")
            if os.path.isfile(preprocessed_file):
                preprocessed_distance = None
                with open(preprocessed_file, "r") as f:
                    lines = f.read()
                    for i, l in enumerate(lines.split("\n")):
                        if not l:
                            continue
                        elif i == 0:
                            preprocessed_distance = float(l.split(":")[1])
                        else:
                            entries = l.split(",")
                            stop_node = int(entries[0])
                            boarding_node_distance_list = []
                            if len(entries) > 0 and entries[1] != "":
                                for entry in entries[1:]:
                                    node, distance = entry.split(";") 
                                    if self.boarding_point_node_positions.get(routing_engine.return_node_position(int(node))) is None:
                                        continue
                                    boarding_node_distance_list.append( (routing_engine.return_node_position(int(node)), float(distance) ) )
                            self.computed_possible_boarding_points[routing_engine.return_node_position(stop_node)] = {preprocessed_distance : boarding_node_distance_list}

            nearest_bp_file = os.path.join(infrastructure_dir, "nearest_bp.csv")
            if os.path.isfile(nearest_bp_file):
                #node_index,closest_bp_index,walking_distance
                nearest_df = pd.read_csv(nearest_bp_file)
                for node_index, closest_bp_index, walking_distance in zip(nearest_df["node_index"].values, nearest_df["closest_bp_index"].values, nearest_df["walking_distance"].values):
                    self.preprocessed_nearest_boarding_points[routing_engine.return_node_position(node_index)] = (routing_engine.return_node_position(closest_bp_index), walking_distance)
                #LOG.info("preprocessed nearest bps: {}".format(self.preprocessed_nearest_boarding_points))

    def return_boarding_points_in_walking_range(self, origin_position, max_walking_range, max_boarding_points = None, return_nearest_else = True):
        """ finds boarding points in walking range by computing the distanced shortest forward and backward dijkstra from the origin_position to boarding point nodes in range
        returns only closest boarding point if none can be found in walking range
        :param origin_position: network position in question
        :param max_walking_range: max walking distance in m from origin_position
        :param max_boarding_points: maximum number of (closest) boarding nodes to return
        :param return_nearest_else: if True, the nearest boarding point is added to the return list in case no boarding points are found within max_walking_range
        :return: list of (boarding_node_position, walking_distance)
        """
        preprocessed_list = self.computed_possible_boarding_points.get(origin_position, {}).get(max_walking_range, None)
        if preprocessed_list is not None:
            #print("found!", max_walking_range)
            possible_boarding_points = {boarding_node_pos : dis for boarding_node_pos, dis in preprocessed_list}
        else:
            #print("not found!", max_walking_range)
            possible_boarding_points = {}
            res_hin = self.routing_engine.return_travel_costs_1toX(origin_position, self.boarding_point_node_positions.keys(), max_routes=max_boarding_points,
                                    max_cost_value=max_walking_range, customized_section_cost_function = routing_min_distance_cost_function)
            res_back = self.routing_engine.return_travel_costs_Xto1(self.boarding_point_node_positions.keys(), origin_position, max_routes=max_boarding_points,
                                    max_cost_value=max_walking_range, customized_section_cost_function = routing_min_distance_cost_function)
            for boarding_node_pos, dis, _, _ in res_hin + res_back:
                if possible_boarding_points.get(boarding_node_pos, 99999999.9) > dis:
                    possible_boarding_points[boarding_node_pos] = dis
        return_list = []
        for boarding_node_pos, dis in sorted(possible_boarding_points.items(), key = lambda x : x[1]):
            return_list.append( (boarding_node_pos, dis) )
            if max_boarding_points is not None and len(return_list) >= max_boarding_points:
                break
        if len(return_list) == 0 and return_nearest_else:
            nearest_preprocessed = self.preprocessed_nearest_boarding_points.get(origin_position)
            if nearest_preprocessed is not None:
                return_list.append(nearest_preprocessed)
            else:
                LOG.info("still havent found what im looking for {}".format(origin_position))
                res_hin = self.routing_engine.return_travel_costs_1toX(origin_position, self.boarding_point_node_positions.keys(), max_routes=1,
                                        customized_section_cost_function = routing_min_distance_cost_function)
                res_back = self.routing_engine.return_travel_costs_Xto1(self.boarding_point_node_positions.keys(), origin_position, max_routes=1,
                                        customized_section_cost_function = routing_min_distance_cost_function)
                try:
                    shortest = min(res_hin + res_back, key = lambda x:x[1])
                    return_list.append( (shortest[0], shortest[1]) )
                except ValueError:
                    pass
        return return_list

    def return_walking_distance(self, origin_pos, target_pos):
        """ returns the walking distance from an origin network position to an target network position
        :param origin_pos: network position of origin
        :param target_pos: network position of target
        :return: walking distance in m
        """
        walking_distance = None
        for walking_range_list in self.computed_possible_boarding_points.get(origin_pos, {}).values():
            for bp, dis in walking_range_list:
                if bp == target_pos:
                    walking_distance = dis
                    return walking_distance
        if walking_distance is None:
            _, _, dis1 = self.routing_engine.return_travel_costs_1to1(origin_pos, target_pos, customized_section_cost_function = routing_min_distance_cost_function)
            _, _, dis2 = self.routing_engine.return_travel_costs_1to1(target_pos, origin_pos, customized_section_cost_function = routing_min_distance_cost_function)
            if dis1 < dis2:
                walking_distance = dis1
            else:
                walking_distance = dis2
        return walking_distance