# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging
from typing import Callable, List, Tuple, Dict

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np
from pyproj import Transformer

# src imports
# -----------
from src.routing.NetworkBasic import NetworkBasic, Edge, Node

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

def read_node_line(columns):
    # @Daniel -> todo ? 
    return Node(int(columns["node_index"]), int(columns["is_stop_only"]), float(columns["pos_x"]), float(columns["pos_y"]))

def external_cost_section_cost_function(travel_time : float, travel_distance : float, edge_obj) -> float:
    """returns the external cost of an edge @Daniel -> todo?

    :param travel_time: travel_time of a section
    :type travel time: float
    :param travel_distance: travel_distance of a section
    :type travel_distance: float
    :param edge_obj: current edge object (dependent on network implementation) in dijkstra computation that is currently checked
    :type edge_obj: edge object
    :return: travel_cost_value of section
    :rtype: float
    """
    return edge_obj.get_external_cost()

class EdgeDaniel(Edge):
    def __init__(self, edge_index: Tuple[int, int], distance: float, travel_time: float, external_cost:float):
        super().__init__(edge_index, distance, travel_time)
        self._external_cost = external_cost
        
    def get_external_cost(self):
        return self._external_cost
    
class NetworkDaniel(NetworkBasic):
    def __init__(self, network_name_dir: str, network_dynamics_file_name: str = None, scenario_time: int = None):
        super().__init__(network_name_dir, network_dynamics_file_name, scenario_time)
        
    def loadNetwork(self, network_name_dir:str, network_dynamics_file_name:str=None, scenario_time:int=None):
        nodes_f = os.path.join(network_name_dir, "base", "nodes.csv")
        print(f"Loading nodes from {nodes_f} ...")
        nodes_df = pd.read_csv(nodes_f)
        self.nodes = nodes_df.apply(read_node_line, axis=1)
        #
        edges_f = os.path.join(network_name_dir, "base", "edges.csv")
        print(f"Loading edges from {edges_f} ...")
        edges_df = pd.read_csv(edges_f)
        for _, row in edges_df.iterrows():  # @ Daniel: Todo!
            o_node = self.nodes[row[G_EDGE_FROM]]
            d_node = self.nodes[row[G_EDGE_TO]]
            tmp_edge = EdgeDaniel((o_node, d_node), row[G_EDGE_DIST], row[G_EDGE_TT])
            o_node.add_next_edge_to(d_node, tmp_edge)
            d_node.add_prev_edge_from(o_node, tmp_edge)
        print("... {} nodes loaded!".format(len(self.nodes)))
        if scenario_time is not None:
            latest_tt = None
            if len(self.travel_time_file_folders.keys()) > 0:
                tts = sorted(list(self.travel_time_file_folders.keys()))
                for tt in tts:
                    if tt > scenario_time:
                        break
                    latest_tt = tt
                self.load_tt_file(latest_tt)
                
    def load_tt_file(self, scenario_time:int): # @ Daniel: todo -> in case external costs change dynamically
        """
        loads new travel time files for scenario_time
        """
        self._reset_internal_attributes_after_travel_time_update()
        f = self.travel_time_file_folders[scenario_time]
        tt_file = os.path.join(f, "edges_td_att.csv")
        tmp_df = pd.read_csv(tt_file)
        tmp_df.set_index(["from_node","to_node"], inplace=True)
        for edge_index_tuple, new_tt in tmp_df["edge_tt"].iteritems():
            self._set_edge_tt(edge_index_tuple[0], edge_index_tuple[1], new_tt) # @ Daniel -> you then might have to adopt this function
    
    # @ Daniel: all routing functions (that have input customized_section_cost_function) have to be adopted the following way (note the "external_cost_section_cost_function"):      
    def return_travel_costs_1to1(self, origin_position: Tuple[int, int, float], destination_position: Tuple[int, int, float],
                                 customized_section_cost_function: Callable[[float, float, Edge], float] = external_cost_section_cost_function) -> Tuple[float, float, float]:
        return super().return_travel_costs_1to1(origin_position, destination_position, customized_section_cost_function = customized_section_cost_function)
    
    # @ Daniel: repeat for other methods that have nput customized_section_cost_function