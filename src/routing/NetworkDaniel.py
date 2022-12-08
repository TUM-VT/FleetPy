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

class EdgeDaniel(Edge):
    def __init__(self, edge_index: Tuple[int, int], distance: float, travel_time: float, external_cost:float):
        super().__init__(edge_index, distance, travel_time)
        self._external_cost = external_cost
        
    def get_external_cost(self):
        return self._external_cost