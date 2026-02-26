from __future__ import annotations
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
from abc import abstractmethod, ABCMeta
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

# -------------------------------------------------------------------------------------------------------------------- #
# local imports
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    
    
LOG = logging.getLogger(__name__)

class MLZoneBasedRepositioning(RepositioningBase):
    def __init__(self, fleetctrl : FleetControlBase, operator_attributes : dict, dir_names : dict, solver : str="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver)
        self._list_vid_that_changed_plans = []
    
    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        super().determine_and_create_repositioning_plans(sim_time, lock)
        return_list = self._list_vid_that_changed_plans.copy()
        self._list_vid_that_changed_plans = []
        return return_list
    
    def register_vid_that_changed_plan(self, vid):
        self._list_vid_that_changed_plans.append(vid)
        