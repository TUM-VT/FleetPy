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
    from src.fleetctrl.planning.PlanRequest import PlanRequest
    
    
LOG = logging.getLogger(__name__)

class MLZoneBasedRepositioning(RepositioningBase):
    def __init__(self, fleetctrl : FleetControlBase, operator_attributes : dict, dir_names : dict, solver : str="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver)
        self._list_vid_that_changed_plans = []
        self._rejected_customer_origins_since_last_step = []
    
    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        super().determine_and_create_repositioning_plans(sim_time, lock)
        return_list = self._list_vid_that_changed_plans.copy()
        self._list_vid_that_changed_plans = []
        self._rejected_customer_origins_since_last_step = []
        return return_list
    
    def register_vid_that_changed_plan(self, vid):
        self._list_vid_that_changed_plans.append(vid)

    def register_rejected_customer(self, planrequest : PlanRequest, sim_time):
        LOG.debug(f"new rejected customer at {planrequest.get_o_stop_info()[0][0]} time {sim_time}")
        super().register_rejected_customer(planrequest, sim_time)
        self._rejected_customer_origins_since_last_step.append(planrequest.get_o_stop_info()[0][0])
        