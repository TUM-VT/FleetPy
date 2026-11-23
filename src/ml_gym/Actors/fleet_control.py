from abc import abstractmethod
from src.ml_gym.MLClasses.MLZoneBasedRepositioning import MLZoneBasedRepositioning
from src.ml_gym.MLClasses.RLPricingIRSOnly import RLPricingIRSOnly
from src.fleetctrl.PoolingIRSOnly import PoolingInsertionHeuristicOnly
from src.ml_gym.Actors import AbstractActor
import random, logging
from queue import Queue
from typing import TYPE_CHECKING
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.planning.VehiclePlan import VehiclePlan

LOG = logging.getLogger(__name__)

class ImmediateVehPlanSelector(AbstractActor):

    def compute_action(self, observation, process_id, **kwargs) -> tuple[str, str, float]:
        raise NotImplementedError("The compute_action method not implemented!. If you are using FleetPy as "
                                  "gymnasium.Env then the code should not have reached here. Otherwise, if you want to"
                                  "manually calculate the action, then override this method with you custom logic")

    def _act(self, observation, fleetpy_module, hook_id, process_id: int = None, in_queue: Queue = None,
             out_queue: Queue = None, **kwargs):
        assert isinstance(fleetpy_module, PoolingInsertionHeuristicOnly), "the actor VehPlanSelector can only be used with the fleet controller PoolingInsertionHeuristicOnly"
        self._apply_od_assignment(observation, fleetpy_module, hook_id, process_id, in_queue, out_queue, **kwargs)

    def _apply_od_assignment(self, observation, fleet_controller: PoolingInsertionHeuristicOnly, hook_id: int, process_id: int,
                             in_queue: Queue, out_queue: Queue, **kwargs):

        rid, vid, new_fare = self._compute_action_via_master_process(observation, hook_id, process_id, in_queue, out_queue)
        plans: dict[str, dict[str, tuple[VehiclePlan, float, float]]] = kwargs["candidate_plans"]
        veh_plan = plans["rid"]["vid"][0]
        fleet_controller.set_best_candidate_plan(rid, vid, veh_plan, new_fare)

