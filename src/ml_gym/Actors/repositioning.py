from abc import abstractmethod
from src.ml_gym.MLClasses.MLZoneBasedRepositioning import MLZoneBasedRepositioning
from src.ml_gym.Actors import AbstractActor
import random, logging
from multiprocessing.connection import PipeConnection
from typing import List, Tuple

LOG = logging.getLogger(__name__)

class ZoneBasedRepositioningActor(AbstractActor):

    def compute_action(self, observation, process_id) -> List[Tuple[int, int]]:
        raise NotImplementedError("The compute_action method not implemented!. If you are using FleetPy as "
                                  "gymnasium.Env then the code should not have reached here. Otherwise, if you want to"
                                  "manually calculate the action, then override this method with you custom logic")

    def _act(self, observation, fleetpy_module, hook_id, process_id: int = None, conn: PipeConnection = None):
        assert isinstance(fleetpy_module, MLZoneBasedRepositioning), "the fleetpy_module ZoneBasedRepositioningActor can only be used MLZoneBasedRepositioning"
        self._apply_od_assignment(observation, fleetpy_module, hook_id, process_id, conn)

    def _apply_od_assignment(self, observation, repo_module: MLZoneBasedRepositioning, hook_id: int, process_id: int,
                             conn: PipeConnection):
        """ apply externally computed od assignment for repositioning
        :param repo_module: repositioning module
        """
        od_reposition_trips = self._compute_action_via_master_process(observation, hook_id, process_id, conn)
        print("\napply_od_assignment - od_reposition_trips: ", od_reposition_trips)
        list_veh_with_changes = []
        sim_time = repo_module.sim_time
        lock = repo_module.lock_repo_assignments
        t0 = sim_time + repo_module.list_horizons[0]
        t1 = sim_time + repo_module.list_horizons[1]
        cplan_arrival_idle_dict = repo_module._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1)

        random.seed(sim_time)
        random.shuffle(od_reposition_trips)
        for (origin_zone_id, destination_zone_id) in od_reposition_trips:
            list_idle_veh = cplan_arrival_idle_dict[origin_zone_id][2]
            if len(list_idle_veh) == 0:
                LOG.warning(
                    f"No idle vehicles available for repositioning from zone {origin_zone_id} to zone {destination_zone_id}!")
                continue
            LOG.info(f"repo from {origin_zone_id} to {destination_zone_id}")
            if origin_zone_id == destination_zone_id:
                rand_veh = random.choice(list_idle_veh)
                cplan_arrival_idle_dict[origin_zone_id][2].remove(rand_veh)
                continue
            list_veh_obj_with_repos = repo_module._od_to_veh_plan_assignment(sim_time, origin_zone_id,
                                                                             destination_zone_id, list_idle_veh,
                                                                             lock=lock)
            list_veh_with_changes.extend([veh_obj.vid for veh_obj in list_veh_obj_with_repos])
            for veh_obj in list_veh_obj_with_repos:
                cplan_arrival_idle_dict[origin_zone_id][2].remove(veh_obj)

        for vid in list_veh_with_changes:
            repo_module.register_vid_that_changed_plan(vid)
