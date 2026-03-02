from __future__ import annotations
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
from typing import Dict, List, Any, Tuple, TYPE_CHECKING
import random

# -------------------------------------------------------------------------------------------------------------------- #
# local imports
if TYPE_CHECKING:
    from src.ml_gym.MLClasses.MLZoneBasedRepositioning import MLZoneBasedRepositioning
    
    
LOG = logging.getLogger(__name__)

def apply_od_assignment(repo_module: MLZoneBasedRepositioning, od_reposition_trips: List[Tuple[int,int]]):
    """
    apply externally computed od assignment for repositioning
    :param repo_module: repositioning module
    :param od_assignment: list of tuple of (origin_zone_id, destination_zone_id) [does not have to be unique!]
    """
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
            LOG.warning(f"No idle vehicles available for repositioning from zone {origin_zone_id} to zone {destination_zone_id}!")
            continue
        LOG.info(f"repo from {origin_zone_id} to {destination_zone_id}")
        if origin_zone_id == destination_zone_id:
            rand_veh = random.choice(list_idle_veh)
            cplan_arrival_idle_dict[origin_zone_id][2].remove(rand_veh)
            continue
        list_veh_obj_with_repos = repo_module._od_to_veh_plan_assignment(sim_time, origin_zone_id,
                                                                    destination_zone_id, list_idle_veh, lock=lock)
        list_veh_with_changes.extend([veh_obj.vid for veh_obj in list_veh_obj_with_repos])
        for veh_obj in list_veh_obj_with_repos:
            cplan_arrival_idle_dict[origin_zone_id][2].remove(veh_obj)
            
    for vid in list_veh_with_changes:
        repo_module.register_vid_that_changed_plan(vid)