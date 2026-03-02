import random    
from src.ml_gym.MLEnvs.MLEnv import MLEnv

class RandomRepositioningEnv(MLEnv):
    def __init__(self):
        super().__init__(None, None)
        
    def _compute_action(self, observation):
        # compute random repositioning action based on observation
        print("\nRandomRepositioningEnv: compute random repositioning action")
        print(f"Observation: {observation}")
        sim_time = observation["sim_time"]
        zone_to_fc_rq_origins = observation["zone_to_fc_rq_origins"]
        zone_to_fc_rq_destinations = observation["zone_to_fc_rq_destinations"]
        zone_to_idle_vehilces = observation["zone_to_idle_vehilces"]
        zone_to_overall_available_vehilces = observation["zone_to_overall_available_vehilces"]
        zone_to_current_repositioning_vehicles = observation["zone_to_current_repositioning_vehicles"]
        
        list_repo_targets = []
        for zone_id,value in zone_to_fc_rq_origins.items():
            if zone_id >= 0:
                for _ in range(int(value)):
                    list_repo_targets.append(zone_id)
        list_repo_origins = []
        for zone_id, value in zone_to_idle_vehilces.items():
            if zone_id > 0:
                for _ in range(int(value)):
                    list_repo_origins.append(zone_id)
        
        list_repo_actions = []
        while len(list_repo_targets) > 0 and len(list_repo_origins) > 0:
            origin = random.choice(list_repo_origins)
            target = random.choice(list_repo_targets)
            list_repo_actions.append((origin, target))
            list_repo_targets.remove(target)
            list_repo_origins.remove(origin)
        return list_repo_actions