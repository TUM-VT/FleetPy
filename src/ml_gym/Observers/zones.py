import numpy as np
from src.ml_gym.Observers import AbstractObserver
from src.misc.globals import *


class TravelTimeMatrixObserver(AbstractObserver):
    def __init__(self):
        super().__init__()
        self.tt_matrix = None

    # travel time is calculated only once
    def observe(self, fleetpy_module, event, **kwargs):
        _, repo, zone_system, _ = self.extract_modules(fleetpy_module, event)

        if self.tt_matrix is None:
            zones = sorted(zone_system.get_all_zones())
            if -1 in zones:
                zones.remove(-1)
            Z = len(zones)
            tt_matrix = np.zeros((Z, Z))
            for i_idx, i in enumerate(zones):
                for j_idx, j in enumerate(zones):
                    tt, _ = repo._get_od_zone_travel_info(repo.sim_time, i, j)
                    tt_matrix[i_idx, j_idx] = tt
            self.tt_matrix = tt_matrix

        return {"tt_matrix": self.tt_matrix}