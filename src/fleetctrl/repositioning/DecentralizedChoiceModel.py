from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.misc.distributions import draw_from_distribution_dict
from src.misc.globals import *

if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.fleetctrl.planning.PlanRequest import PlanRequest

LOG = logging.getLogger(__name__)

# global parameters from study https://doi.org/10.1080/19427867.2023.2192581 (Scenario 1, Primary Model)
# not all model parts available in FleetPy! only used following parameters
# - ASC_waiting
# - expected demand in surge/high demand area
# - surge pricing (in Dollar)
# - drive time to surge/high demand area
ASC_WAITING = 0.207
DRIVE_TIME_DEMAND = -0.025 / 60
# important attributes that are not modelled here:
# - SURGE_PRICING = 0.177
# - DRIVE_TIME_SURGE = -0.020
# - prior waiting time in minutes
# - driver location with respect to centrality
# - parking availability
# randomly cruising is also not modelled
# - familiarity with neighborhood
# - working shift (0: end, 1: beginning)
# the term for "pre-booked rides around the drop-off location [min]" needs to be translated into
#   expected number of trips
# - time until expected pre-booked trip [min]: -0.020
# the following is an uncalibrated parameter
NUMBER_EXP_TRIPS_OVER_FS = 1

INPUT_PARAMETERS_DecentralizedChoiceModel = {
    "doc" :     """ this class implements a pro-active decentralized repositioning strategy that could be applied
    by freelance drivers. It is based on the discrete choice model presented in
    https://doi.org/10.1080/19427867.2023.2192581
    The repositioning stops are not locked.
    """,
    "inherit": "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class DecentralizedChoiceModel(RepositioningBase):
    """ This class implements a pro-active decentralized repositioning strategy that could be applied
        by freelance drivers. It is based on the discrete choice model presented in
        https://doi.org/10.1080/19427867.2023.2192581
        The repositioning stops are not locked.
    """
    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """This method determines and creates new repositioning plans. The repositioning plans are directly assigned
        to the vehicles.
        In order to allow further database processes, the vids of vehicles with new plans are returned.

        :param sim_time: current simulation time
        :param lock: indicates if vehplans should be locked
        :return: list[vid] of vehicles with changed plans
        """
        self.sim_time = sim_time
        list_veh_with_changes = []
        t0, t1 = 0, 0  # TODO # think about reasonable times!
        demand_fc_dict = self._get_demand_forecasts(t0, t1)
        cplan_arrival_idle_dict = self._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1)
        for origin_zone_id, tmp_dict in cplan_arrival_idle_dict.items():
            list_idle_veh_obj = tmp_dict.get(2)
            nr_idle_veh_in_zone = len(list_idle_veh_obj)
            # compute probability distribution for destination zones and draw nr_idle_veh times
            destination_utilities = {}
            for destination_zone_id in self.zone_system.all_zones:
                if destination_zone_id == origin_zone_id:
                    util = ASC_WAITING + NUMBER_EXP_TRIPS_OVER_FS * demand_fc_dict.get(destination_zone_id, 0) / \
                           self.fleetctrl.nr_vehicles
                else:
                    tt, td = self._get_od_zone_travel_info(sim_time, origin_zone_id, destination_zone_id)
                    util = NUMBER_EXP_TRIPS_OVER_FS * demand_fc_dict.get(destination_zone_id, 0) / \
                           self.fleetctrl.nr_vehicles + DRIVE_TIME_DEMAND * tt
                destination_utilities[destination_zone_id] = util
            list_destinations = draw_from_distribution_dict(destination_utilities, nr_idle_veh_in_zone)
            # draw list_idle_veh_obj
            for veh_obj, destination_zone_id in zip(list_idle_veh_obj, list_destinations):
                if origin_zone_id != destination_zone_id:
                    # choose random node -> destination_node = -1
                    self._od_to_veh_plan_assignment(self, sim_time, origin_zone_id, destination_zone_id, [veh_obj],
                                                    lock=False, destination_node=-1)
                    list_veh_with_changes.append(veh_obj.vid)
        return list_veh_with_changes
