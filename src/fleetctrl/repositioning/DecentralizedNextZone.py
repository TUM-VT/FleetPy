from __future__ import annotations
import logging
import numpy as np
from typing import TYPE_CHECKING

from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.misc.distributions import draw_from_distribution_dict
from src.misc.globals import *

if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase

LOG = logging.getLogger(__name__)
NEIGHBOR_COL = "neighbors"


INPUT_PARAMETERS_DecentralizedChoiceModel = {
    "doc" :     """ this class implements a pro-active decentralized repositioning strategy that could be applied
    by freelance drivers. Idle vehicles/drivers are looking at neighboring zones and drive to each with a probability
    related to the demand expected in the respective zone.
    The repositioning stops are not locked.
    
    BEWARE: the zone system needs an additional column "neighbors", in which the neighboring zone_ids are written,
            separated by a ";"
    """,
    "inherit": "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


def func(x, sum_x):
    return np.exp(x)


class DecentralizedNextZone(RepositioningBase):
    """ This class implements a pro-active decentralized repositioning strategy that could be applied
        by freelance drivers. Idle vehicles/drivers are looking at neighboring zones and drive to each with a probability
        related to the demand expected in the respective zone.
        The repositioning stops are not locked.

        BEWARE: the zone system needs an additional column "neighbors", in which the neighboring zone_ids are written,
                separated by a ";"
    """
    def __init__(self, fleetctrl: FleetControlBase, operator_attributes: dict, dir_names: dict):
        super().__init__(fleetctrl, operator_attributes, dir_names)
        # check whether zone system actually has neighbor column and raise Error otherwise
        if self.zone_system is None or self.zone_system.general_info_df is None or \
                NEIGHBOR_COL not in self.zone_system.general_info_df.columns:
            raise AssertionError("DecentralizedNextZone repositioning strategy requires a column neighbors in the"
                                 " general_information.csv indicating the neighbors of every zone.")

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
            destination_utilities = {}
            destination_utilities[origin_zone_id] = demand_fc_dict.get(origin_zone_id, 0)
            for d_zone_id in self.zone_system.general_info_df[NEIGHBOR_COL].split(";"):
                destination_utilities[d_zone_id] = demand_fc_dict.get(d_zone_id, 0)
            sum_demand = sum(destination_utilities.values())
            destination_probs = {k: func(v, sum_demand) for k,v in destination_utilities.items()}
            list_destinations = draw_from_distribution_dict(destination_probs, nr_idle_veh_in_zone)
            # draw list_idle_veh_obj
            for veh_obj, destination_zone_id in zip(list_idle_veh_obj, list_destinations):
                if origin_zone_id != destination_zone_id:
                    # choose random node -> destination_node = -1
                    self._od_to_veh_plan_assignment(self, sim_time, origin_zone_id, destination_zone_id, [veh_obj],
                                                    lock=False, destination_node=-1)
                    list_veh_with_changes.append(veh_obj.vid)
        return list_veh_with_changes


