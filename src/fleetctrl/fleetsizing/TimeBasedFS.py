from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from src.misc.globals import *
LOG = logging.getLogger(__name__)
from src.fleetctrl.fleetsizing.DynamicFleetSizingBase import DynamicFleetSizingBase
from src.misc.distributions import draw_from_distribution_dict

if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    
INPUT_PARAMETERS_DynamicFleetSizingBase = {
"doc" :  """information about the timedependent fleetsize has to be given with the file "op_act_fs_file"
        according to the active fleetsize curve this strategy activates and deactivates vehicles to fit this curve
        inactive vehicles in depots are selected to charge if a charging station is present
            - vehicles are sent to the nearest depots (idle vehicles have priority; other vehicles have to finish their task first (locks their plan))
            - the presence of a charging station has no influence on the depot to be picked! """,
"inherit" : "DynamicFleetSizingBase",
"input_parameters_mandatory": [G_OP_ACT_FLEET_SIZE],
"input_parameters_optional": [],
"mandatory_modules": [],
"optional_modules": []
}

class TimeBasedFS(DynamicFleetSizingBase):
    def __init__(self, fleetctrl: FleetControlBase, operator_attributes: dict, solver: str="Gurobi"):
        """Initialization of fleetsizing class.
        
        information about the timedependent fleetsize has to be given with the file "op_act_fs_file"
        according to the active fleetsize curve this strategy activates and deactivates vehicles to fit this curve
        inactive vehicles in depots are selected to charge if a charging station is present
            - vehicles are sent to the nearest depots (idle vehicles have priority; other vehicles have to finish their task first (locks their plan))
            - the presence of a charging station has no influence on the depot to be picked!

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes)
        
    def add_init(self, operator_attributes: dict):
        """ additional loading for stuff that has to be initialized after full init of fleetcontrol"""
        active_vehicle_file_name = operator_attributes.get(G_OP_ACT_FLEET_SIZE)
        if active_vehicle_file_name is None:
            raise EnvironmentError("TimeBased FleetSizing selected but no input file given! specify parameter {}!".format(G_OP_ACT_FLEET_SIZE))
        else:
            active_vehicle_file = os.path.join(self.fleetctrl.dir_names[G_DIR_FCTRL], "elastic_fleet_size",
                                               active_vehicle_file_name)
            if not os.path.isfile(active_vehicle_file):
                raise FileNotFoundError(f"Could not find active vehicle file {active_vehicle_file}")
        self._read_active_vehicle_file(active_vehicle_file)

    def check_and_change_fleet_size(self, sim_time):
        """This method checks whether fleet vehicles should be activated or deactivated.

        :param sim_time: current simulation time
        :return: net change in fleet size
        """
        activate_list = self.time_triggered_activate(sim_time)
        deactivate_list = self.time_triggered_deactivate(sim_time)
        LOG.debug("refill charging units at depot")
        self.fill_charging_units_at_depot(sim_time)
        return len(activate_list) - len(deactivate_list)
    
    def _read_active_vehicle_file(self, active_vehicle_file):
        """This method reads a csv file which has at least two columns (time, share_active_veh_change). For times with a
        positive value for active_veh_change, self.time_activate receives an entry, such that the time trigger will
        activate the respective number of vehicles during the simulation. For a negative value, the time trigger will
        deactivate the respective number of vehicles.

        :param active_vehicle_file: csv-file containing the change in active vehicles
        :return: None
        """

        df = pd.read_csv(active_vehicle_file, index_col=0)
        LOG.debug("read active vehicle file")
        # remove entries before simulation start time
        sim_start_time = self.fleetctrl.sim_time
        remove_indices = []
        last_smaller = None
        for t in df.index:
            if t < sim_start_time:
                if last_smaller is not None:
                    remove_indices.append(last_smaller)
                last_smaller = t
            else:
                break
        df.drop(remove_indices, inplace=True)
        #
        depot_distribution = {}
        for depot_id, depot in self.op_charge_depot_infra.depot_by_id.items():
            depot_distribution[depot_id] = depot.free_parking_spots
        last_number_active = None
        def share_to_act_veh(row):
            return min( max(int(np.round(row[G_ACT_VEH_SHARE] * self.fleetctrl.nr_vehicles)), 0), self.fleetctrl.nr_vehicles)
        df["number_active_vehicles"] = df.apply(share_to_act_veh, axis=1)
        LOG.debug("act veh df: {}".format(df))
        for sim_time, number_active in df["number_active_vehicles"].items():
            if last_number_active is None:
                # initial inactive vehicles have to be set here; the chosen vehicles will start in the depot
                # -> will not be overwritten by FleetSimulation.set_initial_state() as veh_obj.pos is not None
                number_inactive = self.fleetctrl.nr_vehicles - number_active
                LOG.debug("init inactive vehicles: {}".format(number_inactive))
                for vid in range(number_inactive):
                    veh_obj = self.fleetctrl.sim_vehicles[vid]
                    drawn_depot_id = draw_from_distribution_dict(depot_distribution)
                    drawn_depot = self.op_charge_depot_infra.depot_by_id[drawn_depot_id]
                    veh_obj.status = 0
                    veh_obj.soc = 1.0
                    veh_obj.pos = drawn_depot.pos
                    depot, ps = self.deactivate_vehicle(veh_obj, sim_start_time, drawn_depot)
                    if depot is None:
                        LOG.warning("init active vehicle file: no empty parking space can be found anymore!!")
                        continue
                    # beam vehicle to depot if necessary
                    if ps.pos != veh_obj.pos:
                        veh_obj.pos = ps.pos
                        veh_obj.soc = 1.0
                active_veh_change = 0
            else:
                active_veh_change = number_active - last_number_active
            if active_veh_change > 0:
                add_active_veh = active_veh_change
                if add_active_veh > 0:
                    self.add_time_triggered_activate(sim_time, add_active_veh)
            elif active_veh_change < 0:
                rem_active_veh = -active_veh_change
                if rem_active_veh > 0:
                    self.add_time_triggered_deactivate(sim_time, rem_active_veh)
            last_number_active = number_active
        LOG.info(f"Loaded nr-of-active-vehicles curve from {active_vehicle_file}.")
        LOG.debug("-> time triggered activate: {}".format(self.sorted_time_activate))
        LOG.debug("-> time triggered deactivate: {}".format(self.sorted_time_deactivate))
