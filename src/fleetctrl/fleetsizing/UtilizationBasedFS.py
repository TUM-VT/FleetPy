from __future__ import annotations

import logging
import numpy as np

from src.fleetctrl.fleetsizing.DynamicFleetSizingBase import DynamicFleetSizingBase
from src.misc.globals import *

import typing as tp
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_UtilizationBasedFS = {
"doc" :  """ this strategy dynamically adopts the active fleet size based on the current fleet utilization
            - a target utilization has to be given. fleet size is adopted to fit this utilization ("op_dyfs_target_utilization")
            - an interval around this utilization has to be given which specifies the reaction time of the algorithm ("op_dyfs_target_utilization_interval")
            - if the current utilization exceeds target_utilization + target_utilization intervall
                vehicles are added until the utilization afterwards corresponds to target_utilization - target_utilization_interval
            - for vehicles to be deactivated the current utilization has to deceed target_utilization - target_utilization_interval
                for a time interval specified by "op_dyfs_underutilization_interval"
            - if the active fleet size falls below a minimum, specified by "op_dyfs_minimum_active_fleetsize" no more vehicles are deactivated

        inactive vehicles in depots are selected to charge if a charging station is present
            - vehicles are sent to the nearest depots (idle vehicles have priority; other vehicles have to finish their task first (locks their plan))
            - the presence of a charging station has no influence on the depot to be picked! """,
"inherit" : "DynamicFleetSizingBase",
"input_parameters_mandatory": [G_OP_DYFS_TARGET_UTIL, G_OP_DYFS_TARGET_UTIL_INT, G_OP_DYFS_UNDER_UTIL_DUR, G_OP_DYFS_MIN_ACT_FS],
"input_parameters_optional": [],
"mandatory_modules": [],
"optional_modules": []
}

class UtilizationBasedFS(DynamicFleetSizingBase):
    def __init__(self, fleetctrl: FleetControlBase, operator_attributes: dict, solver: str="Gurobi"):
        """Initialization of fleetsizing class.
        
        this strategy dynamically adopts the active fleet size based on the current fleet utilization
            - a target utilization has to be given. fleet size is adopted to fit this utilization ("op_dyfs_target_utilization")
            - an interval around this utilization has to be given which specifies the reaction time of the algorithm ("op_dyfs_target_utilization_interval")
            - if the current utilization exceeds target_utilization + target_utilization intervall
                vehicles are added until the utilization afterwards corresponds to target_utilization - target_utilization_interval
            - for vehicles to be deactivated the current utilization has to deceed target_utilization - target_utilization_interval
                for a time interval specified by "op_dyfs_underutilization_interval"
            - if the active fleet size falls below a minimum, specified by "op_dyfs_minimum_active_fleetsize" no more vehicles are deactivated

        inactive vehicles in depots are selected to charge if a charging station is present
            - vehicles are sent to the nearest depots (idle vehicles have priority; other vehicles have to finish their task first (locks their plan))
            - the presence of a charging station has no influence on the depot to be picked!

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes, solver=solver)
        self.target_utilization = operator_attributes[G_OP_DYFS_TARGET_UTIL]
        self.target_utilization_interval = operator_attributes[G_OP_DYFS_TARGET_UTIL_INT]
        self.max_duration_underutilized = operator_attributes[G_OP_DYFS_UNDER_UTIL_DUR]
        self.minimum_active_fleetsize = operator_attributes[G_OP_DYFS_MIN_ACT_FS]
        self.start_time_underutilization = self.fleetctrl.sim_time

    def check_and_change_fleet_size(self, simulation_time):
        """This method checks whether fleet vehicles should be activated or deactivated.

        :param sim_time: current simulation time
        :return: net change in fleet size
        """
        util, n_eff_utilized, n_active = self.fleetctrl.compute_current_fleet_utilization(simulation_time)
        LOG.debug(f"current util at time {simulation_time}: util {util} n_eff_util {n_eff_utilized} n_active {n_active} | first underutilization: {self.start_time_underutilization}")
        output_activate = 0
        if util > self.target_utilization + self.target_utilization_interval:
            to_activate = n_eff_utilized/self.target_utilization - n_active
            to_activate = int(np.ceil(to_activate))
            if len(self.fleetctrl.sim_vehicles) < n_active + to_activate:
                to_activate = len(self.fleetctrl.sim_vehicles) - n_active
            self.add_time_triggered_activate(simulation_time, int(to_activate))
            self.start_time_underutilization = simulation_time
            output_activate = to_activate
            LOG.debug(f" -> activate {to_activate}")
        elif util < self.target_utilization - self.target_utilization_interval:
            if simulation_time - self.start_time_underutilization > self.max_duration_underutilized:
                to_deactivate = n_active - n_eff_utilized/self.target_utilization
                to_deactivate = int(np.ceil(to_deactivate))
                if n_active - to_deactivate < self.minimum_active_fleetsize:
                    to_deactivate = n_active - self.minimum_active_fleetsize
                self.add_time_triggered_deactivate(simulation_time, int(to_deactivate))
                LOG.debug(f" -> deactivate {to_deactivate}")
                self.start_time_underutilization = simulation_time
                output_activate = -to_deactivate
        else:
            self.start_time_underutilization = simulation_time

        if n_active + output_activate < self.minimum_active_fleetsize: #emergancy
            out2 = self.minimum_active_fleetsize - n_active - output_activate
            LOG.debug("emergency activation of {}".format(out2))
            assert out2 > 0
            self.add_time_triggered_activate(simulation_time, int(out2))
            output_activate += out2
        #dynamic output
        dyn_output_dict = {
            "utilization" : util,
            "effective utilized vehicles" : n_eff_utilized,
            "active vehicles" : n_active,
            "activate vehicles" : output_activate,
        }
        self.fleetctrl._add_to_dynamic_fleetcontrol_output(simulation_time, dyn_output_dict)
        
        activate_list = self.time_triggered_activate(simulation_time)
        deactivate_list = self.time_triggered_deactivate(simulation_time)
        LOG.debug("refill charging units at depot")
        self.fill_charging_units_at_depot(simulation_time)
        return len(activate_list) - len(deactivate_list)
