import logging

from src.fleetctrl.charging.ChargingBase import ChargingBase
from src.misc.globals import *
LOG = logging.getLogger(__name__)


class ChargingThreshold(ChargingBase):
    def __init__(self, fleetctrl, operator_attributes):
        super().__init__(fleetctrl, operator_attributes)
        self.soc_threshold = operator_attributes.get(G_OP_APS_SOC, 0.1)
        if operator_attributes.get(G_OP_CHARGE_PUBLIC_ONLY):
            self.charge_in_depots = False
        else:
            self.charge_in_depots = True

    def _call_specific_charging_strategy(self, sim_time):
        if self.charge_in_depots:
            self._low_soc_veh_to_depot(sim_time)
        else:
            self._low_soc_veh_to_public_cs(sim_time)

    def _low_soc_veh_to_depot(self, sim_time):
        """This charging strategy sends low SOC vehicles to the depot and reactivates another vehicle at the depot.

        :param sim_time: current simulation time
        :return: None
        """
        # deactivate vehicles with SOC below threshold
        reactivate = {}  # (future time, depot_id) -> nr_activate
        found_depots = {}  # depot_id -> depot
        for veh_obj in self.fleetctrl.sim_vehicles:
            # do not consider inactive vehicles
            if veh_obj.status == 5:
                continue
            current_plan = self.fleetctrl.veh_plans[veh_obj.vid]
            if current_plan.list_plan_stops:
                last_pstop = current_plan.list_plan_stops[-1]
                if last_pstop.get_charging_power() == 0 and last_pstop.get_planned_arrival_soc() < self.soc_threshold and \
                        not last_pstop.is_inactive():
                    # search depot with replacement vehicle
                    dep_rv = self.fleetctrl.charging_management.find_nearest_depot_replace_veh(last_pstop.get_pos(),
                                                                                               self.fleetctrl.op_id)
                    #
                    depot, depot_ps = self.fleetctrl.charging_management.deactivate_vehicle(self.fleetctrl, veh_obj,
                                                                                            sim_time, dep_rv)
                    LOG.info(f"Operator {self.fleetctrl.op_id} sending vehicle {veh_obj.vid} to "
                             f"depot {depot} for charging.")
                    arrival_time = depot_ps.get_planned_arrival_and_departure_time()[0]
                    found_depots[depot.cstat_id] = depot
                    try:
                        reactivate[(arrival_time, depot.cstat_id)] += 1
                    except KeyError:
                        reactivate[(arrival_time, depot.cstat_id)] = 1
        # reactivate drivers with other vehicles at depot
        for k, nr_activate in reactivate.items():
            activate_time, depot_id = k
            depot = found_depots[depot_id]
            self.fleetctrl.charging_management.add_time_triggered_activate(activate_time, self.fleetctrl.op_id,
                                                                           nr_activate, depot)

    def _low_soc_veh_to_public_cs(self, sim_time):
        """This charging strategy sends low SOC vehicles to the next available public charging infrastructure and
        charges it there.

        :param sim_time: current simulation time
        :return: None
        """
        raise NotImplementedError("Threshold charging with public charging infrastructure not yet implemented.")
