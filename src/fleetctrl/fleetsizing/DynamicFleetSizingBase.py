from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, List
import logging

from src.misc.globals import *
from src.fleetctrl.planning.VehiclePlan import RoutingTargetPlanStop, VehiclePlan
if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.simulation.Vehicles import SimulationVehicle
    from src.infra.ChargingInfrastructure import Depot

LOG = logging.getLogger(__name__)

LARGE_INT = 10000000
REACTIVATE_TIME = 15*60 # reactivation of another vehicle if it was not possible

INPUT_PARAMETERS_DynamicFleetSizingBase = {
    "doc" :  """This sub-module is used to dynamically active or deactive vehicles. Deactivated vehicles are sent back to the depot
        and are not available for customer assignments. This can be used model driver shift for example. """,
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class DynamicFleetSizingBase(ABC):
    def __init__(self, fleetctrl: FleetControlBase, operator_attributes: dict, solver: str="Gurobi"):
        """Initialization of repositioning class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        self.fleetctrl = fleetctrl
        self.routing_engine = fleetctrl.routing_engine
        self.op_charge_depot_infra = self.fleetctrl.op_charge_depot_infra
        if self.op_charge_depot_infra is None:
            raise EnvironmentError("For Dynamic fleet sizing depots for the operators have to be specified!")
        self.solver_key = solver
        self.sorted_time_activate = []
        self.sorted_time_deactivate = []
        self.keep_free_depot_cu = 0 # TODO (parameter to keep free depot charging spots otherwise filled with parking vehicles)
        # children classes:
        # - check of additionally required attributes from operator_attributes
        # - save these as class attributes
        
    def add_init(self, operator_attributes: dict):
        """ additional loading for stuff that has to be initialized after full init of fleetcontrol"""
        pass

    @abstractmethod
    def check_and_change_fleet_size(self, sim_time):
        """This method checks whether fleet vehicles should be activated or deactivated.

        :param sim_time: current simulation time
        :return: net change in fleet size
        """
        return 0

    def deactivate_vehicle(self, veh_obj : SimulationVehicle, sim_time: int, depot: Depot=None):
        """This method is used to send a vehicle to a depot and make it inactive. If not depot is given, the nearest
        depot with free parking lots is chosen. The out-of-service PlanStop is generated and has to be assigned by
        the fleet control. The respective fleet control classes are responsible to remove all other hypothetical
        vehicle plans for that vehicle.

        :param veh_obj: simulation vehicle to send into inactive status
        :param sim_time: current simulation time
        :param depot: optional parameter to choose a depot to pick a vehicle from
        :return: Depot, PlanStop or None, None if no free depot can be found
        """
        if depot is not None and depot.free_parking_spots > 0:
            next_free_depot = depot
        else:
            final_veh_pos = veh_obj.pos
            if veh_obj.assigned_route:
                final_veh_pos = veh_obj.assigned_route[-1].destination_pos
            next_free_depot = self.op_charge_depot_infra.find_nearest_free_depot(final_veh_pos, veh_obj.op_id)
        if not next_free_depot:
            LOG.warning(f"Could not find a free depot for vehicle {veh_obj} at time {sim_time}.")
            return None, None
        LOG.info(f"Deactivating vehicle {veh_obj} at depot {next_free_depot} (plan time: {sim_time})")
        next_free_depot.schedule_inactive(veh_obj)
        ps = RoutingTargetPlanStop(next_free_depot.pos, duration=LARGE_INT, locked=True, planstop_state=G_PLANSTOP_STATES.INACTIVE)
        ass_plan = self.fleetctrl.veh_plans[veh_obj.vid]
        ass_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
        self.fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
        self.fleetctrl.assign_vehicle_plan(veh_obj, ass_plan, sim_time)
        return next_free_depot, ps
    
    def activate_vehicle(self, sim_time: int, depot: Depot=None) -> SimulationVehicle:
        """This method activates a vehicle at a depot. Either the depot is given (and has vehicles) or the depot with
        the most vehicles will be chosen. The method will call the end_current_leg() function for the respective vehicle
        and call the receive_status_update() method of the fleet control.

        :param sim_time: simulation time
        :param depot: optional parameter to choose a depot to pick a vehicle from
        :return: vehicle object
        """
        depot_input = depot
        if depot is None:
            most_parking_veh = 0
            for possible_depot in self.op_charge_depot_infra.depot_by_id.values():
                if possible_depot.parking_vehicles > most_parking_veh:
                    depot = possible_depot
                    most_parking_veh = depot.parking_vehicles
        if depot is None:
            return None
        veh_obj = depot.pick_vehicle_to_be_active()
        LOG.info(f"Activating vehicle {veh_obj} from depot {depot} (plan time: {sim_time})")
        if veh_obj is not None:
            if len(self.fleetctrl.veh_plans[veh_obj.vid].list_plan_stops) > 1:
                # for ps in self.fleetctrl.veh_plans[veh_obj.vid].list_plan_stops[1:]:
                #     if ps.get_state() == G_PLANSTOP_STATES.CHARGING:
                #         charging_process_id = ps.get_charging_task_id()
                #         charging_process = self.fleetctrl._active_charging_processes[charging_process_id]
                #         LOG.debug("cancel charging process {}".format(charging_process))
                #         charging_process.station.cancel_booking(sim_time, charging_process)
                new_veh_plan = VehiclePlan(veh_obj, sim_time, self.fleetctrl.routing_engine, self.fleetctrl.veh_plans[veh_obj.vid].list_plan_stops[:1])
                self.fleetctrl.assign_vehicle_plan(veh_obj, new_veh_plan, sim_time, force_assign=True)
            _, inactive_vrl = veh_obj.end_current_leg(sim_time)
            self.fleetctrl.receive_status_update(veh_obj.vid, sim_time, [inactive_vrl])
            depot.schedule_active(veh_obj)
        else:
            LOG.info("Activation failed!")
            if depot_input is not None:
                LOG.warning(f"Activation failed! Trying to activate again in {REACTIVATE_TIME/60} min")
                new_activate_time = sim_time + REACTIVATE_TIME
                self.add_time_triggered_activate(new_activate_time, 1, depot=depot_input)
        return veh_obj
    
    def add_time_triggered_activate(self, activate_time, nr_activate, depot=None):
        """This method can be called if vehicles should be activated at a certain point later in the simulation. This
        can be useful when a vehicle has to return to depot due to low charge, but the fleet size should not change,
        i.e. the driver changes vehicle.

        :param activate_time: time to activate a vehicle
        :param nr_activate: nr of vehicles to activate
        :param depot: (optional) depot where vehicle should be activated
        :return: None
        """
        self.sorted_time_activate.append((activate_time, nr_activate, depot))
        LOG.debug("sorted_time_activate {}".format(self.sorted_time_activate))
        self.sorted_time_activate.sort(key=lambda x:x[0])

    def add_time_triggered_deactivate(self, deactivate_time, nr_deactivate, depot=None):
        """This method can be called if vehicles should be deactivated at a certain point later in the simulation.

        :param deactivate_time: time to deactivate a vehicle
        :param nr_deactivate: nr of vehicles to deactivate
        :param depot: (optional) depot where vehicle should be activated
        :return: None
        """
        self.sorted_time_deactivate.append((deactivate_time, nr_deactivate, depot))
        LOG.debug("sorted_time_deactivate {}".format(self.sorted_time_deactivate))
        self.sorted_time_deactivate.sort(key=lambda x:x[0])
    
    def time_triggered_deactivate(self, simulation_time, list_veh_obj : List[SimulationVehicle]=None):
        """This method can be utilized to deactivate a certain number of vehicles in a time-controlled fashion. This
        can be useful if the fleet size should be limited in a time-controlled fashion. This method calls the depot
        method for a subset of list_veh_obj: preferably idle vehicles, or the vehicles with few VRLs in their assigned
        route. The method adapts the number of free parking facilities and returns a list of (veh_obj, PlanStop) tuples
        that can be assigned by the fleet control.

        :param simulation_time: current simulation time
        :param list_veh_obj: optional: list of simulation vehicles (subset of all); if not given, all vehicles are used
        :return: list of (veh_obj, depot, PlanStop)
        """
        if not self.sorted_time_deactivate:
            return []
        if self.sorted_time_deactivate[0][0] < simulation_time:
            return []
        number_deactivate = 0
        list_remove_index = []
        for i in range(len(self.sorted_time_deactivate)):
            if self.sorted_time_deactivate[i][0] <= simulation_time:
                number_deactivate += self.sorted_time_deactivate[i][1]
                list_remove_index.append(i)
        for i in reversed(list_remove_index):
            del self.sorted_time_deactivate[i]
        if list_veh_obj is None:
            list_veh_obj = self.fleetctrl.sim_vehicles
        return_list = []
        list_next_prio = []     # list of (number_vrl, veh_obj) where no charging or repositioning tasks are assigned
        list_other = []         # list of other veh_obj that are not considered inactive yet
        LOG.info("to deactivate: {}".format(number_deactivate))
        for veh_obj in list_veh_obj:
            # stop if enough vehicles are chosen
            if number_deactivate == 0:
                break
            # pick idle to deactivate
            if not veh_obj.assigned_route:
                depot, depot_ps = self.deactivate_vehicle(veh_obj, simulation_time)
                if depot is None:
                    continue
                return_list.append((veh_obj, depot, depot_ps))
                number_deactivate -= 1
            else:
                # sort other vehicles
                prio = True
                other = True
                vrl_counter = 0
                for vrl in veh_obj.assigned_route:
                    vrl_counter += 1
                    if vrl.status == VRL_STATES.OUT_OF_SERVICE:
                        prio = False
                        other = False
                        break
                    elif vrl.status in [VRL_STATES.BOARDING, VRL_STATES.CHARGING]:
                        prio = False
                if prio:
                    list_next_prio.append([vrl_counter, veh_obj])
                elif other:
                    list_other.append(veh_obj)
        if number_deactivate > 0:
            # deactivate prioritized vehicles next
            list_next_prio.sort(key=lambda x:x[0])
            number_prio_picks = min(number_deactivate, len(list_next_prio))
            for i in range(number_prio_picks):
                veh_obj = list_next_prio[i][1]
                depot, depot_ps = self.deactivate_vehicle(veh_obj, simulation_time)
                if depot is None:
                    continue
                return_list.append((veh_obj, depot, depot_ps))
                number_deactivate -= 1
            # deactivate other vehicles next
            number_other_picks = min(number_deactivate, len(list_other))
            for i in range(number_other_picks):
                veh_obj = list_other[i]
                depot, depot_ps = self.deactivate_vehicle(veh_obj, simulation_time)
                if depot is None:
                    continue
                return_list.append((veh_obj, depot, depot_ps))
                number_deactivate -= 1
        if number_deactivate > 0:
            LOG.warning(f"Depot-Management of Operator {self.fleetctrl}: could not deactivate as many vehicles as planned."
                        f"{number_deactivate} de-activations could not be conducted.")
        return return_list

    def time_triggered_activate(self, simulation_time):
        """This method can be utilized to activate a certain number of vehicles in a time-controlled fashion. This can
        for example be useful if a driver drives to a depot to deactivate a vehicle and use another one after a certain
        time. The method will call the end_current_leg() function for the respective vehicle
        and call the receive_status_update() method of the fleet control.

        :param fleetctrl: FleetControl class
        :param simulation_time: current simulation time
        :return: list of activated vehicles
        """
        list_remove_index = []
        return_veh_list = []
        for i in range(len(self.sorted_time_activate)):
            activate_time, nr_activate, depot = self.sorted_time_activate[i]
            if simulation_time >= activate_time:
                list_remove_index.append(i)
                for _ in range(nr_activate):
                    return_veh_list.append(self.activate_vehicle(simulation_time, depot))
            else:
                break
        for i in reversed(list_remove_index):
            del self.sorted_time_activate[i]
        return return_veh_list

    def fill_charging_units_at_depot(self, simulation_time):
        """This method automatically fills empty charging units at a depot with vehicles parking there.
        This method creates the VCLs, assigns them to the charging units and also assigns the VCLs and PlanStops to
        the vehicles. The charging tasks are locked and are followed by a status 5 (out of service) stop.

        :param fleetctrl: FleetControl class
        :param simulation_time: current simulation time
        :return: None
        """
        for depot_obj in self.op_charge_depot_infra.depot_by_id.values():
            depot_obj.refill_charging(self.fleetctrl, simulation_time, keep_free_for_short_term=self.keep_free_depot_cu)