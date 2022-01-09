# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
from __future__ import annotations
import os
import logging

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING

# src imports
# -----------
from src.misc.config import decode_config_str
from src.misc.distributions import draw_from_distribution_dict
from src.simulation.Legs import VehicleChargeLeg
from src.simulation.Vehicles import SimulationVehicle
from src.fleetctrl.planning.VehiclePlan import ChargingPlanStop, RoutingTargetPlanStop, PlanStopBase
if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)
LARGE_INT = 10000000
REACTIVATE_TIME = 15*60 # reactivation of another vehicle if it was not possible


class ChargingUnit:
    def __init__(self, cu_id : int, node_index : int, max_power : float):
        self.cu_id = cu_id  # tuple (charging_station_id, charging_unit subid)
        self.node_index = node_index
        self.pos = (node_index, None, None)
        self.max_power = max_power
        self.charging_schedule : List[VehicleChargeLeg] = []  # time-sorted list of VehicleChargeLeg
        self.counter = 0

    def get_id(self):
        return self.cu_id

    def get_pos(self):
        return self.pos

    def get_max_power(self):
        return self.max_power

    def is_currently_available(self, sim_time, min_charging_duration=None):
        """This method checks whether a charging unit is free for at least min_charging_time seconds.

        :param: sim_time: current simulation time
        :param min_charging_duration: minimum charging duration
        :return: True / False
        """
        av = True
        if self.charging_schedule:
            if self.charging_schedule[0].is_active() or self.charging_schedule[0].earliest_start_time == sim_time:
                av = False
            elif min_charging_duration:
                if sim_time + min_charging_duration > self.charging_schedule[0].earliest_start_time:
                    av = False
        return av

    def check_charging_possibility(self, veh_obj : SimulationVehicle, estimated_arrival_time : float, estimated_start_soc : float,
                                   desired_end_soc : float, stop_unlocked : bool=True, power : float=None,
                                   min_charging_duration : float=0) -> Tuple[float, float, float, float, VehicleChargeLeg]:
        """This method checks when the next charging slot after estimated_arrival_time is available and how long the
        availability lasts. The algorithm assumes that not locked charging jobs can be stopped if stop_unlocked is True.
        Either desired_end_soc or desired_end_time have to be set!

        :param veh_obj: battery information; methods compute_soc_charging() and get_charging_duration()
        :param estimated_arrival_time: arrival time according to current route plan
        :param estimated_start_soc: arrival soc according to current route plan
        :param desired_end_soc: desired end soc for charging process
        :param stop_unlocked: if True, scheduled charging processes without lock can be shortened
        :param power: charging assumes self.max_power if not further specified (None)
        :param min_charging_duration: minimum time that a charging unit schedule should be free to be considered
        :return: start_time, end_time, end_soc, power, VCL_to_terminate_at_estimated_arrival_time
        """
        last_possible_start_time = estimated_arrival_time
        charging_job_to_terminate_at_estimated_arrival_time = None
        charging_max_end_time = float("inf")
        for i in range(len(self.charging_schedule)):
            scheduled_job = self.charging_schedule[i]
            # 1) check current schedule for first task ending after planned arrival time
            job_start_reserved_time, job_end_reserved_time = scheduled_job.get_current_schedule_info()
            if job_end_reserved_time < estimated_arrival_time:
                last_possible_start_time = job_end_reserved_time
                continue
            # 2) check whether this job is locked
            if not stop_unlocked or scheduled_job.locked:
                last_possible_start_time = job_end_reserved_time
            # 3) check whether there is a job following this one
            # TODO # include possibility to shorten next charging job from front
            try:
                next_job_start_time = self.charging_schedule[i + 1].get_current_schedule_info()[0]
            except IndexError:
                next_job_start_time = float("inf")
            # append charging task later if min charging duration cannot be kept
            if next_job_start_time - last_possible_start_time > min_charging_duration:
                if not scheduled_job.locked:
                    charging_job_to_terminate_at_estimated_arrival_time = scheduled_job
                charging_max_end_time = next_job_start_time
                break
        #
        start_time = last_possible_start_time
        # check if time or soc will end first
        if power is None:
            charging_power = self.max_power
        else:
            charging_power = min(power, self.max_power)
        desired_soc_charging_duration = veh_obj.get_charging_duration(charging_power, estimated_start_soc,
                                                                      desired_end_soc)
        if start_time + desired_soc_charging_duration < charging_max_end_time:
            end_time = np.ceil(start_time + desired_soc_charging_duration)
            end_soc = desired_end_soc
        else:
            end_time = charging_max_end_time
            c_duration = end_time - start_time
            end_soc = round(estimated_start_soc + veh_obj.compute_soc_charging(power, c_duration), 3)
        return start_time, end_time, end_soc, power, charging_job_to_terminate_at_estimated_arrival_time

    def schedule_charging_job(self, veh_obj : SimulationVehicle, power : float, earliest_start_time : float, desired_soc : float,
                              latest_end_time : float, locked : bool) -> VehicleChargeLeg:
        """This method creates a VehicleChargingLeg and inserts it in the schedule at the correct position (according
        to time). Returns the respective VCL, which still has to be assigned by the fleet control in
        VehiclePlan.buildVR().

        :param veh_obj: vehicle object to be assigned
        :param power: desired power; will be used unless power > ChargingUnit.max_power
        :param earliest_start_time: earliest start time for charging process
        :param desired_soc: desired state of charge at end of charging process
        :param latest_end_time: possible latest time because of later obligations (vehicle or charging unit)
        :param locked: locked or unlocked charging process
        :return: VCL
        """
        # TODO # think about change in process flow: use plan-stop or vcl as input?
        new_vcl_id = (self.cu_id, veh_obj.op_id, veh_obj.vid, self.counter)
        self.counter += 1
        new_job_position = 0
        for scheduled_vcl in self.charging_schedule:
            if scheduled_vcl.is_active():
                new_job_position += 1
            else:
                if earliest_start_time > scheduled_vcl.earliest_start_time:
                    new_job_position += 1
        if power > self.max_power:
            power = self.max_power
        new_vcl = VehicleChargeLeg(new_vcl_id, self, power, earliest_start_time, desired_soc, latest_end_time, locked)
        self.charging_schedule.insert(new_job_position, new_vcl)
        # charging_ps = PlanStop(self.pos, {}, {}, {self.cu_id: earliest_start_time}, {}, 0,
        #                        planned_departure={G_FCTRL_DEP_DUR: new_vcl.duration, G_FCTRL_DEP_TIME: latest_end_time},
        #                        planned_arrival_soc=desired_soc, locked=locked, charging_power=power)
        return new_vcl

    def remove_job_from_schedule(self, vcl_id):
        """This method removes a schedule charging process from the charging unit schedule.

        :param vcl_id: VehicleChargeLeg id
        :return: None
        """
        counter = 0
        return_index = None
        for scheduled_vcl in self.charging_schedule:
            if scheduled_vcl.get_vcl_id() == vcl_id:
                return_index = counter
            counter += 1
        if return_index is not None:
            del self.charging_schedule[return_index]


class ChargingStation:
    def __init__(self, cstat_id, cstat_row):
        self.cstat_id = cstat_id
        self.node_index = cstat_row[G_NODE_ID]
        self.pos = (self.node_index, None, None)
        self.owners = []
        self.max_parking = None         # for depots
        self.scheduled_parking : List[SimulationVehicle] = []     # list veh_obj (best keep sorted by SOC)
        self.pub_util = None            # curve for public utilization of charging station
        cunit_dict = decode_config_str(cstat_row[G_INFRA_CU_DEF])
        if cunit_dict is None:
            cunit_dict = {}
        self.c_units : Dict[int, ChargingUnit] = {}               # cunit_id -> cunit_obj
        self.c_units_per_power : Dict[float, List[ChargingUnit]] = {}     # power -> list cunit_obj
        c_counter = 0
        for power, amount in cunit_dict.items():
            for i in range(amount):
                cunit_id = (self.cstat_id, c_counter)
                c_counter += 1
                tmp_cunit = ChargingUnit(cunit_id, self.node_index, power)
                self.c_units[cunit_id] = tmp_cunit
                try:
                    self.c_units_per_power[power].append(tmp_cunit)
                except:
                    self.c_units_per_power[power] = [tmp_cunit]

    def __str__(self):
        return f"{self.cstat_id} (at node {self.node_index})"

    def __lt__(self,other):
        if other is None:
            return False
        return (self.cstat_id<other.cstat_id)
    def __le__(self,other):
        if other is None:
            return False
        return(self.cstat_id<=other.cstat_id)
    def __gt__(self,other):
        if other is None:
            return True
        return(self.cstat_id>other.cstat_id)
    def __ge__(self,other):
        if other is None:
            return False
        return(self.cstat_id>=other.cstat_id)
    def __eq__(self,other):
        if other is None:
            return False
        return (self.cstat_id==other.cstat_id)
    def __ne__(self,other):
        if other is None:
            return True
        return not(self.__eq__(other))

    def get_charging_units(self):
        return self.c_units

    def check_charging_unit_availabilities(self, veh_obj, estimated_arrival_time, estimated_start_soc, desired_end_soc,
                                           stop_unlocked=True, power=None, return_no_terminate_only=False,
                                           prioritize_no_terminate=True, return_first=True,
                                           min_charging_duration=0) -> List[Tuple[bool, float, float, float, float, int, VehicleChargeLeg]]:
        """This method calls check_charging_possibility() for its charging units. This method can already make a
        selection of its charging units and return only a subset of charging units, for which no other VCLs have to be
        terminated. Moreover, there is an option to stop the search after the first valid search.
        If power is given as an argument, charging units with increasing max_power above power are searched first.

        :param veh_obj: simulation vehicle instance
        :param estimated_arrival_time: estimated arrival time at charging station
        :param estimated_start_soc: estimated soc at arrival
        :param desired_end_soc: planned end soc after charging
        :param stop_unlocked: if True, already scheduled unlocked VCLs can be terminated/started later
        :param power: charging assumes self.max_power if not further specified (None)
        :param return_no_terminate_only: if True, does not consider charging units if VCLs would have to be terminated
        :param prioritize_no_terminate: if True, only returns charging units with terminating VCLS if necessary
        :param return_first: return first (prioritized/considered) solution
        :param min_charging_duration: minimum time window length to be considered for charging
        :return: list of (terminate_flag, end_soc, power, start_time, end_time, cunit,
                            VCL_to_terminate_at_estimated_arrival_time) tuples
        """
        if self.owners is not None and veh_obj.op_id not in self.owners:
            LOG.debug(f"Vehicle {veh_obj.vid} of operator {veh_obj.op_id} cannot charge at charging station "
                      f"{self.cstat_id} with ownership attribute {self.owners}.")
            return []
        list_prio_charging_options = []
        list_no_prio_charging_options = []
        if power:
            # start with max_power options that are above the wished power in ascending order -> do not waste power
            # add units with max_power below the desired power in descending order -> get as close as possible
            power_keys = sorted([x for x in self.c_units_per_power.keys() if x >= power]) +\
                         sorted([x for x in self.c_units_per_power.keys() if x < power], reverse=True)
        else:
            power_keys = sorted(self.c_units_per_power.keys(), reverse=True)
        for max_power in power_keys:
            for cunit in self.c_units_per_power[max_power]:
                start_time, end_time, end_soc, power, charging_job_to_terminate_at_estimated_arrival_time = \
                    cunit.check_charging_possibility(veh_obj, estimated_arrival_time, estimated_start_soc,
                                                     desired_end_soc, stop_unlocked, power, min_charging_duration)
                if charging_job_to_terminate_at_estimated_arrival_time is not None:
                    terminate_flag = True
                else:
                    terminate_flag = False
                res_tuple = (terminate_flag, end_soc, power, start_time, end_time, cunit,
                             charging_job_to_terminate_at_estimated_arrival_time)
                # go through different cases depending on function argument flags
                if return_no_terminate_only:
                    if terminate_flag:
                        if return_first and end_soc == desired_end_soc:
                            return [res_tuple]
                        list_prio_charging_options.append(res_tuple)
                elif prioritize_no_terminate:
                    if terminate_flag:
                        if return_first and end_soc == desired_end_soc:
                            return [res_tuple]
                        list_prio_charging_options.append(res_tuple)
                    else:
                        list_no_prio_charging_options.append(res_tuple)
                else:
                    list_prio_charging_options.append(res_tuple)
                    if return_first and end_soc == desired_end_soc:
                        return [res_tuple]
        # combination of prioritized and not prioritized options
        if prioritize_no_terminate and list_prio_charging_options:
            return_list = list_prio_charging_options
        else:
            return_list = list_prio_charging_options + list_no_prio_charging_options
        if return_first:
            return_list.sort()
            return return_list[:1]
        return return_list

    def optimize_schedules_by_shifting_vcl(self, simulation_time, horizon, objective="Default"):
        """This method can be used to shift VCL between charging units of one station in order to optimize a given
        objective.

        :param simulation_time: current simulation time
        :param horizon: time horizon for optimization
        :param objective: optional parameter if non-default optimization
        :return: None
        """
        # TODO # check ownership
        # TODO # method to shift vehicles between charging units to optimize schedule (later)
        pass


class Depot(ChargingStation):
    def __init__(self, depot_id, cstat_row):
        super().__init__(depot_id, cstat_row)
        # assumption: there are at least as many parking lots as charging units
        max_parking = cstat_row[G_INFRA_MAX_PARK]
        if max_parking:
            self.max_parking = max(max_parking, len(self.c_units))
        else:
            self.max_parking = len(self.c_units)

    def set_ownership(self, list_owners):
        """This method sets the ownership of depots, which can be set by the scenario attribute G_INFRA_OP_OWNER.

        :param list_owners: either None (available to all) or list of operator-ids
        :return: None
        """
        self.owners = list_owners

    def check_free_parking(self, op_id):
        """This method checks whether a free parking lot is available at a depot. For simplicity, all vehicles
        scheduled to arrive at a depot are already counted.

        :param op_id: operator_id -> ownership has to be checked
        :return: True/False
        """
        if self.owners is not None and op_id not in self.owners:
            LOG.debug(f"Operator {op_id} cannot park at depot {self.cstat_id} with ownership attribute {self.owners}.")
            return False
        if not self.max_parking:
            return True
        if len(self.scheduled_parking) >= self.max_parking:
            return False
        return True

    def schedule_inactive(self, veh_obj):
        """This method schedules a vehicle for the depot, thereby increasing the number of scheduled parking vehicles.

        :param veh_obj: simulation vehicle
        :return: None
        """
        self.scheduled_parking.append(veh_obj)
        self.scheduled_parking.sort(key=lambda x: x.soc)

    def pick_vehicle_to_be_active(self, op_id, force=False, check_only=False) -> SimulationVehicle:
        """This method picks a vehicle to change status from 5 ('out of service') to 0 ('idle'). The algorithm picks
        the vehicle with status 5 with the highest SOC. If force=True, the charging vehicle with highest SOC would be
        picked, if no status 5 vehicle is available. It would even ignore the locked status of a charging process.
        Moreover, the veh_obj of the respective vehicle is returned. None is returned if no vehicle can become active.

        :param op_id: only activate vehicle from this operator
        :param force: enable forcing a vehicle to stop charging in order to become active
        :param check_only: vehicle is not actually removed from scheduled_parking
        :return: veh_obj / None
        """
        pop_nr = None
        list_possible = []
        for i in range(len(self.scheduled_parking)):
            veh_obj = self.scheduled_parking[i]
            if veh_obj.op_id == op_id:
                if veh_obj.status == 5:
                    pop_nr = i
                    break
                list_possible.append(i)
        if force and pop_nr is None and self.scheduled_parking:
            pop_nr = list_possible[0]
        if pop_nr is not None and not check_only:
            return self.scheduled_parking.pop(pop_nr)
        elif pop_nr and check_only:
            return self.scheduled_parking[pop_nr]
        else:
            return None

    def refill_charging(self, fleetctrl : FleetControlBase, simulation_time, keep_free_for_short_term=0,
                        min_charging_duration=0):
        """This method fills empty charging slots in a depot with the lowest SOC parking (status 5) vehicles. The
        vehicles receive a locked PlanStop/VCL followed, which will be followed by another status 5 PlanStop/VRL.
        These will directly be assigned to the vehicle. The VCL will also be assigned to the charging unit.

        :param fleetctrl: FleetControl class
        :param simulation_time: current simulation time
        :param keep_free_for_short_term: optional parameter in order to keep short-term charging capacity
        :param min_charging_duration:
        :return: None
        """
        # check for vehicles that require charging
        list_consider_charging = []
        for veh_obj in self.scheduled_parking:
            if veh_obj.soc == 1.0 or veh_obj.status != 5:
                continue
            # check whether veh_obj already has vcl
            consider_charging = True
            for vrl in veh_obj.assigned_route:
                if vrl.status == 2:
                    consider_charging = False
            if consider_charging:
                list_consider_charging.append(veh_obj)
        if not list_consider_charging:
            return

        for power in sorted(self.c_units_per_power.keys(), reverse=True):
            if not list_consider_charging:
                break
            for cunit in self.c_units_per_power[power]:
                if not list_consider_charging:
                    break
                # 1) check free charging units
                if cunit.is_currently_available(simulation_time, min_charging_duration):
                    if keep_free_for_short_term > 0:
                        keep_free_for_short_term -= 1
                        continue
                    # 2) fill them from lowest SOC vehicle
                    remove_from_consideration = []
                    for veh_obj in list_consider_charging:
                        # check whether veh_obj already has vcl
                        consider_charging = True
                        for vrl in veh_obj.assigned_route:
                            if vrl.status == 2:
                                consider_charging = False
                        if consider_charging:
                            # 3) create PlanStops, VCL and VRL
                            desired_end_soc = 1.0
                            # get cunit availability information
                            start_time, end_time, end_soc, _, _ = \
                                cunit.check_charging_possibility(veh_obj, simulation_time, veh_obj.soc, desired_end_soc,
                                                                 stop_unlocked=False,
                                                                 min_charging_duration=min_charging_duration)
                            # schedule charging first (in order to add VCL-Id and cunit-id to PlanStop)
                            vcl = cunit.schedule_charging_job(veh_obj, power, simulation_time, end_soc, end_time, True)
                            # create PlanStop
                            duration = end_time - start_time
                            ps = ChargingPlanStop(cunit.pos, duration=duration,
                                                  locked=True, charging_power=power, existing_vcl=vcl.get_vcl_id(), charging_unit_id=cunit.cu_id) # TODO planned attributes have been specified here as input before! why?
                            if start_time == simulation_time:
                                # finish current status 5 task
                                veh_obj.end_current_leg(simulation_time)
                                # modify veh-plan: insert charging before list position -1
                                fleetctrl.veh_plans[veh_obj.vid].add_plan_stop(ps, veh_obj, simulation_time,
                                                                               fleetctrl.routing_engine,
                                                                               return_copy=False, position=-1)
                            else:
                                # modify veh-plan:
                                # set inactive task duration correctly
                                current_inactive_stop = fleetctrl.veh_plans[veh_obj.vid].list_plan_stops[-1]
                                last_start = current_inactive_stop.get_started_at()
                                if last_start is None:
                                    last_start = current_inactive_stop.get_planned_arrival_and_departure_time()[0]
                                complete_duration = start_time - last_start
                                current_inactive_stop.set_duration_and_earliest_end_time(duration=complete_duration)
                                # insert charging before list after current stop
                                fleetctrl.veh_plans[veh_obj.vid].add_plan_stop(ps, veh_obj, simulation_time,
                                                                               fleetctrl.routing_engine,
                                                                               return_copy=False)
                                # append another inactive task after that
                                inactive_ps = RoutingTargetPlanStop(self.pos, locked=True, duration=LARGE_INT, planstop_state=G_PLANSTOP_STATES.INACTIVE)
                                fleetctrl.veh_plans[veh_obj.vid].add_plan_stop(inactive_ps, veh_obj, simulation_time,
                                                                               fleetctrl.routing_engine,
                                                                               return_copy=False)
                            fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
                            # assign vehicle plan
                            fleetctrl.assign_vehicle_plan(veh_obj, fleetctrl.veh_plans[veh_obj.vid], simulation_time)
                            remove_from_consideration.append(veh_obj)
                            # only 1 vehicle per charging unit
                            break
                    for veh_obj in remove_from_consideration:
                        list_consider_charging.remove(veh_obj)


class PublicChargingStation(ChargingStation):
    # TODO # use negative ids to distinguish ids from Depots!
    # TODO # consider public utilization profile
    pass


class ChargingAndDepotManagement:
    def __init__(self, depot_f, pub_cs_f, routing_engine, scenario_pars, list_operator_attributes):
        """This class serves as an interface between FleetControl and Depots and Charging-Stations with its
        Charging-Units.

        :param depot_f: file that contains the depots of operators
        :param pub_cs_f: file that contains the public charging infrastructure
        :param routing_engine: routing engine
        :param scenario_pars: input-parameter dictionary
        :param list_operator_attributes: list of operator input-parameter dictionaries
        """
        self.routing_engine = routing_engine
        self.sim_start_time = scenario_pars[G_SIM_START_TIME]
        self.depots : Dict[int, Depot] = {}    # depot_id = charging_station_id -> Depot
        if os.path.isfile(depot_f):
            tmp_df = pd.read_csv(depot_f, index_col=0)
            for depot_id, row in tmp_df.iterrows():
                self.depots[depot_id] = Depot(depot_id, row)
        self.public_cs : Dict[int, PublicChargingStation] = {} # pcs = charging_station_id -> PublicChargingStation [use negative ids]
        if os.path.isfile(pub_cs_f):
            tmp_df = pd.read_csv(pub_cs_f, index_col=0)
            for pub_cstat_id, row in tmp_df.iterrows():
                self.public_cs[pub_cstat_id] = PublicChargingStation(pub_cstat_id, row)
        self.charging_units : Dict[Tuple[int, int], ChargingUnit] = {}  # tuple: (charging_station_id, charging_station_unit_id) -> ChargingUnit
        for cs in self.depots.values():
            self.charging_units.update(cs.get_charging_units())
        for cs in self.public_cs.values():
            self.charging_units.update(cs.get_charging_units())
        self.keep_free_depot_cu = scenario_pars.get(G_INFRA_DEPOT_FOR_ST, 0)
        self.sorted_time_deactivate = []    # list of (sim_time, op_id, nr_deactivate) tuples
        self.sorted_time_activate = []  # list of (sim_time, op_id, nr_activate, depot) tuples | depot=None default
        # TODO # ownership can be set via scenario_pars!
        ownerships = {}
        for op_id in range(len(list_operator_attributes)):
            operator_attributes = list_operator_attributes[op_id]
            owned_depots = operator_attributes.get(G_INFRA_OP_OWNER)
            if type(owned_depots) == list:
                for depot_id in owned_depots:
                    try:
                        ownerships[depot_id].append(op_id)
                    except KeyError:
                        ownerships[depot_id] = [op_id]
            elif owned_depots is not None and owned_depots == "all":
                for depot_id in self.depots.keys():
                    try:
                        ownerships[depot_id].append(op_id)
                    except KeyError:
                        ownerships[depot_id] = [op_id]
        for depot_id, list_owners in ownerships.items():
            depot = self.depots.get(depot_id)
            if depot:
                depot.set_ownership(list_owners)
        #
        self.allowed_parking_pos = {}
        self.allow_street_parking = True
        if not scenario_pars.get(G_INFRA_ALLOW_SP, True):
            for depot in self.depots.values():
                self.allowed_parking_pos[depot.pos] = True
        self.node2depot = {}
        # TODO # include electric prices, electricity peak regularization constraints

    def read_active_vehicle_file(self, fleetctrl, active_vehicle_file, scenario_parameters):
        """This method reads a csv file which has at least two columns (time, share_active_veh_change). For times with a
        positive value for active_veh_change, self.time_activate receives an entry, such that the time trigger will
        activate the respective number of vehicles during the simulation. For a negative value, the time trigger will
        deactivate the respective number of vehicles.

        :param fleetctrl: FleetControl class
        :param active_vehicle_file: csv-file containing the change in active vehicles
        :param scenario_parameters: scenario parameter
        :return: None
        """
        df = pd.read_csv(active_vehicle_file, index_col=0)
        LOG.debug("read active vehicle file")
        last_share = None
        # remove entries before simulation start time
        sim_start_time = scenario_parameters[G_SIM_START_TIME]
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
        LOG.debug(df)
        #
        depot_distribution = {}
        for depot_id, depot in self.depots.items():
            if fleetctrl.op_id in depot.owners:
                depot_distribution[depot_id] = depot.max_parking - len(depot.scheduled_parking)
        for sim_time, share_active_veh in df[G_ACT_VEH_SHARE].items():
            if last_share is None:
                # initial inactive vehicles have to be set here; the chosen vehicles will start in the depot
                # -> will not be overwritten by FleetSimulation.set_initial_state() as veh_obj.pos is not None
                number_inactive = min(int(np.math.floor((1 - share_active_veh) * fleetctrl.nr_vehicles)), fleetctrl.nr_vehicles)
                for vid in range(number_inactive):
                    veh_obj = fleetctrl.sim_vehicles[vid]
                    veh_plan = fleetctrl.veh_plans[vid]
                    drawn_depot_id = draw_from_distribution_dict(depot_distribution)
                    drawn_depot = self.depots[drawn_depot_id]
                    veh_obj.status = 0
                    veh_obj.soc = 1.0
                    veh_obj.pos = drawn_depot.pos
                    depot, ps = self.deactivate_vehicle(fleetctrl, veh_obj, self.sim_start_time, drawn_depot)
                    if depot is None:
                        LOG.warning("init active vehicle file: no empty parking space can be found anymore!!")
                        continue
                    # beam vehicle to depot if necessary
                    if ps.pos != veh_obj.pos:
                        veh_obj.pos = ps.pos
                        veh_obj.soc = 1.0
                share_active_veh_change = 0
            else:
                share_active_veh_change = share_active_veh - last_share
            if share_active_veh_change > 0:
                add_active_veh = int(np.round(share_active_veh_change * fleetctrl.nr_vehicles, 0))
                if add_active_veh > 0:
                    self.sorted_time_activate.append((sim_time, fleetctrl.op_id, add_active_veh, None))
            elif share_active_veh_change < 0:
                rem_active_veh = int(np.round(-share_active_veh_change * fleetctrl.nr_vehicles, 0))
                if rem_active_veh > 0:
                    self.sorted_time_deactivate.append((sim_time, fleetctrl.op_id, rem_active_veh))
            last_share = share_active_veh
        # better safe than sorry
        self.sorted_time_activate.sort()
        self.sorted_time_deactivate.sort()
        LOG.debug(f"sorted time activate: {self.sorted_time_activate}")
        LOG.debug(f"sorted time deactivate: {self.sorted_time_deactivate}")
        LOG.info(f"Loaded nr-of-active-vehicles curve from {active_vehicle_file}.")

    def get_charging_unit(self, charging_unit_id):
        """This method returns the charging unit instance with id charging_unit_id.

        :param charging_unit_id: id of charging unit
        :return: reference to charging unit instance or None
        """
        return self.charging_units.get(charging_unit_id)

    def get_vcl(self, charging_unit_id, vcl_id):
        """This method returns the VCL with id vcl_id

        :param charging_unit_id: id of charging unit
        :param vcl_id: id of vehicle charge leg
        :return: VehicleChargeLeg with vcl_id that is assigned to charging unit with charging_unit_id or None
        """
        for check_vcl in self.charging_units[charging_unit_id].charging_schedule:
            if check_vcl.get_vcl_id() == vcl_id:
                return check_vcl
        return None

    def create_new_vcl(self, veh_obj : SimulationVehicle, plan_stop : PlanStopBase):
        """This method creates a new VCL based on plan-stop information.

        :param veh_obj: simulation vehicle instance
        :param plan_stop: PlanStop
        :return: VehicleChargeLeg, power of charging process, cu_id, vcl_id
        """
        charging_unit = self.charging_units[plan_stop.charging_unit_id]
        if plan_stop.get_charging_power() > charging_unit.max_power:
            power = charging_unit.max_power
        else:
            power = plan_stop.power
        latest_end = plan_stop.get_planned_arrival_and_departure_time()[1]
        target_soc = plan_stop.get_planned_arrival_and_departure_soc()[1]
        vcl = charging_unit.schedule_charging_job(veh_obj, power, plan_stop.get_earliest_start_time(), target_soc,
                                                  latest_end, plan_stop.is_locked())
        return vcl, power, charging_unit.get_id(), vcl.get_vcl_id()

    def find_nearest_free_depot(self, pos, op_id, check_free=True):
        """This method can be used to send a vehicle to the next depot.

        :param pos: final vehicle position
        :param op_id: operator id
        :param check_free: if set to False, the check for free parking is ignored
        :return: Depot
        """
        free_depot_positions = {}
        for depot in self.depots.values():
            if depot.check_free_parking(op_id) or not check_free:
                free_depot_positions[depot.pos] = depot
        re_list = self.routing_engine.return_travel_costs_1toX(pos, free_depot_positions.keys(), max_routes=1)
        if re_list:
            destination_pos = re_list[0][0]
            depot = free_depot_positions[destination_pos]
        else:
            depot = None
        return depot

    def find_nearest_depot_replace_veh(self, last_pos, op_id):
        """This method can be used to send a vehicle to a nearby depot that has a free parking spot and is likely to
        have a replacement vehicle. Since pick_vehicle_to_be_active(check_only=True) does not actually reserve a vehicle
        for the replacement, there are no guarantees.

        :param last_pos: final vehicle position
        :param op_id: operator id
        :return: Depot
        """
        possible_depot_positions = {}
        for depot in self.depots.values():
            if depot.check_free_parking(op_id) and depot.pick_vehicle_to_be_active(op_id, check_only=True) is not None:
                possible_depot_positions[depot.pos] = depot
        re_list = self.routing_engine.return_travel_costs_1toX(last_pos, possible_depot_positions.keys(), max_routes=1)
        if re_list:
            destination_pos = re_list[0][0]
            depot = possible_depot_positions[destination_pos]
        else:
            depot = None
        return depot

    def deactivate_vehicle(self, fleetctrl : FleetControlBase, veh_obj : SimulationVehicle, sim_time, depot=None):
        """This method is used to send a vehicle to a depot and make it inactive. If not depot is given, the nearest
        depot with free parking lots is chosen. The out-of-service PlanStop is generated and has to be assigned by
        the fleet control. The respective fleet control classes are responsible to remove all other hypothetical
        vehicle plans for that vehicle.

        :param fleetctrl: FleetControl class
        :param veh_obj: simulation vehicle to send into inactive status
        :param sim_time: current simulation time
        :param depot: optional parameter to choose a depot to pick a vehicle from
        :return: Depot, PlanStop or None, None if no free depot can be found
        """
        if depot is not None and depot.check_free_parking(veh_obj.op_id):
            next_free_depot = depot
        else:
            final_veh_pos = veh_obj.pos
            if veh_obj.assigned_route:
                final_veh_pos = veh_obj.assigned_route[-1].destination_pos
            next_free_depot = self.find_nearest_free_depot(final_veh_pos, veh_obj.op_id)
        if not next_free_depot:
            LOG.warning(f"Could not find a free depot for vehicle {veh_obj} at time {sim_time}.")
            return None, None
        LOG.info(f"Deactivating vehicle {veh_obj} at depot {next_free_depot} (plan time: {sim_time})")
        next_free_depot.schedule_inactive(veh_obj)
        # TODO # is this sufficient to ignore vehicle from further tasks? -> inactive attribute on vehicle level?
        ps = RoutingTargetPlanStop(next_free_depot.pos, duration=LARGE_INT, locked=True, planstop_state=G_PLANSTOP_STATES.INACTIVE)
        # TODO # -> adapt search functions to ignore status=5 vehicles in search
        # TODO # check that this works for all classes -> fleetctrl.veh_plans always holds currently assigned plans?
        ass_plan = fleetctrl.veh_plans[veh_obj.vid]
        ass_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
        fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
        fleetctrl.assign_vehicle_plan(veh_obj, ass_plan, sim_time)
        return next_free_depot, ps

    def time_triggered_deactivate(self, fleetctrl : FleetControlBase, simulation_time, list_veh_obj : List[SimulationVehicle]=None):
        """This method can be utilized to deactivate a certain number of vehicles in a time-controlled fashion. This
        can be useful if the fleet size should be limited in a time-controlled fashion. This method calls the depot
        method for a subset of list_veh_obj: preferably idle vehicles, or the vehicles with few VRLs in their assigned
        route. The method adapts the number of free parking facilities and returns a list of (veh_obj, PlanStop) tuples
        that can be assigned by the fleet control.

        :param fleetctrl: FleetControl class
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
                if fleetctrl.op_id == self.sorted_time_deactivate[i][1]:
                    number_deactivate += self.sorted_time_deactivate[i][2]
                    list_remove_index.append(i)
        for i in reversed(list_remove_index):
            del self.sorted_time_deactivate[i]
        if list_veh_obj is None:
            list_veh_obj = fleetctrl.sim_vehicles
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
                depot, depot_ps = self.deactivate_vehicle(fleetctrl, veh_obj, simulation_time)
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
                    if vrl.status == 5:
                        prio = False
                        other = False
                        break
                    elif vrl.status in [2, 11]:
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
                depot, depot_ps = self.deactivate_vehicle(fleetctrl, veh_obj, simulation_time)
                if depot is None:
                    continue
                return_list.append((veh_obj, depot, depot_ps))
                number_deactivate -= 1
            # deactivate other vehicles next
            number_other_picks = min(number_deactivate, len(list_other))
            for i in range(number_other_picks):
                veh_obj = list_other[i]
                depot, depot_ps = self.deactivate_vehicle(fleetctrl, veh_obj, simulation_time)
                if depot is None:
                    continue
                return_list.append((veh_obj, depot, depot_ps))
                number_deactivate -= 1
        if number_deactivate > 0:
            LOG.warning(f"Depot-Management of Operator {fleetctrl}: could not deactivate as many vehicles as planned."
                        f"{number_deactivate} de-activations could not be conducted.")
        return return_list

    def activate_vehicle(self, fleetctrl, sim_time, depot=None):
        """This method activates a vehicle at a depot. Either the depot is given (and has vehicles) or the depot with
        the most vehicles will be chosen. The method will call the end_current_leg() function for the respective vehicle
        and call the receive_status_update() method of the fleet control.

        :param fleetctrl: FleetControl class
        :param sim_time: simulation time
        :param depot: optional parameter to choose a depot to pick a vehicle from
        :return: vehicle object
        """
        depot_input = depot
        if depot is None:
            most_parking_veh = 0
            for possible_depot in self.depots.values():
                parking_veh = len([veh_obj for veh_obj in possible_depot.scheduled_parking
                                   if veh_obj.op_id == fleetctrl.op_id])
                if parking_veh > most_parking_veh:
                    depot = possible_depot
                    most_parking_veh = parking_veh
        if depot is None:
            return None
        veh_obj = depot.pick_vehicle_to_be_active(fleetctrl.op_id)
        LOG.info(f"Activating vehicle {veh_obj} from depot {depot} (plan time: {sim_time})")
        if veh_obj is not None:
            _, inactive_vrl = veh_obj.end_current_leg(sim_time)
            fleetctrl.receive_status_update(veh_obj.vid, sim_time, [inactive_vrl])
        else:
            LOG.info("Activation failed!")
            if depot_input is not None:
                LOG.warning(f"Activation failed! Trying to activate again in {REACTIVATE_TIME/60} min")
                new_activate_time = sim_time + REACTIVATE_TIME
                self.add_time_triggered_activate(new_activate_time, fleetctrl.op_id, 1, depot=depot_input)
        return veh_obj

    def time_triggered_activate(self, fleetctrl, simulation_time):
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
            activate_time, op_id, nr_activate, depot = self.sorted_time_activate[i]
            if simulation_time >= activate_time and op_id == fleetctrl.op_id:
                list_remove_index.append(i)
                for _ in range(nr_activate):
                    return_veh_list.append(self.activate_vehicle(fleetctrl, simulation_time, depot))
            else:
                break
        for i in reversed(list_remove_index):
            del self.sorted_time_activate[i]
        return return_veh_list

    def add_time_triggered_activate(self, activate_time, op_id, nr_activate, depot=None):
        """This method can be called if vehicles should be activated at a certain point later in the simulation. This
        can be useful when a vehicle has to return to depot due to low charge, but the fleet size should not change,
        i.e. the driver changes vehicle.

        :param activate_time: time to activate a vehicle
        :param op_id: operator id
        :param nr_activate: nr of vehicles to activate
        :param depot: (optional) depot where vehicle should be activated
        :return: None
        """
        self.sorted_time_activate.append((activate_time, op_id, nr_activate, depot))
        LOG.debug("sorted_time_activate {}".format(self.sorted_time_activate))
        self.sorted_time_activate.sort()

    def add_time_triggered_deactivate(self, deactivate_time, op_id, nr_deactivate, depot=None):
        """This method can be called if vehicles should be deactivated at a certain point later in the simulation.

        :param deactivate_time: time to deactivate a vehicle
        :param op_id: operator id
        :param nr_deactivate: nr of vehicles to deactivate
        :param depot: (optional) depot where vehicle should be activated
        :return: None
        """
        self.sorted_time_deactivate.append((deactivate_time, op_id, nr_deactivate, depot))
        LOG.debug("sorted_time_deactivate {}".format(self.sorted_time_activate))
        self.sorted_time_deactivate.sort()

    def compute_activation_for_dynamic_fleetsizing(self, fleetctrl : FleetControlBase, simulation_time):
        """ this function computes vehicles activation/deactivation dynamically based on the current fleet utilization
        activation or deactivation will happen in time_triggered_activate (_deactivate)
        :param fleetctrl: fleet controlf ref
        :param simulation_time: simulation time
        :return: None
        """

        util, n_eff_utilized, n_active = fleetctrl.compute_current_fleet_utilization(simulation_time)
        LOG.info(f"current util at time {simulation_time}: util {util} n_eff_util {n_eff_utilized} n_active {n_active} | first underutilization: {fleetctrl.start_time_underutilization}")
        output_activate = 0
        if util > fleetctrl.target_utilization + fleetctrl.target_utilization_interval:
            to_activate = n_eff_utilized/fleetctrl.target_utilization - n_active
            to_activate = int(np.ceil(to_activate))
            if len(fleetctrl.sim_vehicles) < n_active + to_activate:
                to_activate = len(fleetctrl.sim_vehicles) - n_active
            self.add_time_triggered_activate(simulation_time, fleetctrl.op_id, int(to_activate))
            fleetctrl.start_time_underutilization = simulation_time
            output_activate = to_activate
            LOG.info(f" -> activate {to_activate}")
        elif util < fleetctrl.target_utilization - fleetctrl.target_utilization_interval:
            if simulation_time - fleetctrl.start_time_underutilization > fleetctrl.max_duration_underutilized:
                to_deactivate = n_active - n_eff_utilized/fleetctrl.target_utilization
                to_deactivate = int(np.ceil(to_deactivate))
                if n_active - to_deactivate < fleetctrl.minimum_active_fleetsize:
                    to_deactivate = n_active - fleetctrl.minimum_active_fleetsize
                self.add_time_triggered_deactivate(simulation_time, fleetctrl.op_id, int(to_deactivate))
                LOG.info(f" -> deactivate {to_deactivate}")
                fleetctrl.start_time_underutilization = simulation_time
                output_activate = -to_deactivate
        else:
            fleetctrl.start_time_underutilization = simulation_time

        #dynamic output
        dyn_output_dict = {
            "utilization" : util,
            "effective utilized vehicles" : n_eff_utilized,
            "active vehicles" : n_active,
            "activate vehicles" : output_activate,
        }
        fleetctrl._add_to_dynamic_fleetcontrol_output(simulation_time, dyn_output_dict)

    def fill_charging_units_at_depot(self, fleetctrl, simulation_time):
        """This method automatically fills empty charging units at a depot with vehicles parking there.
        This method creates the VCLs, assigns them to the charging units and also assigns the VCLs and PlanStops to
        the vehicles. The charging tasks are locked and are followed by a status 5 (out of service) stop.

        :param fleetctrl: FleetControl class
        :param simulation_time: current simulation time
        :return: None
        """
        for depot_obj in self.depots.values():
            depot_obj.refill_charging(fleetctrl, simulation_time, self.keep_free_depot_cu)

    def find_next_charging_possibility(self, fleetctrl, veh_obj, desired_soc, min_soc_after_charging=None,
                                       min_power=None, desired_power=None, desired_end_time=None):
        """This method searches for the charging station and unit closest to the final vehicle position that allows
        scheduling a VCL to at least charge until min_soc_after_charging. The result will always be a list of PlanStops.
        The length of this list is either one or two:
        1) If min_soc_after_charging is None, one PlanStop with unlocked charging process will be generated.
        2) If min_soc_after_charging == desired_soc, one PlanStop with locked charging process will be generated.
        3) If min_soc_after_charging < desired_soc, one PlanStop with locked charging process followed by one PlanStop
                                                    with unlocked charging process will be generated.
        The charging process will be conducted with ChargingUnit.max_power unless stated differently.

        :param fleetctrl: FleetControl class
        :param veh_obj: simulation vehicle object
        :param desired_soc: desired SOC at end of charging process(es)
        :param min_soc_after_charging: minimum SOC after charging process (optional)
        :param min_power: search will only consider charging stations with ChargingUnit.max_power > min_power
        :param desired_power: overwriting charging unit power setting (only if <= ChargingUnit.max_power)
        :param desired_end_time: can be used to set a clear deadline.
        :return: list of PlanStops at next charging station
        """
        # TODO # find_next_charging_possibility()
        # -> best approach would be Dijkstra which checks charging stations on arrival before continuing
        # simplified chosen approach: -> find nearest charging station and schedule up to min_soc_after_charging
        #                                whenever this is possible
        pass

    def find_XtoX_charging_possibilities(self, list_veh_obj, min_soc_after_charging, power=None):
        """This method can be used to find all possibilities to assign a list of vehicles to all charging stations in
        the respective ranges of their final planned stop; only charging units that allow charging until
        min_soc_after_charging are considered; furthermore, a maximum of 1 charging unit per charging station is
        returned. Each possible assignment returns possible objective function characteristics. The assignments can
        be made with ChargingManagement.create_new_vcl() afterwards.

        :param list_veh_obj: list of vehicles that are considered for charging
        :param min_soc_after_charging: minimum SOC after charging processes (schedules have to allow enough time)
        :param power: if power is given, the algorithm will limit the power of charging processes
        :return: dict {}: veh_obj -> {}: charging unit_id -> (charging trip tt, charging trip distance, end_soc,
                                    end_time, PlanStop), where end_soc is 1.0 or originates from schedule constraints
        """
        # TODO # find_XtoX_charging_possibilities()
        pass

    def move_idle_vehicles_to_nearest_depots(self, sim_time, fleetctrl : FleetControlBase):
        """This method can be called to direct idle vehicles to nearest depots if on-street parking is not allowed.

        :param sim_time: current simulation time
        :param fleetctrl: FleetControl class
        :return: None
        """
        for vid in range(fleetctrl.nr_vehicles):
            veh_obj = fleetctrl.sim_vehicles[vid]
            if not veh_obj.assigned_route:
                if veh_obj.pos not in self.allowed_parking_pos.keys():
                    next_depot = self.node2depot.get(veh_obj.pos[0])
                    if not next_depot:
                        next_depot = self.find_nearest_free_depot(veh_obj.pos, fleetctrl.op_id, check_free=False)
                        self.node2depot[veh_obj.pos[0]] = next_depot
                    # make assignment (non-locked repositioning)
                    veh_plan = fleetctrl.veh_plans[veh_obj.vid]
                    LOG.debug(f"moving idle vehicle {veh_obj.vid} to nearest depot at {next_depot.pos}")
                    ps = RoutingTargetPlanStop(next_depot.pos)
                    veh_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
                    fleetctrl.assign_vehicle_plan(veh_obj, veh_plan, sim_time)
