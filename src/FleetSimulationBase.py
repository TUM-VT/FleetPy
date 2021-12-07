# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import importlib
import logging
import random
import time
import datetime
# import traceback
from abc import abstractmethod
from tqdm import tqdm
import typing as tp
from pathlib import Path
from multiprocessing import Manager
from src.python_plots.plot_classes import PyPlot

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np
# from IPython import embed

# src imports
# -----------
from src.misc.init_modules import load_fleet_control_module, load_routing_engine
from src.demand.demand import Demand, SlaveDemand
from src.routing.NetworkBase import return_position_str
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.routing.NetworkBase import NetworkBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
# set log level to logging.DEBUG or logging.INFO for single simulations
from src.misc.globals import *
DEFAULT_LOG_LEVEL = logging.INFO
LOG = logging.getLogger(__name__)
BUFFER_SIZE = 10
PROGRESS_LOOP = "demand"
PROGRESS_LOOP_VEHICLE_STATUS = [VRL_STATES.IDLE,VRL_STATES.CHARGING,VRL_STATES.REPOSITION]
# check for computation on LRZ cluster
if os.environ.get('SLURM_PROCID'):
    PROGRESS_LOOP = "off"


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------
def create_or_empty_dir(dirname):
    if os.path.isdir(dirname):
        "Removes all files from top"
        if(dirname == '/' or dirname == "\\"): return
        else:
            for root, dirs, files in os.walk(dirname, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception as err:
                        print(err)
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception as err:
                        print(err)
    else:
        os.makedirs(dirname)


def build_operator_attribute_dicts(parameters, n_op, prefix="op_"):
    """
    Extracts elements of parameters dict whose keys begin with prefix and generates a list of dicts.
    The values of the relevant elements of parameters must be either single values or a list of length n_op, or else
    an exception will be raised.

    :param parameters: dict (or dict-like config object) containing a superset of operator parameters
    :type parameters: dict
    :param n_op: number of operators expected
    :type n_op: int
    :param prefix: prefix by which to filter out operator parameters
    :type prefix: str
    """
    list_op_dicts = [dict() for i in range(n_op)]  # initialize list of empty dicts
    for k in [x for x in parameters if x.startswith(prefix)]:
        # if only a single value is given, use it for all operators
        if type(parameters[k]) in [str, int, float, bool, type(None), dict]:
            for di in list_op_dicts:
                di[k] = parameters[k]
        # if a list of values is given and the length matches the number of operators, use them respectively
        elif len(parameters[k]) == n_op:
            for i, op in enumerate(list_op_dicts):
                op[k] = parameters[k][i]
        elif k == G_OP_REPO_TH_DEF: # TODO # lists as inputs for op
            for di in list_op_dicts:
                di[k] = parameters[k]
        # if parameter has invalid number of values, raise exception
        else:
            raise ValueError("Number of values for parameter", k, "equals neither n_op nor 1.", type(parameters[k]))
    return list_op_dicts


# -------------------------------------------------------------------------------------------------------------------- #
# Simulation Vehicle Route Leg class
# ----------------------------------
class VehicleRouteLeg:
    def __init__(self, status, destination_pos, rq_dict, power=0.0, duration=None, route=[], locked=False,
                 earliest_start_time=-1000, charging_station=None):
        """
        This class summarizes the minimal information for a a route leg. It only reflects a complete state
        with the information of the initial state at the start of the leg.

        :param status: numbers reflecting the state of the leg (see misc/globals.py for specification)
        :param destination_pos: end of driving leg
        :param rq_dict: +/-1 -> [rq] ('+' for boarding, '-' for alighting)
        :param duration: required for non-driving legs
        :param route: list of nodes
        :param locked: locked tasks cannot be changed anymore
        :param earliest_start_time: for e.g. boarding processes
        """
        self.status = status
        self.rq_dict = rq_dict
        self.destination_pos = destination_pos
        self.power = power
        self.earliest_start_time = earliest_start_time
        self.duration = duration
        self.route = route
        self.locked = locked
        if duration is not None:
            try:
                x = int(duration)
            except:
                raise TypeError("wrong type for duration: {}".format(duration))
        self.charging_station = charging_station

    def __eq__(self, other):
        """Comparison of two VehicleRouteLegs.

        :param other: other vehicle route leg
        :type other: VehicleRouteLeg
        :return: True if equal, False else
        :rtype: bool
        """
        if self.status == other.status and set(self.rq_dict.get(1,[])) == set(other.rq_dict.get(1,[]))\
                and set(self.rq_dict.get(-1,[])) == set(other.rq_dict.get(-1,[]))\
                and self.destination_pos == other.destination_pos and self.locked == other.locked:
            return True
        else:
            return False

    def __str__(self):
        return "VRL: status {} dest {} duration {} earliest start time {} locked {} bd: {}".format(self.status, self.destination_pos, self.duration, self.earliest_start_time, self.locked, {key: [rq.get_rid_struct() for rq in val] for key, val in self.rq_dict.items()})

    def additional_str_infos(self):
        return "{}".format(self.rq_dict)

    def update_lock_status(self, lock=True):
        self.locked = lock


class VehicleChargeLeg(VehicleRouteLeg):
    def __init__(self, vcd_id, charging_unit, power, earliest_start_time, desired_soc, latest_end, locked=False):
        """This class is used for stops which focus on charging. These are characterized by a desired SOC goal or a
        certain duration which they should charge for. These VRL are of status 2, in contrast to status 3
        board and charging VRL, which are characterized by the boarding process and allow charging as a "byproduct"
        during the short stopping process.

        :param charging_unit: reference to charging unit
        :param power: desired charging power (<= charging_unit.max_power)
        :param earliest_start_time: time at which charging can begin (charging unit may be occupied before that)
        :param desired_soc: target SOC for the VCL
        :param latest_end: time at which charging has to end latest (charging unit may be occupied after that)
        :param locked: this parameter determines whether this VCL can be cut short or started later
        """
        status = VRL_STATES.CHARGING
        route = []
        rq_dict = {}
        duration = None
        self.vcd_id = vcd_id
        self.started_at = None
        self.charging_unit = charging_unit
        self.desired_soc = desired_soc
        self.latest_end = latest_end
        super().__init__(status, charging_unit.get_pos(), rq_dict, power, duration, route, locked, earliest_start_time)

    def __eq__(self, other):
        """Compares a VCL with another VRL/VCL.

        :param other: other VRL, VCL
        :return: True if VCL is equal
        """
        if type(other) == VehicleChargeLeg and self.vcd_id == other.vcd_id:
            return True
        else:
            return False

    def get_vcl_id(self):
        return self.vcd_id

    def is_active(self):
        if self.started_at is None:
            return False
        else:
            return True

    def set_duration_at_start(self, sim_time, veh_obj):
        """This method is called when the vehicle starts to charge. It is used to adapt the duration of the charging
        process in case a vehicle arrives late or with a different SOC. In these cases, shorter/longer charging
        durations to reach desired_soc or max_duration are considered as long as the charging unit schedule allows it.
        If the charging schedule does constrains an extension, the maximum available time is used.

        :param sim_time: simulation time
        :param veh_obj: contains the battery and SOC information
        :return: None; self.duration is set
        """
        self.started_at = sim_time
        desired_soc_duration = veh_obj.get_charging_duration(self.power, veh_obj.soc, self.desired_soc)
        if self.latest_end and self.started_at + desired_soc_duration > self.latest_end:
                self.duration = self.latest_end - self.started_at
                if self.duration < 0:
                    self.duration = 1
        else:
            self.duration = desired_soc_duration

    def get_current_schedule_info(self):
        """This method returns the earliest start time and the latest end time for a charging leg. The charging unit
        will be blocked in between these times for this task regardless of the vehicle actually being there.

        :return: earliest_time, latest_time
        """
        return self.earliest_start_time, self.latest_end

    def adapt_schedule(self, new_earliest_start_time=None, new_latest_end=None):
        """This method adapts the earliest start time and the latest end time of a VCL. The duration of an already
        running VCL is also cut short if necessary.

        :param new_earliest_start_time: can be used to let a task start later than originally planned
        :param new_latest_end: can be used to cut a charging task short, e.g. to start another charging process
        :return: None
        """
        if new_earliest_start_time:
            self.earliest_start_time = new_earliest_start_time
        if new_latest_end:
            self.latest_end = new_latest_end
            if self.started_at and self.started_at + self.duration > self.latest_end:
                self.duration = self.latest_end - self.started_at


# Simulation Vehicle class
# ------------------------
# > guarantee consistent movements in simulation and output
class SimulationVehicle:
    def __init__(self, operator_id, vehicle_id, vehicle_data_dir, vehicle_type, routing_engine, rq_db, op_output,
                 record_route_flag, replay_flag):
        """
        Initialization of vehicle in the simulation environment.
        :param operator_id: id of fleet operator the vehicle belongs to
        :param vehicle_id: id of the vehicle within the operator's fleet
        :param vehicle_data_dir: vehicle data directory
        :param vehicle_type: checks vehicle data base for existing model
        :param routing_engine: routing engine for queries
        :param rq_db: simulation request database (for conversion from plan requests)
        :param op_output: output for VRL records
        :param record_route_flag: generates path output if true
        :param replay_flag: generates extra output for replay visualization
        """
        self.op_id = operator_id
        self.vid = vehicle_id
        self.routing_engine = routing_engine
        self.rq_db = rq_db
        self.op_output = op_output
        self.record_route_flag = record_route_flag
        self.replay_flag = replay_flag
        #
        veh_data_f = os.path.join(vehicle_data_dir, f"{vehicle_type}.csv")
        veh_data = pd.read_csv(veh_data_f, header=None, index_col=0, squeeze=True)
        self.veh_type = veh_data[G_VTYPE_NAME]
        self.max_pax = int(veh_data[G_VTYPE_MAX_PAX])
        self.daily_fix_cost = float(veh_data[G_VTYPE_FIX_COST])
        self.distance_cost = float(veh_data[G_VTYPE_DIST_COST])
        self.battery_size = float(veh_data[G_VTYPE_BATTERY_SIZE])
        self.range = float(veh_data[G_VTYPE_RANGE])
        self.soc_per_m = 1/(self.range*1000)
        # current info
        self.status = VRL_STATES.IDLE
        self.pos = None
        self.soc = None
        self.pax = []  # rq_obj
        # assigned route = list of assigned vehicle legs
        self.assigned_route: tp.List[VehicleRouteLeg] = []
        # current leg (cl) info
        self.cl_start_time = None
        self.cl_start_pos = None
        self.cl_start_soc = None
        self.cl_toll_costs = 0
        self.cl_driven_distance = 0.0
        self.cl_driven_route = []  # list of passed node_indices
        self.cl_driven_route_times = []  # list of times at which nodes were passed; only filled for replay flag
        self.cl_remaining_route = []  # list of remaining nodes to next stop
        self.cl_remaining_time = None
        self.cl_locked = False
        # TODO # check and think about consistent way for large time steps -> will vehicles wait until next update?
        self.start_next_leg_first = False   # flag, if True, a new assignment has been made, which has to be activated first in the next call of update_veh_state

    def __str__(self):
        return f"veh {self.vid} at pos {self.pos} with soc {self.soc} leg status {self.status} remaining time {self.cl_remaining_time} number remaining legs: {len(self.assigned_route)} ob : {[rq.get_rid_struct() for rq in self.pax]}"

    def reset_current_leg(self):
        # current info
        self.status = VRL_STATES.IDLE
        # current leg (cl) info
        self.cl_start_time = None
        self.cl_start_pos = None
        self.cl_start_soc = None
        self.cl_toll_costs = 0
        self.cl_driven_distance = 0.0
        self.cl_driven_route = []  # list of passed node_indices
        self.cl_driven_route_times = []  # list of times at which nodes were passed; only filled for replay flag
        self.cl_remaining_route = []  # list of remaining nodes to next stop
        self.cl_remaining_time = None
        self.cl_locked = False

    def set_initial_state(self, fleetctrl, routing_engine, state_dict, start_time, veh_init_blocking=True):
        """
        This method positions the vehicle at the beginning of a simulation. This method has to be called after
        the initialization of the operator FleetControl classes!

        :param fleetctrl: FleetControl of vehicle operator
        :param routing_engine: Routing Engine
        :param state_dict: see documentation/init_state.md for specification
        :param start_time: simulation start time
        :param veh_init_blocking: if this flag is set, the vehicles are blocked until final_time; otherwise,
                            only the position and soc are used at the start_time of the simulation
        :return:
        """
        self.status = VRL_STATES.IDLE
        self.pos = routing_engine.return_node_position(state_dict[G_V_INIT_NODE])
        self.soc = state_dict[G_V_INIT_SOC]
        if veh_init_blocking:
            final_time = state_dict[G_V_INIT_TIME]
            if final_time > start_time:
                init_blocked_duration = final_time-start_time
                self.status = VRL_STATES.BLOCKED_INIT
                fleetctrl.set_init_blocked(self, start_time, routing_engine, init_blocked_duration)
                self.assigned_route.append(VehicleRouteLeg(VRL_STATES.BLOCKED_INIT, state_dict[G_V_INIT_NODE], {},
                                                           duration=init_blocked_duration, locked=True))
                # self.start_next_leg(start_time)
                self.start_next_leg_first = True

    def return_final_state(self, sim_end_time):
        """
        This method returns the required information to save the currently assigned route. For simplicity, the
        start node is used for vehicles that end up in the middle of a link and the soc at the end of the simulation
        run is used. Only the time is checked.
        :return: {}: key > value
        """
        final_state_dict = {G_V_OP_ID:self.op_id, G_V_VID:self.vid, G_V_INIT_SOC:self.soc}
        if self.assigned_route:
            final_state_dict[G_V_INIT_NODE] = self.assigned_route[-1].destination_pos[0]
            last_time = sim_end_time
            last_pos = self.pos
            for leg in self.assigned_route:
                leg_end_pos = leg.destination_pos
                if leg_end_pos != last_pos:
                    _, tt, _ = self.routing_engine.return_travel_costs_1to1(last_pos, leg_end_pos)
                    last_pos = leg_end_pos
                    last_time += tt
                elif leg.duration:
                    last_time += leg.duration
            end_time = last_time
        else:
            final_state_dict[G_V_INIT_NODE] = self.pos[0]
            end_time = sim_end_time
        # beware of not blocking a vehicle for a whole day!
        final_state_dict[G_V_INIT_TIME] = end_time % (24*3600)
        return final_state_dict

    def start_next_leg(self, simulation_time):
        """
        This function resets the current task attributes of a vehicle. Furthermore, it returns a list of currently
        boarding requests.
        :param simulation_time
        :return: list of currently boarding requests, list of starting to alight requests
        """
        if self.assigned_route:
            LOG.debug(f"start_next_leg {self.vid} : {self.assigned_route[0]}")
        else:
            LOG.debug(f"start_next_leg {self.vid} : no")
        if self.assigned_route and simulation_time >= self.assigned_route[0].earliest_start_time:
            LOG.debug(f"Vehicle {self.vid} starts new VRL {self.assigned_route[0].__dict__} at time {simulation_time}")
            self.cl_start_time = simulation_time
            self.cl_start_pos = self.pos
            self.cl_start_soc = self.soc
            self.cl_driven_distance = 0.0
            self.cl_driven_route = []
            ca = self.assigned_route[0]
            self.status = ca.status
            if self.pos != ca.destination_pos:
                if ca.route and self.pos[0] == ca.route[0]:
                    self.cl_remaining_route = ca.route
                else:
                    self.cl_remaining_route = self.routing_engine.return_best_route_1to1(self.pos, ca.destination_pos)
                try:
                    self.cl_remaining_route.remove(self.pos[0])
                except ValueError:
                    # TODO # check out after ISTTT
                    LOG.warning(f"First node in position {self.pos} not found in currently assigned route {ca.route}!")
                self.cl_driven_route.append(self.pos[0])
                self.cl_driven_route_times.append(simulation_time)
                self.cl_remaining_time = None
                list_boarding_pax = []
                list_start_alighting_pax = []
            else:
                # VehicleChargeLeg: check whether duration should be changed (other SOC value or late arrival)
                if ca.status == VRL_STATES.CHARGING:
                    ca.set_duration_at_start(simulation_time, self)
                self.cl_remaining_route = []
                if ca.duration is not None:
                    self.cl_remaining_time = ca.duration
                    if ca.status in G_LOCK_DURATION_STATUS:
                        ca.locked = True
                else:
                    if not ca.locked:
                        # this will usually only happen for waiting tasks, which can be stopped at any time
                        self.cl_remaining_time = None
                    else:
                        # raise AssertionError(f"Current locked task of vehicle {self.vid}"
                        #                      f"(operator {self.op_id}) has no stop criterion!")
                        LOG.info(f"Current locked task of vehicle {self.vid} (operator {self.op_id}) has no stop"
                                 f" criterion!")
                list_boarding_pax = [rq.get_rid_struct() for rq in ca.rq_dict.get(1, [])]
                list_start_alighting_pax = [rq.get_rid_struct() for rq in ca.rq_dict.get(-1, [])]
                for rq_obj in ca.rq_dict.get(1, []):
                    self.pax.append(rq_obj)
                LOG.debug(f"boarding the vehicle: bd: {list_boarding_pax} db: {list_start_alighting_pax} pax {self.pax}")
            LOG.debug("veh start next leg boarding at time {} remaining {}: {}".format(simulation_time, self.cl_remaining_time, self))
            return list_boarding_pax, list_start_alighting_pax
        elif self.assigned_route and simulation_time < self.assigned_route[0].earliest_start_time:
            # insert waiting leg
            waiting_vrl = VehicleRouteLeg(VRL_STATES.WAITING, self.pos, {}, duration=self.assigned_route[0].earliest_start_time - simulation_time)
            self.assigned_route = [waiting_vrl] + self.assigned_route
            LOG.debug("veh start next leg waiting at time {} remaining {} : {}".format(simulation_time, self.cl_remaining_time, self))
            return self.start_next_leg(simulation_time)
        else:
            LOG.debug("veh start next leg at time {} remaining {}: {}".format(simulation_time, self.cl_remaining_time, self))
            return [], []

    def end_current_leg(self, simulation_time):
        """
        This method stops the current leg, creates the record dictionary, and shifts the list of legs in the
        assigned route. It returns a list of alighting passengers and the record dictionary.
        :param simulation_time
        :return: (list_alight, passed_VRL)
            [rid1, rid2] given by int value
            passed_VRL: VRL object
        """
        if self.assigned_route and not self.start_next_leg_first:
            ca = self.assigned_route[0]
            LOG.debug(f"Vehicle {self.vid} ends a VRL and starts {ca.__dict__} at time {simulation_time}")
            # record
            record_dict = {}
            record_dict[G_V_OP_ID] = self.op_id
            record_dict[G_V_VID] = self.vid
            record_dict[G_V_TYPE] = self.veh_type
            record_dict[G_VR_STATUS] = G_VEHICLE_STATUS_DICT[self.status]
            record_dict[G_VR_LOCKED] = ca.locked
            record_dict[G_VR_LEG_START_TIME] = self.cl_start_time
            record_dict[G_VR_LEG_END_TIME] = simulation_time
            if self.cl_start_pos is None:
                LOG.error(f"current cl starting point not set before! {self.vid} {G_VEHICLE_STATUS_DICT[self.status]} {self.cl_start_time}")
                raise EnvironmentError
            record_dict[G_VR_LEG_START_POS] = self.routing_engine.return_position_str(self.cl_start_pos)
            record_dict[G_VR_LEG_END_POS] = self.routing_engine.return_position_str(self.pos)
            record_dict[G_VR_LEG_DISTANCE] = self.cl_driven_distance
            record_dict[G_VR_LEG_START_SOC] = self.cl_start_soc
            record_dict[G_VR_LEG_END_SOC] = self.soc
            record_dict[G_VR_CHARGING_POWER] = ca.power
            if type(ca) == VehicleChargeLeg:
                cu_id = ca.charging_unit.get_id()
                ca.charging_unit.remove_job_from_schedule(ca.get_vcl_id())
                record_dict[G_VR_CHARGING_UNIT] = f"{cu_id[0]}-{cu_id[1]}"
            else:
                record_dict[G_VR_CHARGING_UNIT] = ""
            record_dict[G_VR_TOLL] = self.cl_toll_costs
            record_dict[G_VR_OB_RID] = ";".join([str(rq.get_rid_struct()) for rq in self.pax])
            record_dict[G_VR_NR_PAX] = sum([rq.nr_pax for rq in self.pax])
            # remove and record alighting passengers
            list_boarding_pax = [rq.get_rid_struct() for rq in ca.rq_dict.get(1, [])]
            list_alighting_pax = [rq.get_rid_struct() for rq in ca.rq_dict.get(-1, [])]
            for rq_obj in ca.rq_dict.get(-1, []):
                try:
                    self.pax.remove(rq_obj)
                except:
                    LOG.warning(f"Could not remove passenger {rq_obj.get_rid_struct()} from vehicle {self.vid}"
                                f" at time {simulation_time}")
            record_dict[G_VR_BOARDING_RID] = ";".join([str(rid) for rid in list_boarding_pax])
            record_dict[G_VR_ALIGHTING_RID] = ";".join([str(rid) for rid in list_alighting_pax])
            if self.record_route_flag:
                record_dict[G_VR_NODE_LIST] = ";".join([str(x) for x in self.cl_driven_route])
            if self.replay_flag:
                route_length = len(self.cl_driven_route)
                route_replay_str = ";".join([f"{self.cl_driven_route[i]}:{self.cl_driven_route_times[i]}"\
                                             for i in range(route_length)])
                record_dict[G_VR_REPLAY_ROUTE] = route_replay_str
            # default status and shift to next leg
            self.reset_current_leg()
            self.assigned_route = self.assigned_route[1:]
            self.op_output.append(record_dict)
            return list_alighting_pax, ca
        elif self.start_next_leg_first:
            # this can happen when user_request makes assignment and batch optimization in same time step makes
            # different assignment
            LOG.debug(f"vid {self.vid}: another assignment in same step? end_leg not called")
            return ([], {})
        else:
            LOG.warning(f"Trying to stop a leg of vehicle {self.vid} at time {simulation_time} "
                        f"even though no route is assigned!")
            return ([], {})

    def assign_vehicle_plan(self, list_route_legs, sim_time, force_ignore_lock = False):
        """This method enables the fleet control modules to assign a plan to the simulation vehicles. It ends the
        previously assigned leg and starts the new one if necessary.
        :param list_route_legs: list of legs to assign to vehicle
        :type list_route_legs: list of VehicleRouteLeg
        :param sim_time: current simulation time
        :type sim_time: int
        :param force_ignore_lock: this parameter allows overwriting locked VehicleRouteLegs, except a boarding process allready started
        :type force_ignore_lock: bool
        """
        # transform rq from PlanRequest to SimulationRequest (based on RequestBase)
        # LOG.info(f"Vehicle {self.vid} before new assignment: {[str(x) for x in self.assigned_route]} at time {sim_time}")
        for vrl in list_route_legs:
            boarding_list = [self.rq_db[prq.get_rid()] for prq in vrl.rq_dict.get(1,[])]
            alighting_list = [self.rq_db[prq.get_rid()] for prq in vrl.rq_dict.get(-1,[])]
            vrl.rq_dict = {1:boarding_list, -1:alighting_list}
        #LOG.debug(f"Vehicle {self.vid} received new VRLs {[str(x) for x in list_route_legs]} at time {sim_time}")
        #LOG.debug(f"  -> current assignment: {self.assigned_route}")
        start_flag = True
        if self.assigned_route:
            if not list_route_legs or list_route_legs[0] != self.assigned_route[0]:
                if not self.assigned_route[0].locked:
                    self.end_current_leg(sim_time)
                else:
                    if force_ignore_lock and not self.assigned_route[0].status == VRL_STATES.BOARDING:
                        self.end_current_leg(sim_time)
                    else:
                        LOG.error("vid : {}".format(self.vid))
                        LOG.error("currently assigned: {}".format([str(x) for x in self.assigned_route]))
                        LOG.error("new: {}".format([str(x) for x in list_route_legs]))
                        LOG.error("current additional infos: {}".format(self.assigned_route[0].additional_str_infos()))
                        LOG.error("new additional infos: {}".format(list_route_legs[0].additional_str_infos()))
                        raise AssertionError("assign_vehicle_plan(): Trying to assign new VRLs instead of a locked VRL.")
            else:
                start_flag = False

        self.assigned_route = list_route_legs
        if list_route_legs:
            if start_flag:
                self.start_next_leg_first = True
                # print("THIS IS NOT ALLOWED TO HAPPEN HERE IN CASE OF IMIDIATE BOARDINGS/DEBOARDINGS") # TODO #
                # exit()
                # self.start_next_leg(sim_time)
        # LOG.info(f"Vehicle {self.vid} after new assignment: {[str(x) for x in self.assigned_route]} at time {sim_time}")

    def update_veh_state(self, current_time, next_time):
        """This method updates the current state of a simulation vehicle. This includes moving, boarding etc.
        The method updates the vehicle position, soc. Additionally, it triggers the end and start of VehicleRouteLegs.
        It returns a list of boarding request, alighting requests.

        :param current_time: this time corresponds to the current state of the vehicle
        :type current_time: float
        :param next_time: the time until which the state should be updated
        :type next_time: float
        :return:(dict of boarding requests -> (time, position), dict of alighting request objects -> (time, position), list of passed VRL, dict_start_alighting)
        :rtype: list
        """
        LOG.debug(f"update veh state {current_time} -> {next_time} : {self}")
        dict_boarding_requests = {}
        dict_start_alighting = {}
        dict_alighting_requests = {}
        list_passed_VRL = []
        c_time = current_time
        remaining_step_time = next_time - current_time
        if self.start_next_leg_first:
            add_boarding_rids, start_alighting_rids = self.start_next_leg(c_time)
            for rid in add_boarding_rids:
                dict_boarding_requests[rid] = (c_time, self.pos)
            for rid in start_alighting_rids:
                dict_start_alighting[rid] = (c_time, self.pos)
            self.start_next_leg_first = False
        # LOG.debug(f"veh update state from {current_time} to {next_time}")
        while remaining_step_time > 0:
            # LOG.debug(f" veh moving c_time {c_time} remaining time step {remaining_step_time}")
            if self.status in G_DRIVING_STATUS:
                # 1) moving: update pos and soc (call move along route with record_node_times=replay_flag)
                # LOG.debug(f"Vehicle {self.vid} is driving between {c_time} and {next_time}")
                (new_pos, driven_distance, arrival_in_time_step, passed_nodes, passed_node_times) = \
                    self.routing_engine.move_along_route(self.cl_remaining_route, self.pos, remaining_step_time,
                                                         sim_vid_id=(self.op_id, self.vid),
                                                         new_sim_time=c_time,
                                                         record_node_times=self.replay_flag)
                last_node = self.pos[0]
                self.pos = new_pos
                self.cl_driven_distance += driven_distance
                self.soc -= self.compute_soc_consumption(driven_distance)
                self.compute_energy_source(current_time)
                if passed_nodes:
                    self.cl_driven_route.extend(passed_nodes)
                for node in passed_nodes:
                    self.cl_remaining_route.remove(node)
                    tmp_toll_route = [last_node, node]
                    # TODO #
                    _, toll_costs, _ = \
                        self.routing_engine.get_zones_external_route_costs(current_time,
                                                                           tmp_toll_route,
                                                                           park_origin=False, park_destination=False)
                    self.cl_toll_costs += toll_costs
                    last_node = node
                if passed_node_times:
                    self.cl_driven_route_times.extend(passed_node_times)
                if arrival_in_time_step == -1:
                    #   a) move until next_time; do nothing
                    remaining_step_time = 0
                else:
                    #   b) move up to destination [compute required time]
                    #       end task, start next task; continue with remaining time
                    remaining_step_time -= (arrival_in_time_step - c_time)
                    c_time = arrival_in_time_step
                    # LOG.debug(f"arrival in time step {arrival_in_time_step}")
                    add_alighting_rq, passed_VRL = self.end_current_leg(c_time)
                    for rid in add_alighting_rq:
                        dict_alighting_requests[rid] = c_time
                    if isinstance(passed_VRL, list):
                        list_passed_VRL.extend(passed_VRL)
                    else:
                        list_passed_VRL.append(passed_VRL)
                    if self.assigned_route:
                        add_boarding_rids, start_alighting_rids = self.start_next_leg(c_time)
                        for rid in add_boarding_rids:
                            dict_boarding_requests[rid] = (c_time, self.pos)
                        for rid in start_alighting_rids:
                            dict_start_alighting[rid] = (c_time, self.pos)
            elif self.status != VRL_STATES.IDLE: #elif self.status != 0 and not self.status in G_IDLE_BUSY_STATUS:
                # 2) non-moving:
                # LOG.debug(f"Vehicle {self.vid} performs non-moving task between {c_time} and {next_time}")
                if remaining_step_time < self.cl_remaining_time:
                    #   a) duration is ongoing: do nothing
                    self.cl_remaining_time -= remaining_step_time
                    if self.assigned_route[0].power > 0:
                        self.soc += self.compute_soc_charging(self.assigned_route[0].power, remaining_step_time)
                        self.soc = min(self.soc, 1.0)
                    remaining_step_time = 0
                else:
                    #   b) duration is passed; end task, start next task; continue with remaining time
                    c_time += self.cl_remaining_time
                    # LOG.debug(f"cl remaining time {self.cl_remaining_time}")
                    remaining_step_time -= self.cl_remaining_time
                    if self.assigned_route[0].power > 0:
                        self.soc += self.compute_soc_charging(self.assigned_route[0].power, self.cl_remaining_time)
                        self.soc = min(self.soc, 1.0)
                    add_alighting_rq, passed_VRL = self.end_current_leg(c_time)
                    for rid in add_alighting_rq:
                        dict_alighting_requests[rid] = c_time
                    list_passed_VRL.append(passed_VRL)
                    if self.assigned_route:
                        add_boarding_rids, start_alighting_rids = self.start_next_leg(c_time)
                        for rid in add_boarding_rids:
                            dict_boarding_requests[rid] = (c_time, self.pos)
                        for rid in start_alighting_rids:
                            dict_start_alighting[rid] = (c_time, self.pos)
            else:
                # 3) idle without VRL
                remaining_step_time = 0
        return dict_boarding_requests, dict_alighting_requests, list_passed_VRL, dict_start_alighting

    def update_route(self):
        if self.assigned_route:
            ca = self.assigned_route[0]
            if self.pos != ca.destination_pos:
                self.cl_remaining_route = self.routing_engine.return_best_route_1to1(self.pos, ca.destination_pos)
                try:
                    self.cl_remaining_route.remove(self.pos[0])
                except ValueError:
                    # TODO # check out after ISTTT
                    LOG.warning(f"First node in position {self.pos} not found in currently assigned route {ca.route}!")
                # LOG.info(f"veh {self.vid} update route {self.pos} -> {ca.destination_pos} : {self.cl_remaining_route}")

    def compute_soc_consumption(self, distance):
        """This method returns the SOC change for driving a certain given distance.

        :param distance: driving distance in meters
        :type distance: float
        :return: delta SOC (positive value!)
        :rtype: float
        """
        return distance * self.soc_per_m

    def compute_energy_source(self, sim_time):
        """ This function checks the validity of the current energy source for the vehicle """
        pass

    def compute_soc_charging(self, power, duration):
        """This method returns the SOC change for charging a certain amount of power for a given duration.

        :param power: power of charging process in kW(!)
        :type power: float
        :param duration: duration of charging process in seconds(!)
        :type duration: float
        :return: delta SOC
        :rtype: float
        """
        return power * duration / (self.battery_size * 3600)

    def get_charging_duration(self, power, init_soc, final_soc=1.0):
        """This method computes the charging duration required to charge the vehicle from init_soc to final_soc.

        :param power: power of charging process in kW(!); result should be in seconds
        :type power: float
        :param init_soc: soc at beginning of charging process
        :type init_soc: float in [0,1]
        :param final_soc: soc at end of charging process
        :type final_soc: float in [0,1]
        :return: duration in seconds
        :rtype: float
        """
        return (final_soc - init_soc) * self.battery_size * 3600 / power

    def get_nr_pax_without_currently_boarding(self):
        """ this method returns the current number of pax for the use of setting the inititial stats for 
        the update_tt_... function in fleetControlBase.py.
        In case the vehicle is currently boarding, this method doesnt count the number of currently boarding customers
        the reason is that boarding and deboarding of customers is recognized during different timesteps of the vcl
        :return: number of pax without currently boarding ones
        :rtype: int
        """
        if self.status == VRL_STATES.BOARDING:
            return sum([rq.nr_pax for rq in self.pax]) - sum([rq.nr_pax for rq in self.assigned_route[0].rq_dict.get(1, [])])
        else:
            return sum([rq.nr_pax for rq in self.pax])

# -------------------------------------------------------------------------------------------------------------------- #
# Traveler Offer Class
# -----------


class TravellerOffer:
    def __init__(self, traveler_id, operator_id, offered_waiting_time, offered_driving_time, fare,
                 additional_parameters=None):
        """ this class collects all information of a trip offered by an operator for a specific customer request
        TravellerOffer entities will be created by mobility operators and send to travellers, who perform mode choices
        based on the corresponding entries
        if at least the offered_waiting_time is set to None the offer is treated as a decline by the operator
        :param traveler_id: traveler_id this offer is sent to
        :type traveler_id: int
        :param operator_id: id of operator who made the offer
        :type operator_id: int
        :param offered_waiting_time: absolute time [s] from request-time until expected pick-up time
        :type offered_waiting_time: float or None
        :param offered_driving_time: time [s] a request is expected to drive from origin to destination
        :type offered_driving_time: float or None
        :param fare: fare of the trip [ct]
        :type fare: int or None
        :param additional_parameters: dictionary of other offer-attributes that might influence the simulation flow
        :type additional_parameters: dict or None
        """
        if additional_parameters is None:
            additional_parameters = {}
        self.traveler_id = traveler_id
        self.operator_id = operator_id
        self.offered_waiting_time = offered_waiting_time
        self.offered_driving_time = offered_driving_time
        self.fare = fare
        self.additional_offer_parameters = additional_parameters.copy()

    def extend_offer(self, additional_offer_parameters):
        """ this function can be used to add parameters to the offer
        :param additional_offer_parameters: dictionary offer_variable (globals!) -> value
        :type additional_offer_parameters: dict
        """
        self.additional_offer_parameters.update(additional_offer_parameters)

    def service_declined(self):
        """ this function evaluates if the offer should be treated as a decline because the service is not possible
        :return: True if operator decline the service, False else
        :rtype: bool
        """
        if self.offered_waiting_time is None:
            return True
        else:
            return False

    def __getitem__(self, offer_attribute_str):
        """ this function can be used to access specific attributes of the offer
        :param offer_attribute_str: attribute_str of the offer parameter (see globals!)
        :type offer_attribute_str: str
        :return: value of the specific attribute within the offer. raises error if not specified!
        :rtype: not defined
        """   
        if offer_attribute_str == G_OFFER_WAIT:
            return self.offered_waiting_time
        elif offer_attribute_str == G_OFFER_DRIVE:
            return self.offered_driving_time
        elif offer_attribute_str == G_OFFER_FARE:
            return self.fare
        else:
            try:
                return self.additional_offer_parameters[offer_attribute_str]
            except KeyError:
                pass
        raise KeyError(type(self).__name__+" object has no attribute '"+offer_attribute_str+"'")

    def get(self, offer_attribute_str, other_wise=None):
        """ this function can be used to access specific attributes of the offer
        :param offer_attribute_str: attribute_str of the offer parameter (see globals!)
        :type offer_attribute_str: str
        :param other_wise: value of the corresponding offer_attribute_str in case it is not specified in the offer
        :type other_wise: not defined
        :return: value of the specific attribute within the offer
        :rtype: not defined
        """
        if offer_attribute_str == G_OFFER_WAIT:
            return self.offered_waiting_time
        elif offer_attribute_str == G_OFFER_DRIVE:
            return self.offered_driving_time
        elif offer_attribute_str == G_OFFER_FARE:
            return self.fare
        else:
            return self.additional_offer_parameters.get(offer_attribute_str, other_wise)

    def __contains__(self, offer_attribute_str):
        """ this function overwrites the "in" operator and can be used to test
        if the offer attribute is within the allready defined offer attributes
        :param offer_attribute_str: specific offer attribute key (globals!)
        :type offer_attribute_str: str
        :return: true, if offer attribute defined in offer; else false
        :rtype: bool
        """
        if offer_attribute_str == G_OFFER_WAIT or offer_attribute_str == G_OFFER_DRIVE or offer_attribute_str == G_OFFER_FARE:
            return True
        elif self.additional_offer_parameters.get(offer_attribute_str, None) is not None:
            return True
        else:
            return False

    def to_output_str(self):
        """ this function creates a string of the offer parameters for the output file
        in the form offer_param1:offer_value1;offer_param2_offer_value2;...
        if no service was offered an empty str is returned
        :return: string of the offer to write to the outputfile
        :rtype: str
        """
        if not self.service_declined():
            offer_info = [f"{G_OFFER_WAIT}:{self.offered_waiting_time}", f"{G_OFFER_DRIVE}:{self.offered_driving_time}", f"{G_OFFER_FARE}:{self.fare}"]
            for k, v in self.additional_offer_parameters.items():
                if k == G_OFFER_PU_POS or k == G_OFFER_DO_POS:
                    v = return_position_str(v)
                offer_info.append(f"{k}:{v}")
            return ";".join(offer_info)
        else:
            return ""

    def __str__(self):
        if self.service_declined():
            return "declined"
        else:
            return self.to_output_str()


class Rejection(TravellerOffer):
    """This class takes minimal input and creates an offer that represents a rejection."""
    def __init__(self, traveler_id, operator_id):
        super().__init__(traveler_id, operator_id, offered_waiting_time=None, offered_driving_time=None, fare=None)


# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----


class FleetSimulationBase:
    def __init__(self, scenario_parameters: dict):
        self.t_init_start = time.perf_counter()
        # config
        self.scenario_name = scenario_parameters[G_SCENARIO_NAME]
        print("-"*80 + f"\nSimulation of scenario {self.scenario_name}")
        LOG.info(f"General initialization of scenario {self.scenario_name}...")
        self.dir_names = self.get_directory_dict(scenario_parameters)
        self.scenario_parameters: dict = scenario_parameters

        # check whether simulation already has been conducted -> use final_state.csv to check
        final_state_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "final_state.csv")
        if self.scenario_parameters.get("keep_old", False) and os.path.isfile(final_state_f):
            prt_str = f"Simulation {self.scenario_name} results available and keep_old flag is True!" \
                      f" Not starting the simulation!"
            print(prt_str)
            LOG.info(prt_str)
            self._started = True
            return
        else:
            self._started = False

        # general parameters
        self.start_time = self.scenario_parameters[G_SIM_START_TIME]
        self.end_time = self.scenario_parameters[G_SIM_END_TIME]
        self.time_step = self.scenario_parameters.get(G_SIM_TIME_STEP, 1)
        self.check_sim_env_spec_inputs(self.scenario_parameters)
        self.n_op = self.scenario_parameters[G_NR_OPERATORS]
        self._manager: tp.Optional[Manager] = None
        self._shared_dict: dict = {}
        self._plot_class_instance: tp.Optional[PyPlot] = None
        self.realtime_plot_flag = self.scenario_parameters.get(G_SIM_REALTIME_PLOT_FLAG, 0)

        # build list of operator dictionaries  # TODO: this could be eliminated with a new YAML-based config system
        self.list_op_dicts: tp.Dict[str,str] = build_operator_attribute_dicts(scenario_parameters, self.n_op,
                                                                              prefix="op_")

        # take care of random seeds at beginning of simulations
        random.seed(self.scenario_parameters[G_RANDOM_SEED])
        np.random.seed(self.scenario_parameters[G_RANDOM_SEED])

        # empty output directory
        create_or_empty_dir(self.dir_names[G_DIR_OUTPUT])

        # write scenario config file in output directory
        self.save_scenario_inputs()

        # remove old log handlers (otherwise sequential simulations only log to first simulation)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # start new log file
        logging.VERBOSE = 5
        logging.addLevelName(logging.VERBOSE, "VERBOSE")
        logging.Logger.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
        logging.LoggerAdapter.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
        logging.verbose = lambda msg, *args, **kwargs: logging.log(logging.VERBOSE, msg, *args, **kwargs)
        if self.scenario_parameters.get("log_level", "info"):
            level_str = self.scenario_parameters["log_level"]
            if level_str == "verbose":
                log_level = logging.VERBOSE
            elif level_str == "debug":
                log_level = logging.DEBUG
            elif level_str == "info":
                log_level = logging.INFO
            elif level_str == "warning":
                log_level = logging.WARNING
            else:
                log_level = DEFAULT_LOG_LEVEL
        else:
            log_level = DEFAULT_LOG_LEVEL
            pd.set_option("mode.chained_assignment", None)
        self.log_file = os.path.join(self.dir_names[G_DIR_OUTPUT], f"00_simulation.log")
        if log_level < logging.INFO:
            streams = [logging.FileHandler(self.log_file), logging.StreamHandler()]
        else:
            print("Only minimum output to console -> see log-file")
            streams = [logging.FileHandler(self.log_file)]
        # TODO # log of subsequent simulations is saved in first simulation log
        logging.basicConfig(handlers=streams,
                            level=log_level, format='%(process)d-%(name)s-%(levelname)s-%(message)s')

        # set up output files
        self.user_stat_f = os.path.join(self.dir_names[G_DIR_OUTPUT], f"1_user-stats.csv")
        self.network_stat_f = os.path.join(self.dir_names[G_DIR_OUTPUT], f"3_network-stats.csv")
        self.pt_stat_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "4_pt_stats.csv")

        # init modules
        # ------------
        # zone system
        # TODO # after ISTTT: enable multiple zone systems
        # TODO # after ISTTT: bring init of modules in extra function (-> parallel processing)
        self.zones = None
        if self.dir_names.get(G_DIR_ZONES, None) is not None:
            if self.scenario_parameters.get(G_FC_TYPE) and self.scenario_parameters[G_FC_TYPE] == "perfect":
                from src.infra.PerfectForecastZoning import PerfectForecastZoneSystem
                self.zones = PerfectForecastZoneSystem(self.dir_names[G_DIR_ZONES], self.scenario_parameters, self.dir_names)
            else:
                from src.infra.Zoning import ZoneSystem
                self.zones = ZoneSystem(self.dir_names[G_DIR_ZONES], self.scenario_parameters, self.dir_names)

        # routing engine
        LOG.info("Initialization of network and routing engine...")
        network_type = self.scenario_parameters[G_NETWORK_TYPE]
        network_dynamics_file = self.scenario_parameters.get(G_NW_DYNAMIC_F, None)
        # TODO # check consistency of scenario inputs / another way to refactor add_init_data ?
        self.routing_engine: NetworkBase = load_routing_engine(network_type, self.dir_names[G_DIR_NETWORK],
                                                               network_dynamics_file_name=network_dynamics_file)
        if network_type == "NetworkDynamicNFDClusters":
            self.routing_engine.add_init_data(self.start_time, self.time_step,
                                              self.scenario_parameters[G_NW_DENSITY_T_BIN_SIZE],
                                              self.scenario_parameters[G_NW_DENSITY_AVG_DURATION], self.zones,
                                              self.network_stat_f)
        # public transportation module
        LOG.info("Initialization of line-based public transportation...")
        pt_type = self.scenario_parameters.get(G_PT_TYPE)
        gtfs_data_dir = self.dir_names.get(G_DIR_PT)
        if pt_type is None or gtfs_data_dir is None:
            self.pt = None
        elif pt_type == "PTMatrixCrowding":
            pt_module = importlib.import_module("src.pubtrans.PtTTMatrixCrowding")
            self.pt = pt_module.PublicTransportTravelTimeMatrixWithCrowding(gtfs_data_dir, self.pt_stat_f,
                                                                            self.scenario_parameters,
                                                                            self.routing_engine, self.zones)
        elif pt_type == "PtCrowding":
            pt_module = importlib.import_module("src.pubtrans.PtCrowding")
            self.pt = pt_module.PublicTransportWithCrowding(gtfs_data_dir, self.pt_stat_f, self.scenario_parameters,
                                                            self.routing_engine, self.zones)
        else:
            raise IOError(f"Public transport module {pt_type} not defined for current simulation environment.")

        # attribute for demand, charging and zone module
        self.demand = None
        self.cdp = None
        self._load_demand_module()
        self._load_charging_modules()

        # take care of charging stations, depots and initially inactive vehicles
        if self.dir_names.get(G_DIR_INFRA):
            depot_fname = self.scenario_parameters.get(G_INFRA_DEP)
            if depot_fname is not None:
                depot_f = os.path.join(self.dir_names[G_DIR_INFRA], depot_fname)
            else:
                depot_f = os.path.join(self.dir_names[G_DIR_INFRA], "depots.csv")
            pub_cs_fname = self.scenario_parameters.get(G_INFRA_PBCS)
            if pub_cs_fname is not None:
                pub_cs_f = os.path.join(self.dir_names[G_DIR_INFRA], pub_cs_fname)
            else:
                pub_cs_f = os.path.join(self.dir_names[G_DIR_INFRA], "public_charging_stations.csv")
            from src.infra.ChargingStation import ChargingAndDepotManagement
            self.cdp = ChargingAndDepotManagement(depot_f, pub_cs_f, self.routing_engine, self.scenario_parameters,
                                                  self.list_op_dicts)
            LOG.info("charging stations and depots initialzied!")
        else:
            self.cdp = None

        # attributes for fleet controller and vehicles
        self.sim_vehicles: tp.Dict[tp.Tuple[int, int], SimulationVehicle] = {}
        self.sorted_sim_vehicle_keys: tp.List[tp.Tuple[int, int]] = sorted(self.sim_vehicles.keys())
        self.operators: tp.List[FleetControlBase] = []
        self.op_output = {}
        self._load_fleetctr_vehicles()

        # call additional simulation environment specific init
        LOG.info("Simulation environment specific initializations...")
        self.init_blocking = True
        self.add_init(self.scenario_parameters)

        # load initial state depending on init_blocking attribute
        # HINT: it is important that this is done at the end of initialization!
        LOG.info("Creating or loading initial vehicle states...")
        np.random.seed(self.scenario_parameters[G_RANDOM_SEED])
        self.load_initial_state()
        LOG.info(f"Initialization of scenario {self.scenario_name} successful.")

        # self.routing_engine.checkNetwork()

    def _load_demand_module(self):
        """ Loads some demand modules """

        # demand module
        LOG.info("Initialization of travelers...")
        if self.scenario_parameters[G_SIM_ENV] != "MobiTopp":
            self.demand = Demand(self.scenario_parameters, self.user_stat_f, self.routing_engine, self.zones)
            self.demand.load_demand_file(self.scenario_parameters[G_SIM_START_TIME],
                                         self.scenario_parameters[G_SIM_END_TIME], self.dir_names[G_DIR_DEMAND],
                                         self.scenario_parameters[G_RQ_FILE], self.scenario_parameters[G_RANDOM_SEED],
                                         self.scenario_parameters.get(G_RQ_TYP1, None),
                                         self.scenario_parameters.get(G_RQ_TYP2, {}),
                                         self.scenario_parameters.get(G_RQ_TYP3, {}),
                                         simulation_time_step=self.time_step)
        else:
            self.demand = SlaveDemand(self.scenario_parameters, self.user_stat_f)

        if self.zones is not None:
            self.zones.register_demand_ref(self.demand)

    def _load_charging_modules(self):
        """ Loads necessary modules for charging """

        # take care of charging stations, depots and initially inactive vehicles
        if self.dir_names.get(G_DIR_INFRA):
            depot_fname = self.scenario_parameters.get(G_INFRA_DEP)
            if depot_fname is not None:
                depot_f = os.path.join(self.dir_names[G_DIR_INFRA], depot_fname)
            else:
                depot_f = os.path.join(self.dir_names[G_DIR_INFRA], "depots.csv")
            pub_cs_fname = self.scenario_parameters.get(G_INFRA_PBCS)
            if pub_cs_fname is not None:
                pub_cs_f = os.path.join(self.dir_names[G_DIR_INFRA], pub_cs_fname)
            else:
                pub_cs_f = os.path.join(self.dir_names[G_DIR_INFRA], "public_charging_stations.csv")
            from src.infra.ChargingStation import ChargingAndDepotManagement
            self.cdp = ChargingAndDepotManagement(depot_f, pub_cs_f, self.routing_engine, self.scenario_parameters,
                                                  self.list_op_dicts)
            LOG.info("charging stations and depots initialzied!")
        else:
            self.cdp = None

    def _load_fleetctr_vehicles(self):
        """ Loads the fleet controller and vehicles """

        # simulation vehicles and fleet control modules
        LOG.info("Initialization of MoD fleets...")
        route_output_flag = self.scenario_parameters.get(G_SIM_ROUTE_OUT_FLAG, True)
        replay_flag = self.scenario_parameters.get(G_SIM_REPLAY_FLAG, False)
        veh_type_list = []
        for op_id in range(self.n_op):
            operator_attributes = self.list_op_dicts[op_id]
            operator_module_name = operator_attributes[G_OP_MODULE]
            self.op_output[op_id] = []  # shared list among vehicles
            if not operator_module_name == "PtFleetControl":
                fleet_composition_dict = operator_attributes[G_OP_FLEET]
                list_vehicles = []
                vid = 0
                for veh_type, nr_veh in fleet_composition_dict.items():
                    for _ in range(nr_veh):
                        veh_type_list.append([op_id, vid, veh_type])
                        tmp_veh_obj = SimulationVehicle(op_id, vid, self.dir_names[G_DIR_VEH], veh_type,
                                                        self.routing_engine, self.demand.rq_db,
                                                        self.op_output[op_id], route_output_flag,
                                                        replay_flag)
                        list_vehicles.append(tmp_veh_obj)
                        self.sim_vehicles[(op_id, vid)] = tmp_veh_obj
                        vid += 1
                OpClass = load_fleet_control_module(operator_module_name)
                self.operators.append(OpClass(op_id, operator_attributes, list_vehicles, self.routing_engine, self.zones,
                                            self.scenario_parameters, self.dir_names, self.cdp))
            else:
                from src.pubtrans.PtFleetControl import PtFleetControl
                OpClass = PtFleetControl(op_id, gtfs_data_dir, self.routing_engine, self.zones, scenario_parameters, self.dir_names, charging_management=self.cdp)
                init_vids = OpClass.return_vehicles_to_initialize()
                list_vehicles = []
                for vid, veh_type in init_vids.items():
                    tmp_veh_obj = SimulationVehicle(op_id, vid, self.dir_names[G_DIR_VEH], veh_type,
                                                        self.routing_engine, self.demand.rq_db,
                                                        self.op_output[op_id], route_output_flag,
                                                        replay_flag)
                    list_vehicles.append(tmp_veh_obj)
                    self.sim_vehicles[(op_id, vid)] = tmp_veh_obj
                OpClass.continue_init(list_vehicles, self.start_time)
                self.operators.append(OpClass)
        veh_type_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "2_vehicle_types.csv")
        veh_type_df = pd.DataFrame(veh_type_list, columns=[G_V_OP_ID, G_V_VID, G_V_TYPE])
        veh_type_df.to_csv(veh_type_f, index=False)

    @staticmethod
    def get_directory_dict(scenario_parameters):
        """
        This function provides the correct paths to certain data according to the specified data directory structure.
        :param scenario_parameters: simulation input (pandas series)
        :return: dictionary with paths to the respective data directories
        """
        return get_directory_dict(scenario_parameters)

    def save_scenario_inputs(self):
        config_f = os.path.join(self.dir_names[G_DIR_OUTPUT], G_SC_INP_F)
        config = {"scenario_parameters": self.scenario_parameters, "list_operator_attributes": self.list_op_dicts,
                  "directories": self.dir_names}
        with open(config_f, "w") as fh_config:
            json.dump(config, fh_config, indent=4)

    def evaluate(self):
        """Runs standard and simulation environment specific evaluations over simulation results."""
        output_dir = self.dir_names[G_DIR_OUTPUT]
        # standard evaluation
        from src.evaluation.standard import standard_evaluation
        standard_evaluation(output_dir)
        self.add_evaluate()

    def load_initial_state(self):
        """This method initializes the simulation vehicles. It can consider an initial state file. Moreover, an
        active_vehicle files would be considered as the FleetControl already set the positions of vehicles in the depot
        and therefore the "if veh_obj.pos is None:" condition does not trigger.
        The VehiclePlans of the respective FleetControls are also adapted for blocked vehicles.

        :return: None
        """
        init_f_flag = False
        init_state_f = None
        if self.scenario_parameters.get(G_INIT_STATE_SCENARIO):
            init_state_f = os.path.join(self.dir_names[G_DIR_MAIN], "results",
                                        self.scenario_parameters[G_STUDY_NAME],
                                        str(self.scenario_parameters.get(G_INIT_STATE_SCENARIO, "None")),
                                        "final_state.csv")
            init_f_flag = True
        set_unassigned_vid = set([(veh_obj.op_id, veh_obj.vid) for veh_obj in self.sim_vehicles.values()
                                  if veh_obj.pos is None])
        if init_f_flag and os.path.isfile(init_state_f):
            # set according to initial state if available
            init_state_df = pd.read_csv(init_state_f)
            init_state_df.set_index([G_V_OP_ID, G_V_VID], inplace=True)
            for sim_vid, veh_obj in self.sim_vehicles.items():
                if veh_obj.pos is None:
                    op_fleetctrl = self.operators[veh_obj.op_id]
                    init_state_info = init_state_df.loc[sim_vid]
                    if init_state_info is not None:
                        veh_obj.set_initial_state(op_fleetctrl, self.routing_engine, init_state_info,
                                                  self.scenario_parameters[G_SIM_START_TIME], self.init_blocking)
                        set_unassigned_vid.remove(sim_vid)
        if len(set_unassigned_vid) > 0:
            op_init_distributions = {}
            for op_id in range(self.n_op):
                if self.list_op_dicts[op_id].get(G_OP_INIT_VEH_DIST) is not None:   #specified random distribution
                    init_dist_df = pd.read_csv(os.path.join(self.dir_names[G_DIR_FCTRL], "initial_vehicle_distribution", self.scenario_parameters[G_NETWORK_NAME], self.list_op_dicts[op_id][G_OP_INIT_VEH_DIST]), index_col=0)
                    op_init_distributions[op_id] = init_dist_df["probability"].to_dict()
                else:   # randomly uniform
                    boarding_nodes = self.routing_engine.get_must_stop_nodes()
                    if not boarding_nodes:
                        boarding_nodes = list(range(self.routing_engine.get_number_network_nodes()))
                    op_init_distributions[op_id] = {bn : 1.0/len(boarding_nodes) for bn in boarding_nodes}
            LOG.debug("init distributons: {}".format(op_init_distributions))
            for sim_vid in set_unassigned_vid:
                veh_obj = self.sim_vehicles[sim_vid]
                if veh_obj.pos is None:
                    op_fleetctrl = self.operators[veh_obj.op_id]
                    init_dist = op_init_distributions[veh_obj.op_id]
                    r = np.random.random()
                    s = 0.0
                    init_node = None
                    for n, prob in init_dist.items():
                        s += prob
                        if s >= r:
                            init_node = n
                            break
                    if init_node is None:
                        LOG.error(f"No init node found for random val {r} and init dist {init_dist}")
                    # randomly position all vehicles
                    init_state_info = {}
                    init_state_info[G_V_INIT_NODE] = init_node# np.random.choice(init_node)
                    init_state_info[G_V_INIT_TIME] = self.scenario_parameters[G_SIM_START_TIME]
                    init_state_info[G_V_INIT_SOC] = 0.5 * (1 + np.random.random())
                    veh_obj.set_initial_state(op_fleetctrl, self.routing_engine, init_state_info,
                                                self.scenario_parameters[G_SIM_START_TIME], self.init_blocking)

    def save_final_state(self):
        """
        Records the state at the end of the simulation; can be used as initial state for other simulations.
        """
        LOG.info("Saving final simulation state.")
        final_state_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "final_state.csv")
        sorted_sim_vehicle_keys = sorted(self.sim_vehicles.keys())
        list_vehicle_states = [self.sim_vehicles[sim_vid].return_final_state(self.end_time)
                               for sim_vid in sorted_sim_vehicle_keys]
        fs_df = pd.DataFrame(list_vehicle_states)
        fs_df.to_csv(final_state_f)

    def record_remaining_assignments(self):
        """
        This method simulates the remaining assignments at the end of the simulation in order to get them recorded
        properly. This is necessary for a consistent evaluation.
        """
        c_time = self.end_time# - self.time_step
        LOG.info("record_remaining_assignments()")
        remaining_tasks = -1
        while remaining_tasks != 0:
            self.update_sim_state_fleets(c_time - self.time_step, c_time)
            remaining_tasks = 0
            for veh_obj in self.sim_vehicles.values():
                if veh_obj.assigned_route and veh_obj.assigned_route[0].status == VRL_STATES.OUT_OF_SERVICE:
                    veh_obj.end_current_leg(c_time)
                remaining_tasks += len(veh_obj.assigned_route)
                # if len(veh_obj.assigned_route) > 0:
                    # LOG.debug(f"vid {veh_obj.vid} has remaining assignments:")
                    # LOG.debug("{}".format([str(x) for x in veh_obj.assigned_route]))
            LOG.info(f"\t time {c_time}, remaining tasks {remaining_tasks}")
            c_time += self.time_step
            self.routing_engine.update_network(c_time, update_state=False)
            if c_time > self.end_time + 2*7200:
                # # alternative 1: force end of tasks
                # LOG.warning(f"remaining assignments could not finish! Forcing end of assignments.")
                # for veh_obj in self.sim_vehicles.values():
                #     if veh_obj.assigned_route:
                #         veh_obj.end_current_leg(c_time)
                # alternative 2: just break loop
                LOG.warning(f"remaining assignments could not finish! Break Loop")
                break
        self.record_stats()

    def record_stats(self, force=True):
        """This method records the stats at the end of the simulation."""
        self.demand.save_user_stats(force)
        for op_id in range(self.n_op):
            current_buffer_size = len(self.op_output[op_id])
            if (current_buffer_size and force) or current_buffer_size > BUFFER_SIZE:
                op_output_f = os.path.join(self.dir_names[G_DIR_OUTPUT], f"2-{op_id}_op-stats.csv")
                if os.path.isfile(op_output_f):
                    write_mode = "a"
                    write_header = False
                else:
                    write_mode = "w"
                    write_header = True
                tmp_df = pd.DataFrame(self.op_output[op_id])
                tmp_df.to_csv(op_output_f, index=False, mode=write_mode, header=write_header)
                self.op_output[op_id].clear()
                # LOG.info(f"\t ... just wrote {current_buffer_size} entries from buffer to stats of operator {op_id}.")
                LOG.debug(f"\t ... just wrote {current_buffer_size} entries from buffer to stats of operator {op_id}.")
            self.operators[op_id].record_dynamic_fleetcontrol_output(force=force)

    def update_sim_state_fleets(self, last_time, next_time, force_update_plan=False):
        """
        This method updates the simulation vehicles, records, ends and starts tasks and returns some data that
        will be used for additional state updates (fleet control information, demand, network, ...)
        :param last_time: simulation time before the state update
        :param next_time: simulation time of the state update
        :param force_update_plan: flag that can force vehicle plan to be updated
        """
        LOG.debug(f"updating MoD state from {last_time} to {next_time}")
        for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
            op_id, vid = opid_vid_tuple
            boarding_requests, alighting_requests, passed_VRL, dict_start_alighting =\
                veh_obj.update_veh_state(last_time, next_time)
            for rid, boarding_time_and_pos in boarding_requests.items():
                boarding_time, boarding_pos = boarding_time_and_pos
                LOG.debug(f"rid {rid} boarding at {boarding_time} at pos {boarding_pos}")
                self.demand.record_boarding(rid, vid, op_id, boarding_time, pu_pos=boarding_pos)
                self.operators[op_id].acknowledge_boarding(rid, vid, boarding_time)
            for rid, alighting_start_time_and_pos in dict_start_alighting.items():
                # record user stats at beginning of alighting process
                alighting_start_time, alighting_pos = alighting_start_time_and_pos
                LOG.debug(f"rid {rid} deboarding at {alighting_start_time} at pos {alighting_pos}")
                self.demand.record_alighting_start(rid, vid, op_id, alighting_start_time, do_pos=alighting_pos)
            for rid, alighting_end_time in alighting_requests.items():
                # # record user stats at end of alighting process
                self.demand.user_ends_alighting(rid, vid, op_id, alighting_end_time)
                self.operators[op_id].acknowledge_alighting(rid, vid, alighting_end_time)
            # send update to operator
            if len(boarding_requests) > 0 or len(dict_start_alighting) > 0:
                self.operators[op_id].receive_status_update(vid, next_time, passed_VRL, True)
            else:
                self.operators[op_id].receive_status_update(vid, next_time, passed_VRL, force_update_plan)
        # TODO # after ISTTT: live visualization: send vehicle states (self.live_visualization_flag==True)

    def update_vehicle_routes(self, sim_time):
        """ this method can be used to recalculate routes of currently driving vehicles in case
        network travel times changed and shortest paths need to be re-set
        """
        for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
            veh_obj.update_route()

    def _rid_chooses_offer(self, rid, rq_obj, sim_time):
        """This method performs all actions that derive from a mode choice decision.

        :param rid: request id
        :param rq_obj: request object
        :param sim_time: current simulation time
        :return: chosen operator
        """
        chosen_operator = rq_obj.choose_offer(self.scenario_parameters, sim_time)
        LOG.debug(f" -> chosen operator: {chosen_operator}")
        if chosen_operator is None: # undecided
            if rq_obj.leaves_system(sim_time):
                for i, operator in enumerate(self.operators):
                    operator.user_cancels_request(rid, sim_time)
                self.demand.record_user(rid)
                del self.demand.rq_db[rid]
                try:
                    del self.demand.undecided_rq[rid]
                except KeyError:
                    # raises KeyError if request decided right away
                    pass
            else:
                self.demand.undecided_rq[rid] = rq_obj
        elif chosen_operator < 0:
            # if chosen_operator == G_MC_DEC_PV:
            #     # TODO # self.routing_engine.assign_route_to_network(rq_obj, sim_time)
            #     # TODO # computation of route only when necessary
            #     self.routing_engine.assign_route_to_network(rq_obj, sim_time)
            #     # TODO # check if following method is necessary
            #     self.demand.user_chooses_PV(rid, sim_time)
            # elif chosen_operator == G_MC_DEC_PT:
            #     pt_offer = rq_obj.return_offer(G_MC_DEC_PT)
            #     pt_start_time = sim_time + pt_offer.get(G_OFFER_ACCESS_W, 0) + pt_offer.get(G_OFFER_WAIT, 0)
            #     pt_end_time = pt_start_time + pt_offer.get(G_OFFER_DRIVE, 0)
            #     self.pt.assign_to_pt_network(pt_start_time, pt_end_time)
            #     # TODO # check if following method is necessary
            #     self.demand.user_chooses_PT(rid, sim_time)
            for i, operator in enumerate(self.operators):
                operator.user_cancels_request(rid, sim_time)
            self.demand.record_user(rid)
            del self.demand.rq_db[rid]
            try:
                del self.demand.undecided_rq[rid]
            except KeyError:
                # raises KeyError if request decided right away
                pass
        else:
            for i, operator in enumerate(self.operators):
                if i != chosen_operator:
                    operator.user_cancels_request(rid, sim_time)
                else:
                    operator.user_confirms_booking(rid, sim_time)
                    self.demand.waiting_rq[rid] = rq_obj
            try:
                del self.demand.undecided_rq[rid]
            except KeyError:
                # raises KeyError if request decided right away
                pass
        return chosen_operator

    def _check_waiting_request_cancellations(self, sim_time):
        """This method builds the interface for traveler models, where users can cancel their booking after selecting
        an operator.

        :param sim_time: current simulation time
        :return: None
        """
        for rid, rq_obj in self.demand.waiting_rq.items():
            chosen_operator = rq_obj.get_chosen_operator()
            in_vehicle = rq_obj.get_service_vehicle()
            if in_vehicle is None and chosen_operator is not None and rq_obj.cancels_booking(sim_time):
                self.operators[chosen_operator].user_cancels_request(rid, sim_time)
                self.demand.record_user(rid)
                del self.demand.rq_db[rid]
                del self.demand.waiting_rq[rid]

    def run(self):
        self._start_realtime_plot()
        t_run_start = time.perf_counter()
        if not self._started:
            self._started = True
            if PROGRESS_LOOP == "off":
                for sim_time in range(self.start_time, self.end_time, self.time_step):
                    self.step(sim_time)
                    self._update_realtime_plots_dict(sim_time)
            elif PROGRESS_LOOP == "demand":
                # loop over time with progress bar scaling according to future demand
                all_requests = sum([len(x) for x in self.demand.future_requests.values()])
                with tqdm(total=100) as pbar:
                    for sim_time in range(self.start_time, self.end_time, self.time_step):
                        remaining_requests = sum([len(x) for x in self.demand.future_requests.values()])
                        self.step(sim_time)
                        cur_perc = int(100 * (1 - remaining_requests/all_requests))
                        pbar.update(cur_perc - pbar.n)
                        vehicle_counts = self.count_fleet_status()
                        info_dict = {"simulation_time": sim_time,
                                     "driving": sum([vehicle_counts[x] for x in G_DRIVING_STATUS])}
                        info_dict.update({G_VEHICLE_STATUS_DICT[x]: vehicle_counts[x]
                                          for x in PROGRESS_LOOP_VEHICLE_STATUS})
                        pbar.set_postfix(info_dict)
                        self._update_realtime_plots_dict(sim_time)
            else:
                # loop over time with progress bar scaling with time
                for sim_time in tqdm(range(self.start_time, self.end_time, self.time_step)):
                    self.step(sim_time)
                    self._update_realtime_plots_dict(sim_time)

            # record stats
            self.record_stats()

            # save final state, record remaining travelers and vehicle tasks
            self.save_final_state()
            self.record_remaining_assignments()
        t_run_end = time.perf_counter()
        # call evaluation
        self.evaluate()
        t_eval_end = time.perf_counter()
        # short report
        t_init = datetime.timedelta(seconds=int(t_run_start - self.t_init_start))
        t_sim = datetime.timedelta(seconds=int(t_run_end - t_run_start))
        t_eval = datetime.timedelta(seconds=int(t_eval_end - t_run_end))
        prt_str = f"Scenario {self.scenario_name} finished:\n" \
                  f"{'initialization':>20} : {t_init} h\n" \
                  f"{'simulation':>20} : {t_sim} h\n" \
                  f"{'evaluation':>20} : {t_eval} h\n"
        print(prt_str)
        LOG.info(prt_str)
        self._end_realtime_plot()

    def _start_realtime_plot(self):
        """ This method starts a separate process for real time python plots """
        if self.realtime_plot_flag in {1, 2}:
            bounding = self.routing_engine.return_network_bounding_box()
            lons, lats = list(zip(*bounding))
            if self.realtime_plot_flag == 1:
                self._manager = Manager()
                self._shared_dict = self._manager.dict()
                self._plot_class_instance = PyPlot(self.dir_names["network"], self._shared_dict, plot_extent=lons+lats)
                self._plot_class_instance.start()
            else:
                plot_dir = Path(self.dir_names["output"], "real_time_plots")
                if plot_dir.exists() is False:
                    plot_dir.mkdir()
                self._plot_class_instance = PyPlot(self.dir_names["network"], self._shared_dict,
                                                   plot_extent=lons + lats, plot_folder=str(plot_dir))

    def _end_realtime_plot(self):
        """ Closes the process for real time plots """
        if self.realtime_plot_flag == 1:
            self._shared_dict["stop"] = True
            self._plot_class_instance.join()
            self._manager.shutdown()

    def _update_realtime_plots_dict(self, sim_time):
        """ This method updates the shared dict with the realtime plot process """
        if self.realtime_plot_flag in {1, 2}:
            veh_ids = list(self.sim_vehicles.keys())
            possible_states = self.scenario_parameters.get(G_SIM_REALTIME_PLOT_VEHICLE_STATUS,
                                                           G_VEHICLE_STATUS_DICT.keys())
            possible_states = [G_VEHICLE_STATUS_DICT[x] for x in possible_states]
            veh_status = [self.sim_vehicles[veh].status for veh in veh_ids]
            veh_status = [G_VEHICLE_STATUS_DICT[state] for state in veh_status]
            veh_positions = [self.sim_vehicles[veh].pos for veh in veh_ids]
            veh_positions = self.routing_engine.return_positions_lon_lat(veh_positions)
            df = pd.DataFrame({"status": veh_status,
                               "coordinates": veh_positions})
            self._shared_dict.update({"veh_coord_status_df": df,
                                      "possible_status": possible_states,
                                      "simulation_time": f"simulation time: {datetime.timedelta(seconds=sim_time)}"})
            if self.realtime_plot_flag == 2:
                self._plot_class_instance.save_single_plot(str(sim_time))

    def count_fleet_status(self) -> dict:
        """ This method counts the number of vehicles in each of the vehicle statuses

        :return: dictionary of vehicle codes as keys and number of vehicles in those status as values
        """
        vehicles = self.sim_vehicles.values()
        count = {key: 0 for key in G_VEHICLE_STATUS_DICT.keys()}
        for v in vehicles:
            count[v.status] += 1
        return count

    @abstractmethod
    def step(self, sim_time):
        """This method determines the simulation flow in a time step.

        :param sim_time: new simulation time
        :return: None
        """
        LOG.warning("abstractmethod not overwritten! When defined as ABC in next commits, this will raise an error!")
        pass

    @abstractmethod
    def check_sim_env_spec_inputs(self, scenario_parameters):
        # TODO # delete? -> part of init if necessary
        LOG.warning("abstractmethod not overwritten! When defined as ABC in next commits, this will raise an error!")
        return scenario_parameters

    def add_init(self, scenario_parameters):
        for op_id, op in enumerate(self.operators):
            operator_attributes = self.list_op_dicts[op_id]
            op.add_init(operator_attributes, self.scenario_parameters)

    @abstractmethod
    def add_evaluate(self):
        LOG.warning("abstractmethod not overwritten! When defined as ABC in next commits, this will raise an error!")
        pass
