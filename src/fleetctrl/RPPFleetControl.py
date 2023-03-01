# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
import time
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.rideparcelpooling.objectives import return_parcel_pooling_objective_function
from src.fleetctrl.pooling.immediate.insertion import insertion_with_heuristics, insert_prq_in_selected_veh_list
from src.fleetctrl.rideparcelpooling.immediate.insertion import insert_parcel_prq_in_selected_veh_list, insert_prq_in_selected_veh_list_route_with_parcels, insert_parcel_o_in_selected_veh_list_route_with_parcels, insert_parcel_d_in_selected_veh_list_route_with_parcels
from src.simulation.Offers import TravellerOffer
if TYPE_CHECKING:
    from src.demand.TravelerModels import RequestBase, ParcelRequestBase
    from src.routing.NetworkBase import NetworkBase
    from src.simulation.Vehicles import SimulationVehicle

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

LARGE_INT = 1000000000000

def get_veh_plan_distance(veh_obj, veh_plan, routing_engine):
    """This function evaluates the driven distance according to a vehicle plan.

    :param veh_obj: simulation vehicle object
    :param veh_plan: vehicle plan in question
    :param routing_engine: for routing queries
    :return: objective function value
    """
    sum_dist = 0
    last_pos = veh_obj.pos
    for ps in veh_plan.list_plan_stops:
        pos = ps.get_pos()
        if pos != last_pos:
            sum_dist += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
            last_pos = pos
    return sum_dist

class ParcelPlanRequest(PlanRequest):
    """ this class is the main class describing customer requests with hard time constraints which is
    used for planing in the fleetcontrol modules. in comparison to the traveler classes additional parameters
    and attributes can be defined which are unique for each operator defining the service of the operator (i.e. time constraints, walking, ...)"""
    def __init__(self, rq : ParcelRequestBase, routing_engine : NetworkBase, earliest_pickup_time : int=0, 
                 latest_pickup_time : int=LARGE_INT, earliest_drop_off_time : int = -1, latest_drop_off_time : int = LARGE_INT,
                 boarding_time : int=0, pickup_pos : tuple=None, dropoff_pos : tuple=None, 
                 walking_time_start : float=0, walking_time_end : float=0, sub_rid_id=None):
        """
        :param rq: reference to parcel-traveller object which is requesting a trip
        :param routing_engine: reference to network object
        :param earliest_pickup_time: defines earliest pickup_time from request time
        :param latest_pickup_time: defines latest pickup time from request time
        :param earliest_drop_off_time: defines earliest drop off time from request time
        :param latest_drop_off_time: defines latest drop off time from request time
        :param boarding_time: time needed for parcel to board the vehicle
        :param pickup_pos: network position tuple of pick up (used if pickup differs from request origin)
        :param dropoff_pos: network position tuple of drop off (used if dropoff differs from request destination)
        :param walking_time_start: walking time from origin to pickup
        :param walking_time_end: walking time from dropoff to destination
        :param sub_rid_id: id of this plan request that differs from the traveller id; usefull if one customer can be represented by multiple plan requests
        """
        # TODO: Variable Paketgröße erstellen

        # für Wartezeiten und Umwege etc.
        # (min_wait_time, max_wait_time, max_detour_time_factor, max_constant_detour_time)
        # entsprechende globale Parameter erstellen

        # Zeigervariable für Champions League (passenger und parcel auf selber Fahrt) einführen
        # Locations für post offices/parcel lockers für parcel hopping importieren u. implementieren (Wie und wo im Code?)
        # Time windows für pu und drop-off

        # What to do with pre-reserved requests? --> Pickup time, delivery time etc. for those

        # Priority variable for parcel delivery (range 1:5 descending priority e.g.)
        
        # copy from rq
        self.rid = rq.get_rid_struct()
        self.nr_pax = 0 # no persons
        self.parcel_size = rq.parcel_size
        if sub_rid_id is not None:
            self.sub_rid_struct = (self.rid, sub_rid_id)
        else:
            self.sub_rid_struct = self.rid
        self.rq_time = rq.rq_time
        #
        if pickup_pos is None:
            self.o_pos = rq.o_pos
        else:
            self.o_pos = pickup_pos
        if dropoff_pos is None:
            self.d_pos = rq.d_pos
        else:
            self.d_pos = dropoff_pos
        #
        self.walking_time_start = walking_time_start
        self.walking_time_end = walking_time_end
        #
        _, self.init_direct_tt, self.init_direct_td = routing_engine.return_travel_costs_1to1(self.o_pos, self.d_pos)
        # decision/output
        self.service_vehicle = None
        self.pu_time = None
        # constraints -> only in operator rq-class [pu: pick-up | do: drop-off, both start with boarding process]

        self.reservation_flag = False
        # earliest pu
        if earliest_pickup_time is None:
            earliest_pickup_time = 0
        self.t_pu_earliest = self.rq_time + earliest_pickup_time
        if rq.earliest_start_time is not None and rq.earliest_start_time > self.t_pu_earliest:
            self.t_pu_earliest = rq.earliest_start_time
        # latest pu
        if latest_pickup_time is None:
            latest_pickup_time = LARGE_INT
        self.t_pu_latest = self.rq_time + latest_pickup_time
        if rq.latest_start_time is not None and rq.latest_start_time < self.t_pu_latest:
            self.t_pu_latest = rq.latest_start_time
        # earliest drop off
        self.t_do_earliest = self.rq_time + earliest_drop_off_time
        if rq.earliest_drop_off_time is not None and rq.earliest_drop_off_time > self.t_do_earliest:
            self.t_do_earliest = rq.earliest_drop_off_time
        if self.t_do_earliest > self.rq_time:
            LOG.warning("earliest drop off time contraint not implemented!")
        # latest drop off
        self.t_do_latest = self.rq_time + latest_drop_off_time
        if rq.latest_drop_off_time is not None and self.t_do_latest < rq.latest_drop_off_time:
            self.t_do_latest = rq.latest_drop_off_time
        
        if self.t_pu_earliest + self.init_direct_tt + boarding_time > self.t_do_latest:
            raise EnvironmentError(f"cant be fulfilled within time constraints: {self.t_pu_earliest} + {self.init_direct_tt} > {self.t_do_latest}")
        
        self.max_trip_time = float("inf")
        self.locked = False
        # offer
        self.offer = None
        self.status = G_PRQS_NO_OFFER
        self.expected_pickup_time = None
        self.expected_dropoff_time = None

    def is_parcel(self):
        return True
    
    def get_d_stop_info(self):
        """ returns a three tuple defining information about request drop off
        :return: tuple of (destination pos, latest drop off time, maximum trip time)"""
        return self.d_pos, self.t_do_latest, LARGE_INT
    
INPUT_PARAMETERS_RPPFleetControlFullInsertion = {
    "doc" :     """
    this class is used in the paper
    Integrating Parcel Deliveries into a Ride-Pooling Service - An Agent-Based Simulation Study; Fehn, Engelhardt, Dandl, Bogenberger, Busch (2022)
    parcel deliveries are integrated in a ride-pooling service. No explicit time constraints are assumed for parcel pick-ups and drop-offs.
    Parcels are picked up and dropped off, when vehicles pass by origin or destination during passenger transport
    the "closeness" of vehicle routes to parcels is defined by the parameter G_OP_PA_ASSTH (between 0 and 1, with 1 reflecting that parcel o and d is directly on the vehicle route)
    G_OP_PA_OBASS (bool) describes the integration. if False, an insertion of parcels into vehicle routes is only allowed, when no passenger is on board
    This class represents the CDPA strategy in the paper. origin and destination of parcels are directly inserted, when vehicle pass by.
    the fleet control requires an immediate descision process in the simulation class
    """,
    "inherit" : "FleetControlBase",
    "input_parameters_mandatory": [G_OP_PA_CONST_BT, G_OP_PA_ASSTH, G_OP_PA_OBASS],
    "input_parameters_optional": [G_OP_PA_EPT, G_OP_PA_LPT, G_OP_PA_EDT, G_OP_PA_LDT, G_OP_PA_ADD_BT],
    "mandatory_modules": [],
    "optional_modules": []
}

class RPPFleetControlFullInsertion(FleetControlBase):
    """
    this class is used in the paper
    Integrating Parcel Deliveries into a Ride-Pooling Service - An Agent-Based Simulation Study; Fehn, Engelhardt, Dandl, Bogenberger, Busch (2022)
    parcel deliveries are integrated in a ride-pooling service. No explicit time constraints are assumed for parcel pick-ups and drop-offs.
    Parcels are picked up and dropped off, when vehicles pass by origin or destination during passenger transport
    the "closeness" of vehicle routes to parcels is defined by the parameter G_OP_PA_ASSTH (between 0 and 1, with 1 reflecting that parcel o and d is directly on the vehicle route)
    G_OP_PA_OBASS (bool) describes the integration. if False, an insertion of parcels into vehicle routes is only allowed, when no passenger is on board
    This class represents the CDPA strategy in the paper. origin and destination of parcels are directly inserted, when vehicle pass by.
    the fleet control requires an immediate descision process in the simulation class
    """
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=None, list_pub_charging_infra=[]):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=op_charge_depot_infra, list_pub_charging_infra=list_pub_charging_infra)
        self.rid_to_assigned_vid = {} # rid -> vid
        self.pos_veh_dict = {}  # pos -> list_veh
        self.vr_ctrl_f = return_parcel_pooling_objective_function(operator_attributes[G_OP_VR_CTRL_F])
        self.sim_time = scenario_parameters[G_SIM_START_TIME]
        self.optimisation_time_step = operator_attributes[G_RA_REOPT_TS]
        # parcel service attributes
        # min_wait_time : int=0, 
        #                 max_wait_time : int=LARGE_INT, earliest_drop_off_time : int = -1, latest_drop_off_time : int = LARGE_INT,
        self.parcel_earliest_pickup_time = operator_attributes.get(G_OP_PA_EPT, 0)
        self.parcel_latest_pickup_time = operator_attributes.get(G_OP_PA_LPT, LARGE_INT)
        self.parcel_earliest_dropoff_time = operator_attributes.get(G_OP_PA_EDT, 0)
        self.parcel_latest_dropoff_time = operator_attributes.get(G_OP_PA_LDT, LARGE_INT)
        self.parcel_const_bt = operator_attributes[G_OP_PA_CONST_BT]
        self.parcel_add_bt = operator_attributes.get(G_OP_PA_ADD_BT, 0)
        #
        self.parcel_assignment_threshold = operator_attributes[G_OP_PA_ASSTH] # assignment of parcel if additional distance by assignment < self.parcel_assignment_threshold * direct trip of parcel
        self.allow_parcel_pu_with_ob_cust = operator_attributes[G_OP_PA_OBASS]
        self.max_number_parcels_scheduled_per_veh = self.sim_vehicles[0].max_parcels # TODO
        self.vehicle_assignment_changed = {}    # vid -> 1 if assignment changed in last step
        #
        # others # TODO # standardize IRS assignment memory?
        self.tmp_assignment = {}  # rid -> VehiclePlan
        self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_RQU)
        # parcels
        self.parcel_dict : Dict[Any, ParcelPlanRequest] = {} 
        self.unassigned_parcel_dict : Dict[Any, ParcelPlanRequest] = {}
        
    def receive_status_update(self, vid, simulation_time, list_finished_VRL, force_update=True):
        """This method can be used to update plans and trigger processes whenever a simulation vehicle finished some
         VehicleRouteLegs.

        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        :param list_finished_VRL: list of VehicleRouteLeg objects
        :type list_finished_VRL: list
        :param force_update: indicates if also current vehicle plan feasibilities have to be checked
        :type force_update: bool
        """
        veh_obj = self.sim_vehicles[vid]
        if simulation_time%self.optimisation_time_step == 0:
            force_update=True
        super().receive_status_update(vid, simulation_time, list_finished_VRL, force_update=force_update)
        try:
            self.pos_veh_dict[veh_obj.pos].append(veh_obj)
        except KeyError:
            self.pos_veh_dict[veh_obj.pos] = [veh_obj]
        LOG.debug(f"veh {veh_obj} | after status update: {self.veh_plans[vid]}")

    def user_request(self, rq : RequestBase, simulation_time : int):
        if rq.is_parcel:    # request of parcel
            return self._parcel_request(rq, simulation_time)
        else:   # request of person
            return self._person_request(rq, simulation_time)

    def _parcel_request(self, parcel_traveler : RequestBase, sim_time : int):
        """ this function should do the same as the usual user request function just for parcels
        no direct parcel customer - operator interaction is modelled, instead it is assumed that all parcels are already booked. 
        therefore parcels are just added to the operator database and a quasi-offer is created (an offer with fixed entries to enable the customer acceptance in the fleetsim class)
        :param parcel_traveler: request class representing a parcel
        :param sim_time: current simulation time"""
        t0 = time.perf_counter()
        LOG.debug(f"Incoming request {parcel_traveler.__dict__} at time {sim_time}")
        parcel_prq = ParcelPlanRequest(parcel_traveler, self.routing_engine, earliest_pickup_time=self.parcel_earliest_pickup_time,
                                       latest_pickup_time=self.parcel_latest_pickup_time, earliest_drop_off_time=self.parcel_earliest_dropoff_time,
                                       latest_drop_off_time=self.parcel_latest_dropoff_time, boarding_time=self.parcel_const_bt)
        rid_struct = parcel_prq.get_rid_struct()
        self.parcel_dict[rid_struct] = parcel_prq
        self.rq_dict[rid_struct] = parcel_prq
        self.unassigned_parcel_dict[rid_struct] = parcel_prq
        
        if parcel_prq.o_pos == parcel_prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(parcel_prq, sim_time)
            return
        
        self._create_parcel_offer(parcel_prq, sim_time)

    def _person_request(self, person_request : RequestBase, sim_time : int):
        """This method is triggered for a new incoming request. It generally adds the rq to the database. It has to
        return an offer to the user. This operator class only works with immediate responses and therefore either
        sends an offer or a rejection. this function uses a simple insertion heuristic to assign customers and create offers

        :param person_request: request object containing all request information
        :type person_request: RequestBase
        :param sim_time: current simulation time
        :type sim_time: int
        """
        t0 = time.perf_counter()
        LOG.debug(f"Incoming request {person_request.__dict__} at time {sim_time}")
        self.sim_time = sim_time
        prq = PlanRequest(person_request, self.routing_engine, min_wait_time=self.min_wait_time,
                          max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt)

        rid_struct = person_request.get_rid_struct()
        self.rq_dict[rid_struct] = prq

        if prq.o_pos == prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(prq, sim_time)
            return

        o_pos, t_pu_earliest, t_pu_latest = prq.get_o_stop_info()
        if t_pu_earliest - sim_time > self.opt_horizon:
            self.reservation_module.add_reservation_request(prq, sim_time)
            offer = self.reservation_module.return_reservation_offer(prq.get_rid_struct(), sim_time)
            LOG.debug(f"reservation offer for rid {rid_struct} : {offer}")
        else:
            list_tuples = insert_prq_in_selected_veh_list_route_with_parcels(self.sim_vehicles, self.veh_plans, prq, self.vr_ctrl_f,
                                                               self.routing_engine, self.rq_dict, sim_time, self.const_bt, self.add_bt,
                                                               allow_parcel_pu_with_ob_cust=self.allow_parcel_pu_with_ob_cust)
            #list_tuples = insertion_with_heuristics(sim_time, prq, self, force_feasible_assignment=True)
            LOG.debug("list insertion solutions: {}".format(list_tuples))
            if len(list_tuples) > 0:
                (vid, vehplan, delta_cfv) = min(list_tuples, key=lambda x:x[2])
                self.tmp_assignment[rid_struct] = vehplan
                offer = self._create_user_offer(prq, sim_time, vehplan)
                LOG.debug(f"new offer for rid {rid_struct} : {offer}")
            else:
                LOG.debug(f"rejection for rid {rid_struct}")
                self._create_rejection(prq, sim_time)
        # record cpu time
        dt = round(time.perf_counter() - t0, 5)
        old_dt = self._get_current_dynamic_fleetcontrol_value(sim_time, G_FCTRL_CT_RQU)
        if old_dt is None:
            new_dt = dt
        else:
            new_dt = old_dt + dt
        output_dict = {G_FCTRL_CT_RQU: new_dt}
        self._add_to_dynamic_fleetcontrol_output(sim_time, output_dict)
        
    def user_confirms_booking(self, rid, simulation_time):
        """This method is used to confirm a customer booking. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        super().user_confirms_booking(rid, simulation_time)
        LOG.debug(f"user confirms booking {rid} at {simulation_time}")
        prq = self.rq_dict[rid]
        if prq.is_parcel():
            return
        if prq.get_reservation_flag():
            self.reservation_module.user_confirms_booking(rid, simulation_time)
        else:
            new_vehicle_plan = self.tmp_assignment[rid]
            vid = new_vehicle_plan.vid
            veh_obj = self.sim_vehicles[vid]
            self.assign_vehicle_plan(veh_obj, new_vehicle_plan, simulation_time)
            del self.tmp_assignment[rid]

    def user_cancels_request(self, rid, simulation_time):
        """This method is used to confirm a customer cancellation. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        LOG.debug(f"user cancels request {rid} at {simulation_time}")
        prq = self.rq_dict[rid]
        if prq.is_parcel():
            if prq.o_pos == prq.d_pos:
                try:
                    del self.rq_dict[rid]
                except KeyError:
                    pass
                try:
                    del self.parcel_dict[rid]
                except KeyError:
                    pass
                try:
                    del self.unassigned_parcel_dict[rid]
                except:
                    pass
            else:
                raise NotImplementedError
        else:
            if prq.get_reservation_flag():
                self.reservation_module.user_cancels_request(rid, simulation_time)
            else:
                prev_assignment = self.tmp_assignment.get(rid)
                if prev_assignment:
                    del self.tmp_assignment[rid]
            del self.rq_dict[rid]
        
    def acknowledge_boarding(self, rid, vid, simulation_time):
        """This method can trigger some database processes whenever a passenger is starting to board a vehicle.

        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        LOG.debug(f"acknowledge boarding {rid} in {vid} at {simulation_time}")
        self.rq_dict[rid].set_pickup(vid, simulation_time)

    def acknowledge_alighting(self, rid, vid, simulation_time):
        """This method can trigger some database processes whenever a passenger is finishing to alight a vehicle.

        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        LOG.debug(f"acknowledge alighting {rid} from {vid} at {simulation_time}")
        del self.rq_dict[rid]
        del self.rid_to_assigned_vid[rid]
        
    def _prq_from_reservation_to_immediate(self, rid, sim_time):
        """This method is triggered when a reservation request becomes an immediate request.
        All database relevant methods can be triggered from here.

        :param rid: request id
        :param sim_time: current simulation time
        :return: None
        """
        for base_rid, epa in sorted(self.reserved_base_rids.items(), key=lambda x: x[1]):
            if epa - sim_time > self.opt_horizon:
                break
            else:
                LOG.debug(f"activate {base_rid} with epa {epa} for global optimisation at time {sim_time}!")
                del self.reserved_base_rids[base_rid]

    def _call_time_trigger_request_batch(self, simulation_time):
        """this is the main functionality for the rpp assignment control
        it checks for all unassigned parcels all new assigned vehicle plans
            1) if the number of currently assigned parcels does not not exceed a certain number (to avoid long scheduling of parcels that dont have time constraints)
            2) a fast pretest of the insertion if the route fits the parcel origin and destination
            3) the real insertion of the parcel into the route
            4) if the additional distance to deliver the parcel in this route < (1 - assignment_threshold) * direct parcel delivery distance
            5) of all possible vehicle assignments the parcel is assigned to the vehicle that results in the smalltest additional driven distance

        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        self.pos_veh_dict = {}  # pos -> list_veh
        if simulation_time % self.optimisation_time_step == 0:
            new_assigned_parcel = {}    # p_rid -> 1
            for p_rid, parcel_prq in self.unassigned_parcel_dict.items():
                list_options = []
                for vid in self.vehicle_assignment_changed.keys():
                    veh_plan = self.veh_plans[vid]
                    veh_obj = self.sim_vehicles[vid]
                    number_scheduled_parcels = sum([self.rq_dict[x].parcel_size for x in veh_plan.pax_info.keys() if type(x) == str and x.startswith("p")])
                    if number_scheduled_parcels >= self.max_number_parcels_scheduled_per_veh:
                        LOG.debug("too many scheduled parcels! {} {}".format(vid, number_scheduled_parcels))
                        continue
                    if not self._pre_test_insertion(parcel_prq, vid):
                        continue
                    status_quo_dd = get_veh_plan_distance(veh_obj, veh_plan, self.routing_engine)
                    res = insert_parcel_prq_in_selected_veh_list([veh_obj], {vid : veh_plan}, parcel_prq, self.vr_ctrl_f,
                                                    self.routing_engine, self.rq_dict, simulation_time, self.parcel_const_bt, self.parcel_add_bt,
                                                    allow_parcel_pu_with_ob_cust=self.allow_parcel_pu_with_ob_cust)
                    if len(res) == 0:
                        continue
                    veh_plan_with_parcel = res[0][1]
                    LOG.debug("possible plan with prq: {}".format(veh_plan_with_parcel))
                    inserted_dd = get_veh_plan_distance(veh_obj, veh_plan_with_parcel, self.routing_engine)
                    additional_parcel_distance = inserted_dd - status_quo_dd
                    rel_saved = (parcel_prq.init_direct_td - additional_parcel_distance)/parcel_prq.init_direct_td
                    if rel_saved > self.parcel_assignment_threshold:
                        LOG.warning("to adopt!")
                    #if additional_parcel_distance < parcel_prq.init_direct_td:
                        list_options.append( (vid, veh_plan_with_parcel, additional_parcel_distance - parcel_prq.init_direct_td) )
                if len(list_options) > 0:
                    best_option = min(list_options, key = lambda x:x[2])
                    best_vid, best_plan, diff_distance = best_option
                    self.assign_vehicle_plan(self.sim_vehicles[best_vid], best_plan, simulation_time)
                    new_assigned_parcel[p_rid] = 1
                    LOG.debug(f"assigned parcel {p_rid} to vid {vid} with diff distance {diff_distance} : {best_plan}")
                else:
                    LOG.debug(f"no option found for parcel {p_rid}")
            for p_rid in new_assigned_parcel.keys():
                del self.unassigned_parcel_dict[p_rid]
            self.vehicle_assignment_changed = {}
                    
    def compute_VehiclePlan_utility(self, simulation_time, veh_obj, vehicle_plan):
        """This method computes the utility of a given plan and returns the value.

        :param simulation_time: current simulation time
        :type simulation_time: float
        :param veh_obj: vehicle object
        :type veh_obj: SimulationVehicle
        :param vehicle_plan: vehicle plan in question
        :type vehicle_plan: VehiclePlan
        :return: utility of vehicle plan
        :rtype: float
        """
        return self.vr_ctrl_f(simulation_time, veh_obj, vehicle_plan, self.rq_dict, self.routing_engine)
    
    def _create_user_offer(self, prq : PlanRequest, simulation_time : int, assigned_vehicle_plan : VehiclePlan=None, offer_dict_without_plan : dict={}):
        """ creating the offer for a requests

        :param prq: plan request
        :type prq: PlanRequest obj
        :param simulation_time: current simulation time
        :type simulation_time: int
        :param assigned_vehicle_plan: vehicle plan of initial solution to serve this request
        :type assigned_vehicle_plan: VehiclePlan None
        :param offer_dict_without_plan: can be used to create an offer that is not derived from a vehicle plan
                    entries will be used to create/extend offer
        :type offer_dict_without_plan: dict or None
        :return: offer for request
        :rtype: TravellerOffer
        """
        if prq.is_parcel():
            LOG.warning("direct parcel offer created")
        if assigned_vehicle_plan is not None:
            pu_time, do_time = assigned_vehicle_plan.pax_info.get(prq.get_rid_struct())
            # offer = {G_OFFER_WAIT: pu_time - simulation_time, G_OFFER_DRIVE: do_time - pu_time,
            #          G_OFFER_FARE: int(prq.init_direct_td * self.dist_fare + self.base_fare)}
            offer = TravellerOffer(prq.get_rid_struct(), self.op_id, pu_time - prq.rq_time, do_time - pu_time,
                                   self._compute_fare(simulation_time, prq, assigned_vehicle_plan))
            prq.set_service_offered(offer)  # has to be called
        else:
            offer = self._create_rejection(prq, simulation_time)
        return offer
    
    def _create_parcel_offer(self, prq : ParcelPlanRequest, simulation_time : int, assigned_vehicle_plan : VehiclePlan=None, offer_dict_without_plan : dict={}):
        """ creating the offer for a parcel requests
        this is just a quasi-offer. entries cannot be used for mode choice. all offers just get standard values to enforce the customer to accept the service because
        customer-operator interactions are not included for these fleet controls

        :param prq: plan request
        :type prq: PlanRequest obj
        :param simulation_time: current simulation time
        :type simulation_time: int
        :param assigned_vehicle_plan: vehicle plan of initial solution to serve this request
        :type assigned_vehicle_plan: VehiclePlan None
        :param offer_dict_without_plan: can be used to create an offer that is not derived from a vehicle plan
                    entries will be used to create/extend offer
        :type offer_dict_without_plan: dict or None
        :return: offer for request
        :rtype: TravellerOffer
        """
        LOG.warning("parcel offer not implemented correctly")
        wait_time = 0
        driving_time = 0
        fare = 0
        offer = TravellerOffer(prq.get_rid_struct(), self.op_id, wait_time, driving_time, fare)
        prq.set_service_offered(offer)  # has to be called
        return offer
    
    def change_prq_time_constraints(self, sim_time, rid, new_lpt, new_ept=None):
        """This method should be called when the hard time constraints of a customer should be changed.
        It changes the PlanRequest attributes. Moreover, this method called on child classes should adapt the
        PlanStops of VehiclePlans containing this PlanRequest and recheck feasibility. The VehiclePlan method
        update_prq_hard_constraints() can be used for this purpose.

        :param sim_time: current simulation time
        :param rid: request id
        :param new_lpt: new latest pickup time, None is ignored
        :param new_ept: new earliest pickup time, None is ignored
        :return: None
        """
        LOG.debug("change time constraints for rid {}".format(rid))
        prq = self.rq_dict[rid]
        prq.set_new_pickup_time_constraint(new_lpt, new_earliest_pu_time=new_ept)
        ass_vid = self.rid_to_assigned_vid.get(rid)
        if ass_vid is not None:
            self.veh_plans[ass_vid].update_prq_hard_constraints(self.sim_vehicles[ass_vid], sim_time,
                                                                self.routing_engine, prq, new_lpt, new_ept=new_ept,
                                                                keep_feasible=True)

    def assign_vehicle_plan(self, veh_obj, vehicle_plan, sim_time, force_assign=False, add_arg=None):
        super().assign_vehicle_plan(veh_obj, vehicle_plan, sim_time, force_assign, add_arg)
        self.vehicle_assignment_changed[veh_obj.vid] = 1

    def lock_current_vehicle_plan(self, vid):
        super().lock_current_vehicle_plan(vid)

    def _lock_vid_rid_pickup(self, sim_time, vid, rid):
        super()._lock_vid_rid_pickup(sim_time, vid, rid)
        
    def _compute_fare(self, sim_time: int, prq: PlanRequest, assigned_veh_plan: VehiclePlan = None) -> float:
        if prq.is_parcel():
            return self._compute_parcel_fare(sim_time, prq, assigned_veh_plan=assigned_veh_plan)
        return super()._compute_fare(sim_time, prq, assigned_veh_plan=assigned_veh_plan)
    
    def _compute_parcel_fare(self, sim_time, parcel_plan_request, assigned_veh_plan = None):
        LOG.warning("no parcel fare system defined")
        return 0
    
    def _pre_test_insertion(self, parcel_request : PlanRequest, vid):
        """ to reduce computationally complexity for the check of insertions for parcels, this function is used as a filter for checks.
        an insertion into a vehicle routes is only possible within the current implementations if the following check returns true."""
        min_o_dd = float("inf")
        min_d_dd = float("inf")
        s_pos = self.sim_vehicles[vid].pos
        po = parcel_request.get_o_stop_info()[0]
        pd = parcel_request.get_d_stop_info()[0]
        for ps in self.veh_plans[vid].list_plan_stops:
            c_pos = ps.get_pos()
            o_dd = self.routing_engine.return_travel_costs_1to1(s_pos, po)[2] + self.routing_engine.return_travel_costs_1to1(po, c_pos)[2] - self.routing_engine.return_travel_costs_1to1(s_pos, c_pos)[2]
            d_dd = self.routing_engine.return_travel_costs_1to1(s_pos, pd)[2] + self.routing_engine.return_travel_costs_1to1(pd, c_pos)[2] - self.routing_engine.return_travel_costs_1to1(s_pos, c_pos)[2]
            if o_dd < min_o_dd:
                min_o_dd = o_dd
            if d_dd < min_d_dd:
                min_d_dd = d_dd
            s_pos = c_pos
        o_dd = self.routing_engine.return_travel_costs_1to1(s_pos, po)[2]
        d_dd = self.routing_engine.return_travel_costs_1to1(s_pos, pd)[2]
        if o_dd < min_o_dd:
            min_o_dd = o_dd
        if d_dd < min_d_dd:
            min_d_dd = d_dd
        additional_parcel_distance = min_o_dd + min_d_dd
        rel_saved = (parcel_request.init_direct_td - additional_parcel_distance)/parcel_request.init_direct_td
        LOG.debug(f"test {parcel_request.get_rid_struct()} -> vid {vid}: rel_saved {rel_saved}")
        if rel_saved > self.parcel_assignment_threshold:
            LOG.debug(f"-> True")
            return True
        else:
            LOG.debug(f" -> False")
            return False

INPUT_PARAMETERS_RPPFleetControlSingleStopInsertion = {
    "doc" :     """
    this class is used in the paper
    Integrating Parcel Deliveries into a Ride-Pooling Service - An Agent-Based Simulation Study; Fehn, Engelhardt, Dandl, Bogenberger, Busch (2022)
    parcel deliveries are integrated in a ride-pooling service. No explicit time constraints are assumed for parcel pick-ups and drop-offs.
    Parcels are picked up and dropped off, when vehicles pass by origin or destination during passenger transport
    the "closeness" of vehicle routes to parcels is defined by the parameter G_OP_PA_ASSTH (between 0 and 1, with 1 reflecting that parcel o and d is directly on the vehicle route)
    G_OP_PA_OBASS (bool) describes the integration. if False, an insertion of parcels into vehicle routes is only allowed, when no passenger is on board
    This class represents the SDPA strategy in the paper. origin and destination of parcels are added to vehicle schedules independently from each other. if a vehicle passes by origin, 
    it is inserted into the route to pick up the parcel. The destination is only inserted if the corresponding vehicle passes also the destination some time later during the simulation.
    in case not all parcels could be delivered (but are already pick up), remaining parcels are delivered at the time specified by G_OP_PA_REDEL
    this class reflects the SDPA strategy in the paper
    the fleet control requires an immediate descision process in the simulation class
    """,
    "inherit" : "RPPFleetControlFullInsertion",
    "input_parameters_mandatory": [G_OP_PA_REDEL],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class RPPFleetControlSingleStopInsertion(RPPFleetControlFullInsertion):
    """
    this class is used in the paper
    Integrating Parcel Deliveries into a Ride-Pooling Service - An Agent-Based Simulation Study; Fehn, Engelhardt, Dandl, Bogenberger, Busch (2022)
    parcel deliveries are integrated in a ride-pooling service. No explicit time constraints are assumed for parcel pick-ups and drop-offs.
    Parcels are picked up and dropped off, when vehicles pass by origin or destination during passenger transport
    the "closeness" of vehicle routes to parcels is defined by the parameter G_OP_PA_ASSTH (between 0 and 1, with 1 reflecting that parcel o and d is directly on the vehicle route)
    G_OP_PA_OBASS (bool) describes the integration. if False, an insertion of parcels into vehicle routes is only allowed, when no passenger is on board
    This class represents the SDPA strategy in the paper. origin and destination of parcels are added to vehicle schedules independently from each other. if a vehicle passes by origin, 
    it is inserted into the route to pick up the parcel. The destination is only inserted if the corresponding vehicle passes also the destination some time later during the simulation.
    in case not all parcels could be delivered (but are already pick up), remaining parcels are delivered at the time specified by G_OP_PA_REDEL
    this class reflects the SDPA strategy in the paper
    the fleet control requires an immediate descision process in the simulation class
    """
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=None, list_pub_charging_infra=[]):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=op_charge_depot_infra, list_pub_charging_infra=list_pub_charging_infra)
        self.vid_to_inserted_parcel_id = {} # vid -> p_rid -> 1
        self.deliver_remaining_parcel_time = operator_attributes[G_OP_PA_REDEL]
        
    def _call_time_trigger_request_batch(self, simulation_time):
        """This method implements the main functionality of the parcel assignment strategy
        the function is seperated into parcel destination assignment and origin assignment
        
        first it is checked if an on-board parcel can be delivered
        therefore for all vehicles with a new assignment the on-board parcels are checked
        the insertion for each on-board parcel is checked and if the driven distance does not increase by more then (1-assignment_threshold)*direct parcel distance/2, the assignment is made
        
        second, a pick-up of parcels is checked
        similarly, for all vehicles with a new assignment the pick-up of a new parcel is checked
        for each parcel it is checked if one of the vehicle can accomodate the parcel pickup
        after insertion it is checked if the driven distance does not increase by more then (1-assignment_threshold)*direct parcel distance/2
        if more than one vehilce is found, the route with minimal additonal driven distance is assigned

        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        self.pos_veh_dict = {}  # pos -> list_veh
        if simulation_time % self.optimisation_time_step == 0:
            if simulation_time >= self.deliver_remaining_parcel_time:
                return
            new_assigned_parcel = {}    # p_rid -> 1
            for vid in self.vehicle_assignment_changed.keys():
                veh_plan = self.veh_plans[vid]
                veh_obj = self.sim_vehicles[vid]
                status_quo_dd = get_veh_plan_distance(veh_obj, veh_plan, self.routing_engine)
                for p_rid in list(self.vid_to_inserted_parcel_id.get(vid, {}).keys()):
                    parcel_prq = self.rq_dict[p_rid]
                    if not self._pre_test_insertion(parcel_prq, vid, False):
                        continue
                    LOG.debug(f"try inserting d of {p_rid} -> {vid}")
                    res = insert_parcel_d_in_selected_veh_list_route_with_parcels([veh_obj], {vid : veh_plan}, parcel_prq, self.vr_ctrl_f,
                                                    self.routing_engine, self.rq_dict, simulation_time, self.parcel_const_bt, self.parcel_add_bt,
                                                    allow_parcel_pu_with_ob_cust=self.allow_parcel_pu_with_ob_cust)
                    if len(res) == 0:
                        continue
                    veh_plan_with_parcel = res[0][1]
                    LOG.debug("possible plan with destination of prq: {}".format(veh_plan_with_parcel))
                    inserted_dd = get_veh_plan_distance(veh_obj, veh_plan_with_parcel, self.routing_engine)
                    additional_parcel_distance = inserted_dd - status_quo_dd
                    rel_saved = (parcel_prq.init_direct_td/2.0 - additional_parcel_distance)/parcel_prq.init_direct_td*2.0
                    if rel_saved > self.parcel_assignment_threshold:
                        LOG.warning("to adopt!")
                        best_vid, best_plan, _ = res[0]
                        self.assign_vehicle_plan(self.sim_vehicles[best_vid], best_plan, simulation_time)
                        status_quo_dd = get_veh_plan_distance(veh_obj, best_plan, self.routing_engine)
                        veh_plan = self.veh_plans[best_vid]
                        del self.vid_to_inserted_parcel_id[vid][p_rid]
                        LOG.debug(f"assigned parcel d {p_rid} to vid {vid} with rel saved {rel_saved} : {best_plan}")
                    
            for p_rid, parcel_prq in self.unassigned_parcel_dict.items():
                list_options = []
                for vid in self.vehicle_assignment_changed.keys():
                    veh_plan = self.veh_plans[vid]
                    number_scheduled_parcels = sum([self.rq_dict[x].parcel_size for x in veh_plan.pax_info.keys() if type(x) == str and x.startswith("p")])
                    if number_scheduled_parcels >= self.max_number_parcels_scheduled_per_veh:
                        LOG.debug("too many scheduled parcels! {} {}".format(vid, number_scheduled_parcels))
                        continue
                    if not self._pre_test_insertion(parcel_prq, vid, True):
                        continue
                    veh_obj = self.sim_vehicles[vid]
                    status_quo_dd = get_veh_plan_distance(veh_obj, veh_plan, self.routing_engine)
                    res = insert_parcel_o_in_selected_veh_list_route_with_parcels([veh_obj], {vid : veh_plan}, parcel_prq, self.vr_ctrl_f,
                                                    self.routing_engine, self.rq_dict, simulation_time, self.parcel_const_bt, self.parcel_add_bt,
                                                    allow_parcel_pu_with_ob_cust=self.allow_parcel_pu_with_ob_cust)
                    if len(res) == 0:
                        continue
                    veh_plan_with_parcel = res[0][1]
                    LOG.debug("possible plan with orign of prq: {}".format(veh_plan_with_parcel))
                    inserted_dd = get_veh_plan_distance(veh_obj, veh_plan_with_parcel, self.routing_engine)
                    additional_parcel_distance = inserted_dd - status_quo_dd
                    rel_saved = (parcel_prq.init_direct_td/2.0 - additional_parcel_distance)/parcel_prq.init_direct_td*2.0
                    LOG.debug("init dis {} additional dis {} rel saved {}".format(parcel_prq.init_direct_td, additional_parcel_distance, rel_saved))
                    if rel_saved > self.parcel_assignment_threshold:
                        LOG.warning("to adopt!")
                        LOG.debug("rel saved {} for plan {}".format(rel_saved, veh_plan_with_parcel))
                    #if additional_parcel_distance < parcel_prq.init_direct_td:
                        list_options.append( (vid, veh_plan_with_parcel, additional_parcel_distance - parcel_prq.init_direct_td) )
                if len(list_options) > 0:
                    best_option = min(list_options, key = lambda x:x[2])
                    best_vid, best_plan, diff_distance = best_option
                    self.assign_vehicle_plan(self.sim_vehicles[best_vid], best_plan, simulation_time)
                    new_assigned_parcel[p_rid] = 1
                    try:
                        self.vid_to_inserted_parcel_id[best_vid][p_rid] = 1
                    except KeyError:
                        self.vid_to_inserted_parcel_id[best_vid] = {p_rid : 1}
                    LOG.debug(f"assigned parcel o {p_rid} to vid {vid} with diff distance {diff_distance} : {best_plan}")
                else:
                    LOG.debug(f"no option found for parcel {p_rid}")
            for p_rid in new_assigned_parcel.keys():
                del self.unassigned_parcel_dict[p_rid]
            self.vehicle_assignment_changed = {}
            
    def _call_time_trigger_additional_tasks(self, sim_time):
        r = super()._call_time_trigger_additional_tasks(sim_time)
        if sim_time >= self.deliver_remaining_parcel_time:
            self.deliver_remaining_parcels(sim_time)
        return r
            
    def assign_vehicle_plan(self, veh_obj : SimulationVehicle, vehicle_plan : VehiclePlan, sim_time : int, force_assign : bool=False, add_arg : Any=None):
        """ this method should be used to assign a new vehicle plan to a vehicle

        WHEN OVERWRITING THIS FUNCTION MAKE SURE TO CALL AT LEAST THE LINES BELOW (i.e. super())

        :param veh_obj: vehicle obj to assign vehicle plan to
        :type veh_obj: SimulationVehicle
        :param vehicle_plan: vehicle plan that should be assigned
        :type vehicle_plan: VehiclePlan
        :param sim_time: current simulation time in seconds
        :type sim_time: int
        :param force_assign: this parameter can be used to enforce the assignment, when a plan is (partially) locked
        :type force_assign: bool
        :param add_arg: possible additional argument if needed
        :type add_arg: not defined here
        """
        LOG.debug(f"assign to {veh_obj.vid} at time {sim_time} : {vehicle_plan}")
        vehicle_plan.update_tt_and_check_plan(veh_obj, sim_time, self.routing_engine, keep_feasible=True)
        new_vrl = self._build_VRLs(vehicle_plan, veh_obj, sim_time)
        veh_obj.assign_vehicle_plan(new_vrl, sim_time, force_ignore_lock=force_assign)
        self.veh_plans[veh_obj.vid] = vehicle_plan
        for rid in list(vehicle_plan.pax_info.keys()):
            pax_info = vehicle_plan.get_pax_info(rid)
            if len(pax_info) == 2:
                self.rq_dict[rid].set_assigned(pax_info[0], pax_info[1])
            else:
                LOG.debug("no destination for rid {}?".format(rid))
                self.rq_dict[rid].set_assigned(pax_info[0], float("inf"))
            self.rid_to_assigned_vid[rid] = veh_obj.vid
        self.vehicle_assignment_changed[veh_obj.vid] = 1
        
    def _pre_test_insertion(self, parcel_request : PlanRequest, vid, o_flag):
        min_dd = float("inf")
        s_pos = self.sim_vehicles[vid].pos
        if o_flag:
            p = parcel_request.get_o_stop_info()[0]
        else:
            p = parcel_request.get_d_stop_info()[0]
        LOG.debug(f"test {parcel_request} {o_flag}")
        LOG.debug(f" -> for {[str(x) for x in self.veh_plans[vid].list_plan_stops]}")
        for ps in self.veh_plans[vid].list_plan_stops:
            c_pos = ps.get_pos()
            dd = self.routing_engine.return_travel_costs_1to1(s_pos, p)[2] + self.routing_engine.return_travel_costs_1to1(p, c_pos)[2] - self.routing_engine.return_travel_costs_1to1(s_pos, c_pos)[2]
            LOG.debug(f"{ps} -> {dd}")
            if dd < min_dd:
                min_dd = dd
            s_pos = c_pos
        dd = self.routing_engine.return_travel_costs_1to1(s_pos, p)[2]
        if dd < min_dd:
            min_dd = dd
        additional_parcel_distance = min_dd
        LOG.debug(f"min dd: {min_dd} | pdd {parcel_request.init_direct_td}")
        rel_saved = (parcel_request.init_direct_td/2.0 - additional_parcel_distance)/parcel_request.init_direct_td*2.0
        LOG.debug(f"test {parcel_request.get_rid_struct()} oflag {o_flag} -> vid {vid}: rel_saved {rel_saved}")
        if rel_saved > self.parcel_assignment_threshold:
            LOG.debug(f"-> True")
            return True
        else:
            LOG.debug(f" -> False")
            return False

    def deliver_remaining_parcels(self, simulation_time):
        LOG.info("deliver remaining parcels!")
        to_del = []
        for vid, list_ob_parcels in self.vid_to_inserted_parcel_id.items():
            if len(list_ob_parcels.keys()) > 0:
                cur_plan = self.veh_plans[vid]
                veh_obj = self.sim_vehicles[vid]
                remaining_parcels = {}
                for p_rid in list_ob_parcels.keys():
                    parcel_prq = self.rq_dict[p_rid]
                    res = insert_parcel_d_in_selected_veh_list_route_with_parcels([veh_obj], {vid : cur_plan}, parcel_prq, self.vr_ctrl_f,
                                                    self.routing_engine, self.rq_dict, simulation_time, self.parcel_const_bt, self.parcel_add_bt,
                                                    allow_parcel_pu_with_ob_cust=self.allow_parcel_pu_with_ob_cust)
                    if len(res) == 0:
                        LOG.warning(f"no option to deliver parcel {p_rid} with plan {cur_plan} ?")
                        remaining_parcels[p_rid] = 1
                        continue
                    cur_plan = res[0][1]
                self.assign_vehicle_plan(veh_obj, cur_plan, simulation_time)  
                self.vid_to_inserted_parcel_id[vid] = remaining_parcels
                if len(remaining_parcels) == 0:
                    to_del.append(vid)
        for vid in to_del:
            del self.vid_to_inserted_parcel_id[vid]            

INPUT_PARAMETERS_RPPFleetControlSingleStopInsertionGuided = {
    "doc" :     """
    this class is used in the paper
    Integrating Parcel Deliveries into a Ride-Pooling Service - An Agent-Based Simulation Study; Fehn, Engelhardt, Dandl, Bogenberger, Busch (2022)
    parcel deliveries are integrated in a ride-pooling service. No explicit time constraints are assumed for parcel pick-ups and drop-offs.
    Parcels are picked up and dropped off, when vehicles pass by origin or destination during passenger transport
    the "closeness" of vehicle routes to parcels is defined by the parameter G_OP_PA_ASSTH (between 0 and 1, with 1 reflecting that parcel o and d is directly on the vehicle route)
    G_OP_PA_OBASS (bool) describes the integration. if False, an insertion of parcels into vehicle routes is only allowed, when no passenger is on board
    This class represents the SDPA strategy in the paper. origin and destination of parcels are added to vehicle schedules independently from each other. if a vehicle passes by origin, 
    it is inserted into the route to pick up the parcel. The destination is only inserted if the corresponding vehicle passes also the destination some time later during the simulation.
    in comparison to the class RPPFleetControlSingleStopInsertion, the descision to deliver a parcel that is onboard of a vehicle is coupled to assignment process of passengers.
    Thereby, when assigning passengers it is also checked if a route with also delivering the parcels would fit the "closeness" criteria.
    in case not all parcels could be delivered (but are already pick up), remaining parcels are delivered at the time specified by G_OP_PA_REDEL
    this class reflects the CDPA strategy in the paper
    the fleet control requires an immediate descision process in the simulation class
    """,
    "inherit" : "RPPFleetControlSingleStopInsertion",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}                
        
class RPPFleetControlSingleStopInsertionGuided(RPPFleetControlSingleStopInsertion):
    """
    this class is used in the paper
    Integrating Parcel Deliveries into a Ride-Pooling Service - An Agent-Based Simulation Study; Fehn, Engelhardt, Dandl, Bogenberger, Busch (2022)
    parcel deliveries are integrated in a ride-pooling service. No explicit time constraints are assumed for parcel pick-ups and drop-offs.
    Parcels are picked up and dropped off, when vehicles pass by origin or destination during passenger transport
    the "closeness" of vehicle routes to parcels is defined by the parameter G_OP_PA_ASSTH (between 0 and 1, with 1 reflecting that parcel o and d is directly on the vehicle route)
    G_OP_PA_OBASS (bool) describes the integration. if False, an insertion of parcels into vehicle routes is only allowed, when no passenger is on board
    This class represents the SDPA strategy in the paper. origin and destination of parcels are added to vehicle schedules independently from each other. if a vehicle passes by origin, 
    it is inserted into the route to pick up the parcel. The destination is only inserted if the corresponding vehicle passes also the destination some time later during the simulation.
    in comparison to the class RPPFleetControlSingleStopInsertion, the descision to deliver a parcel that is onboard of a vehicle is coupled to assignment process of passengers.
    Thereby, when assigning passengers it is also checked if a route with also delivering the parcels would fit the "closeness" criteria.
    in case not all parcels could be delivered (but are already pick up), remaining parcels are delivered at the time specified by G_OP_PA_REDEL
    this class reflects the CDPA strategy in the paper
    the fleet control requires an immediate descision process in the simulation class
    """
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=None, list_pub_charging_infra=[]):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=op_charge_depot_infra, list_pub_charging_infra=list_pub_charging_infra)
        
    def _pre_test_insertion(self, parcel_request : PlanRequest, vid, o_flag, vehplan = None):
        min_dd = float("inf")
        s_pos = self.sim_vehicles[vid].pos
        if o_flag:
            p = parcel_request.get_o_stop_info()[0]
        else:
            p = parcel_request.get_d_stop_info()[0]
        if vehplan is None:
            vehplan = self.veh_plans[vid]
        LOG.debug(f"test {parcel_request} {o_flag}")
        LOG.debug(f" -> for {[str(x) for x in vehplan.list_plan_stops]}")
        for ps in vehplan.list_plan_stops:
            c_pos = ps.get_pos()
            dd = self.routing_engine.return_travel_costs_1to1(s_pos, p)[2] + self.routing_engine.return_travel_costs_1to1(p, c_pos)[2] - self.routing_engine.return_travel_costs_1to1(s_pos, c_pos)[2]
            LOG.debug(f"{ps} -> {dd}")
            if dd < min_dd:
                min_dd = dd
            s_pos = c_pos
        dd = self.routing_engine.return_travel_costs_1to1(s_pos, p)[2]
        if dd < min_dd:
            min_dd = dd
        additional_parcel_distance = min_dd
        LOG.debug(f"min dd: {min_dd} | pdd {parcel_request.init_direct_td}")
        rel_saved = (parcel_request.init_direct_td/2.0 - additional_parcel_distance)/parcel_request.init_direct_td*2.0
        LOG.debug(f"test {parcel_request.get_rid_struct()} oflag {o_flag} -> vid {vid}: rel_saved {rel_saved}")
        if rel_saved > self.parcel_assignment_threshold:
            LOG.debug(f"-> True")
            return True
        else:
            LOG.debug(f" -> False")
            return False
        
    def _person_request(self, person_request : RequestBase, sim_time : int):
        """This method includes the main functionality for assigning the parcel delivery for this strategy.
        when a new person requests a trip, first possible insertions for the request is checked
        if a solution is found, it is also checked if on-board parcel can be included in the route
        an assignment with parcel is made if the distance of the route with parcel - distance of the best route without parcel < (1-assignment_threshold)/2 direct parcel delivery distance
        """
        t0 = time.perf_counter()
        LOG.debug(f"Incoming request {person_request.__dict__} at time {sim_time}")
        self.sim_time = sim_time
        prq = PlanRequest(person_request, self.routing_engine, min_wait_time=self.min_wait_time,
                          max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt)

        rid_struct = person_request.get_rid_struct()
        self.rq_dict[rid_struct] = prq

        if prq.o_pos == prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(prq, sim_time)
            return

        o_pos, t_pu_earliest, t_pu_latest = prq.get_o_stop_info()
        if t_pu_earliest - sim_time > self.opt_horizon:
            self.reservation_module.add_reservation_request(prq, sim_time)
            offer = self.reservation_module.return_reservation_offer(prq.get_rid_struct(), sim_time)
            LOG.debug(f"reservation offer for rid {rid_struct} : {offer}")
        else:
            list_tuples = insert_prq_in_selected_veh_list_route_with_parcels(self.sim_vehicles, self.veh_plans, prq, self.vr_ctrl_f,
                                                               self.routing_engine, self.rq_dict, sim_time, self.const_bt, self.add_bt,
                                                               allow_parcel_pu_with_ob_cust=self.allow_parcel_pu_with_ob_cust)
            #list_tuples = insertion_with_heuristics(sim_time, prq, self, force_feasible_assignment=True)
            if len(list_tuples) > 0:
                LOG.debug("start parcel insertion")
                (best_np_vid, best_np_vehplan, best_np_delta_cfv) = min(list_tuples, key=lambda x:x[2])
                np_dis = get_veh_plan_distance(self.sim_vehicles[best_np_vid], best_np_vehplan, self.routing_engine)
                with_parcel_option = None
                for vid, veh_plan, delta_cfv in list_tuples:
                    for p_rid in self.vid_to_inserted_parcel_id.get(vid, []):
                        LOG.debug(" -> parcel {} in vid {}".format(p_rid, vid))
                        parcel_prq = self.rq_dict[p_rid]
                        if not self._pre_test_insertion(parcel_prq, vid, False, vehplan=veh_plan):
                            continue
                        LOG.debug(f"try inserting d of {p_rid} -> {vid}")
                        res = insert_parcel_d_in_selected_veh_list_route_with_parcels([self.sim_vehicles[vid]], {vid : veh_plan}, parcel_prq, self.vr_ctrl_f,
                                                        self.routing_engine, self.rq_dict, sim_time, self.parcel_const_bt, self.parcel_add_bt,
                                                        allow_parcel_pu_with_ob_cust=self.allow_parcel_pu_with_ob_cust)
                        if len(res) == 0:
                            continue
                        veh_plan_with_parcel = res[0][1]
                        with_parcel_dis = get_veh_plan_distance(self.sim_vehicles[vid], veh_plan_with_parcel, self.routing_engine)
                        additional_parcel_distance = with_parcel_dis - np_dis
                        rel_saved = (parcel_prq.init_direct_td/2.0 - additional_parcel_distance)/parcel_prq.init_direct_td*2.0
                        LOG.debug(f"test {parcel_prq.get_rid_struct()} -> vid {vid}: rel_saved {rel_saved}")
                        if rel_saved > self.parcel_assignment_threshold:
                            LOG.debug(" -> option found")
                            if with_parcel_option is None:
                                with_parcel_option = (vid, veh_plan_with_parcel, p_rid, additional_parcel_distance)
                            elif with_parcel_option[3] > additional_parcel_distance:
                                with_parcel_option = (vid, veh_plan_with_parcel, p_rid, additional_parcel_distance)
                if with_parcel_option is not None:
                    vid, veh_plan_with_parcel, p_rid, rel_saved = with_parcel_option
                    vehplan = veh_plan_with_parcel  
                    del self.vid_to_inserted_parcel_id[vid][p_rid]   
                    LOG.debug(" -> assign veh plan with parcel {}: {}!".format(p_rid, veh_plan_with_parcel)) 
                else:
                    vehplan = best_np_vehplan                 
                self.tmp_assignment[rid_struct] = vehplan
                offer = self._create_user_offer(prq, sim_time, vehplan)
                LOG.debug(f"new offer for rid {rid_struct} : {offer}")
            else:
                LOG.debug(f"rejection for rid {rid_struct}")
                self._create_rejection(prq, sim_time)
        # record cpu time
        dt = round(time.perf_counter() - t0, 5)
        old_dt = self._get_current_dynamic_fleetcontrol_value(sim_time, G_FCTRL_CT_RQU)
        if old_dt is None:
            new_dt = dt
        else:
            new_dt = old_dt + dt
        output_dict = {G_FCTRL_CT_RQU: new_dt}
        self._add_to_dynamic_fleetcontrol_output(sim_time, output_dict)