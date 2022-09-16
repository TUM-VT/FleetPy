import logging
import time

from src.simulation.Offers import TravellerOffer
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.pooling.objectives import return_pooling_objective_function
from src.fleetctrl.pooling.immediate.insertion import insertion_with_heuristics
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000

INPUT_PARAMETERS_PoolingInsertionHeuristicOnly = {
    "doc" : "this class represents a ride-pooling MoD-operator. the operators uses an insertion heuristic for assignment",
    "inherit" : "FleetControlBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class PoolingInsertionHeuristicOnly(FleetControlBase):
    """This class applies an Insertion Heuristic, in which new requests are inserted in the currently assigned
    vehicle plans and the insertion with the best control objective value is selected.

    IMPORTANT NOTE:
    Both the new and the previously assigned plan are stored and await an instant response of the request. Therefore,
    this fleet control module is only consistent for the ImmediateOfferSimulation class.
    """
    # TODO # clarify dependency to fleet simulation module
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                 dir_names, op_charge_depot_infra=None, list_pub_charging_infra= []):
        """The specific attributes for the fleet control module are initialized. Strategy specific attributes are
        introduced in the children classes.

        :param op_id: operator id
        :type op_id: int
        :param operator_attributes: dictionary with keys from globals and respective values
        :type operator_attributes: dict
        :param list_vehicles: simulation vehicles; their assigned plans should be instances of the VehicleRouteLeg class
        :type list_vehicles: list
        :param routing_engine: routing engine
        :type routing_engine: Network
        :param scenario_parameters: access to all scenario parameters (if necessary)
        :type scenario_parameters: dict
        :param dirnames: directories for output and input
        :type dirnames: dict
        :param op_charge_depot_infra: reference to a OperatorChargingAndDepotInfrastructure class (optional) (unique for each operator)
        :type OperatorChargingAndDepotInfrastructure: OperatorChargingAndDepotInfrastructure
        :param list_pub_charging_infra: list of PublicChargingInfrastructureOperator classes (optional) (accesible for all agents)
        :type list_pub_charging_infra: list of PublicChargingInfrastructureOperator
        """
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names=dir_names, op_charge_depot_infra=op_charge_depot_infra, list_pub_charging_infra=list_pub_charging_infra)
        # TODO # make standard in FleetControlBase
        self.rid_to_assigned_vid = {} # rid -> vid
        self.pos_veh_dict = {}  # pos -> list_veh
        self.vr_ctrl_f = return_pooling_objective_function(operator_attributes[G_OP_VR_CTRL_F])
        self.sim_time = scenario_parameters[G_SIM_START_TIME]
        # others # TODO # standardize IRS assignment memory?
        self.tmp_assignment = {}  # rid -> VehiclePlan
        self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_RQU)

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
        super().receive_status_update(vid, simulation_time, list_finished_VRL, force_update=force_update)
        veh_obj = self.sim_vehicles[vid]
        try:
            self.pos_veh_dict[veh_obj.pos].append(veh_obj)
        except KeyError:
            self.pos_veh_dict[veh_obj.pos] = [veh_obj]
        LOG.debug(f"veh {veh_obj} | after status update: {self.veh_plans[vid]}")

    def user_request(self, rq, sim_time):
        """This method is triggered for a new incoming request. It generally adds the rq to the database. It has to
        return an offer to the user. This operator class only works with immediate responses and therefore either
        sends an offer or a rejection.

        :param rq: request object containing all request information
        :type rq: RequestDesign
        :param sim_time: current simulation time
        :type sim_time: float
        :return: offer
        :rtype: TravellerOffer
        """
        t0 = time.perf_counter()
        LOG.debug(f"Incoming request {rq.__dict__} at time {sim_time}")
        self.sim_time = sim_time
        prq = PlanRequest(rq, self.routing_engine, min_wait_time=self.min_wait_time,
                          max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt)

        rid_struct = rq.get_rid_struct()
        self.rq_dict[rid_struct] = prq

        if prq.o_pos == prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(prq, sim_time)
            return

        o_pos, t_pu_earliest, t_pu_latest = prq.get_o_stop_info()
        if t_pu_earliest - sim_time > self.opt_horizon:
            self.reservation_module.add_reservation_request(prq, sim_time)
            offer = self.reservation_module.return_immediate_reservation_offer(prq.get_rid_struct(), sim_time)
            LOG.debug(f"reservation offer for rid {rid_struct} : {offer}")
        else:
            list_tuples = insertion_with_heuristics(sim_time, prq, self, force_feasible_assignment=True)
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
        LOG.debug(f"activate {rid} for global optimisation at time {sim_time}!")
        self.rq_dict[rid].set_reservation_flag(False)

    def _call_time_trigger_request_batch(self, simulation_time):
        """This method can be used to perform time-triggered proccesses, e.g. the optimization of the current
        assignments of simulation vehicles of the fleet.

        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        self.pos_veh_dict = {}  # pos -> list_veh

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

    def _create_user_offer(self, prq, simulation_time, assigned_vehicle_plan=None, offer_dict_without_plan={}):
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

    def assign_vehicle_plan(self, veh_obj, vehicle_plan, sim_time, force_assign=False, assigned_charging_task=None, add_arg=None):
        super().assign_vehicle_plan(veh_obj, vehicle_plan, sim_time, force_assign=force_assign, assigned_charging_task=assigned_charging_task, add_arg=add_arg)

    def lock_current_vehicle_plan(self, vid):
        super().lock_current_vehicle_plan(vid)

    def _lock_vid_rid_pickup(self, sim_time, vid, rid):
        super()._lock_vid_rid_pickup(sim_time, vid, rid)

