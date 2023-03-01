import logging
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.simulation.Offers import TravellerOffer
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000

INPUT_PARAMETERS_RidePoolingBatchAssignmentFleetcontrol = {
    "doc" : """Batch assignment fleet control (i.e. BMW study, ITSC paper 2019)
        ride pooling optimisation is called after every optimisation_time_step and offers are created in the time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldnt be assigned in the first try, it will be considered again in the next opt-step with this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time with an interval of the size of this parameter""",
    "inherit" : "RidePoolingBatchOptimizationFleetControlBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        G_OP_MAX_WT_2, G_OP_OFF_TW
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class RidePoolingBatchAssignmentFleetcontrol(RidePoolingBatchOptimizationFleetControlBase):
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                 dir_names, op_charge_depot_infra=None, list_pub_charging_infra= []):
        """Batch assignment fleet control (i.e. BMW study, ITSC paper 2019)
        ride pooling optimisation is called after every optimisation_time_step and offers are created in the time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldnt be assigned in the first try, it will be considered again in the next opt-step with this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time with an interval of the size of this parameter

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
        self.max_wait_time_2 = operator_attributes.get(G_OP_MAX_WT_2, None)
        # if np.isnan(self.max_wait_time_2):
        #     self.max_wait_time_2 = None
        self.offer_pickup_time_interval = operator_attributes.get(G_OP_OFF_TW, None)
        # if np.isnan(self.offer_pickup_time_interval):
        #     self.offer_pickup_time_interval = None
        self.unassigned_requests_1 = {}
        self.unassigned_requests_2 = {}

    def user_request(self, rq, sim_time):
        super().user_request(rq, sim_time)
        if not self.rq_dict[rq.get_rid_struct()].get_reservation_flag():
            self.unassigned_requests_1[rq.get_rid_struct()] = 1
        return {}

    def user_cancels_request(self, rid, simulation_time):
        """This method is used to confirm a customer cancellation.
        in the first step the assigned vehicle plan for the rid is updated (rid is removed from plan/v2rb)
        in the second step higher level data base processes are triggered to delete request

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        assigned_vid = self.rid_to_assigned_vid.get(rid, None)
        if assigned_vid is not None:
            veh_obj = self.sim_vehicles[assigned_vid]
            assigned_plan = self.RPBO_Module.get_current_assignment(assigned_vid)
            new_best_plan = self.RPBO_Module.get_vehicle_plan_without_rid(veh_obj, assigned_plan, rid, simulation_time)
            if new_best_plan is not None:
                self.assign_vehicle_plan(veh_obj, new_best_plan, simulation_time, force_assign=True)
            else:
                assigned_plan = VehiclePlan(veh_obj, self.sim_time, self.routing_engine, [])
                self.assign_vehicle_plan(veh_obj, assigned_plan, simulation_time, force_assign=True)
        super().user_cancels_request(rid, simulation_time)

    def user_confirms_booking(self, rid, simulation_time):
        """This method is used to confirm a customer booking.
        in a first step the pick-up time constraints are updated based on the offer mad, if "user_offer_time_window" is given
        in the second step higher level database processes are triggered to fix request

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        pu_offer_tuple = self._get_offered_time_interval(rid)
        if pu_offer_tuple is not None:
            new_earliest_pu, new_latest_pu = pu_offer_tuple
            self.change_prq_time_constraints(simulation_time, rid, new_latest_pu, new_ept=new_earliest_pu)

        super().user_confirms_booking(rid, simulation_time)

    def _call_time_trigger_request_batch(self, simulation_time):
        """ this function first triggers the upper level batch optimisation
        based on the optimisation solution offers to newly assigned requests are created in the second step with following logic:
        declined requests will recieve an empty dict
        unassigned requests with a new assignment try in the next opt-step dont get an answer
        new assigned request will recieve a non empty offer-dict

        a retry is only made, if "user_max_wait_time_2" is given

        every request as to answer to an (even empty) offer to be deleted from the system!

        :param simulation_time: current time in simulation
        :return: dictionary rid -> offer for each unassigned request, that will recieve an answer. (offer: dictionary with plan specific entries; empty if no offer can be made)
        :rtype: dict
        """
        super()._call_time_trigger_request_batch(simulation_time)
        # embed()
        rid_to_offers = {}
        if self.sim_time % self.optimisation_time_step == 0:
            new_unassigned_requests_2 = {}
            # rids to be assigned in first try
            for rid in self.unassigned_requests_1.keys():
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                prq = self.rq_dict[rid]
                if assigned_vid is None:
                    if self.max_wait_time_2 is not None and self.max_wait_time_2 > 0:    # retry with new waiting time constraint (no offer returned)
                        new_unassigned_requests_2[rid] = 1
                        self.RPBO_Module.delRequest(rid)
                        _, earliest_pu, _ = prq.get_o_stop_info()
                        new_latest_pu = earliest_pu + self.max_wait_time_2
                        self.change_prq_time_constraints(simulation_time, rid, new_latest_pu)
                        self.RPBO_Module.add_new_request(rid, prq)
                    else:   # no retry, rid declined
                        self._create_user_offer(prq, simulation_time)
                else:
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            for rid in self.unassigned_requests_2.keys():   # check second try rids
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                if assigned_vid is None:    # decline
                    self._create_user_offer(self.rq_dict[rid], simulation_time)
                else:
                    prq = self.rq_dict[rid]
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            self.unassigned_requests_1 = {}
            self.unassigned_requests_2 = new_unassigned_requests_2  # retry rids
            LOG.debug("end of opt:")
            LOG.debug("unassigned_requests_2 {}".format(self.unassigned_requests_2))
            LOG.debug("offers: {}".format(rid_to_offers))

    def vehicle_information_needed_for_optimisation(self, sim_time):
        """ needed for coupling with aimsun: tests if new vehicle information should be fetched
        better way to to that?
        """
        self.sim_time = sim_time
        if sim_time % self.optimisation_time_step == 0:
            return True
        else:
            return False

    def _create_user_offer(self, rq, simulation_time, assigned_vehicle_plan = None, offer_dict_without_plan={}):
        if assigned_vehicle_plan is not None:
            pu_time, do_time = assigned_vehicle_plan.pax_info.get(rq.get_rid_struct())
            add_offer = {}
            pu_offer_tuple = self._get_offered_time_interval(rq.get_rid_struct())
            if pu_offer_tuple is not None:
                new_earliest_pu, new_latest_pu = pu_offer_tuple
                add_offer[G_OFFER_PU_INT_START] = new_earliest_pu
                add_offer[G_OFFER_PU_INT_END] = new_latest_pu
            offer = TravellerOffer(rq.get_rid(), self.op_id, pu_time - rq.get_rq_time(), do_time - pu_time, int(rq.init_direct_td * self.dist_fare + self.base_fare),
                    additional_parameters=add_offer)
            rq.set_service_offered(offer)
        else:
            offer = self._create_rejection(rq, simulation_time)
        return offer

    def _get_offered_time_interval(self, rid):
        """ this function creates an offer for the pickup-intervall for a request in case the parameter G_OP_OFF_TW is given depending on the planned pick-up time of the assigned tour
        :param rid: request id
        :return: None, if G_OP_OFF_TW is not given, tuple (new_earlest_pickup_time, new_latest_pick_up_time) for new pickup time constraints
        """
        if self.offer_pickup_time_interval is not None: # set new pickup time constraints based on expected pu-time and offer time interval
            prq = self.rq_dict[rid]
            _, earliest_pu, latest_pu = prq.get_o_stop_info()
            vid = self.rid_to_assigned_vid[rid]
            assigned_plan = self.veh_plans[vid]
            pu_time, _ = assigned_plan.pax_info.get(rid)
            if pu_time - self.offer_pickup_time_interval/2.0 < earliest_pu:
                new_earliest_pu = earliest_pu
                new_latest_pu = earliest_pu + self.offer_pickup_time_interval
            elif pu_time + self.offer_pickup_time_interval/2.0 > latest_pu:
                new_latest_pu = latest_pu
                new_earliest_pu = latest_pu - self.offer_pickup_time_interval
            else:
                new_earliest_pu = pu_time - self.offer_pickup_time_interval/2.0
                new_latest_pu = pu_time + self.offer_pickup_time_interval/2.0
            return new_earliest_pu, new_latest_pu
        else:
            return None