import logging
import time

from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.fleetctrl.pooling.immediate.insertion import insertion_with_heuristics

from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000

INPUT_PARAMETERS_PoolingIRSAssignmentBatchOptimization = {
    "doc" : """Pooling class that combines an immediate response and batch optimization:
        - requests enter system continuously
        - offer has to be created immediately by an insertion heuristic
        - request replies immediately
            -> there can never be 2 requests at the same time waiting for an offer!
        - re-optimization of solution after certain time interval""",
    "inherit" : "RidePoolingBatchOptimizationFleetControlBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class PoolingIRSAssignmentBatchOptimization(RidePoolingBatchOptimizationFleetControlBase):
    """Pooling class that combines an immediate response and batch optimization:
        - requests enter system continuously
        - offer has to be created immediately by an insertion heuristic
        - request replies immediately
            -> there can never be 2 requests at the same time waiting for an offer!
        - re-optimization of solution after certain time interval

    Formerly known as MOIAfleetcontrol_TRB, which was the version for TRB paper 2021:
    'Self-Regulating Demand and Supply Equilibrium in Joint Simulation of Travel Demand and a Ride-Pooling Service'
    """

    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                 dir_names, op_charge_depot_infra=None, list_pub_charging_infra= []):
        """Initialization of FleetControlClass

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
        self.tmp_assignment = {}  # rid -> possible veh_plan

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
        # TODO # think about way to call super().user_request() again! -> add_new_request should not be called twice
        # check if request is already in database (do nothing in this case)
        if self.rq_dict.get(rq.get_rid_struct()):
            return
        t0 = time.perf_counter()
        self.sim_time = sim_time

        LOG.debug(f"Incoming request {rq.__dict__} at time {sim_time}")
        self.sim_time = sim_time
        prq = PlanRequest(rq, self.routing_engine, min_wait_time=self.min_wait_time,
                          max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt)
        rid_struct = rq.get_rid_struct()

        if prq.o_pos == prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(prq, sim_time)
            return

        self.new_requests[rid_struct] = 1
        self.rq_dict[rid_struct] = prq

        o_pos, t_pu_earliest, t_pu_latest = prq.get_o_stop_info()
        if t_pu_earliest - sim_time > self.opt_horizon:
            self.reservation_module.add_reservation_request(prq, sim_time)
            offer = self.reservation_module.return_immediate_reservation_offer(prq.get_rid_struct(), sim_time)
            LOG.debug(f"reservation offer for rid {rid_struct} : {offer}")
            prq.set_reservation_flag(True)
            self.RPBO_Module.add_new_request(rid_struct, prq, consider_for_global_optimisation=False)
        else:
            self.RPBO_Module.add_new_request(rid_struct, prq)
            list_tuples = insertion_with_heuristics(sim_time, prq, self, force_feasible_assignment=True)
            if len(list_tuples) > 0:
                (vid, vehplan, delta_cfv) = min(list_tuples, key=lambda x:x[2])
                LOG.debug(f"before insertion: {vid} | {self.veh_plans[vid]}")
                LOG.debug(f"after insertion: {vid} | {vehplan}")
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
        self.sim_time = simulation_time
        LOG.debug(f"user confirms booking {rid} at {simulation_time}")
        super().user_confirms_booking(rid, simulation_time)
        prq = self.rq_dict[rid]
        if prq.get_reservation_flag():
            self.reservation_module.user_confirms_booking(rid, simulation_time)
        else:
            assigned_plan = self.tmp_assignment[rid]
            vid = assigned_plan.vid
            veh_obj = self.sim_vehicles[vid]
            self.assign_vehicle_plan(veh_obj, assigned_plan, simulation_time)
            del self.tmp_assignment[rid]

    def user_cancels_request(self, rid, simulation_time):
        """This method is used to confirm a customer cancellation. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        LOG.debug(f"user cancels request {rid} at {simulation_time}")
        prq = self.rq_dict[rid]
        if prq.get_reservation_flag():
            self.reservation_module.user_cancels_request(rid, simulation_time)
        else:
            prev_assignment = self.tmp_assignment.get(rid)
            if prev_assignment:
                del self.tmp_assignment[rid]
        super().user_cancels_request(rid, simulation_time)
