from __future__ import annotations

import logging
import os
from src.fleetctrl.pooling.GeneralPoolingFunctions import get_assigned_rids_from_vehplan
from src.fleetctrl.planning.PlanRequest import PlanRequest
import numpy as np
import pandas as pd
import time
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.fleetctrl.PoolingIRSBatchOptimization import PoolingIRSAssignmentBatchOptimization
from src.fleetctrl.pooling.immediate.insertion import single_insertion, insertion_with_heuristics
from src.misc.globals import *
from src.demand.TravelerModels import BasicRequest
from src.simulation.Offers import TravellerOffer

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Callable

if TYPE_CHECKING:
    from src.fleetctrl.planning.VehiclePlan import VehiclePlan
    from src.simulation.Vehicles import SimulationVehicle
    from src.routing.NetworkBase import NetworkBase
    from src.infra.Zoning import ZoneSystem

LOG = logging.getLogger(__name__)
LARGE_INT = 100000


def get_in_vrl_control_included_assignment_reward_func(vr_control_func_dict : dict) -> Callable:
    """ returns a function that calculates just the assignment-reward for a given ride pooling objective function
    :param vr_control_func_dict: dictionary which has to contain "func_key" as switch between possible functions;
            additional parameters of a function can have additional keys.
    :type vr_control_func_dict: dict
    :return: assignment reward function for assigning a single given request
    :rtype: function
    """
    func_key = vr_control_func_dict["func_key"]
    if func_key == "total_distance" or func_key == "total_system_time" or func_key == "distance_and_user_times" or \
       func_key == "distance_and_user_times_with_walk" or func_key == "user_times" or func_key == "distance_and_user_times_man":
        from src.fleetctrl.pooling.objectives import LARGE_INT as const_per_rq_award

        def assignment_reward_for_rq_func(prq, veh_obj, assigned_plan):
            """ for these objective functions the assignment reward is just given by a large constant per assigned request
            :param prq: plan request in question
            :param veh_obj: veh_obj of assigned plan
            :param assigned_plan: assigned vehicle plan
            :return: share of the assignment reward of assigning request rq on the overall objective function
            """
            return const_per_rq_award

        return assignment_reward_for_rq_func
    else:
        LOG.error("THIS OBJECTIVE FUNCTION IS NOT REGISTERED IN EasyRideBrokerFleetControl.py")
        raise NotImplementedError

# --------------------------------------------------------------------------------------------------------------------------------------------------

INPUT_PARAMETERS_BrokerDecisionCtrl = {
    "doc" : """this fleetcontrol is used for simulations with a central broker which assigns users to operators in the publication
    Competition and Cooperation of Autonomous Ridepooling Services: Game-Based Simulation of a Broker Concept; Engelhardt, Malcolm, Dandl, Bogenberger (2022)
    therefore, in this fleetcontrol the attribute "add fleet vmt" (the addtional driving distance to serve a customer calculated after
    the insertion heuristic) is added to the offer parameters which is used by the broker as a decision variable""",
    "inherit" : "PoolingIRSAssignmentBatchOptimization",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class BrokerDecisionCtrl(PoolingIRSAssignmentBatchOptimization):
    """ this fleetcontrol is used for simulations with a central broker which assigns users to operators in the publication
    Competition and Cooperation of Autonomous Ridepooling Services: Game-Based Simulation of a Broker Concept; Engelhardt, Malcolm, Dandl, Bogenberger (2022)
    therefore, in this fleetcontrol the attribute "add fleet vmt" (the addtional driving distance to serve a customer calculated after
    the insertion heuristic) is added to the offer parameters which is used by the broker as a decision variable"""
    
    def user_request(self, rq : PlanRequest, sim_time : int):
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
        # check if request is already in database (do nothing in this case)
        if self.rq_dict.get(rq.get_rid_struct()):
            return
        t0 = time.perf_counter()
        self.sim_time = sim_time
        prq = PlanRequest(rq, self.routing_engine, min_wait_time=self.min_wait_time, max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt)

        rid_struct = rq.get_rid_struct()
        self.rq_dict[rid_struct] = prq
        self.RPBO_Module.add_new_request(rid_struct, prq)
        self.new_requests[rid_struct] = 1

        if prq.o_pos == prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(prq, sim_time)
            return

        o_pos, t_pu_earliest, t_pu_latest = prq.get_o_stop_info()
        if t_pu_earliest - sim_time > self.opt_horizon:
            prq.set_reservation_flag(True)

        list_tuples = insertion_with_heuristics(sim_time, prq, self, force_feasible_assignment=True)
        if len(list_tuples) > 0:
            (vid, vehplan, delta_cfv) = min(list_tuples, key=lambda x:x[2])
            self.tmp_assignment[rid_struct] = vehplan
            prev_plan = self.veh_plans[vid]
            add_km = self._get_change_in_driven_distance(vid, vehplan, prev_plan)
            offer = self._create_user_offer(prq, sim_time, vehplan, offer_dict_without_plan={G_OFFER_ADD_VMT : add_km})
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
        
    def _create_user_offer(self, prq : PlanRequest, simulation_time : int, assigned_vehicle_plan : VehiclePlan, offer_dict_without_plan : dict) -> TravellerOffer:
        """ this method adds the parameter "add_fleet_vmt" to the user offer
        :param prq: PlanRequest obj to send offer to
        :param simulation_time: current simulation time
        :param assigned_vehicle_plan: assigned vehicle plan which serves this request
        :param offer_dict_without_plan: dict of additional offer parameters -> "add_fleet_vmt" is one entry
        :return: corresponding TravellerOffer
        """
        # additional driven kilometers: offer_dict_without_plan["add_fleet_vmt"]
        if assigned_vehicle_plan is not None: # create offer attributes
            pu_time, do_time = assigned_vehicle_plan.pax_info.get(prq.get_rid_struct())
            offer = TravellerOffer(prq.get_rid_struct(), self.op_id, pu_time - prq.rq_time, do_time - pu_time, int(prq.init_direct_td * self.dist_fare + self.base_fare),
                additional_parameters={G_OFFER_ADD_VMT : offer_dict_without_plan[G_OFFER_ADD_VMT]})
            prq.set_service_offered(offer)  # has to be called
        else: # rejection
            offer = TravellerOffer(prq.get_rid(), self.op_id, None, None, None)
        return offer

    def _get_change_in_driven_distance(self, vid : int, new_plan : VehiclePlan, prev_plan : VehiclePlan) -> float:
        """ this function returns the difference between the driven distances of new_plan and prev_plan
        dd(new_plan) - dd(prev_plan)
        :param vid: vehicle id
        :param new_plan: new vehicle plan object
        :param prev_plan: previous vehicle plan object
        """
        sum_dist_new = 0
        last_pos = self.sim_vehicles[vid].pos
        for ps in new_plan.list_plan_stops:
            if ps.get_pos() != last_pos:
                sum_dist_new += self.routing_engine.return_travel_costs_1to1(last_pos, ps.get_pos())[2]
                last_pos = ps.get_pos()
        sum_dist_prev = 0
        last_pos = self.sim_vehicles[vid].pos
        for ps in prev_plan.list_plan_stops:
            if ps.get_pos() != last_pos:
                sum_dist_prev += self.routing_engine.return_travel_costs_1to1(last_pos, ps.get_pos())[2]
                last_pos = ps.get_pos()
        return sum_dist_new - sum_dist_prev


# --------------------------------------------------------------------------------------------------------------------------------------------------

INPUT_PARAMETERS_BrokerExChangeCtrl = {
    "doc" : """Fleet control class for EasyRide Broker Exchange of requests scenario
        request enter system continously
            offer has to be created immediatly by an insertion heuristic, where also following decision is made:
            operator dicides if he can and if he wants to serve the customer
            if he wants and can:
                request is accepted and the offer is created
            if he cant:
                requests is declined and send to other fltctrl
            if he can but doesnt want:
                offer is created, but request is also sent to other fleetctrl
            request replies immediatly 
            -> there can never be 2 requests at the same time waiting for an offer! 
        reoptimisation of solution after certain time interval""",
    "inherit" : "RidePoolingBatchOptimizationFleetControlBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        G_MULTIOP_PREF_OP_PROB, G_MULTIOP_EVAL_METHOD, G_MULTIOP_EXCH_AC_OBS_TIME, G_MULTIOP_EXCH_AC_STD_WEIGHT, G_MULTIOP_EVAL_LOOKAHEAD
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class BrokerExChangeCtrl(RidePoolingBatchOptimizationFleetControlBase):
    def __init__(self, op_id : int, operator_attributes : dict, list_vehicles : List[SimulationVehicle], routing_engine : NetworkBase, 
                 zone_system : ZoneSystem, scenario_parameters : dict, dir_names : dict, op_charge_depot_infra=None, list_pub_charging_infra= []):
        """Fleet control class for EasyRide Broker Exchange of requests scenario
        request enter system continously
            offer has to be created immediatly by an insertion heuristic, where also following decision is made:
            operator dicides if he can and if he wants to serve the customer
            if he wants and can:
                request is accepted and the offer is created
            if he cant:
                requests is declined and send to other fltctrl
            if he can but doesnt want:
                offer is created, but request is also sent to other fleetctrl
            request replies immediatly 
            -> there can never be 2 requests at the same time waiting for an offer! 
        reoptimisation of solution after certain time interval

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
        """
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=op_charge_depot_infra, list_pub_charging_infra=list_pub_charging_infra)
        self.assignment_reward_for_rq_func = get_in_vrl_control_included_assignment_reward_func(operator_attributes[G_OP_VR_CTRL_F])
        self.current_vid_plan_tuple = None  # (vid, veh_plan) veh_plan is current solution for offer which will be destroyed or accepted once a request decides (only one can exist simultanously!)
        
        self.requested_rid_to_assignment_cost = {}  # rid -> assignment-cost (in dictionary as long operator of rid is not fixed; in case of assignment the value will be added to self.last_assginment_costs)
        self.last_assignment_costs = []     # (time_of_rid, assignment-cost)    stores the last x-min of assignment-costs that are/will be served by operator

        self.prob_rq_share = 1.0
        self.fc_type = None
        n_op = scenario_parameters[G_NR_OPERATORS]
        self.prob_rq_share = scenario_parameters.get(G_MULTIOP_PREF_OP_PROB, [1/n_op for o in range(n_op)])[self.op_id]
        self.fc_type = scenario_parameters.get(G_FC_TYPE)
        # parameters to evaluate willingness TODO if method useful add to scenario-parameters
        self.evaluate_willing_ness_method = operator_attributes.get(G_MULTIOP_EVAL_METHOD, "")  # "forecast" or "reactive"

        self.assignment_cost_observation_time = operator_attributes.get(G_MULTIOP_EXCH_AC_OBS_TIME)# 15*60   # time in seconds to consider last assignments for decision of willingness
        self.assignment_cost_std_weight = operator_attributes.get(G_MULTIOP_EXCH_AC_STD_WEIGHT) # 1     # defines the lower bound of willingness decision (lower bound for new request = mean(last_assignment_costs) - self.assignment_cost_std_weight * std(last_assignment_costs)) inf will result in always accepting request
        if self.evaluate_willing_ness_method == "reactive" and (self.assignment_cost_observation_time is None or self.assignment_cost_std_weight is None):
            raise EnvironmentError(f"parameters {G_MULTIOP_EXCH_AC_OBS_TIME} or {G_MULTIOP_EXCH_AC_STD_WEIGHT} not given!")

        self.look_ahead_time = operator_attributes.get(G_MULTIOP_EVAL_LOOKAHEAD)    # time to look in future for sampling future requests
        if self.evaluate_willing_ness_method == "forecast" and self.look_ahead_time is None:
            raise EnvironmentError(f"parameter {G_MULTIOP_EVAL_LOOKAHEAD} not given!")
        self.last_future_sample_time = -1 # last time a new future requests have been sampled
        self.last_future_sample = {}    # rid -> f_prqs last of future requests from the last sample (rids for future requests are negative)
        self.last_future_sample_sol = {}    # vid -> vehplan involving future samples

        self.op_broker_output_file = os.path.join(dir_names[G_DIR_OUTPUT], "broker_output_op_{}.csv".format(self.op_id))
        with open(self.op_broker_output_file, "w") as f:
            f.write("sim_time,rq_id,mean_assignment_costs,std_assignment_costs,max_assignment_cost,rq_assignment_cost\n")

    def user_request(self, rq : PlanRequest, sim_time : int):
        """This method is triggered for a new incoming request. It generally adds the rq to the database. It has to
        return an offer to the user. An empty dictionary means no offer is made!

        this method additionally evaluates the willingness of the operator to serve the request and sets the decision in the offer with attribute G_OFFER_WILLING_FLAG to provide the information to the fleet simulation.

        :param rq: request object containing all request information
        :type rq: RequestDesign
        :param sim_time: current simulation time
        :type sim_time: float
        """
        if self.current_vid_plan_tuple is not None:
            LOG.error(f"new user_request before old request is resolved! {self.current_vid_plan_tuple}")
            raise AssertionError
        super().user_request(rq, sim_time)
        rid_struct = rq.get_rid_struct()
        prq = self.rq_dict[rid_struct]
        if prq.o_pos == prq.d_pos:
            LOG.debug("automatic decline!")
            return {}

        LOG.debug("new user request {}".format(rid_struct))
        assigned_vid, assigned_plan, change_in_objective_value = single_insertion(self.sim_vehicles, self.veh_plans, prq, self.vr_ctrl_f, self.routing_engine, self.rq_dict, sim_time, self.const_bt, self.add_bt)
        prev_assigned_plan = self.veh_plans.get(assigned_vid)
        if assigned_vid is not None:    # fleetoperator can serve
            if self.evaluate_willing_ness_method == "reactive":
                is_willing = self._evaluate_willingness_reactive(change_in_objective_value, prev_assigned_plan, assigned_plan, prq,
                                                        self.sim_vehicles[assigned_vid], sim_time)
            elif self.evaluate_willing_ness_method == "forecast":
                is_willing = self._evaluate_willingness_forecast(sim_time, prq, assigned_vid, assigned_plan, change_in_objective_value)
            else:
                is_willing = True
            if is_willing: #force_accept or is_willing:    # fleetoperator wants to or must serve
                offer = self._create_user_offer(prq, sim_time, assigned_vehicle_plan=assigned_plan, offer_dict_without_plan={G_OFFER_WILLING_FLAG : is_willing})
                self.current_vid_plan_tuple = (assigned_vid, assigned_plan)
                LOG.debug(f"new offer for rid {rid_struct} : {offer}")
            else:
                offer = self._create_user_offer(prq, sim_time, assigned_vehicle_plan=assigned_plan, offer_dict_without_plan={G_OFFER_WILLING_FLAG : is_willing})  # flag that he could?
                self.current_vid_plan_tuple = (assigned_vid, assigned_plan)
        else:
            LOG.debug(f"no offer for rid {rid_struct}")
            offer = self._create_user_offer(prq, sim_time, offer_dict_without_plan={G_OFFER_WILLING_FLAG : False})

    def user_confirms_booking(self, rid : Any, simulation_time : int):
        """This method is used to confirm a customer booking. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        super().user_confirms_booking(rid, simulation_time)

        vid, assigned_plan = self.current_vid_plan_tuple
        self.assign_vehicle_plan(self.sim_vehicles[vid], assigned_plan, simulation_time)

        if self.evaluate_willing_ness_method == "reactive":
            self.last_assignment_costs.append( (simulation_time, self.requested_rid_to_assignment_cost[rid]) )
        elif self.evaluate_willing_ness_method == "forecast":
            if self.last_future_sample_sol.get(vid):
                LOG.info("resolve conflict for {} assigned to {}".format(rid, vid))
                insertion_rids = get_assigned_rids_from_vehplan(assigned_plan)
                future_sol_rids = get_assigned_rids_from_vehplan(self.last_future_sample_sol[vid][0])
                conflict_rids = set(future_sol_rids) - set(insertion_rids)
                LOG.info("conflict future rids {}".format(conflict_rids))

                new_sol = self.veh_plans.copy()
                for l_vid, sol_val in self.last_future_sample_sol.items():
                    if l_vid != vid:
                        vehplan, change_in_objective_value = sol_val
                        new_sol[l_vid] = vehplan
                new_sol[vid] = assigned_plan
                try:
                    del self.last_future_sample_sol[vid]
                except:
                    pass

                new_rq_dict = self.rq_dict.copy()
                new_rq_dict.update(self.last_future_sample)
                for f_rid in conflict_rids:
                    f_prq = self.last_future_sample[f_rid]
                    assigned_vid, new_assigned_plan, change_in_objective_value = single_insertion(self.sim_vehicles, new_sol, f_prq, self.vr_ctrl_f, self.routing_engine, new_rq_dict, simulation_time, self.const_bt, self.add_bt)
                    new_sol[assigned_vid] = new_assigned_plan
                    self.last_future_sample_sol[assigned_vid] = new_assigned_plan, change_in_objective_value
                    LOG.info("resolved: {} -> {} | {}".format(f_rid, assigned_vid, change_in_objective_value))

        try:
            del self.requested_rid_to_assignment_cost[rid]
        except KeyError:
            pass

        self.current_vid_plan_tuple = None

    def user_cancels_request(self, rid : Any, simulation_time : int):
        """This method is used to confirm a customer cancellation. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        super().user_cancels_request(rid, simulation_time)
        try:
            del self.requested_rid_to_assignment_cost[rid]
        except KeyError:
            pass
        self.current_vid_plan_tuple = None

    def _create_user_offer(self, rq : PlanRequest, simulation_time : int, assigned_vehicle_plan : VehiclePlan=None, offer_dict_without_plan : dict={}) -> TravellerOffer:
        """ creating the offer for a requests
        willing_flag will be set afterwards

        :param rq: plan request
        :type rq: PlanRequest obj
        :param simulation time: current simulation time
        :type simulation time: int
        :param assigned_vehicle_plan: vehicle plan of initial solution to serve this request
        :type assigned_vehicle_plan: VehiclePlan
        :param offer_dict_without_plan: can be used to create an offer that is not derived from a vehicle plan
                    entries will be used to create/extend offer
        :return: offer
        :rtype: TravellerOffer
        """
        offer = super()._create_user_offer(rq, simulation_time, assigned_vehicle_plan=assigned_vehicle_plan)
        offer.extend_offer(offer_dict_without_plan)
        return offer

    def _evaluate_willingness_reactive(self, change_in_objective_function : float, prev_assigned_plan : VehiclePlan,
                                       new_assigned_plan : VehiclePlan, plan_request : PlanRequest, assigned_vehicle_obj : SimulationVehicle, sim_time : int):
        """ This function should evaluate if the fleet operator is willing to serve a customer based
        on the change in objective_function_value after an insertion solution is found

        this function uses the following logic:
        - the "willingness" of the operator for assigning a new requests is dependent on the objective function - assignment reward to represent some "effective costs"
        - effective-objective(rq) = change in objective function (rq) after insertion - assignment reward
        - the effective-objective of each later assigned request is stored in a list to evaluate statistics of assigned requests in the near past
        - the operator is willing to serve the new rq if effective-objective(rq) < mean(past assigned effective-objectives) - self.assginment_cost_std_dev_weight * std(past_assigned effective-objectives)
        - the "near past" is defined by the parameter self.assignment_cost_observation_time
        - the idea is to select good request meassured by the second objective (i.e. driven distance, low waiting time) on average relative to meassured sample size on what is possible (mean). on the other hand the standard deviation is used to 
            weigth sample sizes and overall scattering

        :param change_in_objective_function: float < 0 change of the global objective function of the best insertion solution found
        :param prev_assigned_plan: assigned vehicle plan without new rq
        :param new_assigned_plan: new assigned vehicle plan with request insertion
        :param plan_request: plan_request obj of request in question
        :param assigned_vehicle_obj: vehicle_obj corresponding to assigned plan
        :param sim_time: current simulation time
        :return: True: operator willing to serve, False: operator not willing to serve
        """
        LOG.debug("evaluate_willingness")
        assignment_reward_of_rq = self.assignment_reward_for_rq_func(plan_request, assigned_vehicle_obj, new_assigned_plan)
        effective_assignment_cost = change_in_objective_function + assignment_reward_of_rq
        self.requested_rid_to_assignment_cost[plan_request.get_rid_struct()] = effective_assignment_cost
        # update last_assignment_costs
        del_until = 0
        for i, x in enumerate(self.last_assignment_costs):
            assignment_time = x[0]
            if assignment_time >= sim_time - self.assignment_cost_observation_time:
                del_until = i
                break
        if del_until > 0:
            self.last_assignment_costs = self.last_assignment_costs[del_until:]
        # decide for willingness
        LOG.debug("operator {} | change in obj {} | assignment reward {} | effect assignment cost {}".format(self.op_id, change_in_objective_function, assignment_reward_of_rq, effective_assignment_cost))
        #"sim_time,rq_id,mean_assignment_costs,std_assignment_costs,max_assignment_cost,rq_assignment_cost\n"
        if len(self.last_assignment_costs) > 1:
            mean_assignment_costs = sum( (x[1] for x in self.last_assignment_costs) )/len(self.last_assignment_costs)
            std_assignment_costs = np.math.sqrt( sum( (x[1] - mean_assignment_costs)**2 for x in self.last_assignment_costs) / (len(self.last_assignment_costs) - 1) )
            LOG.debug("assignment costs of prev accepted requests: mean {} | std {}".format(mean_assignment_costs, std_assignment_costs))
            max_assignment_costs = mean_assignment_costs + self.assignment_cost_std_weight * std_assignment_costs
            with open(self.op_broker_output_file, "a") as f:
                f.write("{},{},{},{},{},{}\n".format(sim_time, plan_request.get_rid_struct(), mean_assignment_costs, std_assignment_costs, max_assignment_costs, effective_assignment_cost))
            if effective_assignment_cost < max_assignment_costs:
                LOG.debug(" -> sounds good")
                return True
            else:
                LOG.debug(" -> dont like this request")
                return False
        else:
            with open(self.op_broker_output_file, "a") as f:
                f.write("{},{},{},{},{},{}\n".format(sim_time, plan_request.get_rid_struct(), -1, -1, -1, effective_assignment_cost))

        return True

    def _get_new_future_requests(self, sim_time : int) -> Tuple[Dict[Any, PlanRequest], Dict[int, VehiclePlan]]:
        """ this function is used to sample new future requests and insert them into the current solution
        :param sim_time: current simulation time
        :return: request_id -> PlanRequest from future sample, vid -> Vehiclplan with inserted plan requests from future sample"""
        if sim_time != self.last_future_sample_time:
            if self.fc_type == "perfect":
                future_rq_atts = self.zones.draw_future_request_sample(sim_time, sim_time + self.look_ahead_time, request_attribute = "preferred_operator", attribute_value = self.op_id)
            else:
                future_rq_atts = self.zones.draw_future_request_sample(sim_time, sim_time + self.look_ahead_time, scale = self.prob_rq_share)
            self.last_future_sample = {}
            c_rid = -1
            for t, o_node, d_node in future_rq_atts:
                f_rq_row = pd.Series({G_RQ_TIME : t, G_RQ_ORIGIN : o_node, G_RQ_DESTINATION : d_node, G_RQ_ID : c_rid})
                f_rq = BasicRequest(f_rq_row, self.routing_engine, 1)   # TODO # sim_time_step ??
                f_prq = PlanRequest(f_rq, self.routing_engine, min_wait_time = self.min_wait_time, max_wait_time = self.max_wait_time, max_detour_time_factor = self.max_dtf, 
                                    max_constant_detour_time = self.max_cdt, add_constant_detour_time = self.add_cdt, min_detour_time_window=self.min_dtw, boarding_time = self.const_bt)
                self.last_future_sample[f_prq.get_rid_struct()] = f_prq
                c_rid -= 1
            self.last_future_sample_time = sim_time

            self.last_future_sample_sol = {}
            new_sol = self.veh_plans.copy()
            new_rq_dict = self.rq_dict.copy()
            new_rq_dict.update(self.last_future_sample)
            for f_rid, f_prq in self.last_future_sample.items():
                assigned_vid, assigned_plan, change_in_objective_value = single_insertion(self.sim_vehicles, new_sol, f_prq, self.vr_ctrl_f, self.routing_engine, new_rq_dict, sim_time, self.const_bt, self.add_bt)
                self.last_future_sample_sol[assigned_vid] = (assigned_plan, change_in_objective_value)
                new_sol[assigned_vid] = assigned_plan

        return self.last_future_sample, self.last_future_sample_sol

    def _evaluate_willingness_forecast(self, sim_time : int, prq : PlanRequest, insertion_vid : int, insertion_plan : VehiclePlan,
                                       insertion_cfv_change : float) -> bool:
        """ this method evaluates the willingness of the operator to serve a request based on the forecast of future requests
        the current solution of insertion of future sampled requests is therefore compared with the solution of the new request with additionally
        inserting the future sampled requests afterwards. If an decrease in overall objective value is observed, the customer is accepted
        :param sim_time: current simulation time
        :param prq: new PlanRequest
        :param insertion_vid: the vehicle id the planrequest has been inserted in the offer creation
        :param insertion_plan: the vehicle plan the planrequest has been insertion into in the offer createion
        :param insertion_cfv_change: the change in cost function value after inserting the request in the offer creation
        :return: True, if operator is willing to serve the request"""
        future_sample, future_sample_sol = self._get_new_future_requests(sim_time)
        LOG.info("future sample size: {}".format(len(future_sample)))
        # algorithms need to have access to all requests
        new_rq_dict = self.rq_dict.copy()
        new_rq_dict.update(future_sample)

        # solution quality without rq
        delta_cfv_without = 0
        for _, change_in_objective_value in future_sample_sol.values():
            delta_cfv_without += change_in_objective_value

        # solution quality with rq
        delta_cfv_with = insertion_cfv_change
        if future_sample_sol.get(insertion_vid):
            insertion_rids = get_assigned_rids_from_vehplan(insertion_plan)
            future_sol_rids = get_assigned_rids_from_vehplan(future_sample_sol[insertion_vid][0])
            conflict_rids = set(future_sol_rids) - set(insertion_rids)
            LOG.info("conflict! : conflict rids {}".format(conflict_rids))

            new_sol = self.veh_plans.copy()
            for vid, sol_val in future_sample_sol.items():
                if vid != insertion_vid:
                    vehplan, change_in_objective_value = sol_val
                    new_sol[vid] = vehplan
                    delta_cfv_with += change_in_objective_value
            new_sol[insertion_vid] = insertion_plan

            for f_rid in conflict_rids:
                f_prq = future_sample[f_rid]
                assigned_vid, assigned_plan, change_in_objective_value = single_insertion(self.sim_vehicles, new_sol, f_prq, self.vr_ctrl_f, self.routing_engine, new_rq_dict, sim_time, self.const_bt, self.add_bt)
                delta_cfv_with += change_in_objective_value
                new_sol[assigned_vid] = assigned_plan
        else:
            delta_cfv_with += delta_cfv_without

        LOG.info("quality with rq: {} | without: {}".format(delta_cfv_with, delta_cfv_without))
        if delta_cfv_with <= delta_cfv_without:
            return True
        else:
            return False

# --------------------------------------------------------------------------------------------------------------------------------------------------

INPUT_PARAMETERS_BrokerBaseCtrl = {
    "doc" : """Fleet control class for the base scenario of the easyride broker
        it mimics the EasyRide Broker Exchange of requests scenario but willingness is not used the evaluate the operator decisions
        request enter system continously
            offer has to be created immediatly by an insertion heuristic, where also following decision is made:
            operator dicides if he can and if he wants to serve the customer
            if he wants and can:
                request is accepted and the offer is created
            if he cant:
                requests is declined and send to other fltctrl
            if he can but doesnt want:
                offer is created, but request is also sent to other fleetctrl
            request replies immediatly 
            -> there can never be 2 requests at the same time waiting for an offer! 
        reoptimisation of solution after certain time interval""",
    "inherit" : "BrokerExChangeCtrl",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class BrokerBaseCtrl(BrokerExChangeCtrl):
    def __init__(self, op_id : int, operator_attributes : dict, list_vehicles : List[SimulationVehicle], routing_engine : NetworkBase, 
                 zone_system : ZoneSystem, scenario_parameters : dict, dir_names : dict, op_charge_depot_infra=None, list_pub_charging_infra= []):
        """Fleet control class for the base scenario of the easyride broker
        it mimics the EasyRide Broker Exchange of requests scenario but willingness is not used the evaluate the operator decisions
        request enter system continously
            offer has to be created immediatly by an insertion heuristic, where also following decision is made:
            operator dicides if he can and if he wants to serve the customer
            if he wants and can:
                request is accepted and the offer is created
            if he cant:
                requests is declined and send to other fltctrl
            if he can but doesnt want:
                offer is created, but request is also sent to other fleetctrl
            request replies immediatly 
            -> there can never be 2 requests at the same time waiting for an offer! 
        reoptimisation of solution after certain time interval
        """
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=op_charge_depot_infra, list_pub_charging_infra=list_pub_charging_infra)

    def user_request(self, rq, sim_time):
        """This method is triggered for a new incoming request. It generally adds the rq to the database. It has to
        return an offer to the user. An empty dictionary means no offer is made!

        this method additionally evaluates the willingness of the operator to serve the request and sets the decision in the offer with attribute G_OFFER_WILLING_FLAG to provide the information to the fleet simulation.
        different to its parent-class, this fleet control is always willing to serve a customer, if possible

        :param rq: request object containing all request information
        :type rq: RequestDesign
        :param sim_time: current simulation time
        :type sim_time: float
        """
        if self.current_vid_plan_tuple is not None:
            LOG.error(f"new user_request before old request is resolved! {self.current_vid_plan_tuple}")
            raise AssertionError
        _ = super().user_request(rq, sim_time)
        rid_struct = rq.get_rid_struct()
        prq = self.rq_dict[rid_struct]
        if prq.o_pos == prq.d_pos:
            LOG.debug("automatic decline!")
            return {}

        LOG.debug("new user request {}".format(rid_struct))
        assigned_vid, assigned_plan, change_in_objective_value = single_insertion(self.sim_vehicles, self.veh_plans, prq, self.vr_ctrl_f, self.routing_engine, self.rq_dict, sim_time, self.const_bt, self.add_bt)
        prev_assigned_plan = self.veh_plans.get(assigned_vid)
        if assigned_vid is not None:    # fleetoperator can serve
            if self.evaluate_willing_ness_method == "reactive":
                is_willing = self._evaluate_willingness_reactive(change_in_objective_value, prev_assigned_plan, assigned_plan, prq,
                                                        self.sim_vehicles[assigned_vid], sim_time)
            elif self.evaluate_willing_ness_method == "forecast":
                is_willing = self._evaluate_willingness_forecast(sim_time, prq, assigned_vid, assigned_plan, change_in_objective_value)
            else:
                is_willing = True
            if is_willing:    # fleetoperator wants to or must serve
                offer = self._create_user_offer(prq, sim_time, assigned_vehicle_plan=assigned_plan, offer_dict_without_plan={G_OFFER_WILLING_FLAG : is_willing})
                self.current_vid_plan_tuple = (assigned_vid, assigned_plan)
                LOG.debug(f"new offer for rid {rid_struct} : {offer}")
            else:
                offer = self._create_user_offer(prq, sim_time, assigned_vehicle_plan=assigned_plan, offer_dict_without_plan={G_OFFER_WILLING_FLAG : is_willing})  # flag that he could?
        else:
            LOG.debug(f"no offer for rid {rid_struct}")
            offer = self._create_user_offer(prq, sim_time)
