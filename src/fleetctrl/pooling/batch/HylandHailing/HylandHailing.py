from __future__ import annotations
import logging
import pandas as pd
from pathlib import Path

from typing import Dict, List, TYPE_CHECKING

from src.fleetctrl.planning.VehiclePlan import VehiclePlan, RoutingTargetPlanStop
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import BatchAssignmentAlgorithmBase
from src.fleetctrl.pooling.immediate.insertion import simple_insert_hailing
from src.fleetctrl.pooling.immediate.searchVehicles import veh_search_for_immediate_request
from src.misc.globals import *
if TYPE_CHECKING:
    from src.simulation.Vehicles import SimulationVehicle
    from src.simulation.Legs import VehicleRouteLeg

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_HylandHailing = {
    "doc" :  """this class uses the ride hailing methods by Hyland & Mahmassani (2018)  """,
    "inherit" : "BatchAssignmentAlgorithmBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class HylandHailing(BatchAssignmentAlgorithmBase):
    """ this class uses applies the ride hailing methods by Hyland & Mahmassani(2018) Dynamic autonomous vehicle fleet
    operations: Optimization-based strategies to assign AVs to immediate traveler demand requests"""

    def __init__(self, fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes,
                 optimisation_cores = 1, seed = 6061992, veh_objs_to_build = {}, dir_names=None):

        super().__init__(fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes,
                         optimisation_cores=optimisation_cores, seed=seed, veh_objs_to_build=veh_objs_to_build,
                         dir_names=dir_names)
        self.vehicle_inclusion_policy = operator_attributes.get(G_OP_RH_VEH_SEARCH, "all-vehicles")
        possible_policies = {"idle-only", "repo-and-idle-only", "all-vehicles"}
        assert self.vehicle_inclusion_policy in possible_policies, (f" HylandHailing vehicle exclusion policy "
                                                                    f"{self.vehicle_inclusion_policy} not in possible policies {possible_policies}")
        # Optional recording of the vehicle stats at the time of optimization
        self._veh_considered_f = None
        if operator_attributes.get(G_OP_RH_REC_VEH_OPTM, False):
            self._veh_considered_f = Path(dir_names[G_DIR_OUTPUT], f"5-{fleetcontrol.op_id}_op-hailing_optim_veh_states.csv")


    def solve_assignment_problem(self, sim_time, veh_plan_dict: dict[SimulationVehicle, list[VehiclePlan]], n_cpu=1):
        import gurobipy

        model = gurobipy.Model(f"grb_hailing_{sim_time}")
        model.setParam('OutputFlag', False)
        model.setParam(gurobipy.GRB.param.Threads, n_cpu)
        model.setObjective(gurobipy.GRB.MINIMIZE)

        # Create variables for each vehicle to vehicle-plan combination
        var_dict = {veh_obj: [] for veh_obj in veh_plan_dict.keys()}
        for veh, plan_list in veh_plan_dict.items():
            veh_plan_var = []
            for inx, plan in enumerate(plan_list):
                var_name = f"x_{veh}_{inx}"
                var = model.addVar(vtype=gurobipy.GRB.CONTINUOUS, name=var_name, obj=plan.get_utility())
                var_dict[veh].append(var)

            # Vehicle constraints: each vehicle can have at most one plan
            model.addConstr(gurobipy.quicksum(var_dict[veh]) <= 1, name=f"veh_{veh.vid}")

        # Request constraints: each request can be assigned at most once
        request_to_vars: Dict[int, List[gurobipy.Var]] = {}
        for veh, plan_list in veh_plan_dict.items():
            for inx, plan in enumerate(plan_list):
                var = var_dict[veh][inx]
                for rid in plan.get_involved_request_ids():
                    if rid not in request_to_vars:
                        request_to_vars[rid] = []
                    request_to_vars[rid].append(var)
        for rid, vars_list in request_to_vars.items():
            model.addConstr(gurobipy.quicksum(vars_list) <= 1, name=f"req_{rid}")

        model.update()
        model.optimize()

        # get the solution
        assignments = {}
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            for veh, plan_list in veh_plan_dict.items():
                for inx, plan in enumerate(plan_list):
                    var = var_dict[veh][inx]
                    if var.X > 0.99:
                        assert veh not in assignments, f"vehicle {veh.vid} is being assigned multiple plans!"
                        assignments[veh] = plan
        else:
            raise Exception(f"Operator {self.fleetcontrol.op_id}: No Optimal Assignment Solution found for Ride Hailing!")

        return assignments

    def select_vehicles_to_include(self, sim_time : int) -> List[SimulationVehicle]:
        """ selects vehicles to include from the ride hailing search based on the vehicle inclusion policy
        :param sim_time: current simulation time
        :return: list of vehicle objects to exclude from the ride hailing search
        """
        vehicles_to_include = []
        for veh_obj in self.fleetcontrol.sim_vehicles:
            num_plan_stops = 0
            veh_plan = self.fleetcontrol.veh_plans.get(veh_obj.vid, None)
            if veh_plan is not None:
                num_plan_stops = len([ps for ps in veh_plan.list_plan_stops if type(ps) != RoutingTargetPlanStop])
            if self.vehicle_inclusion_policy == "idle-only":
                # Only include vehicles that are idle and have no assigned tasks
                if veh_obj.status == VRL_STATES.IDLE and num_plan_stops == 0:
                    vehicles_to_include.append(veh_obj)
            elif self.vehicle_inclusion_policy == "repo-and-idle-only":
                # Include vehicles that are idle or repositioning and have no assigned tasks
                if veh_obj.status in {VRL_STATES.IDLE, VRL_STATES.REPOSITION} and num_plan_stops == 0:
                    vehicles_to_include.append(veh_obj)
            elif self.vehicle_inclusion_policy == "all-vehicles":
                vehicles_to_include.append(veh_obj)
        return vehicles_to_include

    def lock_on_board_request_dropoffs(self, sim_time: int):
        """ Locks the drop-off stops of requests that are already on-board vehicles to prevent them from being altered during optimization.
        :param sim_time: current simulation time
        """
        for veh_obj in self.fleetcontrol.sim_vehicles:
            onboard_rids = veh_obj.get_rid_list(ignore_parcels=True)
            assert len(onboard_rids) <= 1, f"Vehicle {veh_obj.vid} has multiple requests {onboard_rids} on-board!"
            if len(onboard_rids) > 0:
                current_veh_p = self.fleetcontrol.veh_plans.get(veh_obj.vid, None)
                if current_veh_p is not None:
                    ps = current_veh_p.list_plan_stops[0]
                    if ps.get_list_boarding_rids():
                        assert ps.get_list_boarding_rids()[0] == onboard_rids[0]
                        ps.set_locked(True)
                        assert current_veh_p.list_plan_stops[1].get_list_alighting_rids()[0] == onboard_rids[0]
                        current_veh_p.list_plan_stops[1].set_locked(True)
                    if ps.get_list_alighting_rids():
                        assert ps.get_list_alighting_rids()[0] == onboard_rids[0]
                        ps.set_locked(True)
    
    def compute_new_vehicle_assignments(self, sim_time : int, vid_to_list_passed_VRLs : Dict[int, List[VehicleRouteLeg]],
                                        veh_objs_to_build : Dict[int, SimulationVehicle] = {},
                                        new_travel_times : bool = False, build_from_scratch : bool = False):
        """ this function computes new vehicle assignments based on current fleet information
        param sim_time : current simulation time
        param vid_to_list_passed_VRLs : (dict) vid -> list_passed_VRLs; needed to update database and V2RBs
        :param veh_objs_to_build: only these vehicles will be optimized (all if empty) dict vid -> SimVehicle obj
                                  only for special cases needed in current alonso mora module
        :param new_travel_times : bool; if traveltimes changed in the routing engine
        :param build_from_scratch : only for special cases needed in current alonso mora module
        """

        self.sim_time = sim_time
        if len(veh_objs_to_build) != 0:
            raise NotImplementedError

        if len(list(self.unassigned_requests.keys())) == 0:
            return

        # Lock the drop-off stops of requests that are already on-board
        # TODO: investigate why this is not happening already by the fleet controller
        self.lock_on_board_request_dropoffs(sim_time)

        # Calculate if some vehicles should be excluded from the ride hailing search
        vehicles_to_include = self.select_vehicles_to_include(sim_time)
        vehicles_to_exclude = [veh.vid for veh in self.fleetcontrol.sim_vehicles if veh not in vehicles_to_include]

        current_plans = {}
        for veh_obj in vehicles_to_include:
            # Get the existing plan or create a new empty one if not present
            current_veh_p = self.fleetcontrol.veh_plans.get(veh_obj.vid, VehiclePlan(veh_obj, self.sim_time, self.routing_engine, []))
            current_veh_p.update_tt_and_check_plan(veh_obj, sim_time, self.routing_engine, keep_feasible=True)
            obj = self.fleetcontrol.compute_VehiclePlan_utility(sim_time, veh_obj, current_veh_p)
            current_veh_p.set_utility(obj)
            # The following will remove repositioning stops if they are not locked
            veh_p = current_veh_p.copy_and_remove_empty_planstops(veh_obj, sim_time, self.routing_engine)
            obj = self.fleetcontrol.compute_VehiclePlan_utility(sim_time, veh_obj, veh_p)
            veh_p.set_utility(obj)
            current_plans[veh_obj.vid] = veh_p

        vobj_plan_dict = {}
        plan_rid_dict = {}
        for rid in list(self.unassigned_requests.keys()):
            if self.rid_to_consider_for_global_optimisation.get(rid) is None:
                continue
            rv_vehicles, rv_results_dict = veh_search_for_immediate_request(sim_time, self.active_requests[rid],
                                                                            self.fleetcontrol, vehicles_to_exclude)
            for veh in rv_vehicles:
                feasible_plans_list = simple_insert_hailing(self.fleetcontrol.routing_engine, sim_time, veh,
                                                            current_plans[veh.vid], self.active_requests[rid],
                                                            self.fleetcontrol.const_bt, self.fleetcontrol.add_bt)
                for next_plan in feasible_plans_list:
                    utility = self.fleetcontrol.compute_VehiclePlan_utility(sim_time, veh, next_plan)
                    next_plan.set_utility(utility)
                    if veh.vid in vobj_plan_dict:
                        vobj_plan_dict[veh].append(next_plan)
                    else:
                        vobj_plan_dict[veh] = [next_plan]
                    plan_rid_dict[next_plan] = rid

        # Solve the assignment problem
        assignments = self.solve_assignment_problem(sim_time, vobj_plan_dict, n_cpu=1)
        if self._veh_considered_f is not None:
            self._record_vehicle_states(sim_time, current_plans, assignments, plan_rid_dict)

        sum_obj = 0
        for veh_obj, assigned_plan in assignments.items():
            self.fleetcontrol.assign_vehicle_plan(veh_obj, assigned_plan, sim_time)
            # update utility
            upd_utility_val = self.fleetcontrol.compute_VehiclePlan_utility(sim_time, veh_obj, self.fleetcontrol.veh_plans[veh_obj.vid])
            self.fleetcontrol.veh_plans[veh_obj.vid].set_utility(upd_utility_val)
            sum_obj += upd_utility_val
            rid = plan_rid_dict[assigned_plan]
            LOG.debug(f"request {rid} assigned to vehicle {veh_obj.vid} with ride hailing assignment")
        LOG.info(f"Objective value at time {sim_time} for ride-hailing assignment: {sum_obj}")

        # The unassigned requests are immediately rejected and removed from future consideration
        self.unassigned_requests = {}

    def _record_vehicle_states(self, sim_time, current_plans, assignments, plan_rid_dict):
        record_list = []
        for veh in self.fleetcontrol.sim_vehicles:
            first_available_pos, last_available_pos = veh.pos[0], None
            first_available_time, last_available_time = sim_time, None

            veh_plan = current_plans[veh.vid]
            for ps in veh_plan.list_plan_stops:
                if ps.is_locked() is False:
                    first_available_pos = ps.pos[0]
                    first_available_time = round(ps.get_planned_arrival_and_departure_time()[1], 1)
                    break
            if len(veh_plan.list_plan_stops) > 0:
                last_available_time = round(veh_plan.list_plan_stops[-1].get_planned_arrival_and_departure_time()[1], 1)
                last_available_pos = veh_plan.list_plan_stops[-1].pos[0]

            new_plan = assignments.get(veh, None)
            new_rid = None
            new_rid_inserted_at = None
            if new_plan is not None:
                new_rid = plan_rid_dict[new_plan]
                for inx, ps in enumerate(new_plan.list_plan_stops):
                    if len(ps.get_list_boarding_rids()) > 0 and ps.get_list_boarding_rids()[0] == new_rid:
                        new_rid_inserted_at = inx
                        break

            record_list.append({
                "sim_time": sim_time,
                "vid": veh.vid,
                "status": veh.status.display_name,
                "current_total_stops": len(veh_plan.list_plan_stops),
                "first_unlocked_pos": first_available_pos,
                "first_unlocked_time": first_available_time,
                "last_pos": last_available_pos,
                "last_time": last_available_time,
                "new_assigned_rid": new_rid,
                "new_rid_inserted_at_index": new_rid_inserted_at
            })
        record_df = pd.DataFrame(record_list)
        if self._veh_considered_f.exists():
            write_mode = "a"
            write_header = False
        else:
            write_mode = "w"
            write_header = True
        record_df.to_csv(self._veh_considered_f, index=False, mode=write_mode, header=write_header)
    
    def get_optimisation_solution(self, vid : int) -> VehiclePlan:
        """ returns optimisation solution for vid
        :param vid: vehicle id
        :return: vehicle plan object for the corresponding vehicle
        """
        return self.fleetcontrol.veh_plans[vid]

    def set_assignment(self, vid : int, assigned_plan : VehiclePlan, is_external_vehicle_plan : bool = False):
        """ sets the vehicleplan as assigned in the algorithm database; if the plan is not computed within the this algorithm, the is_external_vehicle_plan flag should be set to true
        :param vid: vehicle id
        :param assigned_plan: vehicle plan object that has been assigned
        :param is_external_vehicle_plan: should be set to True, if the assigned_plan has not been computed within this algorithm
        """
        super().set_assignment(vid, assigned_plan, is_external_vehicle_plan=is_external_vehicle_plan)

    def get_current_assignment(self, vid : int) -> VehiclePlan:
        """ returns the vehicle plan assigned to vid currently
        :param vid: vehicle id
        :return: vehicle plan
        """
        return self.fleetcontrol.veh_plans[vid]