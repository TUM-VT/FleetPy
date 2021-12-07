from src.fleetctrl.pooling.immediate.insertion import simple_insert
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, BoardingPlanStop
import src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment as AlonsoMoraAssignment

import logging
LOG = logging.getLogger(__name__)

def remove_non_routing_planstops_and_copy(veh_plan, veh_obj, routing_engine, sim_time):
    new_ps_list = []
    for ps in veh_plan.list_plan_stops:
        if not ps.locked and len(ps.boarding_dict.get(1,[])) == 0 and len(ps.boarding_dict.get(-1,[])) == 0:
            continue
        new_ps_list.append(ps.copy())
    if len(new_ps_list) == len(veh_plan.list_plan_stops):
        return veh_plan.copy()
    else:
        return VehiclePlan(veh_obj, sim_time, routing_engine, new_ps_list)

class V2RB():
    def __init__(self, routing_engine, rq_dict, sim_time, rtv_key, veh_obj, std_bt, add_bt, obj_function, orig_veh_plans = None, new_prq_obj = None):
        self.veh_plans = []
        self.cost_function_value = None
        self.rtv_key = rtv_key
        self.veh = veh_obj
        if new_prq_obj is not None and orig_veh_plans is None:
            if veh_obj.has_locked_vehplan():
                # init = [ps.copy() for ps in veh_obj.locked_planstops]
                # empty_plan = VehiclePlan(veh_obj, sim_time, routing_engine, init)
                empty_plan = veh_obj.locked_vehplan.copy()
            else:
                empty_plan = VehiclePlan(veh_obj, sim_time, routing_engine, [])
            self.veh_plans = [new_plan for new_plan in simple_insert(routing_engine, sim_time, veh_obj, empty_plan, new_prq_obj, std_bt, add_bt)]
        elif new_prq_obj is not None and orig_veh_plans is not None:
            for veh_plan in orig_veh_plans:
                copy_plan = remove_non_routing_planstops_and_copy(veh_plan, veh_obj, routing_engine, sim_time)# veh_plan.copy()
                self.veh_plans += [new_plan for new_plan in simple_insert(routing_engine, sim_time, veh_obj, copy_plan, new_prq_obj, std_bt, add_bt)]
        if self.isFeasible():
            self.computeCostFunctionValue(sim_time, obj_function, veh_obj, routing_engine, rq_dict)
        elif orig_veh_plans is not None and new_prq_obj is None:
            self.veh_plans = [remove_non_routing_planstops_and_copy(o_plan, veh_obj, routing_engine, sim_time) for o_plan in orig_veh_plans]# [o_plan.copy() for o_plan in orig_veh_plans]
            self.computeCostFunctionValue(sim_time, obj_function, veh_obj, routing_engine, rq_dict)
        # else:
        #     LOG.debug(f"V2RB infeasible: {self.rtv_key}")
        # LOG.debug(f"new V2RB {self.rtv_key}")
        # LOG.debug("\n".join([str(x) for x in self.veh_plans]))

    def __str__(self):
        l = ["V2RB {} wit cfv {}".format(self.rtv_key, self.cost_function_value)]
        for vp in self.veh_plans:
            l.append(str(vp))
        return "\n".join(l)

    def isFeasible(self):  
        if len(self.veh_plans) > 0:
            return True
        else:
            return False

    def addRequestAndCheckFeasibility(self, new_prq_obj, new_rtv_key, routing_engine, obj_function, rq_dict, sim_time, veh_obj, std_bt, add_bt):
        newV2RB = V2RB(routing_engine, rq_dict, sim_time, new_rtv_key, veh_obj, std_bt, add_bt, obj_function, orig_veh_plans=self.veh_plans, new_prq_obj=new_prq_obj)
        if newV2RB.isFeasible():
            return newV2RB
        else:
            return None

    def updateAndCheckFeasibility(self, routing_engine, obj_function, veh_obj, rq_dict, sim_time, list_passed_VRLs=None,
                                  is_assigned=False):
        if list_passed_VRLs is None:
            list_passed_VRLs = []
        new_veh_plans = []
        self.veh = veh_obj
        if not is_assigned:
            for i, veh_plan in enumerate(self.veh_plans):
                veh_plan.update_plan(veh_obj, sim_time, routing_engine, list_passed_VRLs=list_passed_VRLs,
                                     keep_time_infeasible=False)
                if veh_plan.is_feasible():
                    new_veh_plans.append(veh_plan)
        else:
            struct_feasible_veh_plans = []
            for i, veh_plan in enumerate(self.veh_plans):
                if is_assigned and i == 0:
                    veh_plan.update_plan(veh_obj, sim_time, routing_engine, list_passed_VRLs=list_passed_VRLs,
                                         keep_time_infeasible=True)
                    if veh_plan.is_feasible():
                        new_veh_plans.append(veh_plan)
                    elif veh_plan.is_structural_feasible():
                        struct_feasible_veh_plans.append(veh_plan)
                    else:
                        LOG.warning("(assigned) vehicle plan became structural infeasible! {}".format(veh_plan))
                else:
                    veh_plan.update_plan(veh_obj, sim_time, routing_engine, list_passed_VRLs=list_passed_VRLs,
                                         keep_time_infeasible=False)
                    if veh_plan.is_feasible():
                        new_veh_plans.append(veh_plan)
            if len(new_veh_plans) == 0:
                new_veh_plans = struct_feasible_veh_plans
        self.veh_plans = new_veh_plans
        self.computeCostFunctionValue(sim_time, obj_function, veh_obj, routing_engine, rq_dict)

    def computeCostFunctionValue(self, sim_time, obj_function, veh_obj, routing_engine, rq_dict):
        #simulation_time, veh_obj, veh_plan, rq_dict, routing_engine
        if len(self.veh_plans) > 0:
            #LOG.debug("compute CFV {}".format(self.rtv_key))
            for veh_plan in self.veh_plans:
                #LOG.debug(f"utility input : {veh_obj} | {veh_plan} | {[(rid, rq) for rid, rq in rq_dict.items()]}")
                veh_plan.set_utility(obj_function(sim_time, veh_obj, veh_plan, rq_dict, routing_engine))
                #LOG.debug(f" -> utility {veh_plan.utility}")
            self.veh_plans = sorted(self.veh_plans, key = lambda x:x.utility)
            self.cost_function_value = self.veh_plans[0].utility

    def getBestPlan(self):
        return self.veh_plans[0]

    def createLowerV2RB(self, lower_key, sim_time, routing_engine, obj_function, rq_dict, std_bt, add_bt):
        keep_rids = AlonsoMoraAssignment.getRidsFromRTVKey(lower_key)
        #only keep best route
        base_plan = self.veh_plans[0]
        new_plan_list = []
        for ps in base_plan.list_plan_stops:
            new_boarding_dict = {}
            new_max_trip_time_dict = {}
            new_earliest_pickup_time_dict = {}
            new_latest_pickup_time_dict = {}
            change_nr_pax = 0
            ept, lpt, mtt, lat = ps.get_boarding_time_constraint_dicts()
            for rid in ps.get_list_boarding_rids():
                if rid in keep_rids:
                    try:
                        new_boarding_dict[1].append(rid)
                    except:
                        new_boarding_dict[1] = [rid]
                    new_earliest_pickup_time_dict[rid] = ept[rid]
                    new_latest_pickup_time_dict[rid] = lpt[rid]
            change_nr_pax += sum([rq_dict[rid].nr_pax for rid in new_boarding_dict.get(1, [])])
            for rid in ps.get_list_alighting_rids():
                if rid in keep_rids:
                    try:
                        new_boarding_dict[-1].append(rid)
                    except:
                        new_boarding_dict[-1] = [rid]
                    new_max_trip_time_dict[rid] = mtt[rid]
            change_nr_pax -= sum([rq_dict[rid].nr_pax for rid in new_boarding_dict.get(-1, [])])
            if len(new_boarding_dict.keys()) > 0 or ps.is_locked():
                #LOG.warning("not considering new time constraints!!!!") # TODO #
                new_ps = BoardingPlanStop(ps.pos, boarding_dict=new_boarding_dict, max_trip_time_dict=new_max_trip_time_dict,
                                          earliest_pickup_time_dict=new_earliest_pickup_time_dict, latest_pickup_time_dict=new_latest_pickup_time_dict,
                                          change_nr_pax=change_nr_pax, duration=ps.get_duration_and_earliest_departure()[0])
                # new_ps = ps.copy()  
                # new_ps.boarding_dict = new_boarding_dict
                new_plan_list.append(new_ps)
        new_veh_plan = VehiclePlan(self.veh, sim_time, routing_engine, new_plan_list, external_pax_info=base_plan.pax_info.copy())
        return V2RB(routing_engine, rq_dict, sim_time, lower_key, self.veh, std_bt, add_bt, obj_function, orig_veh_plans=[new_veh_plan])

