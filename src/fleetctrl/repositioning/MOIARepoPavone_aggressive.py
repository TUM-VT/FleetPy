import random
import numpy as np
from src.fleetctrl.repositioning.RepositioningBase import RepositionBase
from src.misc.globals import *

import os
import logging
from src.misc.globals import *
LOG = logging.getLogger(__name__)


class MOIARepoPavone_aggressive(RepositionBase):
    """This class implements an adaption of the real-time rebalancing policy formulated in section 4.3 of
    Zhang, R.; Pavone, M. (2016): Control of robotic mobility-on-demand systems. A queueing-theoretical perspective.
    In: The International Journal of Robotics Research 35 (1-3), S. 186–203. DOI: 10.1177/0278364915581863.

    The adaption is that the supply side is forecast using arrival forecast.

    Adoptions from PavoneHailingFc.py:
        - no arrival forecast
        - (no locking)
    """

    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        """Initialization of repositioning class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes, solver=solver)
        # check if two horizon values are available
        if type(self.list_horizons) != list or len(self.list_horizons) != 2:
            raise IOError("PavoneHailingRepositioningFC requires two time horizon values (start and end)!"
                          f"Set them in the {G_OP_REPO_TH_DEF} scenario parameter!")
        self.optimisation_timeout = 30 # TODO #

    def determine_and_create_repositioning_plans(self, sim_time, lock=True):
        """This method determines and creates new repositioning plans. The repositioning plans are directly assigned
        to the vehicles. The repositioning algorithm can choose whether the generated respective plan stops are locked.
        In order to allow further database processes, the vids of vehicles with new plans are returned.

        :param sim_time: current simulation time
        :param lock: indicates if vehplans should be locked
        :return: list[vid] of vehicles with changed plans
        """
        # get forecast values
        # -------------------
        t0 = sim_time + self.list_horizons[0]
        t1 = sim_time + self.list_horizons[1]
        zone_imbalance = {}
        list_zones = self.zone_system.get_all_zones()
        demand_fc_dict = self._get_demand_forecasts(t0, t1)
        supply_fc_dict = {} #self._get_historic_arrival_forecasts(t0, t1)
        # print(demand_fc_dict)
        # print(supply_fc_dict)
        cplan_arrival_idle_dict = self._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1)

        # compute imbalance values and constraints
        # ----------------------------------------
        vehicles_repo_to_zone = {k: len(v[1]) for k,v in cplan_arrival_idle_dict.items()}
        number_current_own_vehicles = {k: v[0] for k, v in cplan_arrival_idle_dict.items()}
        number_idle_vehicles = {k: len(v[2]) for k,v in cplan_arrival_idle_dict.items()}
        # LOG.info("demand_fc_dict: {}".format(demand_fc_dict))
        # LOG.info("reloc: vehicles_repo_to_zone: {}".format(vehicles_repo_to_zone))
        # LOG.info("reloc: number_idle_vehicles: {}".format(number_idle_vehicles))
        # print("")
        # print(vehicles_repo_to_zone)
        # print(number_idle_vehicles)
        # print("")
        total_idle_vehicles = sum(number_idle_vehicles.values())
        nr_regions = len(list_zones)
        v_i_e_dict = {}
        omegas = []
        zone_dict = {}
        zone_counter = 0
        # compute v_i^e and v_i^d
        # modifications:
        # 1) the number of excess vehicles per area cannot be smaller than 0 and is bound from the top by either
        #   - the number of currently owned vehicles
        #   - the number of currently idle vehicles
        # 2) add constraint for max vehicles that can be sent away from a region
        # if number_idle_vehicles.get(-1) is not None: # vehicle outside zone system
        #     list_zones.append(-1)
        for zone_id in list_zones:
            o_d_diff = vehicles_repo_to_zone.get(zone_id, 0) + int(np.math.ceil(supply_fc_dict.get(zone_id, 0)))\
                    - int(np.math.ceil(demand_fc_dict.get(zone_id, 0)))
            number_idle = number_idle_vehicles.get(zone_id, 0)
            v_i_e_dict[zone_id] = max(number_idle + min(o_d_diff, 0), 0)
            omegas.append(number_idle + o_d_diff)
            zone_dict[zone_id] = zone_counter
            zone_counter += 1
        # my rebalancing policy
        v_i_d_dict = {}
        total_excess_vehicles = int(sum(list(v_i_e_dict.values())))
        n_called = 0
        prob_dist = {}
        for zone_id in list_zones:
            v_i_d_dict[zone_id] = 0
            if self.zone_system.get_random_centroid_node(zone_id) < 0:
                prob_dist[zone_id] = 0
            else:
                o_d_diff = vehicles_repo_to_zone.get(zone_id, 0) + int(np.math.ceil(supply_fc_dict.get(zone_id, 0)))\
                        - int(np.math.ceil(demand_fc_dict.get(zone_id, 0)))
                number_idle = number_idle_vehicles.get(zone_id, 0)
                delta = min(number_idle + min(o_d_diff, 0), 0)
                zone_call = int(np.math.ceil(-delta))
                if zone_call == 0:
                    zone_call = 1
                prob_dist[zone_id] = zone_call
                n_called += zone_call
        for z,v in prob_dist.items():
            prob_dist[z] = v/n_called
        for i in range(total_excess_vehicles):
            zone_to_reposition_to = np.random.choice(list(prob_dist.keys()), p=list(prob_dist.values()))
            v_i_d_dict[zone_to_reposition_to] += 1

        # optimization problem
        # --------------------
        if self.solver_key == "Gurobi":
            alpha_od, od_reposition_trips = self._optimization_gurobi(sim_time, list_zones, v_i_e_dict, v_i_d_dict,
                                                                      number_idle_vehicles, zone_dict)
            LOG.info("od repo trips {}".format(od_reposition_trips))
        else:
            raise IOError(f"Solver {self.solver_key} not available!")

        # # heat map evaluations before and after
        # # -------------------------------------
        # if len(omegas) > 0:
        #     sum_idle_vehicles = sum([len(x) for x in idle_vehicles_per_region.values()])
        #     list_sigma_integrals_0 = operator.rep_int_mod.calculate_kernel_integrals(omegas)
        #     delta_omegas = np.sum(alpha_od, axis=0) - np.sum(alpha_od, axis=1)
        #     list_sigma_integrals_1 = operator.rep_int_mod.calculate_kernel_integrals(omegas + delta_omegas)
        #     f_log = os.path.join(operator.scenario_output_dir, f"19_heat_map_int.csv")
        #     if not os.path.isfile(f_log):
        #         fhlog = open(f_log, "w")
        #         fhlog.write("sim_time,state,sigma,sum_idle_vehicles,F_abs,F_pos_omega,F_neg_omega,F2_val\n")
        #     else:
        #         fhlog = open(f_log, "a")
        #     for (sigma, f_abs_0, f_pos_0, f_neg_0, f2_val_0) in list_sigma_integrals_0:
        #         fhlog.write(
        #             f"{current_time},before,{sigma},{sum_idle_vehicles},{f_abs_0},{f_pos_0},{f_neg_0},{f2_val_0}\n")
        #     for (sigma, f_abs_1, f_pos_1, f_neg_1, f2_val_1) in list_sigma_integrals_1:
        #         fhlog.write(
        #             f"{current_time},after,{sigma},{sum_idle_vehicles},{f_abs_1},{f_pos_1},{f_neg_1},{f2_val_1}\n")
        #     fhlog.close()

        list_veh_with_changes = []
        if od_reposition_trips:
            # create assignments
            # ------------------
            random.seed(sim_time)
            random.shuffle(od_reposition_trips)
            for (origin_zone_id, destination_zone_id) in od_reposition_trips:
                list_idle_veh = cplan_arrival_idle_dict[origin_zone_id][2]
                list_veh_obj_with_repos = self._od_to_veh_plan_assignment(sim_time, origin_zone_id,
                                                                          destination_zone_id, list_idle_veh, lock=lock)
                list_veh_with_changes.extend([veh_obj.vid for veh_obj in list_veh_obj_with_repos])
                for veh_obj in list_veh_obj_with_repos:
                    cplan_arrival_idle_dict[origin_zone_id][2].remove(veh_obj)
        return list_veh_with_changes

    def _optimization_gurobi(self, sim_time, list_zones, v_i_e_dict, v_i_d_dict, number_idle_vehicles, zone_dict):
        import gurobipy
        model = gurobipy.Model("PavoneHailingFC")
        model.setParam('OutputFlag', False)
        model.setParam(gurobipy.GRB.param.Threads, self.fleetctrl.n_cpu)
        model.setObjective(gurobipy.GRB.MINIMIZE)
        # if self.optimisation_timeout:
        #     model.setParam('TimeLimit', self.optimisation_timeout)
        # decision variables
        number_regions = len(v_i_e_dict.keys())
        number_vars = number_regions ** 2 - number_regions
        var = [0 for _ in range(number_vars)]
        var_counter = 0
        o2var = {}  # o -> list of var with repositioning trips from o
        d2var = {}  # d -> list of var with repositioning trips to d
        var2od = {}  # var -> (o,d)
        for o_region in v_i_e_dict.keys():
            for d_region in v_i_e_dict.keys():
                if o_region == d_region:
                    continue
                t_od, d_od = self._get_od_zone_travel_info(sim_time, o_region, d_region)
                var[var_counter] = model.addVar(vtype=gurobipy.GRB.INTEGER, obj=t_od, name=f'n[{o_region}-{d_region}]')
                try:
                    o2var[o_region].append(var_counter)
                except:
                    o2var[o_region] = [var_counter]
                try:
                    d2var[d_region].append(var_counter)
                except:
                    d2var[d_region] = [var_counter]
                var2od[var_counter] = (o_region, d_region)
                var_counter += 1
        # constraints
        for region in v_i_e_dict.keys():
            list_o_vars = o2var.get(region, [])
            list_d_vars = d2var.get(region, [])
            # print(v_i_e_dict.get(region, 0), v_i_d_dict.get(region, 0))
            model.addConstr(
                v_i_e_dict.get(region, 0) + gurobipy.quicksum(var[d] for d in list_d_vars) - gurobipy.quicksum(
                    var[o] for o in list_o_vars) >= v_i_d_dict.get(region, 0))
            model.addConstr(
                gurobipy.quicksum(var[o] for o in list_o_vars) <= number_idle_vehicles.get(region, 0))
        for x in range(number_vars):
            model.addConstr(var[x] >= 0)
        # update model
        model.update()
        # optimize
        model.optimize()
        # record model and solution
        # model_f = os.path.join(operator.scenario_output_dir, f"70_{current_time}_mt_stanford_optimization_model.lp")
        # model.write(model_f)

        # retrieve solution and create od-vehicle list
        # --------------------------------------------
        alpha_od = np.zeros((number_regions, number_regions))
        od_reposition_trips = []
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            # record solution
            # sol_f = os.path.join(operator.scenario_output_dir, f"70_{current_time}_mt_stanford_optimization_solution_no_eval.sol")
            # model.write(sol_f)
            solution = [var.X for var in model.getVars()]
            for x in range(number_vars):
                round_solution_integer = int(round(solution[x], 0))
                if round_solution_integer > 0:
                    (o_region, d_region) = var2od[x]
                    i = zone_dict[o_region]
                    j = zone_dict[d_region]
                    alpha_od[i, j] = round_solution_integer
                    # except:
                    #     print("MOIAREPOPAVONE")
                    #     print(i, j, o_region, d_region, alpha_od.shape, round_solution_integer)
                    #     exit()
                    od_reposition_trips.extend([(o_region, d_region)] * round_solution_integer)
            return alpha_od, od_reposition_trips
        else:
            raise Exception(f"Operator {self.fleetctrl.op_id}: No Optimal Solution! status {model.status}")

    def _optimization_cplex(self, list_zones, v_i_e_dict, v_i_d_dict, number_idle_vehicles, zone_dict):
        pass