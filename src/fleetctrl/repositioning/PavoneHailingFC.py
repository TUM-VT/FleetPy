import random
import numpy as np
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.misc.globals import *

# from IPython import embed

import os
import logging
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_PavoneHailingRepositioningFC = {
    "doc" : """This class implements an adaption of the real-time rebalancing policy formulated in section 4.3 of
    Zhang, R.; Pavone, M. (2016): Control of robotic mobility-on-demand systems. A queueing-theoretical perspective.
    In: The International Journal of Robotics Research 35 (1-3), S. 186–203. DOI: 10.1177/0278364915581863.

    The adaption is that the supply side is forecast using arrival forecast and excess vehicles cannot be negative.""",
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class PavoneHailingRepositioningFC(RepositioningBase):
    """This class implements an adaption of the real-time rebalancing policy formulated in section 4.3 of
    Zhang, R.; Pavone, M. (2016): Control of robotic mobility-on-demand systems. A queueing-theoretical perspective.
    In: The International Journal of Robotics Research 35 (1-3), S. 186–203. DOI: 10.1177/0278364915581863.

    The adaption is that the supply side is forecast using arrival forecast and excess vehicles cannot be negative.
    """

    def __init__(self, fleetctrl, operator_attributes, dir_names, solver="Gurobi"):
        """Initialization of repositioning class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param dir_names: directory structure dict
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes, dir_names, solver=solver)
        # check if two horizon values are available
        if type(self.list_horizons) != list or len(self.list_horizons) != 2:
            raise IOError("PavoneHailingRepositioningFC requires two time horizon values (start and end)!"
                          f"Set them in the {G_OP_REPO_TH_DEF} scenario parameter!")
        self.optimisation_timeout = 30 # TODO #

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """This method determines and creates new repositioning plans. The repositioning plans are directly assigned
        to the vehicles. The repositioning algorithm can choose whether the generated respective plan stops are locked.
        In order to allow further database processes, the vids of vehicles with new plans are returned.

        :param sim_time: current simulation time
        :param lock: indicates if vehplans should be locked
        :return: list[vid] of vehicles with changed plans
        """
        self.sim_time = sim_time
        if lock is None:
            lock = self.lock_repo_assignments
        # get forecast values
        # -------------------
        t0 = sim_time + self.list_horizons[0]
        t1 = sim_time + self.list_horizons[1]
        zone_imbalance = {}
        list_zones_all = self.zone_system.get_all_zones()
        list_zones = sorted([zone for zone in list_zones_all if zone != -1])
        demand_fc_dict = self._get_demand_forecasts(t0, t1)
        supply_fc_dict = self._get_historic_arrival_forecasts(t0, t1)
        # print(demand_fc_dict)
        # print(supply_fc_dict)
        cplan_arrival_idle_dict = self._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1)

        # compute imbalance values and constraints
        # ----------------------------------------
        vehicles_repo_to_zone = {k: len(v[1]) for k,v in cplan_arrival_idle_dict.items()}
        number_current_own_vehicles = {k: v[0] for k, v in cplan_arrival_idle_dict.items()}
        number_idle_vehicles = {k: len(v[2]) for k,v in cplan_arrival_idle_dict.items()}
        # print("")
        # print(vehicles_repo_to_zone)
        # print(number_idle_vehicles)
        # print("")
        total_idle_vehicles = sum(number_idle_vehicles.values())
        if total_idle_vehicles == 0:
            return []
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
        for zone_id in list_zones:
            o_d_diff = vehicles_repo_to_zone.get(zone_id, 0) + supply_fc_dict.get(zone_id, 0)\
                    - demand_fc_dict.get(zone_id, 0)
            number_idle = number_idle_vehicles.get(zone_id, 0)
            v_i_e_dict[zone_id] = max(number_idle + min(o_d_diff, 0), 0)
            omegas.append(number_idle + o_d_diff)
            zone_dict[zone_id] = zone_counter
            zone_counter += 1
        # rebalancing policy -> divide excess vehicles evenly among all zones
        v_i_d_dict = {}
        total_excess_vehicles = sum(list(v_i_e_dict.values()))
        try:
            avg_excess_vehicles_per_zone = int(total_excess_vehicles / len(list_zones))
        except:
            avg_excess_vehicles_per_zone = 0
        for zone_id in list_zones:  # TODO #
            if self.zone_system.get_random_centroid_node(zone_id) < 0:
                v_i_d_dict[zone_id] = 0
            else:
                v_i_d_dict[zone_id] = avg_excess_vehicles_per_zone
        # print(v_i_e_dict)
        # print(v_i_d_dict)
        #exit()

        # optimization problem
        # --------------------
        LOG.debug("PavoneHailingRepositioningFC input:")
        LOG.debug("list zones: {}".format(list_zones))
        LOG.debug("total_excess_vehicles: {}".format(total_excess_vehicles))
        LOG.debug("v_i_e_dict: {}".format(v_i_e_dict))
        LOG.debug("v_i_d_dict: {}".format(v_i_d_dict))
        LOG.debug("idle vehicles: {}".format(number_idle_vehicles))
        if self.solver_key == "Gurobi":
            alpha_od, od_reposition_trips = self._optimization_gurobi(sim_time, list_zones, v_i_e_dict, v_i_d_dict,
                                                                      number_idle_vehicles, zone_dict)
        elif self.solver_key == "Cplex":
            alpha_od, od_reposition_trips = self._optimization_cplex(sim_time, list_zones, v_i_e_dict, v_i_d_dict,
                                                                     number_idle_vehicles, zone_dict)
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
        
        LOG.debug("PavoneHailingRepositioningFC results: {}".format(od_reposition_trips))

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
        model_name = f"PavoneHailingFC: _optimization_gurobi {sim_time}"
        with gurobipy.Env(empty=True) as env:
            if self.fleetctrl.log_gurobi:
                with open(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log"), "a") as f:
                    f.write(f"\n\n{model_name}\n\n")
                env.setParam('OutputFlag', 1)
                env.setParam('LogToConsole', 0)
                env.setParam('LogFile', os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log") )
                env.start()
            else:
                env.setParam('OutputFlag', 0)
                env.setParam('LogToConsole', 0)
                env.start()
            model = gurobipy.Model(model_name, env=env)
            model.setParam(gurobipy.GRB.param.Threads, self.fleetctrl.n_cpu)
            model.setObjective(gurobipy.GRB.MINIMIZE)
            # if self.optimisation_timeout:
            #     model.setParam('TimeLimit', self.optimisation_timeout)
            # decision variables
            number_regions = len(list_zones)
            number_vars = number_regions ** 2 - number_regions
            var = [0 for _ in range(number_vars)]
            var_counter = 0
            o2var = {}  # o -> list of var with repositioning trips from o
            d2var = {}  # d -> list of var with repositioning trips to d
            var2od = {}  # var -> (o,d)
            for o_region in list_zones:
                for d_region in list_zones:
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
            for region in list_zones:
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
                        od_reposition_trips.extend([(o_region, d_region)] * round_solution_integer)
            else:
                model_f = os.path.join(self.output_dir, f"70_repo_pav_opt_model_infeasible_{sim_time}.lp")
                model.write(model_f)
                LOG.warning(f"Operator {self.fleetctrl.op_id}: No Optimal Solution! status {model.status}"
                            f" -> no repositioning")
                LOG.info(f"list zones: {list_zones}")
                LOG.info(f"number vehicles: {number_idle_vehicles}")
                LOG.info(f"vie dict: {len(v_i_e_dict.keys())} | {v_i_e_dict}")
                LOG.info(f"vid dict: {len(v_i_d_dict.keys())} | {v_i_d_dict}")
                LOG.info(f"zone dict: {len(zone_dict.keys())} | {zone_dict}")
        return alpha_od, od_reposition_trips

    def _optimization_cplex(self, sim_time, list_zones, v_i_e_dict, v_i_d_dict, number_idle_vehicles, zone_dict):
        from docplex.mp.model import Model
        model = Model(name='st_ua_optimization')
        model.parameters.threads(self.fleetctrl.n_cpu)

        # decision variables
        number_regions = len(list_zones)
        number_vars = number_regions ** 2 - number_regions
        var = [0 for _ in range(number_vars)]
        var_counter = 0
        o2var = {}  # o -> list of var with repositioning trips from o
        d2var = {}  # d -> list of var with repositioning trips to d
        var2od = {}  # var -> (o,d)
        list_of_dec_vars = []
        list_of_values = []
        for o_region in list_zones:
            for d_region in list_zones:
                if o_region == d_region:
                    continue
                t_od, _ = self._get_od_zone_travel_info(sim_time, o_region, d_region)
                list_of_dec_vars.append(model.integer_var(name=f'n[{o_region}-{d_region}]'))
                list_of_values.append(t_od)
                try:
                    o2var[o_region].append(var_counter)
                except KeyError:
                    o2var[o_region] = [var_counter]
                try:
                    d2var[d_region].append(var_counter)
                except KeyError:
                    d2var[d_region] = [var_counter]
                var2od[var_counter] = (o_region, d_region)
                var_counter += 1
        # constraints
        for region in list_zones:
            list_o_vars = o2var.get(region, [])
            list_d_vars = d2var.get(region, [])
            model.add_constraint(v_i_e_dict.get(region, 0) + np.sum(list_of_dec_vars[d] for d in list_d_vars) -
                                 np.sum(list_of_dec_vars[o] for o in list_o_vars) >= v_i_d_dict.get(region, 0))
            model.add_constraint(np.sum(list_of_dec_vars[o] for o in list_o_vars) <= number_idle_vehicles.get(region, 0))
        for x in range(number_vars):
            model.add_constraint(list_of_dec_vars[x] >= 0)
        dec_var_arr = np.array(list_of_dec_vars)
        values_arr = np.array(list_of_values)
        model.minimize(dec_var_arr.dot(values_arr))
        # optimize
        solution = model.solve()

        # retrieve solution and create od-vehicle list
        # --------------------------------------------
        alpha_od = np.zeros((number_regions, number_regions))
        od_reposition_trips = []
        if solution.solve_details.status == 'integer optimal solution':
            solution_list = solution.get_all_values()
            for x in range(number_vars):
                round_solution_integer = int(round(solution_list[x], 0))
                if round_solution_integer > 0:
                    (o_region, d_region) = var2od[x]
                    i = zone_dict[o_region]
                    j = zone_dict[d_region]
                    alpha_od[i, j] = round_solution_integer
                    od_reposition_trips.extend([(o_region, d_region)] * round_solution_integer)
        return alpha_od, od_reposition_trips

INPUT_PARAMETERS_PavoneHailingV2RepositioningFC = {
    "doc" : """This class implements an adaption of the real-time rebalancing policy formulated in section 4.3 of
    Zhang, R.; Pavone, M. (2016): Control of robotic mobility-on-demand systems. A queueing-theoretical perspective.
    In: The International Journal of Robotics Research 35 (1-3), S. 186–203. DOI: 10.1177/0278364915581863.

    The adaption is that the supply side is forecast using arrival forecast. Moreover, the computation of excess
    vehicles is more complex to allow negative values, but consistent solution during global vehicle shortage.""",
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class PavoneHailingV2RepositioningFC(PavoneHailingRepositioningFC):
    """This class implements an adaption of the real-time rebalancing policy formulated in section 4.3 of
    Zhang, R.; Pavone, M. (2016): Control of robotic mobility-on-demand systems. A queueing-theoretical perspective.
    In: The International Journal of Robotics Research 35 (1-3), S. 186–203. DOI: 10.1177/0278364915581863.

    The adaption is that the supply side is forecast using arrival forecast. Moreover, the computation of excess
    vehicles is more complex to allow negative values, but consistent solution during global vehicle shortage.
    """

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """This method determines and creates new repositioning plans. The repositioning plans are directly assigned
        to the vehicles. The repositioning algorithm can choose whether the generated respective plan stops are locked.
        In order to allow further database processes, the vids of vehicles with new plans are returned.

        :param sim_time: current simulation time
        :param lock: indicates if vehplans should be locked
        :return: list[vid] of vehicles with changed plans
        """
        self.sim_time = sim_time
        if lock is None:
            lock = self.lock_repo_assignments
        # get forecast values
        # -------------------
        t0 = sim_time + self.list_horizons[0]
        t1 = sim_time + self.list_horizons[1]
        zone_imbalance = {}
        list_zones = self.zone_system.get_all_zones()
        demand_fc_dict = self._get_demand_forecasts(t0, t1)
        supply_fc_dict = self._get_historic_arrival_forecasts(t0, t1)
        # print(demand_fc_dict)
        # print(supply_fc_dict)
        cplan_arrival_idle_dict = self._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1)

        # compute imbalance values and constraints
        # ----------------------------------------
        vehicles_repo_to_zone = {k: len(v[1]) for k,v in cplan_arrival_idle_dict.items()}
        number_current_own_vehicles = {k: v[0] for k, v in cplan_arrival_idle_dict.items()}
        number_idle_vehicles = {k: len(v[2]) for k,v in cplan_arrival_idle_dict.items()}
        # print("")
        # print(vehicles_repo_to_zone)
        # print(number_idle_vehicles)
        # print("")
        total_idle_vehicles = sum(number_idle_vehicles.values())
        if total_idle_vehicles == 0:
            return []
        #LOG.info(f"total_idle_vehicles {total_idle_vehicles}")
        nr_regions_with_demand = 0
        v_i_e_dict = {} # can also contain negative values now!
        omegas = []
        zone_dict = {}
        zone_counter = 0
        # compute v_i^e and v_i^d
        for zone_id in list_zones:
            o_d_diff = vehicles_repo_to_zone.get(zone_id, 0) + supply_fc_dict.get(zone_id, 0)\
                    - demand_fc_dict.get(zone_id, 0)
            number_idle = number_idle_vehicles.get(zone_id, 0)
            imbalance = number_idle + o_d_diff
            v_i_e_dict[zone_id] = imbalance
            omegas.append(imbalance)
            zone_dict[zone_id] = zone_counter
            zone_counter += 1
        # modifications:
        # 1) the number of total excess vehicles (positive or negative!) has to be smaller than the total number of
        #       idle vehicles!
        total_excess_vehicles = sum(list(v_i_e_dict.values()))
        #LOG.info(f"total_excess_vehicles {total_excess_vehicles}")
        if abs(total_excess_vehicles) > total_idle_vehicles:
            for zone_id, old_v_i_e in v_i_e_dict.items():
                v_i_e_dict[zone_id] = old_v_i_e / total_excess_vehicles * total_idle_vehicles
        # rebalancing policy -> divide excess vehicles evenly among all zones
        v_i_d_dict = {}
        total_excess_vehicles = sum(list(v_i_e_dict.values()))
        #LOG.info(f"total_excess_vehicles {total_excess_vehicles}")
        try:
            avg_excess_vehicles_per_zone = int(total_excess_vehicles / zone_counter)
        except ZeroDivisionError:
            avg_excess_vehicles_per_zone = 0
        for zone_id in list_zones:
            if self.zone_system.get_random_centroid_node(zone_id) < 0:
                v_i_d_dict[zone_id] = min(0, v_i_e_dict[zone_id])
            else:
                v_i_d_dict[zone_id] = avg_excess_vehicles_per_zone
        # print(v_i_e_dict)
        # print(v_i_d_dict)
        #exit()

        # optimization problem
        # --------------------
        if self.solver_key == "Gurobi":
            alpha_od, od_reposition_trips = self._optimization_gurobi(sim_time, list_zones, v_i_e_dict, v_i_d_dict,
                                                                      number_idle_vehicles, zone_dict)
        elif self.solver_key == "Cplex":
            alpha_od, od_reposition_trips = self._optimization_cplex(sim_time, list_zones, v_i_e_dict, v_i_d_dict,
                                                                     number_idle_vehicles, zone_dict)
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
