import numpy as np
import os
import random
import traceback
import logging
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.fleetctrl.planning.VehiclePlan import RoutingTargetPlanStop
from timeit import default_timer
from src.misc.globals import *
LOG = logging.getLogger(__name__)


TIME_LIMIT = 30
LARGE_INT = 100000
WRITE_SOL = False
WRITE_PROBLEM = False


def set_var_value(np_entry):
    try:
        return np_entry.X
    except AttributeError:
        return 0


np_set_gurobi_var_value = np.vectorize(set_var_value)

INPUT_PARAMETERS_DensityRepositioning = {
    "doc" : "This class implements Density Based Repositioning Algorithm from Frontiers paper of Arslan and Florian",
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_OP_REPO_GAMMA, G_OP_REPO_FRONTIERS_M],
    "mandatory_modules": [],
    "optional_modules": []
}


class DensityRepositioning(RepositioningBase):
    """This class implements Density Based Repositioning Algorithm from Frontiers paper of Arslan and Florian """
    def __init__(self, fleetctrl, operator_attributes, dir_names):
        """Initialization of repositioning class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param dir_names: directory structure dict
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes, dir_names)
        self.distance_cost = np.mean([veh_obj.distance_cost for veh_obj in fleetctrl.sim_vehicles])/1000
        self.zone_corr_matrix = np.array(self._return_squared_zone_imbalance_np_array())
        self.gamma = operator_attributes.get(G_OP_REPO_GAMMA, 1.0)
        self.method = operator_attributes.get(G_OP_REPO_FRONTIERS_M, "2-RFRR")
        possible_methods = {"RFRR", "RFRRp", "RFRRf", "2-RFRR", "2-RFRRp", "2-RFRRf"}
        assert self.method in possible_methods, "unknown {} method {} given for repositioning. " \
                                                "Possible options are {}".format(G_OP_REPO_FRONTIERS_M, self.method,
                                                                                 possible_methods)

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """This method determines and creates new repositioning plans. The repositioning plans are directly assigned
        to the vehicles.
        In order to allow further database processes, the vids of vehicles with new plans are returned.

        :param sim_time: current simulation time
        :param lock: indicates if vehplans should be locked
        :return: list[vid] of vehicles with changed plans
        """

        self.sim_time = sim_time
        if lock is None:
            lock = self.lock_repo_assignments
        list_zones_all = self.zone_system.get_complete_zone_list()
        list_zones = sorted([zone for zone in list_zones_all if zone != -1])

        # 1) get required imbalance forecasts
        # ----------------------------------------

        t0 = sim_time + self.list_horizons[0]
        t1 = sim_time + self.list_horizons[1]
        cplan_arrival_idle_dict = self._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1)
        # beware: number_current_own_vehicles include idle vehicle
        number_idle_plus_customer_arrival = {k: v[0] for k, v in cplan_arrival_idle_dict.items()}
        vehicles_repo_to_zone = {k: len(v[1]) for k,v in cplan_arrival_idle_dict.items()}
        number_idle_vehicles = {k: len(v[2]) for k,v in cplan_arrival_idle_dict.items()}
        forecast_arrivals = self.zone_system._get_trip_forecasts("in", t0, t1, None)
        forecast_departures = self.zone_system._get_trip_forecasts("out", t0, t1, None)

        orig_zone_imbalances = []
        for zone in list_zones:
            orig_zone_imbalances.append(number_idle_plus_customer_arrival.get(zone, 0)
                                        + vehicles_repo_to_zone.get(zone, 0)
                                        + forecast_arrivals.get(zone, 0)
                                        - forecast_departures.get(zone, 0)
                                        )

        # 2) set up and solve optimization problem
        # ----------------------------------------

        # output for debug
        # output_dir = self.fleetctrl.dir_names[G_DIR_OUTPUT]
        # output_dir = None
        if self.solver_key == "Gurobi":
            idle_count = [number_idle_vehicles.get(zone, 0) for zone in list_zones]
            alpha_od = self.reposition_g(np.array(orig_zone_imbalances), idle_count)
        else:
            raise NotImplementedError(f"Optimization problem in {self.__class__.__name__} only implemented for Gurobi!")

        # 3) creating vehicle trip assignments out of alpha_od
        # ----------------------------------------------------
        list_veh_with_changes = []
        od_reposition_trips = []
        for x in zip(*np.where(alpha_od != 0)):
            od_reposition_trips.extend([x] * int(alpha_od[x]))
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

    def _od_to_veh_plan_assignment(self, sim_time, origin_zone_id, destination_zone_id, list_veh_to_consider,
                                   destination_node=None, lock = True):
        random.seed(sim_time)
        destination_centroid = self.zone_system.get_random_centroid_node(destination_zone_id)
        dest_pt = (destination_centroid, None, None)
        # Get distance of all vehicles to the zone centroid and select vehicle with minimum distance
        vehicle_distances = np.array([self.routing_engine.return_travel_costs_1to1(v.pos, dest_pt)[2]
                                      for v in list_veh_to_consider])
        veh_obj = list_veh_to_consider[vehicle_distances.argmin()]
        LOG.info("repositioning {} to zone {} with centroid {}".format(veh_obj.vid, destination_zone_id,
                                                                       destination_node))

        # Send the min-distance vehicle to random point inside the destination zone
        destination_node = self._get_random_reachable_node(veh_obj.pos, destination_zone_id)
        ps = RoutingTargetPlanStop((destination_node, None, None), locked=lock)
        veh_plan = self.fleetctrl.veh_plans[veh_obj.vid]
        veh_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
        if lock:
            self.fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
        self.fleetctrl.assign_vehicle_plan(veh_obj, veh_plan, sim_time)
        return [veh_obj]

    def _get_random_reachable_node(self, point, zone_id):
        """ Get random reachable node from provided point"""

        zone_nodes = self.zone_system.get_all_nodes_in_zone(zone_id)
        random.shuffle(zone_nodes)
        travel_time = np.inf
        try:
            while np.isinf(travel_time) and len(zone_nodes) > 0:
                node = zone_nodes.pop()
                travel_time = self.routing_engine.return_travel_costs_1to1(point, (node, None, None))[0]
            return node
        except IndexError:
            raise IndexError(f"No reachable node found in destination zone {zone_id} while repositioning "
                             f"from vehicle position{point}")

    def reposition_g(self, omegas, idle_vehicles):
        """ Calculates the the number of vehicles to be repositioned between zones using kernel functions

        This function determines the number of vehicles to be repositioned between zones based on the predicted vehicle
        surplus and deficits. These predictions are generally made for zones that are smaller than the region an available
        vehicle is able to serve within a certain maximal customer waiting time.
        The aim of this function is to find alpha_ij that balance the surplus and deficits (accounting for contributions
        from nearby zones).

        :param omegas:          a list or numpy array of size n for weight of predicted vehicle surplus/deficit
                                for each centroid (+ for surplus)
        :param idle_vehicles:   number of idle vehicles counted to each centroid
        :return: Tuple of
                - alpha_ij:     numpy array of shape (n,n) for the number of vehicles to be repositioned
                                from zone i to zone j. It does not contain number for the number of vehicle
                                driving into a zone, i.e. it has only positive values.
        """

        start_time = default_timer()
        method = "2-" + self.method if "2-" not in self.method else self.method
        two_step_delta_flow, two_step_kpi = _reposition_two_steps_g(self.zone_corr_matrix, self.distance_cost, omegas,
                                                                    idle_vehicles, TIME_LIMIT, method)
        if "2-" in self.method:
            return two_step_delta_flow
        else:
            timeout = TIME_LIMIT - (default_timer() - start_time)
            orginal_f2_objective = omegas.T.dot(self.zone_corr_matrix.dot(omegas))
            f2_min = two_step_kpi["Heat Map Objective"]
            scale = (orginal_f2_objective - f2_min, two_step_kpi["Distance Objective"])
            delta_flow_vars, delta_wt_vars, kpis = _general_reposition_g(self.zone_corr_matrix, self.distance_cost,
                                                                         omegas, (self.gamma, 1-self.gamma), scale,
                                                                         idle_vehicles, timeout, (f2_min, 0.0),
                                                                         self.method)
            return delta_flow_vars


def _add_rfrr_delta_omega_g(model, omegas, idle_vehicles):
    """ declares variables and puts constraints for the delta omega inside the model """

    mask_greater_zero = omegas > 0.0
    mask_less_zero = omegas < 0.0

    delta_wt_vars = np.zeros_like(omegas, dtype="object")

    delta_wt_vars[mask_greater_zero] = [
        model.addVar(lb=-min(wt, I), ub=0, name="p_var_" + str(index), vtype=GRB.INTEGER)
        for index, wt, I in
        zip(np.argwhere(mask_greater_zero), omegas[mask_greater_zero],
            idle_vehicles[mask_greater_zero])]

    delta_wt_vars[mask_less_zero] = [model.addVar(lb=0, ub=-wt, name="n_var_" + str(index), vtype=GRB.INTEGER)
                                     for index, wt in zip(np.argwhere(mask_less_zero), omegas[mask_less_zero])]
    model.addConstr(np.sum(delta_wt_vars) == 0)
    return delta_wt_vars


def _add_rfrr_broken_delta_omega_g(model, omegas, idle_vehicles, only_positive):
    """ declares variables and puts constraints for the delta omega broken into positive and negative parts """

    positive_wt_sum = np.sum(idle_vehicles)

    delta_wt_vars_pos = np.zeros_like(omegas, dtype="object")
    delta_wt_vars_neg = np.zeros_like(omegas, dtype="object")
    sign_vars = [model.addVar(lb=0, ub=1, name="sign_{}".format(i), vtype=GRB.BINARY) for i in range(omegas.size)]

    delta_wt_vars_pos[:] = [model.addVar(lb=0, ub=positive_wt_sum, name="p_var_" + str(index), vtype=GRB.INTEGER)
                            for index in range(omegas.size)]

    if only_positive is False:
        delta_wt_vars_neg[:] = [model.addVar(lb=0, ub=I, name="n_var_" + str(index), vtype=GRB.INTEGER)
                                for index, I in enumerate(idle_vehicles)]
    else:
        delta_wt_vars_neg[:] = [model.addVar(lb=0, ub=min(max(0, wt), I), name="n_var_" + str(index), vtype=GRB.INTEGER)
                                for index, (I, wt) in enumerate(zip(idle_vehicles, omegas))]

    for i in range(omegas.size):
        model.addConstr(delta_wt_vars_pos[i] <= positive_wt_sum * sign_vars[i])
        model.addConstr(delta_wt_vars_neg[i] <= positive_wt_sum * (1 - sign_vars[i]))
    delta_wt_vars = delta_wt_vars_pos - delta_wt_vars_neg
    model.addConstr(np.sum(delta_wt_vars) == 0)
    return delta_wt_vars_pos, delta_wt_vars_neg


def _add_rfrr_flow_g(model, omegas, idle_vehicles, delta_wt_vars):
    """ declares variables and puts constraints for the flow variables """

    mask_greater_zero = omegas > 0.0
    index_greater_zero = np.argwhere(omegas > 0.0).flatten()
    mask_less_zero = omegas < 0.0
    index_less_zero = np.argwhere(omegas < 0.0).flatten()

    delta_flow_vars = np.zeros((omegas.size, omegas.size), dtype="object")
    for index_1, wt, I in zip(index_greater_zero, omegas[mask_greater_zero],
                              idle_vehicles[mask_greater_zero]):
        delta_flow_vars[index_1, index_less_zero] = [model.addVar(lb=0, ub=min(wt, I),
                                                                  name="flowvar_" + str(index_1) + "_" + str(index_2),
                                                                  vtype=GRB.INTEGER)
                                                     for index_2 in index_less_zero]
        model.addConstr(np.sum(delta_flow_vars[index_1, :]) == -delta_wt_vars[index_1])
        model.addConstr(np.sum(delta_flow_vars[index_1, :]) <= min(wt, I))

    # summation of vehicles driving into the deficiency regions should be equal to delta omega
    for col_index, delta_wt in zip(index_less_zero, delta_wt_vars[mask_less_zero]):
        model.addConstr(np.sum(delta_flow_vars[:, col_index]) == delta_wt)
    return delta_flow_vars


def _add_rfrr_broken_flow_g(model, omegas, idle_vehicles, delta_wt_vars_pos, delta_wt_vars_neg):
    """ declares variables and puts constraints for the flow variables with broken delta omegas """

    # Flow variable from all zones to all zones
    delta_flow_vars = np.zeros((omegas.size, omegas.size), dtype="object")

    for row_index, I in enumerate(idle_vehicles):
        delta_flow_vars[row_index, :] = [
            model.addVar(name="flowvar_" + str(row_index) + "_" + str(index_2), vtype=GRB.INTEGER)
            for index_2 in range(omegas.size)]
        model.addConstr(np.sum(delta_flow_vars[row_index, :]) == delta_wt_vars_neg[row_index])

    # summation of vehicles driving into the regions should be equal to delta omega
    for col_index, delta_wt_pos in enumerate(delta_wt_vars_pos):
        model.addConstr(np.sum(delta_flow_vars[:, col_index]) == delta_wt_pos)
        model.addConstr(delta_flow_vars[col_index, col_index] == 0)
    return delta_flow_vars


def _general_reposition_g(f2_matrix, cost_matrix, omegas, gamma, fact, idle_vehicles, timeout=None,
                          min_tuple=(0.0, 0.0), method="RFRR", return_pos_neg=False):
    model = gp.Model(name='solve heat map and distance flows')
    model.setParam("OutputFlag", 0)
    model.setParam("NonConvex", 2)
    if timeout:
        model.Params.timeLimit = timeout

    assert f2_matrix.shape[0] == len(omegas), "the size of F2 matrix and omegas does not match"

    omegas = np.array(omegas)
    idle_vehicles = np.array(idle_vehicles)
    if method == "RFRR":
        delta_wt_vars = _add_rfrr_delta_omega_g(model, omegas, idle_vehicles)
        delta_flow_vars = _add_rfrr_flow_g(model, omegas, idle_vehicles, delta_wt_vars)
    else:
        if method == "RFRRp":
            delta_wt_vars_pos, delta_wt_vars_neg = _add_rfrr_broken_delta_omega_g(model, omegas, idle_vehicles, True)
        else:
            delta_wt_vars_pos, delta_wt_vars_neg = _add_rfrr_broken_delta_omega_g(model, omegas, idle_vehicles, False)
        delta_flow_vars = _add_rfrr_broken_flow_g(model, omegas, idle_vehicles, delta_wt_vars_pos, delta_wt_vars_neg)
        delta_wt_vars = delta_wt_vars_pos - delta_wt_vars_neg

    # Objective function
    kde_expr_raw = (delta_wt_vars.T + 2 * omegas.T).dot(f2_matrix.dot(delta_wt_vars)) + omegas.T.dot(
        f2_matrix.dot(omegas))
    # c = 3 / (np.pi * bandwidth ** 2)
    # objective_kde = c ** 2 * kde_expr_raw
    objective_kde = kde_expr_raw
    scaled_kde = gamma[0] * (objective_kde - min_tuple[0]) / fact[0]

    objective_distance = np.sum(cost_matrix * delta_flow_vars)
    scaled_distance = gamma[1] * (objective_distance - min_tuple[1]) / fact[1]

    combined_objective = scaled_kde + scaled_distance
    model.setObjective(combined_objective, GRB.MINIMIZE)
    # optimize
    model.update()
    model.optimize()

    # retrieve solution
    if method == "RFRR":
        delta_wt_vars = np_set_gurobi_var_value(delta_wt_vars)
    else:
        delta_wt_vars_pos = np_set_gurobi_var_value(delta_wt_vars_pos)
        delta_wt_vars_neg = np_set_gurobi_var_value(delta_wt_vars_neg)
        delta_wt_vars = delta_wt_vars_pos - delta_wt_vars_neg
    delta_flow_vars = np_set_gurobi_var_value(delta_flow_vars)

    kpis = defaultdict(float)
    kpis["Heat Map Objective"] = objective_kde.getValue()
    kpis["Distance Objective"] = objective_distance.getValue()
    kpis["No. of Vehicles Repositioned"] = np.sum(delta_flow_vars)
    kpis["Scaled Heat Map Objective"] = scaled_kde.getValue()
    kpis["Scaled Distance Objective"] = scaled_distance.getValue()
    kpis["Original Heat Map Objective"] = omegas.T.dot(f2_matrix.dot(omegas))

    if return_pos_neg is True:
        return delta_flow_vars, delta_wt_vars_pos, delta_wt_vars_neg, kpis
    else:
        return delta_flow_vars, delta_wt_vars, kpis


def _reposition_two_steps_g(f2_matrix, cost_matrix, omegas, idle_vehicles, timeout=None, method="RFRR"):
    """ Calculates the the number of vehicles to be repositioned between zones using kernel functions and two steps

    :param f2_matrix:       the constant matrix calculated using calculate_f2_matrix method
    :param cost_matrix:     numpy (n,n)-array or list of lists for distance between the centroids of omegas
    :param omegas:          a list or numpy array of size n for weight of predicted vehicle surplus/deficit
                            for each centroid (+ for surplus)
    :param idle_vehicles:   number of idle vehicles counted to each centroid
    :param method:          Which formulation to use from "RFRR", "RFRRp" or "RFRRf"
    :param timeout:         Calculation timeout in seconds
    :return: Tuple of
            - alpha_ij:     numpy array of shape (n,n) for the number of vehicles to be repositioned
                            from zone i to zone j. It does not contain number for the number of vehicle
                            driving into a zone, i.e. it has only positive values.
            - Dict of Objectives:   Dictionary of objective values with keys
                                    1) Heat Map Objective
                                    2) Distance Objective
                                    3) No. of Vehicles Repositioned
                                    4) Scaled Heat Map Objective
                                    5) Scaled Distance Objective
                                    6) Original Heat Map Objective
    """

    model = gp.Model(name='solve heat map and distance flows')
    model.setParam("OutputFlag", 0)
    model.setParam("NonConvex", 2)
    start_time = default_timer()
    if timeout:
        model.Params.timeLimit = timeout

    assert f2_matrix.shape[0] == len(omegas), "the size of correlation matrix and omegas does not match"

    omegas = np.array(omegas)
    idle_vehicles = np.array(idle_vehicles)

    if method == "2-RFRR":
        delta_wt_vars = _add_rfrr_delta_omega_g(model, omegas, idle_vehicles)
    else:
        if method == "2-RFRRp":
            delta_wt_vars_pos, delta_wt_vars_neg = _add_rfrr_broken_delta_omega_g(model, omegas, idle_vehicles, True)
        else:
            delta_wt_vars_pos, delta_wt_vars_neg = _add_rfrr_broken_delta_omega_g(model, omegas, idle_vehicles, False)
        delta_wt_vars = delta_wt_vars_pos - delta_wt_vars_neg

    # First solve the model for delta omegas only

    # Heat objectives
    kde_expr_raw = (delta_wt_vars.T + 2 * omegas.T).dot(f2_matrix.dot(delta_wt_vars)) + omegas.T.dot(
        f2_matrix.dot(omegas))
    # c = 3 / (np.pi * bandwidth ** 2)
    # objective_kde = c ** 2 * kde_expr_raw
    objective_kde = kde_expr_raw

    model.setObjective(objective_kde, GRB.MINIMIZE)
    # optimize
    model.update()
    model.optimize()

    kpis = defaultdict(float)
    kpis["Heat Map Objective"] = objective_kde.getValue()
    kpis["Scaled Heat Map Objective"] = objective_kde.getValue()

    # Now solve for the flow variables

    model_flow = gp.Model(name='solve heat map and distance flows')
    model_flow.setParam("OutputFlag", 0)
    model_flow.setParam("NonConvex", 2)
    second_timeout = timeout - (default_timer() - start_time)
    if second_timeout < 5:
        LOG.warning(f"Second step of method {method} has only remaining timeout {second_timeout}. "
                    f"Using 15 seconds for second step.")
        second_timeout = 15
    model_flow.Params.timeLimit = second_timeout

    if method == "2-RFRR":
        delta_wt_vars = np_set_gurobi_var_value(delta_wt_vars)
        delta_flow_vars = _add_rfrr_flow_g(model_flow, omegas, idle_vehicles, delta_wt_vars)
    else:
        delta_wt_vars_pos = np_set_gurobi_var_value(delta_wt_vars_pos)
        delta_wt_vars_neg = np_set_gurobi_var_value(delta_wt_vars_neg)
        delta_flow_vars = _add_rfrr_broken_flow_g(model_flow, omegas, idle_vehicles, delta_wt_vars_pos,
                                                 delta_wt_vars_neg)

    # Objective function

    objective_distance = np.sum(cost_matrix * delta_flow_vars)
    model_flow.setObjective(objective_distance, GRB.MINIMIZE)
    # optimize
    model_flow.update()
    model_flow.optimize()
    delta_flow_vars = np_set_gurobi_var_value(delta_flow_vars)

    kpis["Distance Objective"] = objective_distance.getValue()
    kpis["No. of Vehicles Repositioned"] = np.sum(delta_flow_vars)
    kpis["Scaled Distance Objective"] = objective_distance.getValue()
    kpis["Original Heat Map Objective"] = omegas.T.dot(f2_matrix.dot(omegas))

    return delta_flow_vars, kpis