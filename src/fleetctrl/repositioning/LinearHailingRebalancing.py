import random
import numpy as np
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.misc.globals import *
from src.fleetctrl.forecast.AggForecastZoning import AggForecastZoneSystem

# from IPython import embed

import os
import logging
from src.misc.globals import *
LOG = logging.getLogger(__name__)

WRITE_PROBLEM = False

INPUT_PARAMETERS_LinearHailingRebalancing = {
    "doc" :     """     Based on Vehicle Rebalancing for Mobility-on-Demand Systems with Ride-Sharing; Wallar, Alex; van der Zee, Menno; Alonso-Mora, Javier; Rus, Daniela (2018)
    supply and demand is counted and rebalancing trips are made to equalize
    supply and demand while minimzing driven distance
    parameter "op_weight_on_fc" weights how many vehicles are needed per forecasted demand per zone -> should be provided / calibrated
    """,
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_OP_REPO_FC_WEIGHT],
    "mandatory_modules": [],
    "optional_modules": []
}

class LinearHailingRebalancing(RepositioningBase):
    """
    Based on Vehicle Rebalancing for Mobility-on-Demand Systems with Ride-Sharing; Wallar, Alex; van der Zee, Menno; Alonso-Mora, Javier; Rus, Daniela (2018)
    supply and demand is counted and rebalancing trips are made to equalize.
    supply and demand while minimzing driven distance.
    Note that the parameter 'op_weight_on_fc' weights how many vehicles are needed per forecasted demand per zone and is subject to calibration.
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
            raise IOError("LinearHailingRebalancingrequires two time horizon values (start and end)!"
                          f"Set them in the {G_OP_REPO_TH_DEF} scenario parameter!")
        self.optimisation_timeout = 30 # TODO #
        self._weight_on_forecast = operator_attributes.get(G_OP_REPO_FC_WEIGHT, 0.05) # to scale the forecast by this factor (i.e. to approximate sharing)

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """This method determines and creates new repositioning plans. The repositioning plans are directly assigned
        to the vehicles. The repositioning algorithm can choose whether the generated respective plan stops are locked.
        In order to allow further database processes, the vids of vehicles with new plans are returned.

        :param sim_time: current simulation time
        :param lock: indicates if vehplans should be locked
        :return: list[vid] of vehicles with changed plans
        """
        self.zone_system.time_trigger(sim_time)
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
        for zone_id in self.zone_system.get_all_zones():
            self.record_df.loc[(sim_time, zone_id, t0, t1), "tot_fc_supply"] = max([demand_fc_dict.get(zone_id, 0) - supply_fc_dict.get(zone_id, 0), 0])
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

        needed_vehicles = {}
        for zone_id in list_zones:
            o_d_diff = demand_fc_dict.get(zone_id, 0) - number_current_own_vehicles.get(zone_id, 0) + number_idle_vehicles.get(zone_id, 0)
            if o_d_diff > 0:
                needed_vehicles[zone_id] = o_d_diff
        
        od_reposition_trips = []
        
        import gurobipy
        model_name = f"LinearHailingRebal: _optimization_gurobi {sim_time}"
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

            vars = {}
            d_vars = {}
            for o_zone, number_idle_vehs in number_idle_vehicles.items():
                o_zone_vars = []
                for d_zone, needed_supply in needed_vehicles.items():
                    tt = self._get_od_zone_travel_info(sim_time, o_zone, d_zone)[0]
                    arrival = sim_time + tt
                    if arrival > t1:
                        frac_in_hor = 0
                        continue
                    elif arrival < t0:
                        frac_in_hor = 1.0
                    else:
                        frac_in_hor = (arrival - t0)/(t1 - t0)
                    var_cost = - needed_supply * frac_in_hor
                    var_name = f"{o_zone}_{d_zone}"
                    vars[var_name] = model.addVar(vtype=gurobipy.GRB.INTEGER, obj=var_cost, name=var_name)
                    o_zone_vars.append(var_name)
                    try:
                        d_vars[d_zone][var_name] = frac_in_hor
                    except KeyError:
                        d_vars[d_zone] = {var_name : frac_in_hor}
                model.addConstr(
                    gurobipy.quicksum(vars[o] for o in o_zone_vars) <= number_idle_vehs)
            for d_zone, var_dict in d_vars.items(): #gurobipy.quicksum(vars[var] * frac for var, frac in var_dict.items())
                model.addConstr(
                    gurobipy.quicksum(vars[var] * frac for var, frac in var_dict.items()) <= self._weight_on_forecast * needed_vehicles[d_zone])

            for var in vars.values():
                model.addConstr(var >= 0)
            # update model
            #model.update()
            if WRITE_PROBLEM:
                model.write(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"opt_problem_matching_{sim_time}.lp"))
            
            # optimize
            model.setParam('MIPGap', 0.005)
            model.optimize()
            vals = model.getAttr('X', vars)
            #print(vals)
            for x in vals:
                v = vals[x]
                v = int(np.round(v))
                #print("res ", x, v)
                if v == 0:
                    continue
                od_l = x.split("_")
                od = (int(od_l[0]), int(od_l[1]))
                if od[0] == od[1]:
                    continue
                for _ in range(v):
                    od_reposition_trips.append( od )

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
    
