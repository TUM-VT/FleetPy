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

OPT_TIME_LIMIT = 30
WRITE_PROBLEM = False

INPUT_PARAMETERS_PavoneContinuous = {
    "doc" :     """     This class implements an adaption of the real-time rebalancing policy formulated in section 4.3 of
    Zhang, R.; Pavone, M. (2016): Control of robotic mobility-on-demand systems. A queueing-theoretical perspective.
    In: The International Journal of Robotics Research 35 (1-3), S. 186–203. DOI: 10.1177/0278364915581863.

    eq. (13)
    parameter "op_weight_on_fc" weights how many vehicles are needed per forecasted demand per zone -> not too relevant in this algorithm (can be used to account for pooling)
    
    in comparison to the modules PavoneFC/PavoneFCV2 this module implements continous variables resulting in a more stable but less accurate solution.
    """,
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_OP_REPO_FC_WEIGHT],
    "mandatory_modules": [],
    "optional_modules": []
}

class PavoneContinuous(RepositioningBase):
    """This class implements an adaption of the real-time rebalancing policy formulated in section 4.3 of
    Zhang, R.; Pavone, M. (2016): Control of robotic mobility-on-demand systems. A queueing-theoretical perspective.
    In: The International Journal of Robotics Research 35 (1-3), S. 186–203. DOI: 10.1177/0278364915581863.

    eq. (13)
    parameter "op_weight_on_fc" weights how many vehicles are needed per forecasted demand per zone -> not too relevant in this algorithm (can be used to account for pooling)
    
    in comparison to the modules PavoneFC/PavoneFCV2 this module implements continous variables resulting in a more stable but less accurate solution.
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
        self._weight_on_forecast = operator_attributes.get(G_OP_REPO_FC_WEIGHT, None) # to scale the forecast by this factor (i.e. to approximate sharing) 

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
        od_fc = self.zone_system.get_trip_od_forecasts(t0, t1, scale=self._weight_on_forecast)
        dep_rate_s = {zone_id : sum(od_fc.get(zone_id, {}).values()) for zone_id in list_zones}
        arr_rate_s = {zone_id : 0 for zone_id in list_zones}
        for o_zone_id, d_zone_dict in od_fc.items():
            for d_zone_id, trips in d_zone_dict.items():
                arr_rate_s[d_zone_id] += trips
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

        import gurobipy as grp
        
        model_name = f"pavone_new_{sim_time}"
        with grp.Env(empty=True) as env:
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

            m = grp.Model(model_name, env = env)

            m.setParam(grp.GRB.param.Threads, self.fleetctrl.n_cpu)
            m.setParam('TimeLimit', OPT_TIME_LIMIT)
            
            vars = {}
            from_vars = {o_zone : {} for o_zone in list_zones}
            to_vars = {d_zone : {} for d_zone in list_zones}
            for o_zone in list_zones:
                for d_zone in list_zones:
                    if o_zone == d_zone:
                        continue
                    tt, _ = self._get_od_zone_travel_info(sim_time, o_zone, d_zone)
                    var_name = f'{o_zone}_{d_zone}'
                    vars[var_name] = m.addVar(vtype=grp.GRB.CONTINUOUS, obj=tt, name=var_name)
                    m.addConstr(vars[var_name] >= 0, name=f"lb_{var_name}")
                    from_vars[o_zone][var_name] = 1
                    to_vars[d_zone][var_name] = 1
            
            LOG.info(f"input at time {sim_time}:")
            for zone in list_zones:
                LOG.info(f"zone {zone}: dep fc {dep_rate_s[zone]} arr fc {arr_rate_s[zone]} idle {number_idle_vehicles.get(zone, 0)}  mean {total_idle_vehicles/len(list_zones)} ")        
            for zone in list_zones:
                # m.addConstr(grp.quicksum(vars[var_name] for var_name in to_vars[zone].keys()) - grp.quicksum(vars[var_name] for var_name in from_vars[zone].keys()) ==\
                #     arr_rate_s[zone] - dep_rate_s[zone] + total_idle_vehicles/len(list_zones) - number_idle_vehicles.get(zone, 0), name=f"imbalance_{zone}")
                m.addConstr(grp.quicksum(vars[var_name] for var_name in to_vars[zone].keys()) +\
                    arr_rate_s[zone]  +  number_idle_vehicles.get(zone, 0) ==\
                        grp.quicksum(vars[var_name] for var_name in from_vars[zone].keys()) +\
                        dep_rate_s[zone] + total_idle_vehicles/len(list_zones), name=f"imbalance_{zone}")

            if WRITE_PROBLEM:
                m.write(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"opt_problem_matching_{sim_time}.lp"))
                
            m.optimize()

            # retrieve solution

            vals = m.getAttr('X', vars)
            #print(vals)
            od_reposition_trips = []
            LOG.debug("ODs with rebalancing vehicles:")
            for x in vals:
                v = vals[x]
                LOG.debug(f"{v}, {x}")
                v = int(np.round(v))
                if v == 0:
                    continue
                o_zone, d_zone = x.split("_")
                o_zone = int(o_zone)
                d_zone = int(d_zone)
                LOG.info(f"repositioning {v} vehicles from zone {o_zone} to zone {d_zone}")
                for _ in range(v):
                    od_reposition_trips.append((o_zone, d_zone))

        list_veh_with_changes = []
        if od_reposition_trips:
            # create assignments
            # ------------------
            random.seed(sim_time)
            random.shuffle(od_reposition_trips)
            for (origin_zone_id, destination_zone_id) in od_reposition_trips:
                list_idle_veh = cplan_arrival_idle_dict[origin_zone_id][2]
                if len(list_idle_veh) == 0:
                    LOG.warning(f"No idle vehicles available for repositioning from zone {origin_zone_id} to zone {destination_zone_id}!")
                    continue
                LOG.info(f"repo from {origin_zone_id} to {destination_zone_id}")
                if origin_zone_id == destination_zone_id:
                    rand_veh = random.choice(list_idle_veh)
                    cplan_arrival_idle_dict[origin_zone_id][2].remove(rand_veh)
                    continue
                list_veh_obj_with_repos = self._od_to_veh_plan_assignment(sim_time, origin_zone_id,
                                                                          destination_zone_id, list_idle_veh, lock=lock)
                list_veh_with_changes.extend([veh_obj.vid for veh_obj in list_veh_obj_with_repos])
                for veh_obj in list_veh_obj_with_repos:
                    cplan_arrival_idle_dict[origin_zone_id][2].remove(veh_obj)
        return list_veh_with_changes

