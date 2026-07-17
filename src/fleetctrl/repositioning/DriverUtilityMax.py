from __future__ import annotations
import numpy as np
import pandas as pd
import os
import random
import traceback
import logging
from collections import Counter
from typing import TYPE_CHECKING
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.fleetctrl.planning.VehiclePlan import RoutingTargetPlanStop
from src.misc.globals import *

INPUT_PARAMETERS_DriverUtilityMax = {
    "doc" :     """ this class implements the repositioning strategy where drivers choose if and where to reposition to based on expected net revenue
    """,
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [G_V_ALPHA_D],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class DriverUtilityMax(RepositioningBase):
    def __init__(self, fleetctrl, operator_attributes, dir_names, solver = "Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver)
        self.idle_veh_dict = {} # key: vid; val: repos_determined_flag
        self.avg_demand = pd.read_csv(os.path.join(dir_names[G_DIR_DEMAND], f"avg_hourly_od_matrix.csv"), index_col=0).to_numpy()/3600 # trips / s 
        self.beta_f = 2.5/1000 # per m fare rate
        self.C = 0.25 # commission percentage
        self.alpha_d = self.fleetctrl.alpha_d # average driver operating cost euro / s # 0.25/60 
    
    def _load_zone_system(self, operator_attributes, dir_names):

        return super()._load_zone_system(operator_attributes, dir_names)

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """This method determines and creates new repositioning plans. The repositioning plans are directly assigned
        to the vehicles.
        In order to allow further database processes, the vids of vehicles with new plans are returned.

        :param sim_time: current simulation time
        :param lock: indicates if vehplans should be locked
        :return: list[vid] of vehicles with changed plans
        """

        # add any new idle vehicles and remove any vehicles that have found a match
        for vid, current_veh_plan in self.fleetctrl.veh_plans.items():        
            #if not current_veh_plan.list_plan_stops:
            if self.fleetctrl.sim_vehicles[vid].status == VRL_STATES.IDLE:
                if vid not in self.idle_veh_dict.keys():
                    self.idle_veh_dict[vid] = False
            
            elif vid in self.idle_veh_dict.keys():
                if self.fleetctrl.sim_vehicles[vid].status == VRL_STATES.BOARDING:
                    del self.idle_veh_dict[vid]
            
        # assign repositioning plans for unhandled vehicles

        utility_matrix = None
        zone_list = self.zone_system.get_all_zones()

        for vid in self.idle_veh_dict.keys():
            if self.idle_veh_dict[vid] == False:
                if utility_matrix is None: # only calculate utility matrix if needed and once for each time step
                    utility_matrix = self.compute_utility_matrix(sim_time)
                current_zone = self.zone_system.get_zone_from_pos(self.fleetctrl.sim_vehicles[vid].pos)
                i = zone_list.index(current_zone)
                best_zone = zone_list[np.nanargmax(utility_matrix[i, :])]
                if best_zone != i:
                    self._od_to_veh_plan_assignment(sim_time,current_zone,best_zone,[self.fleetctrl.sim_vehicles[vid]])
                    # print('xx: Yes repos vid ' + str(vid) + ', from ' + str(i) + ' to zone ' + str(best_zone))
                else:
                    # print('xx: No repos vid ' + str(vid) + ', stay in same zone ' + str(i))
                    pass
                self.idle_veh_dict[vid] = True # mark vehicle as handled

    
    def compute_utility_matrix(self, sim_time):
        zone_list = self.zone_system.get_all_zones()
        n_zones = len(zone_list)
        travel_time = np.full((n_zones, n_zones), np.nan)
        travel_dist = np.full((n_zones, n_zones), np.nan)
    
        expected_waiting_time_dict = self.fleetctrl.compute_veh_waiting_time(sim_time)
        expected_waiting_time = np.array([expected_waiting_time_dict[z] for z in zone_list])
        for i, o in enumerate(zone_list): 
            for j, d in enumerate(zone_list): 
                tt, dist = self._get_od_zone_travel_info(sim_time,o,d)
                travel_time[i,j] = tt # seconds
                travel_dist[i,j] = dist # in m
        expected_trip_time = np.divide((travel_time*self.avg_demand).sum(axis=1),self.avg_demand.sum(axis=1),out=np.full(n_zones, np.inf),where=self.avg_demand.sum(axis=1) > 0)
        expected_trip_fare = np.divide((1 - self.C) * self.beta_f * (travel_dist * self.avg_demand).sum(axis=1), self.avg_demand.sum(axis=1), out=np.zeros(n_zones), where=self.avg_demand.sum(axis=1) > 0)
        t_bar_arr = np.tile(expected_trip_time, (n_zones, 1))
        f_bar_arr = np.tile(expected_trip_fare, (n_zones, 1))
        veh_wait_arr = np.tile(expected_waiting_time,(n_zones,1))
        t_repos = travel_time.copy()
        np.fill_diagonal(t_repos, 0.0)
        U_repos = (f_bar_arr - self.alpha_d*(veh_wait_arr+t_bar_arr+t_repos))

        return U_repos
