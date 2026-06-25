# Hallo im roman

import numpy as np
from src.ml_gym.Observers import AbstractObserver
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase

from src.misc.globals import *


class SimTimeObserver(AbstractObserver):
    """Reads the current simulation time from the repositioning module."""

    def observe(self, fleetpy_module):
        """Return the current simulation time.

        :param fleetpy_module: active RepositioningBase instance
        :return: {"sim_time": int}
        """
        assert isinstance(fleetpy_module, RepositioningBase), "SimTimeObserver only works with RepositioningBase"
        return {"sim_time": fleetpy_module.sim_time}


class DemandForecastObserver(AbstractObserver):
    """Reads zone-level trip departure and arrival forecasts over the repositioning horizon."""

    def observe(self, fleetpy_module):
        """Return forecasted trip origins and destinations per zone for the next horizon window.

        :param fleetpy_module: active RepositioningBase instance
        :return: {
            "zone_to_fc_rq_origins": dict[zone_id -> forecasted departures],
            "zone_to_fc_rq_destinations": dict[zone_id -> forecasted arrivals]
        }
        """
        assert isinstance(fleetpy_module, RepositioningBase), "DemandForecastObserver only works with RepositioningBase"
        sim_time = fleetpy_module.sim_time
        list_zones = fleetpy_module.zone_system.get_all_zones()
        t0 = sim_time + fleetpy_module.list_horizons[0]
        t1 = sim_time + fleetpy_module.list_horizons[1]
        print("observe_demand_forecast - t0: ", fleetpy_module, fleetpy_module.zone_system)
        dep_rate_s = fleetpy_module.zone_system.get_trip_departure_forecasts(t0, t1)
        arr_rate_s = fleetpy_module.zone_system.get_trip_arrival_forecasts(t0, t1)

        return {"zone_to_fc_rq_origins": dep_rate_s,
                "zone_to_fc_rq_destinations": arr_rate_s}

class ZoneBasedVehicleStatesObserver(AbstractObserver):
    """Reads per-zone vehicle counts (idle, repositioning, overall available) over the repositioning horizon."""

    def observe(self, fleetpy_module):
        """Return per-zone vehicle counts for idle, repositioning, and overall available vehicles.

        :param fleetpy_module: active RepositioningBase instance
        :return: {
            "zone_to_idle_vehilces": dict[zone_id -> number of idle vehicles],
            "zone_to_overall_available_vehilces": dict[zone_id -> total vehicles expected in zone],
            "zone_to_current_repositioning_vehicles": dict[zone_id -> vehicles currently repositioning to zone]
        }
        """
        assert isinstance(fleetpy_module, RepositioningBase), "DemandForecastObserver only works with RepositioningBase"
        sim_time = fleetpy_module.sim_time
        list_zones = fleetpy_module.zone_system.get_all_zones()
        # t0/t1 define the look-ahead window [t0, t1] passed to _get_current_veh_plan_arrivals_and_repo_idle_vehicles.
        # Vehicles whose plans place them inside a zone within this window are counted as "available";
        # vehicles arriving after t1 are excluded. A wider window (larger horizon offset) therefore
        # increases the available vehicle counts, potentially smoothing out short-term imbalances.
        t0 = sim_time + fleetpy_module.list_horizons[0]
        t1 = sim_time + fleetpy_module.list_horizons[1]

        cplan_arrival_idle_dict = fleetpy_module._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1)

        # compute imbalance values and constraints
        # ----------------------------------------
        vehicles_repo_to_zone = {k: len(v[1]) for k, v in cplan_arrival_idle_dict.items()}
        number_current_own_vehicles = {k: v[0] for k, v in cplan_arrival_idle_dict.items()}
        number_idle_vehicles = {k: len(v[2]) for k, v in cplan_arrival_idle_dict.items()}

        return {
            "zone_to_idle_vehilces": number_idle_vehicles,
            "zone_to_overall_available_vehilces": number_current_own_vehicles,
            "zone_to_current_repositioning_vehicles": vehicles_repo_to_zone
        }
        
class ZoneBasedCurrentDemandObserver(AbstractObserver):
    """Reads current zone-level trip departures and arrivals."""

    def observe(self, fleetpy_module: RepositioningBase):
        """Return current trip departures and arrivals per zone.

        :param fleetpy_module: active RepositioningBase instance
        :return: {
            "zone_based_passenger_demand": dict of (origin_zone_id, destination_zone_id) -> number_of_requests
        }
        """
        assert isinstance(fleetpy_module, RepositioningBase), "ZoneBasedCurrentDemandObserver only works with RepositioningBase"
        sim_time = fleetpy_module.sim_time
        list_zones = fleetpy_module.zone_system.get_all_zones()

        zone_based_demand = {}
        for rid, rq in fleetpy_module.fleetctrl.rq_dict.items():
            o_zone = fleetpy_module.zone_system.get_zone_from_pos(rq.get_o_stop_info()[0])
            d_zone = fleetpy_module.zone_system.get_zone_from_pos(rq.get_d_stop_info()[0])
            zone_based_demand[(o_zone, d_zone)] = zone_based_demand.get((o_zone, d_zone), 0) + 1

        return {"zone_based_passenger_demand": zone_based_demand}
    
class ZoneBasedCurrentVehicleStatesObserver(AbstractObserver):
    # TODO Takashi: update doc strings
    """Reads current vehicle states per zone, including idle vehicles and inter-zone movements."""

    def observe(self, fleetpy_module):
        """Return current vehicle distribution and flows between zones.

        :param fleetpy_module: active RepositioningBase instance
        :return: {
            "ozone_to_dzone_to_all_vehilces": dict[origin_zone_id -> dict[destination_zone_id -> number of vehicles driving from origin to destination],
            "ozone_to_dzone_to_occupied_vehilces": dict[origin_zone_id -> dict[destination_zone_id -> number of vehicles driving from origin to destination],
            "zone_to_number_idle_vehilces": dict[zone_id -> number of idle vehicles]
            }
        """
        assert isinstance(fleetpy_module, RepositioningBase), "DemandForecastObserver only works with RepositioningBase"
        sim_time = fleetpy_module.sim_time
        list_zones = fleetpy_module.zone_system.get_all_zones()

        ozone_to_dzone_to_all_vehicles={} # Number of vehicles driving from i to j (x_ij)
        ozone_to_dzone_to_occupied_vehicles={} # Number of vehicles driving from i to j with passengers (y_ij)
        zone_to_number_idle_vehicles={} #Number of idle vehicles in zone i (v_i)

        for vid, current_veh_plan in fleetpy_module.fleetctrl.veh_plans.items():
            veh_obj = fleetpy_module.fleetctrl.sim_vehicles[vid]
            # 1) idle vehicles
            if not current_veh_plan.list_plan_stops:
                zone_id = fleetpy_module.zone_system.get_zone_from_pos(veh_obj.pos)
                zone_to_number_idle_vehicles[zone_id] = zone_to_number_idle_vehicles.get(zone_id, 0) + 1
            else:
                last_ps = current_veh_plan.list_plan_stops[-1]
                veh_zone = fleetpy_module.zone_system.get_zone_from_pos(veh_obj.pos)
                veh_dest_zone = fleetpy_module.zone_system.get_zone_from_pos(last_ps.get_pos())
                if ozone_to_dzone_to_all_vehicles.get(veh_zone) is None:
                    ozone_to_dzone_to_all_vehicles[veh_zone] = {}
                ozone_to_dzone_to_all_vehicles[veh_zone][veh_dest_zone] = ozone_to_dzone_to_all_vehicles[veh_zone].get(veh_dest_zone, 0) + 1

                #if last_ps.get_state() != G_PLANSTOP_STATES.REPO_TARGET:
                if len(veh_obj.pax) > 0 :
                    if ozone_to_dzone_to_occupied_vehicles.get(veh_zone) is None:
                        ozone_to_dzone_to_occupied_vehicles[veh_zone] = {}
                    ozone_to_dzone_to_occupied_vehicles[veh_zone][veh_dest_zone] = ozone_to_dzone_to_occupied_vehicles[veh_zone].get(veh_dest_zone, 0) + 1

        return {
            "ozone_to_dzone_to_all_vehicles": ozone_to_dzone_to_all_vehicles,
            "ozone_to_dzone_to_occupied_vehicles": ozone_to_dzone_to_occupied_vehicles,
            "zone_to_number_idle_vehicles": zone_to_number_idle_vehicles
        }
    
class ZoneBasedTravelTimeObserver(AbstractObserver):
    # TODO Takashi: update doc strings
    """Reads zone-to-zone travel time and distance information."""

    def observe(self, fleetpy_module):
        """Return travel time and distance for all origin-destination zone pairs.

        :param fleetpy_module: active RepositioningBase instance
        :return: {
            "zone_to_zone_to_tt_dis": dict[
            (origin_zone_id, destination_zone_id) -> (travel_time, distance)
            ]
        }
        """
        assert isinstance(fleetpy_module, RepositioningBase), "DemandForecastObserver only works with RepositioningBase"
        sim_time = fleetpy_module.sim_time
        list_zones = fleetpy_module.zone_system.get_all_zones()

        zone_to_zone_to_tt_dis = {}
        for o_zone in list_zones:
            for d_zone in list_zones:
                tt, dis = fleetpy_module._get_od_zone_travel_info(sim_time, o_zone, d_zone)
                zone_to_zone_to_tt_dis[(o_zone, d_zone)] = (tt, dis)

        return {"zone_to_zone_to_tt_dis" : zone_to_zone_to_tt_dis}

class FutureDropoffObserver(AbstractObserver):
    """
    Reads x_i^(t+k): Number of occupied vehicles arriving at zone i during [t+k-1,t+k)
    k = 1,2,...,tau
    
    Counts the number of vehicles using following infomation in VehiclePlan.py:
    :return: {"zone_to_future_dropoffs": {1: dict[zone_id -> x_i^(t+1)], 2: dict[zone_id -> x_i^(t+2)],..., tau: dict[zone_id -> x_i^(t+tau)]}
        
        Dropoff(emptying arrival):
            A dropoff event is identified by a PlanStop where 
                get_list_alighting_rids() is non-empty.
            A dropoff event where all passengers alight is determined:
                - initialize the passenger count as the current number of passengers:
                    pax = len(veh_obj.pax)
                - At each PlanStop, update the passenger count using:
                    pax += stop.get_change_nr_pax()
                - If the stop has alighting passenger and the updated passenger count becomes zero, the stop is determined as the destination
        
        Arrival zone:
            Obtained from PlanStop.get_pos()
            Mapped to a zone ID
        
        Arrival time:
            Obtained from PlanStop.get_planned_arrival_and_departure_time()[0]
            Time intervals [t,t+1), [t+1,t+2), ..., [t+tau-1,t+tau) are applied for x^(t+1),x^(t+2)...,x^tau
            t is the time step index of RL, and simulation time is expressed by t * delta_t (delta_t is time interval of RL, e.g., 900 sec)
            tau is derived from op_repo_horizons[1] in const_cfg_manhattan_case_study.yaml
    """
    def __init__(self):
        self.tau = None
        
    def observe(self, fleetpy_module):
        if self.tau is None:
            horizon = fleetpy_module.list_horizons[1]
            resolution = fleetpy_module.fleetctrl.repo_time_step
            self.tau = int(horizon / resolution)
        sim_time = fleetpy_module.sim_time
        zone_system = fleetpy_module.zone_system
        delta_t = fleetpy_module.fleetctrl.repo_time_step
        
        # initialize dict when k is changed
        zone_to_future_dropoffs = {
            k: {} for k in range(1, self.tau + 1)
        }

        # iteration for each vehicle
        for vid, veh_plan in fleetpy_module.fleetctrl.veh_plans.items():
            veh_obj = fleetpy_module.fleetctrl.sim_vehicles[vid]
            
            # current no. of passengers
            pax = len(veh_obj.pax)
            
            # skip if no plan
            if not veh_plan.list_plan_stops:
                continue

            # check each PlanStop
            for stop in veh_plan.list_plan_stops:
                # arrival time
                arr_time = stop.get_planned_arrival_and_departure_time()[0]
                # time difference
                dt = arr_time - sim_time

                # skip if beyond the range
                if dt <= 0 or dt > self.tau * delta_t:
                    continue

                # calculate k
                k = int(dt / delta_t) + 1

                if k < 1 or k > self.tau:
                    continue

                # determine droppoff
                if stop.get_list_alighting_rids():
                    
                    # update no. of passengers
                    pax += stop.get_change_nr_pax()
                    
                    if pax == 0:
                        zone_id = zone_system.get_zone_from_pos(stop.get_pos())

                        if zone_id is None or zone_id < 0:
                            continue

                        zone_to_future_dropoffs[k][zone_id] = (
                            zone_to_future_dropoffs[k].get(zone_id, 0) + 1
                    )
        return {"zone_to_future_dropoffs": zone_to_future_dropoffs}
    
class FutureRepositioningCompletionObserver(AbstractObserver):
    """Reads y_i^(t+k): Number of repositioning vehicles arriving at zone i during [t+k-1,t+k)
        
        Counts the number of vehicles using following information in VehiclePlan.py:
        :return: {1: dict[zone_id -> y_i^(t+1)], 2: dict[zone_id -> y_i^(t+2)],..., tau: dict[zone_id -> y_i^(t+tau)]}

        Repositioning completion:
            A repositioning event is identified by a PlanStop where
                both get_list_boarding_rids() and get_list_alighting_rids() are empty
        
        Arrival zone:
            Obtained from PlanStop.get_pos()
            Mapped to a zone ID
        
        Arrival time:
            Obtained from PlanStop.get_planned_arrival_and_departure_time()[0]
            Time intervals [t,t+1), [t+1,t+2), ..., [t+tau-1,t+tau) are applied for x^(t+1),x^(t+2)...,x^tau
            t is the time step index of RL, and simulation time is expressed by t * delta_t (delta_t is time interval of RL, e.g., 900 sec)
            tau is derived from op_repo_horizons[1] in const_cfg_manhattan_case_study.yaml
    """
    def __init__(self):
        self.tau = None

    def observe(self, fleetpy_module):
        if self.tau is None:
            horizon = fleetpy_module.list_horizons[1]
            resolution = fleetpy_module.fleetctrl.repo_time_step
            self.tau = int(horizon / resolution)
        
        sim_time = fleetpy_module.sim_time
        zone_system = fleetpy_module.zone_system
        delta_t = fleetpy_module.fleetctrl.repo_time_step

        # create dict for each k
        zone_to_future_repo_completions = {
            k: {} for k in range(1, self.tau + 1)
        }

        # iteration for each vehicle
        for vid, veh_plan in fleetpy_module.fleetctrl.veh_plans.items():

            if not veh_plan.list_plan_stops:
                continue

            # each plan stop
            for stop in veh_plan.list_plan_stops:

                arr_time = stop.get_planned_arrival_and_departure_time()[0]
                dt = arr_time - sim_time

                if dt <= 0 or dt > self.tau * delta_t:
                    continue

                k = int(dt / delta_t) + 1

                if k < 1 or k > self.tau:
                    continue

                if (not stop.get_list_boarding_rids() and
                    not stop.get_list_alighting_rids()):

                    zone_id = zone_system.get_zone_from_pos(stop.get_pos())

                    if zone_id is None or zone_id < 0:
                        continue

                    zone_to_future_repo_completions[k][zone_id] = (
                        zone_to_future_repo_completions[k].get(zone_id, 0) + 1
                    )
                    
        return {"zone_to_future_repo_completions": zone_to_future_repo_completions}        

class IdleVehiclesObserver(AbstractObserver):
    """Reads z_i^t: Number of idle vehicles in zone i at time t
        Counts the number of idle vehicles using following information in VehiclePlan.py
        :return: {"zone_to_idle_vehicles": dict[zone_id -> z_i^t]}

        Idle vehicles:
            No onboard passengers -> len(veh_obj.pax)==0 
            No planned stop -> len(veh_obj.vehicle_plan.list_plan_stops) == 0
        
        Current zone:
            Obtained from current vehicle position -> veh_obj.pos
            Mapped to a zone ID

        Time:
            Current simulation time
    """
    def observe(self, fleetpy_module):
        zone_system = fleetpy_module.zone_system
        zone_to_idle_vehicles = {}

        for veh_obj in fleetpy_module.fleetctrl.sim_vehicles:
            vid = veh_obj.vid
            
            # condition 1: no passenger
            if len(veh_obj.pax) != 0:
                continue

            # condition 2: no plan
            veh_plan = fleetpy_module.fleetctrl.veh_plans.get(vid)
            if veh_plan and veh_plan.list_plan_stops:
                continue

            # current position
            zone_id = zone_system.get_zone_from_pos(veh_obj.pos)

            if zone_id is None or zone_id < 0:
                continue

            zone_to_idle_vehicles[zone_id] = (
                zone_to_idle_vehicles.get(zone_id, 0) + 1
            )

        return {"zone_to_idle_vehicles": zone_to_idle_vehicles}
    
class UnservedRequestsObserver(AbstractObserver):
    """Reads ru_i^t: Number of unserved requests departing from zone i at time t
        Counts the number of unserved requests using self.undecided_rq in demand.py
        :return: {"zone_to_unserved_requests": dict[zone_id -> number of unserved requests]}        
        T is the time step index of FleetPy simulation (e.g., 30 sec)
    """
    def __init__(self):
        self.tau = None
        self.cumulative_unserved = 0

    def observe(self, fleetpy_module):
        if self.tau is None:
            horizon = fleetpy_module.list_horizons[1]
            resolution = fleetpy_module.fleetctrl.repo_time_step
            self.tau = int(horizon / resolution)
        
        zone_system = fleetpy_module.zone_system
        sim_time = fleetpy_module.sim_time

        zone_to_unserved_requests = {}

        undecided_rq_list = fleetpy_module.demand.get_undecided_travelers(sim_time)

        for rid, rq in undecided_rq_list:
            zone_id = zone_system.get_zone_from_pos(rq.o_pos)

            if zone_id is None or zone_id < 0:
                continue
            zone_to_unserved_requests[zone_id] = (
                zone_to_unserved_requests.get(zone_id, 0) + 1                
            )
        
        self.cumulative_unserved += sum(zone_to_unserved_requests.values()) 

        return {
            "zone_to_unserved_requests": zone_to_unserved_requests,
            "cumulative_unserved": self.cumulative_unserved
            }
    
class FutureRequestsObserver(AbstractObserver):
    """Reads rf_i^(t+k): Number of forecasted requests departing from zone i during [t+k-1,t+k)
        Implement by improving class DemandForecastObserver
        t0 -> t, t+1,...t+tau-1
        t1 -> t+1, t+2,...t+tau
    """
    def __init__(self):
        self.tau = None

    def observe(self, fleetpy_module):
        if self.tau is None:
            horizon = fleetpy_module.list_horizons[1]
            resolution = fleetpy_module.fleetctrl.repo_time_step
            self.tau = int(horizon / resolution)

        zone_system = fleetpy_module.zone_system
        sim_time = fleetpy_module.sim_time
        delta_t = fleetpy_module.fleetctrl.repo_time_step

        # dict for each k
        zone_to_forecasted_requests = {
            k: {} for k in range(1, self.tau + 1)            
        }

        future_requests = fleetpy_module.demand.future_requests

        # future requests: {time: {rid, rq}}
        for t, rq_dict in future_requests.items():
            dt = t - sim_time

            if dt <= 0 or dt > self.tau * delta_t:
                continue

            k = int(dt / delta_t) + 1

            if k < 1 or k > self.tau:
                continue

            for rid, rq in rq_dict.items():
                zone_id = zone_system.get_zone_from_pos(rq.o_pos)

                if zone_id is None or zone_id < 0:
                    continue

                zone_to_forecasted_requests[k][zone_id] = (
                    zone_to_forecasted_requests[k].get(zone_id, 0) + 1
                )

        return {"zone_to_forecasted_requests": zone_to_forecasted_requests}
    
class TravelTimeMatrixObserver(AbstractObserver):
    def __init__(self):
        self.tt_matrix = None
    
    # travel time is calculated only once
    def observe(self, fleetpy_module):
        if self.tt_matrix is None:
            zone_system = fleetpy_module.zone_system
            zones = sorted(zone_system.get_all_zones())

            if -1 in zones:
                zones.remove(-1)
            
            Z = len(zones)
            tt_matrix = np.zeros((Z, Z))

            for i_idx, i in enumerate(zones):
                for j_idx, j in enumerate(zones):
                    tt, _ = fleetpy_module._get_od_zone_travel_info(
                        fleetpy_module.sim_time, i, j
                    )

                    tt_matrix[i_idx, j_idx] = tt
            
            self.tt_matrix = tt_matrix
            
        return {"tt_matrix": self.tt_matrix}