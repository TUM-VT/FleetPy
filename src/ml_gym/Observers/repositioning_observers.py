from src.ml_gym.Observers import AbstractObserver
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase

from src.misc.globals import *


class SimTimeObserver(AbstractObserver):
    """リポジションモジュールからシミュレーション時刻の読み込みReads the current simulation time from the repositioning module."""

    def observe(self, fleetpy_module):
        """Return the current simulation time.

        :param fleetpy_module: active RepositioningBase instance
        :return: {"sim_time": int}
        """
        assert isinstance(fleetpy_module, RepositioningBase), "引数のfleetpy_moduleがクラスRepositionBaseに一致しているかチェックSimTimeObserver only works with RepositioningBase"
        return {"sim_time": fleetpy_module.sim_time}


class DemandForecastObserver(AbstractObserver):
    """ゾーン毎の発着予測の読み込みReads zone-level trip departure and arrival forecasts over the repositioning horizon."""

    def observe(self, fleetpy_module):
        """次のタイムステップにおける予測トリップの出発地と目的地を返すReturn forecasted trip origins and destinations per zone for the next horizon window.

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