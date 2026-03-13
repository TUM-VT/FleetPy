from src.ml_gym.observers import AbstractObserver
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase


class SimTimeObserver(AbstractObserver):

    def observe(self, fleetpy_module):
        assert isinstance(fleetpy_module, RepositioningBase), "SimTimeObserver only works with RepositioningBase"
        return {"sim_time": fleetpy_module.sim_time}


class DemandForecastObserver(AbstractObserver):

    def observe(self, fleetpy_module):
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

    def observe(self, fleetpy_module):
        assert isinstance(fleetpy_module, RepositioningBase), "DemandForecastObserver only works with RepositioningBase"
        sim_time = fleetpy_module.sim_time
        list_zones = fleetpy_module.zone_system.get_all_zones()
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