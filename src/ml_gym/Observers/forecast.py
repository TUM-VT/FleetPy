from src.ml_gym.Observers import AbstractObserver
from src.fleetctrl.forecast.ForecastZoneSystemBase import ForecastZoneSystemBase
from src.misc.globals import *


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
    def observe(self, fleetpy_module, event, **kwargs):

        self.extract_modules(fleetpy_module, event)
        zone_to_idle_vehicles = {}

        for veh_obj in self.fleet_controller.sim_vehicles:
            vid = veh_obj.vid
            # condition 1: no passenger
            if len(veh_obj.pax) != 0:
                continue
            # condition 2: no plan
            veh_plan = self.fleet_controller.veh_plans.get(vid)
            if veh_plan and veh_plan.list_plan_stops:
                continue
            # current position
            zone_id = self.zone_system.get_zone_from_pos(veh_obj.pos)
            if zone_id is None or zone_id < 0:
                continue

            zone_to_idle_vehicles[zone_id] = (zone_to_idle_vehicles.get(zone_id, 0) + 1)
        zone_to_idle_vehicles = dict(sorted(zone_to_idle_vehicles.items()))
        return {"zone_to_idle_vehicles": zone_to_idle_vehicles}


class FutureRequestsObserver(AbstractObserver):
    """ Forecasts the future requests using subclasses of ForecastZoneSystemBase """

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def observe(self, fleetpy_module, event, **kwargs):
        self.extract_modules(fleetpy_module, event)
        assert isinstance(self.zone_system, ForecastZoneSystemBase), (f"The observer {self} can only be used with zone "
                                                                      f"forecast. Current zone type: {self.zone_system}")
        sim_time = self.repositioning.sim_time
        trip_origin_forecasts = self.zone_system.get_trip_departure_forecasts(sim_time, sim_time + self.horizon)
        trips_dest_forecasts = self.zone_system.get_trip_arrival_forecasts(sim_time, sim_time + self.horizon)

        return {"zone_to_forecasted_departure": trip_origin_forecasts,
                "zone_to_forecasted_arrivals": trips_dest_forecasts}


class VehicleFutureIdlePosition(AbstractObserver):
    """Forecasts where and when vehicles become idle based on their current vehicle plans."""

    def __init__(self, aggregate_by_zone: bool = False):
        super().__init__()
        self.aggregate_by_zone = aggregate_by_zone

    def observe(self, fleetpy_module, event, **kwargs):
        self.extract_modules(fleetpy_module, event)
        sim_time = fleetpy_module.sim_time
        future_idle_positions = {}

        for vid, veh_plan in self.fleet_controller.veh_plans.items():
            if not veh_plan.list_plan_stops:
                continue
            last_stop = veh_plan.list_plan_stops[-1]
            idle_pos = last_stop.get_pos()
            idle_time = last_stop.get_planned_arrival_and_departure_time()[1]
            if idle_time is None:
                continue
            future_idle_positions[vid] = (idle_pos, max(sim_time, idle_time))
        observation = {"vehicle_to_future_idle_position": future_idle_positions}

        if self.aggregate_by_zone:
            zone_to_future_idle_vehicles = {}
            for vid, (pos, idle_time) in future_idle_positions.items():
                zone = self.zone_system.get_zone_from_pos(pos)
                if zone is None or zone < 0:
                    continue
                zone_to_future_idle_vehicles.setdefault(zone, []).append((vid, idle_time))
            observation["zone_to_future_idle_vehicles"] = zone_to_future_idle_vehicles
        return observation


class VehicleLocationObserver(AbstractObserver):
    """Forecasts vehicle network locations based on current vehicle plans."""

    def __init__(self, tau: int, delta_t: float, aggregate_by_zone: bool = False):
        super().__init__()
        self.tau = tau
        self.delta_t = delta_t
        self.aggregate_by_zone = aggregate_by_zone

    def _get_future_position(self, veh_obj, veh_plan, target_time, sim_time):
        current_pos = veh_obj.pos
        departure_time = sim_time

        for stop in veh_plan.list_plan_stops:
            arrival_time, next_departure = stop.get_planned_arrival_and_departure_time()
            destination = stop.get_pos()

            if arrival_time is None or next_departure is None:
                raise ValueError(f"Vehicle {veh_obj.vid} has a plan stop without scheduled times")

            if target_time < arrival_time:
                if target_time <= departure_time or current_pos == destination:
                    return current_pos

                route = self.network.return_best_route_1to1(current_pos, destination)
                route = route[1:] if route and route[0] == current_pos[0] else route
                travel_time = self.network.return_travel_costs_1to1(current_pos, destination)[1]

                if not route or travel_time <= 0:
                    raise ValueError(f"Invalid route or travel time for vehicle {veh_obj.vid}")

                fraction = (target_time - departure_time) / (arrival_time - departure_time)
                new_pos, _, _, _, _ = self.network.move_along_route(
                    route, current_pos, fraction * travel_time,
                    target_position=destination, new_sim_time=departure_time
                )
                return new_pos

            if target_time < next_departure:
                return destination

            current_pos = destination
            departure_time = next_departure

        return current_pos

    def observe(self, fleetpy_module, event, **kwargs):
        self.extract_modules(fleetpy_module, event)
        sim_time = fleetpy_module.sim_time
        future_positions = {k: {} for k in range(1, self.tau + 1)}

        for vid, veh_plan in self.fleet_controller.veh_plans.items():
            veh_obj = self.fleet_controller.sim_vehicles[vid]
            for k in future_positions:
                pos = self._get_future_position(veh_obj, veh_plan, sim_time + k * self.delta_t, sim_time)
                if self.aggregate_by_zone:
                    zone = self.zone_system.get_zone_from_pos(pos)
                    if zone is not None and zone >= 0:
                        future_positions[k][zone] = future_positions[k].get(zone, 0) + 1
                else:
                    future_positions[k][vid] = pos

        key = "zone_to_future_vehicle_counts" if self.aggregate_by_zone else "future_vehicle_positions"
        return {key: future_positions}







