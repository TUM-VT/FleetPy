from src.ml_gym.Observers import AbstractObserver
from src.fleetctrl.PoolingIRSOnly import PoolingInsertionHeuristicOnly
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.simulation.Vehicles import SimulationVehicle
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.routing.road.NetworkBase import NetworkBase


def _get_plan_distance(veh_obj: SimulationVehicle, veh_plan: VehiclePlan, network: NetworkBase):
    """Calculates total driving distance through all planned stops."""
    distance = 0.0
    last_pos = veh_obj.pos

    for stop in veh_plan.list_plan_stops:
        pos = stop.get_pos()
        if pos != last_pos:
            distance += network.return_travel_costs_1to1(last_pos, pos)[2]
            last_pos = pos
    return distance


def _get_plan_user_times(veh_plan: VehiclePlan, fleet_controller: FleetControlBase):
    """Calculates total request-to-dropoff time for all passengers in the plan."""
    sum_user_times = 0.0
    for rid, boarding_info_list in veh_plan.pax_info.items():
        rq_time = fleet_controller.rq_dict[rid].rq_time
        drop_off_time = boarding_info_list[1]
        sum_user_times += drop_off_time - rq_time
    return sum_user_times


class ImmediateInsertionCandidatePlansObserver(AbstractObserver):
    """Observes insertion candidates, plan distances, user times, pickup delays, and optionally route zones."""

    def __init__(self, od_route_zones: bool = False, additional_plan_info: bool = False):
        super().__init__()
        self.od_route_zones = od_route_zones
        self.additional_plan_info = additional_plan_info

    def observe(self, fleetpy_module, **kwargs):
        assert isinstance(fleetpy_module, PoolingInsertionHeuristicOnly), "Requires PoolingInsertionHeuristicOnly"
        self.extract_modules(fleetpy_module)

        sim_time = self.fleet_controller.sim_time
        candidate_plans: dict[int, dict[str, tuple[VehiclePlan, float, float]]] = kwargs["candidate_plans"]
        prq: PlanRequest = kwargs["plan_req"]
        req_id = prq.get_rid_struct()

        route_zones = None
        if self.od_route_zones and self.zone_system is not None:
            route = self.network.return_best_route_1to1(prq.o_pos, prq.d_pos)
            route_zones = self.zone_system.get_zones_from_route(route or [])

        original_plan_info = {}
        all_plans = {}

        for rid, plans in candidate_plans.items():
            rq_plans = {"req_origin": prq.o_pos, "req_dest": prq.d_pos, "vid": [], "delta_cfv": [], "fare": [],
                        "pickup_delay": []}

            if self.additional_plan_info:
                rq_plans.update({"plan_distance": [], "delta_plan_distance": [], "plan_user_times": [], "delta_plan_user_times": []})

            if self.zone_system is not None:
                rq_plans["req_origin_zone"] = self.zone_system.get_zone_from_pos(prq.o_pos)
                rq_plans["req_dest_zone"] = self.zone_system.get_zone_from_pos(prq.d_pos)
                if route_zones is not None:
                    rq_plans["req_route_zones"] = route_zones.copy()

            for vid, (veh_plan, delta_cfv, fare) in plans.items():
                pickup_info = veh_plan.get_pax_info(req_id)
                pickup_delay = pickup_info[0] - prq.rq_time if pickup_info else None

                rq_plans["vid"].append(vid)
                rq_plans["delta_cfv"].append(delta_cfv)
                rq_plans["fare"].append(fare)
                rq_plans["pickup_delay"].append(pickup_delay)

                if self.additional_plan_info:
                    veh_obj = self.fleet_controller.sim_vehicles[vid]

                    if vid not in original_plan_info:
                        original_plan = self.fleet_controller.veh_plans[vid]
                        original_plan_info[vid] = (_get_plan_distance(veh_obj, original_plan, self.network),
                                                   _get_plan_user_times(original_plan, self.fleet_controller))

                    original_distance, original_user_times = original_plan_info[vid]
                    plan_distance = _get_plan_distance(veh_obj, veh_plan, self.network)
                    plan_user_times = _get_plan_user_times(veh_plan, self.fleet_controller)

                    rq_plans["plan_distance"].append(plan_distance)
                    rq_plans["delta_plan_distance"].append(plan_distance - original_distance)
                    rq_plans["plan_user_times"].append(plan_user_times)
                    rq_plans["delta_plan_user_times"].append(plan_user_times - original_user_times)

            all_plans[rid] = rq_plans

        return {"immediate_plans": all_plans, "sim_time": sim_time}


class CurrentIdleVehicleObserver(AbstractObserver):
    """Observes current idle vehicle positions, optionally aggregated by zone."""

    def __init__(self, aggregate_by_zone: bool = False):
        super().__init__()
        self.aggregate_by_zone = aggregate_by_zone

    def observe(self, fleetpy_module, event, **kwargs):
        self.extract_modules(fleetpy_module, event)
        current_idle_positions = {}
        for vid, veh_plan in self.fleet_controller.veh_plans.items():
            veh_obj = self.fleet_controller.sim_vehicles[vid]
            if veh_plan.list_plan_stops or veh_obj.assigned_route or veh_obj.pax:
                continue
            current_idle_positions[vid] = veh_obj.pos
        observation = {"vehicle_to_current_idle_position": current_idle_positions}

        if self.aggregate_by_zone:
            zone_to_current_idle_vehicles = {}
            for vid, pos in current_idle_positions.items():
                zone = self.zone_system.get_zone_from_pos(pos)
                if zone is None or zone < 0:
                    continue
                zone_to_current_idle_vehicles.setdefault(zone, []).append(vid)
            observation["zone_to_current_idle_vehicles"] = zone_to_current_idle_vehicles

        return observation

