import logging

from src.fleetctrl.charging.ChargingBase import ChargingBase
from src.simulation.StationaryProcess import ChargingProcess
from src.fleetctrl.planning.VehiclePlan import ChargingPlanStop
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_ChargingBase = {
    "doc" :  """this strategy looks through all fleet vehicles an triggers charging tasks in case the soc within a planned route
            of a vehicle drops below a threshold (G_OP_APS_SOC)
            the closest charging station possible from this position is considered for charging
            in case multiple charging operators are present, the offer closest to postion is selected (also with depots) """,
    "inherit" : "ChargingBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_OP_APS_SOC],
    "mandatory_modules": [],
    "optional_modules": []
}


class ChargingThresholdPublicInfrastructure(ChargingBase):
    """ this strategy looks through all fleet vehicles an triggers charging tasks in case the soc within a planned route
    of a vehicle drops below a threshold (G_OP_APS_SOC)
    the closest charging station possible from this position is considered for charging
    in case multiple charging operators are present, the offer closest to postion is selected (also with depots)"""
    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, solver=solver)
        self.soc_threshold = operator_attributes.get(G_OP_APS_SOC, 0.1)

    def time_triggered_charging_processes(self, sim_time):
        LOG.debug("time triggered charging at {}".format(sim_time))
        for veh_obj in self.fleetctrl.sim_vehicles:
            # do not consider inactive vehicles
            if veh_obj.status in {VRL_STATES.OUT_OF_SERVICE, VRL_STATES.BLOCKED_INIT}:
                continue
            current_plan = self.fleetctrl.veh_plans[veh_obj.vid]
            is_charging_required = False
            last_time = sim_time
            last_pos = veh_obj.pos
            last_soc = veh_obj.soc
            if current_plan.list_plan_stops:
                last_pstop = current_plan.list_plan_stops[-1]
                LOG.debug(f"last ps of vid {veh_obj} : {last_pstop}")
                LOG.debug(f" state {last_pstop.get_state()} inactive {last_pstop.is_inactive()} arr dep soc {last_pstop.get_planned_arrival_and_departure_soc()}")
                if not last_pstop.get_state() == G_PLANSTOP_STATES.CHARGING and not last_pstop.is_inactive():
                    _, last_soc = last_pstop.get_planned_arrival_and_departure_soc()
                    if last_soc < self.soc_threshold:
                        charging_planned = False
                        for ps in current_plan.list_plan_stops: # TODO remove at some time but currently this results in bugs
                            if ps.get_state() == G_PLANSTOP_STATES.CHARGING:
                                charging_planned=True
                                LOG.debug(" -> but charging allready planned")
                                break
                        if not charging_planned:
                            _, last_time = last_pstop.get_planned_arrival_and_departure_time()
                            last_pos = last_pstop.get_pos()
                            is_charging_required = True
            elif veh_obj.soc < self.soc_threshold:
                is_charging_required = True

            if is_charging_required is True:
                LOG.debug(f"charging required for vehicle {veh_obj}")
                best_charging_poss = None
                best_ch_op = None
                for ch_op in self.all_charging_infra:
                    charging_possibilities = ch_op.get_charging_slots(sim_time, veh_obj, last_time, last_pos, last_soc, self.target_soc,
                                                                      max_number_charging_stations=self.n_stations_to_query, max_offers_per_station=self.n_offers_p_station)
                    LOG.debug(f"charging possiblilities of ch op {ch_op.ch_op_id}: {charging_possibilities}")
                    if len(charging_possibilities) > 0:
                        # pick those with earliest finish
                        ch_op_best = min(charging_possibilities, key=lambda x:x[3])
                        if best_charging_poss is None or ch_op_best[3] < best_charging_poss[3]:
                            best_charging_poss = ch_op_best
                            best_ch_op = ch_op
                if best_charging_poss is not None:
                    LOG.debug(f" -> best charging possibility: {best_charging_poss}")
                    (station_id, socket_id, possible_start_time, possible_end_time, desired_veh_soc, max_charging_power) = best_charging_poss
                    booking = best_ch_op.book_station(sim_time, veh_obj, station_id, socket_id, possible_start_time, possible_end_time)
                    station = best_ch_op.station_by_id[station_id]
                    start_time, end_time = booking.get_scheduled_start_end_times()
                    charging_task_id = (best_ch_op.ch_op_id, booking.id)
                    ps = ChargingPlanStop(station.pos, earliest_start_time=start_time, duration=end_time-start_time, charging_power=max_charging_power,
                                          charging_task_id=charging_task_id, locked=True)
                    current_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
                    self.fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
                    self.fleetctrl.assign_vehicle_plan(veh_obj, current_plan, sim_time, assigned_charging_task=(charging_task_id, booking))
