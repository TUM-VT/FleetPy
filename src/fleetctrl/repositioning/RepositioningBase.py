from __future__ import annotations

import os
import random
from abc import abstractmethod, ABC
import logging
import numpy as np
import pandas as pd

from src.fleetctrl.planning.VehiclePlan import RoutingTargetPlanStop
from src.misc.globals import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
LOG = logging.getLogger(__name__)
LARGE_INT = 100000000

INPUT_PARAMETERS_RepositioningBase = {
    "doc" : "this class is the base class representing the repositioning module",
    "inherit" : None,
    "input_parameters_mandatory": [G_OP_REPO_TH_DEF, G_OP_REPO_TS],
    "input_parameters_optional": [
        G_OP_REPO_LOCK, G_OP_REPO_SR_F
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class RepositioningBase(ABC):
    def __init__(self, fleetctrl : FleetControlBase, operator_attributes : dict, dir_names : dict, solver : str="Gurobi"):
        """Initialization of repositioning class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param dir_names: directory structure dict
        :param solver: solver for optimization problems
        """
        self.fleetctrl = fleetctrl
        self.routing_engine = fleetctrl.routing_engine
        self.zone_system = fleetctrl.zones
        self.list_horizons = operator_attributes[G_OP_REPO_TH_DEF]
        self.lock_repo_assignments = operator_attributes.get(G_OP_REPO_LOCK, True)
        self.solver_key = solver
        self.output_dir = dir_names[G_DIR_OUTPUT]
        self.sim_time = None
        self.record_f = os.path.join(self.output_dir, f"4-{self.fleetctrl.op_id}_repositioning_info.csv")
        self.record_df_cols = ["sim_time", "zone_id", "horizon_start", "horizon_end",
                               "number_idle", "incoming", "incoming_repo", "tot_fc_demand", "tot_fc_supply"]
        self.record_df_index_cols = self.record_df_cols[:4]
        self.record_df = pd.DataFrame([], columns= self.record_df_cols)
        self.record_df.set_index(self.record_df_index_cols, inplace=True)
        self.zone_sharing_rates = {}
        if operator_attributes.get(G_OP_REPO_SR_F):
            zsn = self.zone_system.get_zone_system_name()
            sharing_rate_f = os.path.join(dir_names[G_DIR_FCTRL], "estimated_sharing_rates", zsn,
                                          operator_attributes[G_OP_REPO_SR_F])
            sr_df = pd.read_csv(sharing_rate_f)
            for o_zone, o_zone_sr_df in sr_df.groupby("o_zone"):
                avg_sr = (o_zone_sr_df["nr_rq"] * o_zone_sr_df["share_of_trips"]).sum()
                self.zone_sharing_rates[o_zone] = avg_sr
        # children classes:
        # - check of additionally required attributes from operator_attributes
        # - save these as class attributes

    def record_repo_stats(self):
        """This function can be called to save the repositioning stats.

        :return: None
        """
        if os.path.isfile(self.record_f):
            write_mode = "a"
            write_header = False
        else:
            write_mode = "w"
            write_header = True
        self.record_df.to_csv(self.record_f, index=True, mode=write_mode, header=write_header)
        self.record_df = pd.DataFrame([], columns=self.record_df_cols)
        self.record_df.set_index(self.record_df_index_cols, inplace=True)

    @abstractmethod
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
        return []
    
    def register_rejected_customer(self, planrequest, sim_time):
        """ this method is used to register and unserved request due to lack of available vehicles. The information can be stored internally
        and used for creating repositioning plans
        :param planrequest: plan request obj that has been rejected
        :param sim_time: simulation time"""
        pass

    def _get_demand_forecasts(self, t0, t1, aggregation_level=None):
        """This method creates a dictionary, which maps the zones to the expected demand between t0 and t1.

        :param t0: future time horizon start
        :param t1: future time horizon end
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :return: {}: zone -> forecast value; 0 should be default and does not have to be saved
        :rtype: dict
        """
        demand_forecasts = self.zone_system.get_trip_departure_forecasts(t0, t1, aggregation_level)
        for zone_id, val in demand_forecasts.items():
            self.record_df.loc[(self.sim_time, zone_id, t0, t1), "tot_fc_demand"] = val
        return demand_forecasts

    def _get_historic_arrival_forecasts(self, t0, t1, aggregation_level=None):
        """This method creates a dictionary, which maps the zones to the expected trip arrivals just based on historic
        data between t0 and t1.

        :param t0: future time horizon start
        :param t1: future time horizon end
        :param aggregation_level: spatial aggregation level, by default zone_id is used
        :return: {}: zone -> forecast value; 0 should be default and does not have to be saved
        :rtype: dict
        """
        arrival_forecasts = self.zone_system.get_trip_arrival_forecasts(t0, t1, aggregation_level)
        for zone_id, val in arrival_forecasts.items():
            self.record_df.loc[(self.sim_time, zone_id, t0, t1), "tot_fc_demand"] = val
        return arrival_forecasts

    def _get_current_veh_plan_arrivals_and_repo_idle_vehicles(self, t0, t1, node_level=False):
        """This method counts a vehicle as available if its final plan stop is supposed to end between t0 and t1. In
        that case, a vehicle is counted to the zone of its final position.
        If node_level = True, the supply forecast should be made on node instead of zone level, i.e. the return
        dictionary is {}: node -> (forecast value, list repo to zone vehicles, list idle vehicles)

        :param t0: future time horizon start
        :param t1: future time horizon end
        :param node_level: if True, final positions are not aggregated on zonal level, but remain on node level
        :return: {}: zone -> (forecast value, repo to zone vehicles, idle vehicles); (0,,[],[]) should be default
                                when zone not found
        :rtype: dict
        """
        zone_dict = {}
        for zone_id in self.zone_system.get_all_zones():
            zone_dict[zone_id] = [0, [], []]
        for vid, current_veh_plan in self.fleetctrl.veh_plans.items():
            veh_obj = self.fleetctrl.sim_vehicles[vid]
            # 1) idle vehicles
            if not current_veh_plan.list_plan_stops:
                zone_id = self.zone_system.get_zone_from_pos(veh_obj.pos)
                if zone_id >= 0:
                    zone_dict[zone_id][2].append(veh_obj)
                    zone_dict[zone_id][0] += 1
                else:
                    # TODO # think about mechanism to bring vehicles back into zone system!
                    LOG.warning("veh outside zonesystem! {}".format(veh_obj))
                    # if zone_dict.get(-1) is None:
                    #     zone_dict[-1] = [0, [], []]
                    # zone_dict[-1][2].append(veh_obj)
                    # zone_dict[-1][0] += 1
            else:
                last_ps = current_veh_plan.list_plan_stops[-1]
                if last_ps.get_state() != G_PLANSTOP_STATES.REPO_TARGET:
                    arr, dep = last_ps.get_planned_arrival_and_departure_time()
                    last_time = arr
                    if dep is not None:
                        last_time = dep
                    if t0 <= last_time < t1:
                        zone_id = self.zone_system.get_zone_from_pos(last_ps.get_pos())
                        if zone_id >= 0:
                            zone_dict[zone_id][0] += 1
                else:
                    # vehicles repositioning
                    zone_id = self.zone_system.get_zone_from_pos(last_ps.get_pos())
                    if zone_id >= 0:
                        zone_dict[zone_id][1].append(veh_obj)
        # record
        for zone_id, info_list in zone_dict.items():
            nr_normal_incoming = info_list[0]
            nr_repo_incoming = len(info_list[1])
            nr_idle = len(info_list[2])
            self.record_df.loc[(self.sim_time, zone_id, t0, t1), ["number_idle", "incoming", "incoming_repo"]] = \
                [nr_idle, nr_normal_incoming, nr_repo_incoming]
        return zone_dict

    def _get_od_zone_travel_info(self, sim_time, o_zone_id, d_zone_id):
        """This method returns OD travel times on zone level.

        :param o_zone_id: origin zone id
        :param d_zone_id: destination zone id
        :return: tt, dist
        """
        # v0) pick random node and compute route
        loop_iter = 0
        while True:
            o_pos = self.routing_engine.return_node_position(self.zone_system.get_random_centroid_node(o_zone_id))
            d_pos = self.routing_engine.return_node_position(self.zone_system.get_random_centroid_node(d_zone_id))
            if o_pos[0] >= 0 and d_pos[0] >= 0:
                route_info = self.routing_engine.return_travel_costs_1to1(o_pos, d_pos)
                if route_info:
                    return route_info[1], route_info[2]
                loop_iter += 1
                if loop_iter == 10:
                    break
            else:
                break
        # TODO # v1) think about the use of centroids!
        return LARGE_INT, LARGE_INT

    def _od_to_veh_plan_assignment(self, sim_time, origin_zone_id, destination_zone_id, list_veh_to_consider,
                                   destination_node=None, lock = True):
        """This method translates an OD repositioning assignment for one of the vehicles in list_veh_to_consider.
        This method adds a PlanStop at a random node in the destination zone, thereby calling
        the fleet control class to make all required data base entries.

        :param sim_time: current simulation time
        :param origin_zone_id: origin zone id
        :param destination_zone_id: destination zone id
        :param list_veh_to_consider: list of (idle/repositioning) simulation vehicle objects
        :param destination_node: if destination node is given, it will be prioritized over picking a random node in
                                    the destination zone
        :param lock: indicates if vehplan should be locked
        :return: list_veh_objs with new repositioning assignments
        """
        # v0) randomly pick vehicle, randomly pick destination node in destination zone
        random.seed(sim_time)
        veh_obj = random.choice(list_veh_to_consider)
        veh_plan = self.fleetctrl.veh_plans[veh_obj.vid]
        destination_node = self.zone_system.get_random_centroid_node(destination_zone_id)
        LOG.debug("repositioning {} to zone {} with centroid {}".format(veh_obj.vid, destination_zone_id,
                                                                        destination_node))
        if destination_node < 0:
            destination_node = self.zone_system.get_random_node(destination_zone_id)
        ps = RoutingTargetPlanStop((destination_node, None, None), locked=lock, planstop_state=G_PLANSTOP_STATES.REPO_TARGET)
        veh_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
        self.fleetctrl.assign_vehicle_plan(veh_obj, veh_plan, sim_time)
        if lock:
            self.fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
        return [veh_obj]

    def _compute_reachability_adjusted_zone_imbalances(self, sim_time, list_zone_imbalance_weights):
        """This method can be called to evaluate the imbalance densities before and after the repositioning method
        was called. This method works on zone level.

        :param sim_time: current simulation time
        :param list_zone_imbalance_weights: initial zone weights I_z'
        :return: K_z,z' * I_z'
        """
        np_imbalance = np.array(list_zone_imbalance_weights)
        return np.matmul(np_imbalance, self.zone_system.get_zone_correlation_matrix())

    def _return_zone_imbalance_np_array(self):
        return self.zone_system.get_zone_correlation_matrix()

    def _compute_reachability_adjusted_squared_zone_imbalance(self, sim_time, list_zone_imbalance_weights):
        """This method can be called to evaluate the imbalance densities before and after the repositioning method
        was called. This method works on zone level.

        :param sim_time: current simulation time
        :param list_zone_imbalance_weights: initial zone weights I_z'
        :return: [K_z,z' * I_z']**2
        """
        np_imbalance = np.array(list_zone_imbalance_weights)
        return np.matmul(np_imbalance, np.matmul(self.zone_system.get_squared_correlation_matrix(), np_imbalance))

    def _return_squared_zone_imbalance_np_array(self):
        return self.zone_system.get_squared_correlation_matrix()