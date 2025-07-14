import pickle
import os
from typing import Dict, List, Callable
from collections import defaultdict

from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignmentOriginal import AlonsoMoraAssignmentOriginal
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment import AlonsoMoraAssignment
from src.misc.globals import *
from src.routing.NetworkBase import NetworkBase
from src.simulation.Legs import VehicleRouteLeg
from src.simulation.Vehicles import SimulationVehicle


class GNNAlonsoMoraAssignment(AlonsoMoraAssignmentOriginal):
    """Extension Alonso Mora Assignment Class for gathering training data for training machine learning models"""

    def __init__(self, fleetcontrol: FleetControlBase, routing_engine: NetworkBase, sim_time: int,
                 obj_function: Callable, operator_attributes: dict, optimisation_cores: int = 1, seed: int = 6061992,
                 veh_objs_to_build: Dict[int, SimulationVehicle] = {}):
        super().__init__(fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes, optimisation_cores,
                         seed, veh_objs_to_build)
        self.train_data_path = os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], G_DIR_TRAIN)

    def compute_new_vehicle_assignments(self, sim_time: int, vid_to_list_passed_VRLs: Dict[int, List[VehicleRouteLeg]],
                                        veh_objs_to_build: Dict[int, SimulationVehicle] = {},
                                        new_travel_times: bool = False, build_from_scratch: bool = False):
        super().compute_new_vehicle_assignments(sim_time, vid_to_list_passed_VRLs, veh_objs_to_build, new_travel_times,
                                                build_from_scratch)
        self.write_train_data(sim_time)
        # TODO save top 10 optimized assignments

    def write_train_data(self, sim_time: int):
        dir_path = os.path.join(self.train_data_path, str(sim_time))
        os.makedirs(dir_path, exist_ok=True)
        train_data = self.get_train_data()
        for name, data in train_data.items():
            path = os.path.join(dir_path, f'{name}.pkl')
            self.write_pickle(path, data)

    def get_train_data(self) -> dict[str, dict]:
        train_data = {
            G_TRAIN_RR_FILE: self.get_rr_graph_with_features(),
            G_TRAIN_VR_FILE: self.get_v2r_graph_with_features(),
            G_TRAIN_ASSIGNMENTS_FILE: self.optimisation_solutions,
            G_TRAIN_INIT_ASSIGNMENTS_FILE: self.current_assignments,
            G_TRAIN_REQ_FEATURES_FILE: self.get_req_features(),
            G_TRAIN_VEH_FEATURES_FILE: self.get_veh_features()
        }
        return train_data

    def get_veh_features(self):
        veh_features = {vid: {
            G_TRAIN_FEATURE_TYPE: vehicle.veh_type,
            G_TRAIN_FEATURE_STATUS: vehicle.status.value,
            G_TRAIN_FEATURE_SOC: vehicle.soc,
            G_TRAIN_FEATURE_V_POS_LAT: self.routing_engine.return_positions_lon_lat([vehicle.pos])[0][0],
            G_TRAIN_FEATURE_V_POS_LON: self.routing_engine.return_positions_lon_lat([vehicle.pos])[0][1]
        }
        for vid, vehicle in self.veh_objs.items()}
        return veh_features

    def get_req_features(self):
        req_features = {
            rid: {G_TRAIN_FEATURE_O_POS_LAT: self.routing_engine.return_positions_lon_lat([req.o_pos])[0][0],
                  G_TRAIN_FEATURE_O_POS_LON: self.routing_engine.return_positions_lon_lat([req.o_pos])[0][1],
                  G_TRAIN_FEATURE_D_POS_LAT: self.routing_engine.return_positions_lon_lat([req.d_pos])[0][0],
                  G_TRAIN_FEATURE_D_POS_LON: self.routing_engine.return_positions_lon_lat([req.d_pos])[0][1],
                  G_TRAIN_FEATURE_RQ_TIME: req.rq_time,
                  G_TRAIN_FEATURE_TW_PE: req.t_pu_earliest,
                  G_TRAIN_FEATURE_TW_PL: req.t_pu_latest, 
                  G_TRAIN_FEATURE_DIRECT_TT: req.init_direct_tt,
                  G_TRAIN_FEATURE_DIRECT_TD: req.init_direct_td,
                  G_TRAIN_FEATURE_MAX_TRIP_TIME: req.max_trip_time, 
                  G_TRAIN_FEATURE_STATUS: req.status}
            for rid, req in self.active_requests.items()}
        return req_features

    def get_travel_time_v2r(self, vid, rid):
        v_pos = self.veh_objs[vid].pos
        r_pos = self.active_requests[rid].get_o_stop_info()[0]
        return {key: val for key, val in
                zip([G_TRAIN_FEATURE_TRAVEL_COST, G_TRAIN_FEATURE_TRAVEL_TIME, G_TRAIN_FEATURE_TRAVEL_DIST],
                    self.routing_engine.return_travel_costs_1to1(v_pos, r_pos))}

    def get_travel_time_r2r(self, rid1, rid2):
        """get all travel times between the 6 combinations of rid1 and rid2 origins and destination positions"""
        req1, req2 = self.active_requests[rid1], self.active_requests[rid2]
        return {G_TRAIN_FEATURE_TRAVEL_COST: [{key: val for key, val in
                                               zip([G_TRAIN_FEATURE_TRAVEL_COST, G_TRAIN_FEATURE_TRAVEL_TIME,
                                                    G_TRAIN_FEATURE_TRAVEL_DIST],
                                                   self.routing_engine.return_travel_costs_1to1(pos1, pos2))} for
                                              pos1, pos2 in
                                              self.get_od_pool_pairs(req1, req2)]}

    @staticmethod
    def get_od_pool_pairs(req1, req2):
        return [(req1.d_pos, req2.o_pos), (req1.d_pos, req2.d_pos), (req1.o_pos, req2.o_pos), (req1.o_pos, req2.d_pos)]

    def get_rr_graph_with_features(self):
        rr_graph = defaultdict(dict)
        for rid1, rid2 in self.rr:
            rr_graph[rid1][rid2] = self.get_travel_time_r2r(rid1, rid2)
        return rr_graph

    def get_v2r_graph_with_features(self):
        return {vid: {rid: self.get_travel_time_v2r(vid, rid) for rid in rids} for vid, rids in self.v2r.items()}

    @staticmethod
    def write_pickle(path, data: Dict):
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
