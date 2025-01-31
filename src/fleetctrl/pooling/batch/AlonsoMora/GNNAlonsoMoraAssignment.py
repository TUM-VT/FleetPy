import pickle
import os
from typing import Dict, List, Callable, Tuple

from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment import AlonsoMoraAssignment
from src.misc.globals import G_DIR_TRAIN, G_DIR_OUTPUT, G_TRAIN_RR_FILE, G_TRAIN_RV_FILE, G_TRAIN_ASSIGNMENTS_FILE, \
    G_TRAIN_INIT_ASSIGNMENTS_FILE, G_TRAIN_FEATURE_TRAVEL_TIME, G_TRAIN_NODE_FEATURES_FILE
from src.routing.NetworkBase import NetworkBase
from src.simulation.Legs import VehicleRouteLeg
from src.simulation.Vehicles import SimulationVehicle


class GNNAlonsoMoraAssignment(AlonsoMoraAssignment):
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
        # TODO explore attributes for features
        # TODO save top 10 optimized assignments

    def write_train_data(self, sim_time):
        dir_path = os.path.join(self.train_data_path, str(sim_time))
        os.makedirs(dir_path, exist_ok=True)
        train_data = self.get_train_data()
        for name, data in train_data:
            path = os.path.join(dir_path, f'{name}.pkl')
            self.write_pickle(path, data)

    def get_train_data(self) -> List[Tuple[int, dict]]:
        train_data = [(G_TRAIN_RR_FILE, self.get_rr_graph_with_features()),
                      (G_TRAIN_RV_FILE, self.get_r2v_graph_with_features()),
                      (G_TRAIN_ASSIGNMENTS_FILE, self.optimisation_solutions),
                      (G_TRAIN_INIT_ASSIGNMENTS_FILE, self.current_assignments),
                      (G_TRAIN_NODE_FEATURES_FILE, self.get_node_features())]
        return train_data

    def get_node_features(self):
        req_features = self.get_req_features()
        veh_features = self.get_veh_features()
        return {**req_features, **veh_features}

    def get_veh_features(self):
        # TODO complete
        veh_features = {vid: {'node_type': 'vehicle', 'pos': vehicle.pos, 'type': vehicle.veh_type,
                              'status': vehicle.status.display_name,
                              'fix_cost': vehicle.daily_fix_cost, 'var_cost': vehicle.distance_cost, 'soc': vehicle.soc}
                        for
                        vid, vehicle in self.veh_objs.items()}
        return veh_features

    def get_req_features(self):
        # TODO complete
        req_features = {
            # demand
            # revenue, profit
            # max_detour_time_factor
            # max_constant_detour_time
            rid: {'node_type': 'request', 'o_pos': req.o_pos[0], 'd_pos': req.d_pos[0], 'tw_pe': req.t_pu_earliest,
                  'tw_pl': req.t_pu_latest, 'tw_dl': req.t_do_latest,
                  'max_trip_time': req.max_trip_time, } for rid, req in self.active_requests.items()}
        return req_features

    def get_travel_time_r2v(self, rid, vid):
        v_pos = self.veh_objs[vid].pos
        r_pos = self.active_requests[rid].get_o_stop_info()[0]
        return {G_TRAIN_FEATURE_TRAVEL_TIME: self.routing_engine.return_travel_costs_1to1(v_pos, r_pos)}

    def get_travel_time_r2r(self, rid1, rid2):
        """get all travel times between the 6 combinations of rid1 and rid2 origins and destination positions"""
        req1, req2 = self.active_requests[rid1], self.active_requests[rid2]
        return {G_TRAIN_FEATURE_TRAVEL_TIME: [self.routing_engine.return_travel_costs_1to1(pos1, pos2) for pos1, pos2 in
                                              self.get_od_pairs(req1, req2)]}

    @staticmethod
    def get_od_pairs(req1, req2):
        return [(req1.o_pos, req1.d_pos), (req2.o_pos, req2.d_pos), (req1.o_pos, req2.d_pos),
                (req1.d_pos, req2.d_pos), (req1.d_pos, req2.o_pos), (req1.o_pos, req2.o_pos)]

    def get_rr_graph_with_features(self):
        return {(rid1, rid2): self.get_travel_time_r2r(rid1, rid2) for rid1, rid2 in self.rr}

    def get_r2v_graph_with_features(self):
        return {rid: {vid: self.get_travel_time_r2v(rid, vid) for vid in vids} for rid, vids in self.r2v.items()}

    @staticmethod
    def write_pickle(path, data: Dict):
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
