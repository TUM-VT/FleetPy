from collections import defaultdict
import pickle
import os
from tqdm import tqdm

class DataProcessor:
    def __init__(self, train_data_dir, save_raw_dir, sim_duration, sim_step):
        self.train_data_dir = train_data_dir
        self.save_raw_dir = save_raw_dir
        self.sim_duration = sim_duration
        self.sim_step = sim_step

    def process_data(self):
        os.makedirs(f'{self.save_raw_dir}', exist_ok=True)
        all_data = []
        for timestep in tqdm(range(0, self.sim_duration, self.sim_step)):
            try:
                data = self.load_data(timestep)
                data = self.add_new_features(data)
                self.save_data(timestep, data)
                all_data.append(data)
            except Exception as e:
                print('Loading data for timestep', timestep, 'failed', e)
        return all_data

    def load_data(self, timestep):
        data = {}
        for file in os.scandir(f'{self.train_data_dir}/{timestep}/'):
            with open(file.path, 'rb') as f:
                data[file.name[:file.name.find('.')]] = pickle.load(f)
        return data

    def save_data(self, timestep, data):
        with open(f'{self.save_raw_dir}/{timestep}.pkl', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def add_new_features(self, data):
        self.n_requests = len(data['req_features'])
        self.add_node_degrees(data)
        nghbs = self.get_node_neighbors(data)
        self.add_common_nghbs(data, nghbs)
        self.add_clustering_coeffs(data, nghbs)
        self.add_assign_to_edges(data)
        return data

    def add_node_degrees(self, data):
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)

        # TODO add fn wrapper for loop(s)
        for node_ind_fn, graph in zip([self.node_rid, self.node_vid], [data['rr_graph'], data['vr_graph']]):
            for id_, rids in graph.items():
                out_degrees[node_ind_fn(id_)] += len(rids)
                for rid in rids:
                    in_degrees[self.node_rid(rid)] += 1

        for node_ind_fn, graph in zip([self.node_rid, self.node_vid], [data['req_features'], data['veh_features']]):
            for id_, feats in graph.items():
                node = node_ind_fn(id_)
                feats['in_degree'] = in_degrees[node]
                feats['out_degree'] = out_degrees[node]
        return data

    def get_node_neighbors(self, data):
        nghbs = defaultdict(set)
        for node_ind_fn, graph in zip([self.node_rid, self.node_vid], [data['rr_graph'], data['vr_graph']]):
            for id_, rids in graph.items():
                node = node_ind_fn(id_)
                nghbs[node].update([self.node_rid(rid) for rid in rids.keys()])
        return nghbs

    def add_common_nghbs(self, data, nghbs):
        for node_ind_fn, graph in zip([self.node_rid, self.node_vid], [data['rr_graph'], data['vr_graph']]):
            for id_, rids in graph.items():
                node1 = node_ind_fn(id_)
                for rid, feats in rids.items():
                    node2 = self.node_rid(rid)
                    feats['common_nghbs'] = len(nghbs[node1].intersection(nghbs[node2]))
                    feats['jaccards_coeff'] = feats['common_nghbs'] / len(nghbs[node1].union(nghbs[node2]))
        return data

    def compute_clustering_coeff(self, node, feats, nghbs):
        neighbors = nghbs[node]
        n_nghbs = len(neighbors)
        if n_nghbs < 2:
            feats['clustering_coeff'] = 0.0
        else:
            links = 0
            for neighbor in neighbors:
                links += len(nghbs[neighbor].intersection(neighbors))
            feats['clustering_coeff'] = links / (n_nghbs * (n_nghbs - 1))

    def add_clustering_coeffs(self, data, nghbs):
        for node_ind_fn, graph in zip([self.node_rid, self.node_vid], [data['req_features'], data['veh_features']]):
            for id_, feats in graph.items():
                self.compute_clustering_coeff(node_ind_fn(id_), feats, nghbs)
        return data

    def node_rid(self, rid):
        return rid

    def node_vid(self, vid):
        return vid + self.n_requests

    def add_assign_to_edges(self, data):
        self.add_labels_from_seq(data, data['init_assignments'], 'init_assign')
        self.add_labels_from_seq(data, data['assignments'], 'opt_assign')

    def add_labels_from_seq(self, data, assigns, feature_name):
        for vid, seq in assigns.items():
            if seq is None:
                continue
            data['vr_graph'][vid][seq[1]][feature_name] = 1
            for rid1, rid2 in zip(seq[1:], seq[2:]):
                data['rr_graph'][rid1][rid2][feature_name] = 1
