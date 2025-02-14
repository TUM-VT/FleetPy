from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
import pandas as pd
import os
import numpy as np


class GraphProcessor:
    def __init__(self, base_data_dir, load_saved_data=True):
        self.base_data_dir = base_data_dir
        self.load_saved_data = load_saved_data

    def load_and_process_graphs(self):
        if self.load_saved_data:
            data = torch.load(f'{self.base_data_dir}processed/graph_data.pt')
        else:
            data = self.load_data(f'{self.base_data_dir}processed/')
            data = self.transform_features(data)
            data = self.split_data_to_separate_graphs(data)
            transform = T.NormalizeFeatures(attrs=['x', 'edge_attr'])
            # TODO check correctness of undirected: add new edge type?
            data = [transform(T.ToUndirected()(graph)) for graph in data]
            torch.save(data, f'{self.base_data_dir}processed/graph_data.pt')
        return data

    def load_data(self, data_dir):
        data = {}
        for file in os.scandir(data_dir):
            if not file.name.endswith('.csv'):
                continue
            with open(file, 'r') as f:
                file_name = file.name[:file.name.find('.')]
                data[file_name] = pd.read_csv(f)
        return data

    def split_data_to_separate_graphs(self, data):
        max_timestep = data['req_features']['timestep'].max()
        split_data = [HeteroData() for _ in range(max_timestep + 1)]
        for name, add_feats_fn in zip(['req_features', 'veh_features', 'rr_graph', 'vr_graph'],
                                      [self.add_r_features, self.add_v_features, self.add_rr_edge_data,
                                       self.add_vr_edge_data]):
            grouped = data[name].groupby('timestep')
            for timestep in range(max_timestep + 1):
                try:
                    rows = grouped.get_group(timestep)
                except:
                    rows = pd.DataFrame()
                add_feats_fn(rows, split_data[timestep])
        return split_data

    def add_r_features(self, rows, hdata):
        hdata['request'].x = torch.tensor(rows.values)

    def add_v_features(self, rows, hdata):
        hdata['vehicle'].x = torch.tensor(rows.values)

    def add_rr_edge_data(self, rows, hdata):
        hdata['request', 'connects', 'request'].edge_index = torch.tensor(np.array([rows['source'].values,
            rows['target'].values]) if not rows.empty else np.array([[], []])).int()
        hdata['request', 'connects', 'request'].edge_attr = torch.tensor(
            rows.drop(columns=['source', 'target', 'label']).values) if not rows.empty else torch.tensor([])
        hdata['request', 'connects', 'request'].y = torch.tensor(
            rows['label'].values if not rows.empty else []).int()

    def add_vr_edge_data(self, rows, hdata):
        hdata['vehicle', 'connects', 'request'].edge_index = torch.tensor(np.array([rows['source'].values,
            rows['target'].values]) if not rows.empty else np.array([[], []])).int()
        hdata['vehicle', 'connects', 'request'].edge_attr = torch.tensor(
            rows.drop(columns=['source', 'target', 'label']).values if not rows.empty else [])
        hdata['vehicle', 'connects', 'request'].y = torch.tensor(
            rows['label'].values if not rows.empty else []).int()

    def transform_features(self, data):
        data['req_features'] = pd.get_dummies(columns=['status', 'o_pos', 'd_pos'], data=data['req_features'], dtype=float)
        data['veh_features'] = pd.get_dummies(columns=['type', 'status', 'pos'], data=data['veh_features'], dtype=float)
        return data
