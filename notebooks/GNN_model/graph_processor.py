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
        self.rr_edge_feature_dim = None
        self.vr_edge_feature_dim = None

    def load_and_process_graphs(self):
        if self.load_saved_data:
            data = torch.load(f'{self.base_data_dir}processed/graph_data.pt')
        else:
            data = self.load_data(f'{self.base_data_dir}processed/')
            data = self.transform_features(data)
            # Calculate feature dimensions from non-empty data
            if not data['rr_graph'].empty:
                self.rr_edge_feature_dim = len(data['rr_graph'].columns) - 3  # Subtract source, target, label
            if not data['vr_graph'].empty:
                self.vr_edge_feature_dim = len(data['vr_graph'].columns) - 3  # Subtract source, target, label
            data = self.split_data_to_separate_graphs(data)
            transform = T.NormalizeFeatures(attrs=['x', 'edge_attr'])
            # TODO check correctness of undirected: add new edge type?
            data = [transform(T.ToUndirected()(graph)) for graph in data]
            torch.save(data, f'{self.base_data_dir}processed/graph_data.pt')
        return data

    @staticmethod
    def load_data(data_dir):
        data = {}
        for file in os.scandir(data_dir):
            if not file.name.endswith('.csv'):
                continue
            with open(file, 'r') as f:
                file_name = file.name[:file.name.find('.')]
                data[file_name] = pd.read_csv(f)
        return data

    def split_data_to_separate_graphs(self, data):
        '''
        Splits the data into separate graphs for each timestep.
        Each graph contains the features and edges for that timestep.
        The data is grouped by timestep, and each group is added to a separate HeteroData object.

        :param data: The data to be split.
        '''
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

    @staticmethod
    def add_r_features(rows, hdata):
        hdata['request'].x = torch.tensor(rows.values)

    @staticmethod
    def add_v_features(rows, hdata):
        hdata['vehicle'].x = torch.tensor(rows.values)

    def add_rr_edge_data(self, rows, hdata):
        if rows.empty:
            hdata['request', 'connects', 'request'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            hdata['request', 'connects', 'request'].edge_attr = torch.zeros((0, self.rr_edge_feature_dim), dtype=torch.float)
            hdata['request', 'connects', 'request'].y = torch.zeros(0, dtype=torch.long)
        else:
            hdata['request', 'connects', 'request'].edge_index = torch.tensor(
                np.array([rows['source'].values, rows['target'].values])).int()
            hdata['request', 'connects', 'request'].edge_attr = torch.tensor(
                rows.drop(columns=['source', 'target', 'label']).values)
            hdata['request', 'connects', 'request'].y = torch.tensor(rows['label'].values).int()

    def add_vr_edge_data(self, rows, hdata):
        if rows.empty:
            hdata['vehicle', 'connects', 'request'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            hdata['vehicle', 'connects', 'request'].edge_attr = torch.zeros((0, self.vr_edge_feature_dim), dtype=torch.float)
            hdata['vehicle', 'connects', 'request'].y = torch.zeros(0, dtype=torch.long)
        else:
            hdata['vehicle', 'connects', 'request'].edge_index = torch.tensor(
                np.array([rows['source'].values, rows['target'].values])).int()
            hdata['vehicle', 'connects', 'request'].edge_attr = torch.tensor(
                rows.drop(columns=['source', 'target', 'label']).values)
            hdata['vehicle', 'connects', 'request'].y = torch.tensor(rows['label'].values).int()

    @staticmethod
    def transform_features(data):
        data['req_features'] = pd.get_dummies(columns=['status', 'o_pos', 'd_pos'], data=data['req_features'],
                                              dtype=float)
        data['veh_features'] = pd.get_dummies(columns=['type', 'status', 'pos'], data=data['veh_features'], dtype=float)
        return data
