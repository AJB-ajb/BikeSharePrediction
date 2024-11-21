import torch
import numpy as np
import pandas as pd
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.data as geomdata
import analysis as an
import tqdm

def z_score(x, mean, std):
    return (x - mean) / std
def un_z_score(x_normed, mean, std):
    return x_normed * std  + mean

def MAPE(v, v_):
    return torch.mean(torch.abs((v_ - v)) /(v + 1e-15) * 100)

def RMSE(v, v_):
    return torch.sqrt(torch.mean((v_ - v) ** 2))

def MAE(v, v_):
    return torch.mean(torch.abs(v_ - v))

def edge_list_from_matrix(adj):
    """
        Create an edge list from an adjacency matrix for an undirected graph.
        The edge list is a 2 × N_edges array with the first row being the source node index and the second row being the target node index.
    """
    adj = adj.copy()
    adj[np.tril_indices(adj.shape[0], k = -1)] = 0
    inds_i, inds_j = np.nonzero(adj)
    return np.stack([inds_i, inds_j], axis = 0)


import pyproj

__geod = pyproj.Geod(ellps='WGS84')
def dst(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface in meters.
    """
    angle1,angle2,distance = __geod.inv(lon1, lat1, lon2, lat2)
    return distance

class BikeGraphDataset(InMemoryDataset):
    """
        Dataset storing projessed bike sharing rate data for GNN training.
        Stores roughly ([InRate, OutRate]_t)_(t) as input and (InRate, OutRate)_(t0 < t) to be predicted as output.
    """
    def __init__(self, config, root = None, transform = None, pre_transform = None):
        self.config = config
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.N_stations, self.μ, self.σ = torch.load(self.processed_paths[0])

    def adjacency_matrix(self, stations, min_stations_connected = 3, max_dst_meters = 500):
        adj = np.zeros((len(stations), len(stations)))
        lats, lons = np.array(stations['lat']), np.array(stations['lon'])
        dst_matrix = np.zeros_like(adj, dtype=np.float32)
        N_stations = len(stations)
        for i in range(len(stations)):
            dst_matrix[i, :] = dst(np.tile(lats[i], N_stations), np.tile(lons[i], N_stations), lats, lons)

        adj[dst_matrix < max_dst_meters] = 1
        for i in range(len(stations)):
            if np.sum(adj[i, :]) < min_stations_connected * 2 + 1: # node degree is (sum of row i - 1) // 2 (we exclude the self-loop)
                # connect the closest stations
                closest = np.argsort(dst_matrix[i, :])
                for j in range(min(2 * min_stations_connected + 1, len(stations))):
                    adj[i, closest[j]] = 1
        return adj
    
    @property
    def data_directory(self):
        dir = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        return dir

    
    @property
    def processed_file_names(self):
        data_dir = self.data_directory
        return [osp.join(data_dir, 'in_out_pred.pt')]

    def process(self):
        config = self.config
        self.data = an.BikeShareData.load(month=config['month'], year=config['year'])
        # remove stations with missing lat/lon
        # after having eliminated ghost stations (missing lat/lon), we have a new sequential index
        stations = self.data.stations[~(self.data.stations['lat'].isna() | self.data.stations['lon'].isna())]
        if self.config['N_stations'] is not None:
            stations = stations.iloc[:self.config['N_stations']]
        # thus we have new_index < old_index
        new2old_idx = np.array([old_idx for old_idx in stations.index])
        stations = stations.reset_index(drop=True)

        in_rates, out_rates = self.data.in_rates[new2old_idx, ::5], self.data.out_rates[new2old_idx, ::5] # eliminate ghost stations and subsample every 5 mins
        N_stations, N_times = in_rates.shape


        inout_rates = np.concatenate((in_rates[..., None], out_rates[..., None]), axis = 2) # (N_stations × N_times × 2)
        μ, σ = np.mean(inout_rates), np.std(inout_rates)
        inout_rates = z_score(inout_rates, μ, σ)
        N_history = self.config['N_history']
        n_window = self.config['N_predictions'] + self.config['N_history']
        
        adj_matrix = self.adjacency_matrix(stations)
        edge_list = edge_list_from_matrix(adj_matrix) # (2 × N_edges), "edge_index" in PyG
        edge_attr = np.ones(edge_list.shape[1], dtype=np.float32) # edge features are all 1 for the GAT version

        # create dataset_len [N_stations × N_features] list of data objects
        # note, that the graphs are all concatenated, as common in GNNs (different than conventional NN practise, where we would have an additional batch dimension)
        # N_features = 2 * (N_history + N_predictions) = 2 * n_window
        # dataset_len = N_times - n_window + 1
        seqs = []
        for i in tqdm.tqdm(range(N_times - n_window + 1), desc = 'Processing dataset'):
            window = inout_rates[:, i:i + n_window, :] # (N_stations × n_window × 2)
            x = window[:, :N_history, :].reshape(N_stations, -1) 
            y = window[:, N_history:, :].reshape(N_stations, -1)
            graphdata = geomdata.Data(x = torch.tensor(x), y = torch.tensor(y), edge_index = torch.tensor(edge_list), edge_attr = torch.tensor(edge_attr))
            seqs.append(graphdata)
        self.data, self.slices = geomdata.InMemoryDataset.collate(seqs)
        torch.save((self.data, self.slices, N_stations, μ, σ), self.processed_paths[0])


        

        