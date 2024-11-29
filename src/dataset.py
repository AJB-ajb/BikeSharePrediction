import torch
import numpy as np
import math
import pandas as pd
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.data as geomdata
import analysis as an
import tqdm
from pathlib import Path

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
import warnings

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
        self.cfg = config
        super().__init__(root, transform, pre_transform)
        warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
        self.data, self.slices, self.N_stations, self.μ, self.σ, self.new2oldidx = torch.load(self.processed_paths[0])

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
        return [osp.join(data_dir, f'in_out_pred_{self.cfg.year}_{self.cfg.month}_{self.cfg.name}.pt')]

    def process(self):
        cfg = self.cfg
        self.data = an.BikeShareData.load(month=cfg['month'], year=cfg['year'], force_reprocess=cfg['reload_bike_data'])
        # remove stations with missing lat/lon
        # after having eliminated ghost stations (missing lat/lon), we have a new sequential index
        stations = self.data.stations[~(self.data.stations['lat'].isna() | self.data.stations['lon'].isna())]
        if self.cfg.N_stations is not None:
            stations = stations.iloc[:self.cfg.N_stations]
        else:
            self.cfg.N_stations = len(stations)
        # thus we have new_index < old_index
        new2old_idx = np.array([old_idx for old_idx in stations.index])
        stations = stations.reset_index(drop=True)

        # eliminate ghost stations and subsample every supsample mins
        in_rates = self.data.in_rates[new2old_idx, ::self.cfg['subsample_minutes']]
        out_rates = self.data.out_rates[new2old_idx, ::self.cfg['subsample_minutes']] 
        N_stations, N_times = in_rates.shape


        inout_rates = np.concatenate((in_rates[..., None], out_rates[..., None]), axis = 2) # (N_stations × N_times × 2)

        at_min_mask = self.data.at_min_mask[new2old_idx, ::self.cfg['subsample_minutes']]
        at_max_mask = self.data.at_max_mask[new2old_idx, ::self.cfg['subsample_minutes']]
        # at_max_mask applies to in_rates, at_min_mask applies to out_rates
        mask = np.concatenate([at_max_mask[..., None], at_min_mask[..., None]], axis = 2) # (N_stations × N_times × 2)


        self.μ, self.σ = np.mean(inout_rates), np.std(inout_rates)
        inout_rates = z_score(inout_rates, self.μ, self.σ)
        N_history = self.cfg['N_history']
        n_window = self.cfg['N_predictions'] + self.cfg['N_history']
        
        adj_matrix = self.adjacency_matrix(stations)
        edge_list = edge_list_from_matrix(adj_matrix) # (2 × N_edges), "edge_index" in PyG
        edge_attr = np.ones(edge_list.shape[1], dtype=np.float32) # edge features are all 1 for the GAT version

        # create dataset_len [N_stations × N_features] list of data objects
        # note, that the graphs are all concatenated, as common in GNNs (different than conventional NN practise, where we would have an additional batch dimension)
        # N_features = 2 * (N_history + N_predictions) = 2 * n_window
        # dataset_len = N_times - n_window + 1
        times_in_month = np.arange(0, N_times * self.cfg.subsample_minutes, step = self.cfg.subsample_minutes)
        daytime_minute = times_in_month % (24 * 60)
        weekday_idx = (times_in_month // (24 * 60)) % 7

        seqs = []
        for i in tqdm.tqdm(range(N_times - n_window + 1), desc = 'Processing dataset'):
            mask_window = mask[:, i:i + n_window, :] # (N_stations × n_window × 2)

            window = inout_rates[:, i:i + n_window, :] # (N_stations × n_window × 2)
            x = window[:, :N_history, :].reshape(N_stations, -1) 
            y = window[:, N_history:, :].reshape(N_stations, -1)

            y_mask = mask_window[:, N_history:, :].reshape(N_stations, -1)

            # get daytime and day in week global features (i.e. independent of station, dependent only on time)
            # i.e. [N_history × 4], each [cos, sin]
            # use sinusoidal encoding for the time of day and day of week to model the periodicity
            daytime_angle = 2 * np.pi * daytime_minute[i:i + N_history] / (24 * 60)
            weekday_angle = 2 * np.pi * weekday_idx[i:i + N_history] / 7
            time_features = np.stack([np.cos(daytime_angle), np.sin(daytime_angle), np.cos(weekday_angle), np.sin(weekday_angle)], axis = 1)

            graphdata = geomdata.Data(x = torch.tensor(x), y = torch.tensor(y), edge_index = torch.tensor(edge_list), edge_attr = torch.tensor(edge_attr), y_mask = torch.tensor(y_mask), time_features = torch.tensor(time_features, dtype = torch.float32))

            seqs.append(graphdata)
        self.data, self.slices = geomdata.InMemoryDataset.collate(seqs)
        torch.save((self.data, self.slices, N_stations, self.μ, self.σ, new2old_idx), self.processed_paths[0])

    def get_day_splits(self, train_frac = 21 / 30, val_frac = 3 / 30, shuffle = True):
        """
            Split the day data randomly in this dataset into train, val and test datasets according to the approximate percentages. I.e. the dataset is split into days and the days are randomly assigned to the datasets. The test dataset is the remainder of the days assigned to train and eval. If shuffle is True, the days are shuffled before assignment, otherwise the first consecutive days are assigned to train and so on, i.e. in these cases, the splits are contiguous.
            Returns train, val, test datasets.
        """
        window_size = self.cfg['N_history'] + self.cfg['N_predictions']
        N_subsamples_per_day = 24 * 60 // self.cfg['subsample_minutes']

        days_in_dataset = len(self) // N_subsamples_per_day
        N_train_days = math.floor(train_frac * days_in_dataset)
        N_val_days = math.floor(val_frac * days_in_dataset)

        day_indices = np.arange(days_in_dataset)
        if shuffle:
            np.random.shuffle(day_indices)

        train_days = day_indices[:N_train_days]
        val_days = day_indices[N_train_days:(N_train_days + N_val_days)]
        test_days = day_indices[(N_train_days + N_val_days):]

        train_DL_indices = np.concat([np.arange(i*N_subsamples_per_day, (i+1) * N_subsamples_per_day) for i in train_days]) # train dataloader (i.e. subsample) indices
        val_DL_indices = np.concat([np.arange(i*N_subsamples_per_day, (i+1) * N_subsamples_per_day)for i in val_days])
        test_DL_indices = np.concat([np.arange(i*N_subsamples_per_day, (i+1) * N_subsamples_per_day) for i in test_days])
        train = self[train_DL_indices]
        val = self[val_DL_indices]
        test = self[test_DL_indices]
        return train, val, test
    
    @property
    def raw_data_dir(self):
        return self.cfg['data_dir'] / f'bikeshare-ridership-{self.cfg["year"]}'

    
    @property
    def raw_file_names(self):
        year = self.cfg['year']
        month = self.cfg['month']
        return self.raw_data_dir / f'Bike share ridership {year}-{month}.csv'
    
    def download(self) -> None:
        import zipfile
        import wget

        data_dir = self.cfg.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        year = self.cfg['year']
        zip_url = f"https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/7e876c24-177c-4605-9cef-e50dd74c617f/resource/9a9a0163-8114-447c-bf66-790b1a92da51/download/bikeshare-ridership-{year}.zip"
        zip_path = data_dir / f'bikeshare-ridership-{year}.zip'
        if zip_path.exists():
            zip_path.unlink()
        wget.download(str(zip_url), str(zip_path))

        print(f"Downloaded {zip_url} to {zip_path}")
        with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
            zip_ref.extractall(str(self.raw_data_dir))
            
        zip_path.unlink()
        print(f"Extracted {zip_path}")
        print("Done")
    