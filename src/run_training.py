from dataset import BikeGraphDataset
from st_gat import STGAT
import analysis as an

import torch as th
import torch.nn as nn
import numpy as np
import torch_geometric.data as geomdata
import torch_geometric.loader as geomloader

def default_config():
    # default configuration
    cfg = {
        'month': '05',
        'year': '2024',
        'N_stations': None, # load all
        'N_history': 12, # take an hour of history
        'N_predictions': 9,
        'batch_size': 32,
        'min_stations_connected': 2,
        'dropout': 0.1,
        'gat_heads': 8,
        'lstm1_hidden_size': 32,
        'lstm2_hidden_size': 128
    }
    in_features_per_node = 2 * cfg['N_history'] 
    cfg.update({
        'in_channels': in_features_per_node,
    })
    return cfg

def test_config():
    "small scale minimal test configuration; absolute minimum for fastest testing"
    # base the test configuration on the default configuration and only override the necessary values
    cfg = default_config()
    cfg.update({
        'N_stations': 5,
        'N_history': 5,
        'N_predictions': 3
    })
    in_features_per_node = 2 * cfg['N_history'] 
    cfg.update({
        'in_channels': in_features_per_node,
        'out_channels' : in_features_per_node
    })
    return cfg


np.random.seed(42)
th.manual_seed(42)

config = test_config()
dataset = BikeGraphDataset(config, root=str())
# dataset.process() # force reprocessing

device = th.device('cpu')
model = STGAT(N_nodes = config['N_stations'], **config).to(device)
dataloader = geomloader.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# test the forward pass
for batch in dataloader:
    model.forward(batch, device)
    break
print("Success!")