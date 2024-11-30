import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, cfg):
        super(LinearModel, self).__init__()
        # the linear model takes the whole flattened input, i.e. N_nodes times all features per node plus the global features
        self.cfg = cfg
        self.N_in_features = cfg.N_in_features_per_step_with_global * cfg.N_history
        # we cannot calculate a demand prediction with this linear model, so we predict only the rates, i.e. // 2
        self.N_out_features = cfg.out_features_per_node * cfg.N_stations // 2 
        self.N_time_features = 4 * cfg.N_history
        self.linear = nn.Linear(self.N_in_features, self.N_out_features)
        
    def forward(self, batch): # assume batch is a torch_geometric.data.Data object
        batch_size = batch.num_graphs
        x = batch.x.reshape(batch_size, -1) # [batch_size * N_nodes × N_history * features]
        time_features = batch.time_features.reshape(batch_size, -1) # [batch_size *seq_length, 4]
        x = torch.cat((x, time_features), dim = -1)
        y = self.linear(x)

        # copy the predictions into the demand predictions in order to return the same shape as the one produced by the STGAT
        y = y.reshape(batch_size, self.cfg.N_stations, self.cfg.N_predictions, 2)
        y = torch.cat((y, y), dim = -1)

        # bring the output in the right shape
        y = y.reshape(batch_size * self.cfg.N_stations, -1)
        return y
    
    def eval(self):
        return
    
    def train(self, train_dataset):
        x = train_dataset.x.numpy()
        y = train_dataset.y.numpy()
        time_features = train_dataset.time_features.numpy()

        self.scipy_linear = LinearRegression()

        x = x.reshape(-1, self.N_in_features - self.N_time_features)
        time_features = time_features.reshape(-1, self.N_time_features)

        y = y.reshape(-1, self.N_out_features)

        x = np.concatenate((x, time_features), axis = -1)

        self.scipy_linear.fit(x, y)
        self.linear.weight = torch.nn.Parameter(torch.tensor(self.scipy_linear.coef_))
