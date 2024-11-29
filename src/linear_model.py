import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class LinearModel(nn.Module):
    def __init__(self, cfg):
        super(LinearModel, self).__init__()
        N_features = cfg.N_history * cfg.N_features_per_node
        N_inputs = cfg.N_stations * (cfg.N_history * cfg.N_features_per_node + )
        self.linear = nn.Linear(input_dim, output_dim)

        
    def forward(self, batch): # assume batch is a torch_geometric.data.Data object
        batch_size = batch.num_graphs
        x = batch.x.reshape(batch_size, -1) # [batch_size * N_nodes × N_history * features]
        time_features = batch.time_features.reshape(batch_size, -1) # [batch_size *seq_length, 4]
        y = self.linear(x)


        return self.linear(x)
    
    def train(self, train_dataset):
        x = train_dataset.x.numpy()
        y = train_dataset.y.numpy()
        time_features = train_dataset.time_features.numpy()

        scipy_linear = LinearRegression()
        N = len(train_dataset)
        x = x.reshape(N, -1)
        y = y.reshape(N, -1)
        time_features = time_features.reshape(N, -1)
        x = torch.cat((x, time_features), dim = -1)

        scipy_linear.fit(x, y)
        self.linear.weight = torch.nn.Parameter(torch.tensor(self.scipy_linear.coef_))
