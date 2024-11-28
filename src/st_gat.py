import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geomnn


import itertools

class STGAT(nn.Module):
    """
        Spatio-Temporal Graph Attention Network

        Parameters:
        - in_channels: number of input channels to the GAT first layer, i.e. number of total input features per node
        - out_features_per_node: number of numbers that are predicted for each node
        normally out_channels := in_channels
    """
    def __init__(self, in_features_per_node, out_features_per_node, N_nodes, gat_heads, dropout, lstm1_hidden_size, lstm2_hidden_size, final_module, **kwargs):
        super(STGAT, self).__init__()

        self.dropout = nn.Dropout(p = dropout)
        self.gat = geomnn.GATConv(in_channels= in_features_per_node, out_channels= in_features_per_node, heads = gat_heads, dropout = 0, concat = False)

        if final_module == 'lstm':
            self.lstm1 = nn.LSTM(input_size=N_nodes, hidden_size=lstm1_hidden_size, num_layers=1) # outputs sequence outputs with lstm1_hidden_size
            self.lstm2 = nn.LSTM(input_size=lstm1_hidden_size, hidden_size = lstm2_hidden_size, num_layers=1)
            self.linear = nn.Linear(lstm2_hidden_size, N_nodes * out_features_per_node) 
        elif final_module == 'transformer':
            raise NotImplementedError('transformer not yet implemented')

        # LSTM parameter initialization
        for name, param in itertools.chain(self.lstm1.named_parameters(), self.lstm2.named_parameters()):
            if 'bias' in name:
                nn.init.constant_(param, 0.)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        # Linear weight initialization
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, batch):
        dev = next(self.parameters()).device
        # batch is a torch_geometric.data.Data object
        x, edge_index = batch.x, batch.edge_index
        batch_size = batch.num_graphs
        # obtain N_nodes from data to allow generalization to unseen graphs
        N_nodes = batch.num_nodes // batch_size 
        seq_length = batch.num_features

        # should be [batch_size * N_nodes × N_history * features]
        x = self.gat(x, edge_index) # -> batch_size * N_nodes × N_history (× N_Features) (?) (i.e. in total N_nodes… × Features…)
        x = self.dropout(x) 
        # for lstm, the sequence length must be the first dimension, i.e. swap and expand
        x = th.reshape(x, (batch_size, N_nodes, seq_length))
        x = th.movedim(x, 2, 0) # i.e. move the sequence dimension to the front # -> [seq_length × batch_size × N_nodes]

        x, _ = self.lstm1(x) # -> [h_1, h_2, …]
        x, _ = self.lstm2(x) # -> [h'_1, h'_2, …]

        # take the last element, i.e. the last output of the second LSTM, i.e. the only one having integrated all historic data
        last_pred = x[-1, ...]
        out = self.linear(last_pred) # [batch_size × N_nodes × N_preds]

        # output size [(batchsize * N_nodes) × N_preds] # (?)
        out = out.reshape(batch_size * N_nodes, -1)
        
        return out 
        




        

        

            

