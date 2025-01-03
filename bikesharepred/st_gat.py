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
    def __init__(self, gat_heads, dropout, N_history, cfg, **kwargs):
        super(STGAT, self).__init__()

        N_nodes = cfg.N_stations

        self.dropout = nn.Dropout(p = dropout)

        self.gat_layers = nn.ModuleList()
        for gat_layer_idx in range(cfg.num_gat_layers):
            gat = geomnn.GATConv(in_channels= cfg.in_features_per_node, out_channels= cfg.in_features_per_node, heads = gat_heads, dropout = 0, concat = False)
            self.gat_layers.append(gat)


        # number of features per node per time step in the input data
        self.N_features = cfg.in_features_per_node // N_history
        self.N_features_with_global = cfg.N_in_features_per_step_with_global

        self.cfg = cfg
        self.final_module = cfg.final_module

        if cfg.final_module == 'lstm':
            lstm1_hidden_size, lstm2_hidden_size = cfg.final_module_params['lstm1_hidden_size'], cfg.final_module_params['lstm2_hidden_size']
            self.lstm1 = nn.LSTM(input_size=self.N_features_with_global, hidden_size=lstm1_hidden_size, num_layers=1) # outputs sequence outputs with lstm1_hidden_size
            self.lstm2 = nn.LSTM(input_size=lstm1_hidden_size, hidden_size = lstm2_hidden_size, num_layers=1)
            self.linear = nn.Linear(lstm2_hidden_size, N_nodes * cfg.out_features_per_node) 

            # LSTM parameter initialization
            for name, param in itertools.chain(self.lstm1.named_parameters(), self.lstm2.named_parameters()):
                if 'bias' in name:
                    nn.init.constant_(param, 0.)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
            # Linear weight initialization
            nn.init.xavier_uniform_(self.linear.weight)

        elif cfg.final_module == 'transformer':
            d_model, nhead, num_layers, dim_feedforward = cfg.final_module_params['d_model'], cfg.final_module_params['n_heads'], cfg.final_module_params['n_layers'], cfg.final_module_params['dim_feedforward']

            self.transformer = DecoderOnlyModel(self.N_features_with_global, d_model, nhead, num_layers, dim_feedforward, N_history, dropout)
            self.linear = nn.Linear(d_model, N_nodes * cfg.out_features_per_node) 

        else: 
            raise NotImplementedError(f'{cfg.final_module} not implemented')

    
    def forward(self, batch):
        dev = next(self.parameters()).device
        # batch is a torch_geometric.data.Data object
        x, edge_index = batch.x, batch.edge_index
        batch_size = batch.num_graphs
        # obtain N_nodes from data to allow generalization to unseen graphs
        
        N_nodes = batch.num_nodes // batch_size 
        seq_length = self.cfg.N_history

        for gat_layer in self.gat_layers:
        # x should be [batch_size * N_nodes × N_history * features]
            x = gat_layer(x, edge_index) # -> batch_size * N_nodes × (N_history * N_Features) (i.e. in total N_nodes… × Features…)
        x = self.dropout(x) 

        # for lstm, the sequence length must be the first dimension, i.e. swap and expand
        x = th.reshape(x, (batch_size, N_nodes, seq_length, self.N_features))

        x = th.movedim(x, 2, 0).reshape(seq_length, batch_size, -1) # i.e. move the sequence dimension to the front 
        # -> [seq_length × batch_size × (N_nodes * N_features)]

        if self.cfg.use_time_features:
            # 
            # time features in input is # [batch_size * seq_length, 4] (seq_length = N_history)
            time_features = batch.time_features.reshape(batch_size, seq_length, 4).movedim(0, 1)
            
            # .expand(batch_size, N_nodes, seq_length, 4) 
            x = th.cat((x, time_features), dim = -1) # -> [ seq_length × batch_size × (N_features + 4)]
            

        if self.final_module == 'lstm':
            x, _ = self.lstm1(x) # -> [h_1, h_2, …]
            x, _ = self.lstm2(x) # -> [h'_1, h'_2, …]
        elif self.final_module == 'transformer':
            x = self.transformer(x)

        # take the last element, i.e. the last output of the second LSTM, i.e. the only one having integrated all historic data
        last_pred = x[-1, ...].reshape(batch_size, -1)
        out = self.linear(last_pred) # [batch_size × N_nodes * N_preds]

        # output size [(batchsize * N_nodes) × (N_predictions * 4)]
        out = out.reshape(batch_size * N_nodes, -1)
        
        return out 
        
class DecoderOnlyModel(nn.Module):
    """
        Implements a Transformer Decoder model for autoregressive sequence generation.
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(DecoderOnlyModel, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)  # Input embedding layer

        self.positional_encoding = nn.Parameter(th.zeros(max_seq_length, 1, d_model))  # Positional encoding
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, input_embds, memory=None):
        """
        Args:
            input embeddings: Input token embeddings of shape (seq_length, batch_size, d_model).

        Returns:
            Tensor: Output logits of shape (seq_length, batch_size, d_model).
        """
        # Input Embedding + Positional Encoding
        seq_length, _, _ = input_embds.shape  # Sequence length, batch size
        embds = self.input_embedding(input_embds)

        embeddings = embds + self.positional_encoding[:, :seq_length, :]
        
        # Generate causal mask for autoregressive attention
        causal_mask = th.triu(th.ones(seq_length, seq_length), diagonal=1).bool().to(input_embds.device)
        
        memory = memory or th.zeros_like(embeddings)
        # Pass through the Transformer Decoder
        decoder_output = self.decoder(tgt=embeddings, memory=memory, tgt_mask=causal_mask)
        
        # Shape: (seq_length, batch_size, d_model)
        return decoder_output