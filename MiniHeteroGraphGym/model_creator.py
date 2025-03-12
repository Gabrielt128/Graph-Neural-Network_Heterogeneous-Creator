# The implementation of the exact model happened here
# (mainly regarding the in_channels & outp_channels specification 

from configs.config import global_cfg
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GatedGraphConv

cfg = global_cfg.network

class CustomGNN(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
        hidden_size = cfg.hidden_neurons
        activation = cfg.activation

        if activation == "relu":
            self.activation = F.relu
        elif activation == "prelu":
            self.activation = F.PReLU

        self.pre_mp_layers = torch.nn.ModuleList([
            torch.nn.Linear(3 if i == 0 else hidden_size, 
                            hidden_size) 
                            for i in range(cfg.pre_mp)
        ])

        self.post_mp_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size,
                            2 if i == cfg.post_mp - 1 else hidden_size)
                            for i in range(cfg.post_mp)
        ])

        # MessagePassing layers:
        self.mp_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList() if cfg.batch_norm else None
        for index in range(cfg.layers):
            in_channels = 3 if index==0 and cfg.pre_mp==0 else hidden_size
            out_channels = 2 if index==cfg.layers-1 and cfg.post_mp==0 else hidden_size
            # if cfg.layer_type == "graph":
            #     # self.mp_layers.append(GraphConv(hidden_size, hidden_size))
            #     # in_channels = 3 if index==0 and cfg.pre_mp==0 else hidden_size
            #     # out_channels = 2 if index==cfg.layers-1 and cfg.post_mp==0 else hidden_size
            #     self.mp_layers.append(GraphConv(in_channels, 
            #                                     out_channels))
            #     if cfg.batch_norm:
            #         self.bn_layers.append(BatchNorm1d(out_channels))
            # else:
            #     raise NotImplementedError
            
            if cfg.layer_type == "graph":
                # Add GraphConv
                self.mp_layers.append(GraphConv(in_channels, out_channels))
            elif cfg.layer_type == "gated":
                # Add GatedGraphConv
                self.mp_layers.append(GatedGraphConv(out_channels, num_layers=1, aggr='add'))
            else:
                raise NotImplementedError(f"Layer type '{cfg.layer_type}' not implemented.")

    def forward(self, x, edge_index):
        for layer in self.pre_mp_layers:
            x = self.activation(layer(x))

        # for layer in self.mp_layers:
        #     x = self.activation(layer(x, edge_index))
        for index, layer in enumerate(self.mp_layers):
            x = layer(x, edge_index)
            if self.bn_layers:
                x = self.bn_layers[index](x)  # Apply BatchNorm if it exists
            x = self.activation(x)

        for layer in self.post_mp_layers:
            x = self.activation(layer(x))
        return x
    

def model_creator():
    pass