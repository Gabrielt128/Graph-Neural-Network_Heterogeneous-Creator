# Here we register the layers & message passing frameworks(e.g. with/without batchnorm) that we allowed

import torch
import torch_geometric as pyg
from torch_geometric.nn import GraphConv, GatedGraphConv, GraphSAGE, GCNConv, GATConv, SAGEConv
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm

# the below BHGraphGym is added afterwards
import BHGraphGym.configs.gnn_registery as register
from BHGraphGym.configs.gnn_registery import register_layer, register_mp_module
from BHGraphGym.configs.config import global_cfg

# cfg = global_cfg.network
network_cfg = global_cfg.network
# --------------------------------------------------------------------------------------
# First, the layer choices that are allowed: GraphConv, GCNConv, GatedGraphConv, GAT/Graph Sage 
# register_layer('graph', GraphConv(flow="target_to_source"))
register_layer('graph', GraphConv)
register_layer('gcn', GCNConv)

register_layer('graph_sage', SAGEConv)

register_layer('gat', GATConv)
register_layer('gated', GatedGraphConv)

# If we implemented our own designed layer, we can register it through decorator 
# @register_layer('GraphConv')
# class SelfConv(GraphConv):
#     pass


# --------------------------------------------------------------------------------------
# Second we register the message passing containers/framework here 
# or shall we implement this in another file? to be decided huh
# I say we implement it here implement it with a layer choice input placeholder
# cfg.output_dim
@register_mp_module('basic')
class MPModule(torch.nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        layer = register.layer_dict[cfg.layer_type]
        self.layers = torch.nn.ModuleList()
        self.activation = F.relu if cfg.activation == "relu" else torch.nn.PReLU()

        for index in range(cfg.layers):
            in_channels = 3 if index == 0 and cfg.pre_mp==0 else cfg.hidden_neurons

            # TODO:
            # out_channels = 2 if index == cfg.layers - 1 and cfg.post_mp==0 else cfg.hidden_neurons
            out_channels = cfg.output_dim if index == cfg.layers - 1 and cfg.post_mp==0 else cfg.hidden_neurons
            # self.layers.append(layer(in_channels, out_channels))

            # here could be modulized a bit you know
            if cfg.layer_type == "gated":
                # self.layers.append(layer(out_channels, 1, flow="target_to_source"))
                self.layers.append(layer(out_channels, 1))
            elif cfg.layer_type == "graph_sage":
                self.layers.append(layer(in_channels, out_channels))
            else:
                # self.layers.append(layer(in_channels, out_channels, flow="target_to_source"))
                self.layers.append(layer(in_channels, out_channels))
            if cfg.batch_norm:
                # self.layers.append(torch.nn.BatchNorm1d(out_channels))
                self.layers.append(BatchNorm(out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    # forward part doesn't matter for hetero representation
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.activation(x)
        return x

# --------------------------------------------------------------------------------------
# The model_creator() implementation
def model_creator(network_cfg) -> torch.nn.Module:
    cfg = network_cfg
    MPModuleClass = register.mp_module_dict[cfg.structure]
    mp_module = MPModuleClass(cfg)

    # cfg.output_dim
    pre_mp_layers = torch.nn.ModuleList([
        torch.nn.Linear(3 if i == 0 else cfg.hidden_neurons, cfg.hidden_neurons)
        for i in range(cfg.pre_mp)
    ])

    # post_mp_layers = torch.nn.ModuleList([
    #     torch.nn.Linear(cfg.hidden_neurons, 2 if i == cfg.post_mp - 1 else cfg.hidden_neurons)
    #     for i in range(cfg.post_mp)
    # ])

    # TODO: 
    post_mp_layers = torch.nn.ModuleList([
        torch.nn.Linear(cfg.hidden_neurons, cfg.output_dim if i == cfg.post_mp - 1 else cfg.hidden_neurons)
        for i in range(cfg.post_mp)
    ])

    class GNNModel(torch.nn.Module):
        def __init__(self, pre_mp_layers, mp_module, post_mp_layers):
            super().__init__()
            self.pre_mp_layers = pre_mp_layers
            self.mp_module = mp_module.layers
            self.post_mp_layers = post_mp_layers
            self.activation = mp_module.activation
            self.use_residual = cfg.use_residual

        def forward(self, x, edge_index):
            for layer in self.pre_mp_layers:
                x = self.activation(layer(x))
            for index, layer in enumerate(self.mp_module):
                if isinstance(layer, BatchNorm):
                    x = layer(x)
                else:
                    if self.use_residual:
                        if len(self.pre_mp_layers) == 0 and index == 0:
                            x = layer(x, edge_index)
                        else:
                            x_res = x  
                            x = layer(x, edge_index)
                            x = x + x_res
                    else:
                        x = layer(x, edge_index)
                x = self.activation(x)

            # for layer in self.mp_module:
            #     if isinstance(layer, BatchNorm):
            #         x = layer(x)
            #     else:
            #         x = layer(x, edge_index)
            #     x = self.activation(x)
            for layer in self.post_mp_layers:
                x = self.activation(layer(x))
            return x

    model = GNNModel(pre_mp_layers, mp_module, post_mp_layers)
    return model