"""Particle Flow Network"""

from typing import Optional

import torch.nn as nn
import torch_geometric.nn as geo_nn

from . import classical


class PhiMessagePassing(geo_nn.MessagePassing):
    def __init__(self, phi: nn.Module, aggr: Optional[str] = 'add'):
        """Undirected message passing model.
        See "Creating Message Passing Networks"
        https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
        
        Args:
            phi : nn.Module
                The correlation function between nodes.
            aggr : str
                Aggregation method, ex: 'max', 'mean', 'add', etc.
        """

        super().__init__(aggr=aggr, flow='target_to_source')

        self.phi = phi
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # `x_i` and `x_j` are source node and target node respectively.
        # PFN does not calculate pairwise information.
        return x_i
    
    def update(self, aggr_out, x):
        # The output of aggrataion of phi will not be used.
        return self.phi(x)


class ParticleFlowNetwork(nn.Module):
    def __init__(
            self,
            score_dim: int,
            parameters: dict
    ):

        super().__init__()

        phi = classical.ClassicalMLP(
            in_channel=parameters['Phi']['in_channel'],
            hidden_channel=parameters['Phi']['hidden_channel'],
            num_layers=parameters['Phi']['num_layers'],
            out_channel=parameters['Phi']['out_channel'],
        )

        self.gnn = PhiMessagePassing(phi, aggr='add')
        
        self.mlp = classical.ClassicalMLP(
            in_channel=parameters['F']['in_channel'],
            hidden_channel=parameters['F']['hidden_channel'],
            num_layers=parameters['F']['num_layers'],
            out_channel=score_dim,
        )

    def forward(self, x, edge_index, batch):
        # Graph neural network.
        x = self.gnn(x, edge_index)

        # Graph aggregation = SUM
        x = geo_nn.global_add_pool(x, batch)

        # Shallow linear model.
        x = self.mlp(x)

        return x