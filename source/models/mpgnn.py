"""Classical Message Passing Graph Neural Network (MPGNN)

This classical model is built with GNN structure followed by a simple
shallow fully connected linear network. The GNN is constructed with
package `torch_geometric`, see 

["PyTorch Geometric"]
(https://pytorch-geometric.readthedocs.io) 

for futher details. The tutorial for creating a `MessagePassing` class
can be found at

["Creating Message Passing Networks"]
(https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn

from . import classical


class MessagePassing(geo_nn.MessagePassing):
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
        # Want to calculate
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # `x_i` and `x_j` are source node and target node respectively.
        return self.phi(torch.cat((x_i, x_j), dim=-1))
    
    def update(self, aggr_out, x):
        # Set `gamma` as identity function, so `x` will not be used.
        return aggr_out


class ClassicalMPGNN(nn.Module):
    def __init__(
            self,
            phi_in: int,
            phi_out: int,
            phi_hidden: int,
            phi_layers: int,
            score_dim: int,
            aggregation: Optional[str] = 'add',
            dropout: Optional[float] = 0.0,
            **kwargs,
    ):
        """Classical model for benchmarking

        Arguments with prefix "phi" are related to the message passing
        function `phi`, which is constructed with a classical MLP.

        Arguments with prefix "mlp" are related to the shallow linear
        model after graph aggregation, which is also constructed with a 
        classical MLP.

        Args:
            phi_in : int
                The input channel dimension of `phi` in MPGNN.
            phi_out : int
                The output channel dimension of `phi` in MPGNN.
            phi_hidden : int
                Number of hidden neurons of `phi` in MPGNN.
            phi_layers : int
                Number of hidden layers of `phi` in MPGNN.
            score_dim : int
                Dimension of the final score output.
            aggregation : str (default='add')
                Aggregation method used in MessagePassing.
            dropout : float (default=0.0)
                Dropout rate for hidden layers.
        """

        super().__init__()

        self.aggr = aggregation
        
        phi = classical.ClassicalMLP(
            in_channel=phi_in,
            out_channel=phi_out,
            hidden_channel=phi_hidden,
            num_layers=phi_layers,
            dropout=dropout,
        )

        self.gnn = MessagePassing(phi, aggr=aggregation)
        
        self.mlp = nn.Sequential(
            nn.Linear(phi_out, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, score_dim),
        )

    def forward(self, x, edge_index, batch):
        # Graph neural network.
        x = self.gnn(x, edge_index)

        # Graph aggregation.
        if self.aggr == 'add':
            x = geo_nn.global_add_pool(x, batch)
        elif self.aggr == 'mean':
            x = geo_nn.global_mean_pool(x, batch)

        # Shallow linear model.
        x = self.mlp(x)

        return x