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
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # `x_i` and `x_j` are source node and target node respectively.
        return self.phi(torch.cat((x_i, x_j), dim=-1))
    
    def update(self, aggr_out, x):
        # Set `gamma` as identity function, so `x` will not be used.
        return aggr_out


class GraphMPGNN(nn.Module):
    def __init__(self, phi: nn.Module, mlp: nn.Module, aggr: Optional[str] = 'add'):
        """MPGNN with SUM as the aggregation function.
        
        Instead of combining this part with the class `ClassicalMPGNN` 
        below, we seperate out for the possibility of other design of 
        mlp.

        Args:
            phi : nn.Module
                See `MessagePassing` above.
            mlp : nn.Module
                Basically just a simple shallow fully connected linear
                model for transforming dimensions.
            aggr : str
                Aggregation method used in MessagePassing.
        """

        super().__init__()

        self.aggr = aggr # Aggregation method.
        self.gnn = MessagePassing(phi, aggr)
        self.mlp = mlp # Multi-layer perceptron.

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


class ParticleFlowNetwork(GraphMPGNN):
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
        
        # See `GraphMPGNN` above.
        phi = classical.ClassicalMLP(
            in_channel=phi_in,
            out_channel=phi_out,
            hidden_channel=phi_hidden,
            num_layers=phi_layers,
            dropout=dropout,
        )
        
        mlp = nn.Sequential(
            nn.Linear(phi_out, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, score_dim),
        )

        super().__init__(phi, mlp, aggr=aggregation)