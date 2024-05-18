"""Classical submodels."""

import torch
import torch.nn as nn

class ClassicalMLP(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            hidden_channel: int,
            num_layers: int
        ):
        """Classical MLP (feed forward neural network).

        Activation functions are default to be `nn.ReLU()`. There will 
        be NO activation functions in the last layer.

        Args:
            in_channel : int
                Dimension of initial input channels.
            out_channel : int
                Dimension of final output channels.
            hidden_channel : int
                Number of hidden neurons, assumed to be the same in each
                hidden layers.
            num_layers : int
                Number of hidden layers. If `num_layers=0`, then only an
                input layer and an output layer without activation
                functions (used for changing dimension of data only).
        """

        super().__init__()

        # Add an activation function for each layer.
        net = [nn.Linear(in_channel, hidden_channel), nn.ReLU()]
        for _ in range(num_layers - 1):
            net += [nn.Linear(hidden_channel, hidden_channel), nn.ReLU()]
        net += [nn.Linear(hidden_channel, out_channel)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ElementwiseLinear(nn.Module):
    """Create a single connected linear layer.

    Unlike `nn.Linear` (which fully connects all neurons), this layer
    intends to perform a linear transformation for each neuron only.
    If the input has two dimensions, e.g., $x=(x_1, x_2)$, it will be 
    transformed seperately as $y_1=w_1*x_1+b_1$ and $y_2=w_2*x_2+b_2$.

    This class is modified from
    https://stackoverflow.com/questions/51980654/pytorch-element-wise-filter-layer
    """

    def __init__(self, in_channel: int):
        super().__init__()
        
        self.w = nn.Parameter(torch.Tensor(in_channel))
        self.b = nn.Parameter(torch.Tensor(in_channel))

    def forward(self, x):
        return self.w * x + self.b