import torch
from torch import nn

@torch.jit.script
def knn(points, K: int = 2):
    """Calculate k-nearest neighbors."""

    # Calculate the distance of all points.
    distance = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)

    # Find out the indices of k nearest neighbor
    _, topk_index = torch.topk(-distance, k=(K + 1))

    return topk_index[..., 1:]


@torch.jit.script
def extend_neighbor_distance(x: torch.Tensor, knn_index: torch.Tensor):
    """Extend with neighbor distance (in feature space).

        Args:
            x : torch.Tensor
                Shape = (N, P, C).
            knn_index : torch.Tensor
                Shape = (N, P, K).
    """

    N = knn_index.shape[0]
    P = knn_index.shape[1]
    K = knn_index.shape[2]
    C = x.shape[2]

    knn_index = knn_index.reshape(N, P * K) # (N, P * K)

    f_neighbor = torch.cat([x[i, sub_index] for i, sub_index in enumerate(knn_index)]) # (N * P * K, C)
    f_neighbor = f_neighbor.reshape(N, P, K, C)

    f_center = torch.tile(torch.unsqueeze(x, dim=-2), (1, 1, K, 1)) # (N, P, K, C)
    f_extend = torch.cat([f_center, torch.subtract(f_center, f_neighbor)], dim=-1) # (N, P, K, 2 * C)

    return torch.permute(f_extend, (0, 3, 1, 2)) # (N, 2 * C, P, K)


class EdgeConv(nn.Module):

    def __init__(self, edge_conv_parameters: list[list[int, list[int, int]]]):
        super().__init__()
        
        self.K, self.channel_list = edge_conv_parameters

        conv_layers = []

        for i, (C_in, C_out) in enumerate(self.channel_list):
            if i == 0:
                # Feature redefinition.
                C_in = 2 * C_in

            conv_layers.append(nn.Conv2d(C_in, C_out, kernel_size=(1, 1), bias=False))
            conv_layers.append(nn.ReLU())

        self.conv_layer = nn.Sequential(*conv_layers)

        self.residual = nn.Conv1d(self.channel_list[0][0], self.channel_list[-1][-1], kernel_size=1, bias=False)

        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, direction: torch.Tensor = None):

        """
            Args:
                x : torch.Tensor
                    Shape = (N, P, C).
                mask : torch.Tensor
                    Shape = (N, P).
                direction : torch.Tensor (default None)
                    Shape = (N, P, 2).
        """

        # For residual.
        x_residual = self.residual(x.transpose(-1, -2)) # (N, C', P)
        x_residual = x_residual.transpose(-1, -2) # (N, P, C')

        # knn method
        if direction is None:
            direction = x.masked_fill(mask.unsqueeze(-1), float('inf'))
        knn_index = knn(direction, K=self.K) # (N, P, K)
        x = extend_neighbor_distance(x, knn_index) # (N, 2 * C, P, K), 2 means x and x'

        # Mask of K
        mask_K = mask.shape[1] - torch.sum(mask, dim=-1) # (N)
        mask_K = (knn_index >= mask_K.unsqueeze(dim=-1).unsqueeze(dim=-1)) # (N, P, K)

        # Convolution layer
        x = self.conv_layer(x) #(N, C', P, K)
        x = x.masked_fill(mask_K.unsqueeze(dim=-3), 0.)

        # Aggregation
        x = torch.sum(x, dim=-1) / (self.K - torch.sum(mask_K, dim=-1).unsqueeze(-2)) # (N, C', P)
        x = x.transpose(-1, -2) # (N, P, C')

        # Residual 
        x = self.final_act(x + x_residual) # (N, P, C')

        return x

class ParticleNet(nn.Module):

    def __init__(self, score_dim: int, parameters: dict):
        super().__init__()

        embedding_dim = parameters['edge_conv'][0][1][0][0]
        self.par_embedding = nn.Sequential(
            nn.Conv1d(parameters['input_dim'], embedding_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.edge_conv = nn.ModuleList([EdgeConv(param) for param in parameters['edge_conv']])
        
        fc_layer = []

        for drop_rate, nodes in parameters['fc']:

            fc_layer.append(nn.Linear(*nodes))
            fc_layer.append(nn.ReLU())
            fc_layer.append(nn.Dropout(p=drop_rate))

        self.fc = nn.Sequential(*fc_layer)
        
        self.output_layer = nn.Linear(parameters['fc'][-1][-1][-1], score_dim)


    # def forward(self, features, direction):
    def forward(self, x: torch.Tensor):
        """
            x : torch.Tensor
                Shape = (N, P, C) with C = 3 for (pt, delta_eta, delta_phi).
        """

        # Mask for padded particles.
        with torch.no_grad():
            mask = torch.isnan(x[..., 0]) # (N, P)
            x = x.masked_fill(torch.isnan(x), 0.)

        # Direction for knn of the first edge convolution.
        direction = x[..., -2:] # Last two are `delta_eta` and `delta_phi`.
        direction = direction.masked_fill(mask.unsqueeze(-1), float('inf'))

        # Transform the dimension of the input.
        x = x.transpose(-1, -2) # (N, P, C) -> (N, C, P)
        x = self.par_embedding(x) # (N, C, P) -> # (N, C', P)
        x = x.transpose(-1, -2) # (N, P, C)
        x = x.masked_fill(mask.unsqueeze(-1), 0.)

        # Edge convolution
        for i, conv in enumerate(self.edge_conv):
            if i == 0:
                x = conv(x, mask, direction) # (N, P, C')
            else:
                x = conv(x, mask) # (N, P, C')
            
            x = x.masked_fill(mask.unsqueeze(-1), 0.)

        # Global average pooling
        x = torch.sum(x, dim=-2) / (mask.shape[-1] - torch.sum(mask, dim=-1).unsqueeze(-1)) # (N, C')

        # Fully connected layer
        x = self.fc(x)

        return self.output_layer(x)