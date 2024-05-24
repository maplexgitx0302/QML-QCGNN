import torch
from torch import nn
from collections import OrderedDict

@torch.jit.script
def point_distance_matrix(A, B):
    r_A = torch.unsqueeze(torch.sum(A*A, dim=2), dim=2)
    r_B = torch.unsqueeze(torch.sum(B*B, dim=2), dim=2)
    A_dot_B = torch.matmul(A, torch.permute(B, (0, 2, 1)))

    return r_A - 2 * A_dot_B + torch.permute(r_B, (0, 2, 1))

@torch.jit.script
def gather_nd(tensor, index):
    '''
        tensor : (N, P, C)
        index  : (N, P, K)
        output : (N, P, K, C)
    '''
    N = index.shape[0]
    P = index.shape[1]
    K = index.shape[2]
    C = tensor.shape[2]

    index = index.reshape(N, P*K) #(N, P*K)

    return torch.cat( [tensor[i,sub_index] for i, sub_index in enumerate(index)] ).reshape(N, P, K, C)

@torch.jit.script
def knn(points, K: int = 2):

    #Calculate the distance of all points
    distance = point_distance_matrix(points, points)

    #Find out the indices of k nearest neighbor
    _, topk_index = torch.topk(-distance, k=K+1)
    return topk_index[:,:,1:]

@torch.jit.script
def feature_redefine(features, knn_index):
    f  = gather_nd(features, knn_index) #(N, P, K, C)
    fc = torch.tile(torch.unsqueeze(features, dim=2), (1,1,f.shape[2],1)) #(N, P, K, C)
    ff = torch.cat([fc, torch.subtract(fc, f)], dim=-1) #(N, P, K, 2*C)

    return torch.permute(ff, (0, 3, 1, 2))

class Edge_Conv(nn.Module):

    def __init__(self, index, edge_conv_parameters):
        super().__init__()

        self.index = index
        self.K, self.channel_list = edge_conv_parameters

        self.conv_layer = self._make_conv_layer()

        self.shortcut = nn.Sequential(
            nn.Conv1d(self.channel_list[0][0], self.channel_list[-1][-1], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channel_list[-1][-1])
        )

        self.final_act = nn.ReLU()

    # Convolution layer maker
    def _make_conv_layer(self):

        '''
            [Conv2d--BatchNorm2d--ReLU] * n
        '''

        layer = []

        for i_conv, (C_in, C_out) in enumerate(self.channel_list):

            layer.append( ('edge_conv_Conv2d_{}_{}'.format(self.index, i_conv), nn.Conv2d(C_in*2 if i_conv == 0 else C_in, C_out, kernel_size=(1,1), bias=False)) )
            layer.append( ('edge_conv_BatchNorm2d_{}_{}'.format(self.index, i_conv), nn.BatchNorm2d(C_out)) )
            layer.append( ('edge_conv_ReLU_{}_{}'.format(self.index, i_conv), nn.ReLU()) ) #(N, C, P, K)

        return nn.Sequential( OrderedDict(layer) )


    def forward(self, features, mask):

        '''
            points  : (N, P, 2)
            feature : (N, P, C) if index == 0
                      (N, C, P) if index > 0
            mask    : (N, P)
        '''

        # If second edge convolution , permutation (N, C, P) ---> (N, P, C)
        if self.index != 0:
            features = torch.permute(features, (0,2,1))

        X = features

        # The first edge convolution chooses (eta, phi) to judge the point distance and the others as input features
        pts = X[:,:,0:2] if self.index == 0 else X
        X = X[:,:,2:] if self.index == 0 else X

        if self.index != 0:
            pts = pts.masked_fill( mask.unsqueeze(2), 10e11 )

        X_shortcut = X

        # knn method
        knn_index = knn(pts, K=self.K) #(N, P, K)
        X = feature_redefine(X, knn_index) #(N, 2*C, P, K), 2 means x and x'

        # Mask of K
        mask_K = mask.shape[1] - torch.sum( mask, dim=1 ) # (N)
        mask_K = (knn_index >= mask_K.unsqueeze(dim=1).unsqueeze(dim=2)) # (N, P, K)

        # Convolution layer
        X = self.conv_layer(X) #(N, C', P, K)
        X = X.masked_fill(mask_K.unsqueeze(dim=1), 0.)

        # Aggregation
        X = torch.sum(X, dim=3) / (self.K - torch.sum(mask_K, dim=2).unsqueeze(1)).clamp(min=1e-8) #(N, C', P)
        X.masked_fill(mask.unsqueeze(1), 0.)

        # Residual 
        init_X = self.shortcut( torch.permute(X_shortcut, (0,2,1)) )  #(N, C', P)
        init_X = init_X.masked_fill(mask.unsqueeze(1), 0.)

        return self.final_act(X+init_X) #(N, C', P)

class Particle_Net(nn.Module):

    def __init__(self, parameters: dict, is_inference: bool = True):
        super().__init__()

        self.is_inference = is_inference

        self.par_embedding = nn.Sequential(
            nn.BatchNorm1d(parameters['input_dim']),
            nn.Conv1d(parameters['input_dim'], 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.edge_conv_parameters = parameters['edge_conv']
        self.fc_parameters = parameters['fc']

        self.Edge_Conv   = self._make_edge_conv()
        self.FC          = self._make_fully_connected_layer()
        self.output_layer = nn.Linear(self.fc_parameters[-1][-1][-1], parameters['nclass'])

        self.make_probability = nn.Softmax(dim=1)

    def _make_edge_conv(self):

        block = [ Edge_Conv(i_block, param) for i_block, param in enumerate(self.edge_conv_parameters) ]
        return nn.ModuleList(block)

    def _make_fully_connected_layer(self):

        layer = []

        for i_layer, param in enumerate(self.fc_parameters):

            drop_rate, nodes = param

            layer.append( ('Linear_{}'.format(i_layer), nn.Linear(nodes[0], nodes[1])) )
            layer.append( ('ReLU_{}'.format(i_layer), nn.ReLU()) )
            layer.append( ('Dropout_{}'.format(i_layer), nn.Dropout(p=drop_rate)) )

        return nn.Sequential( OrderedDict(layer) )


    def forward(self, features, direction):

        '''
            features : (N, P, C)
            direction : (N, P, 2) --> "2" means eta and phi
        '''

        # Mask
        mask = (direction[:,:,0] > 10e9) # (N, P)

        # First batch normalization
        fts = self.par_embedding( torch.permute(features, (0,2,1)) ) # (N, C, P) -> # (N, C', P)
        fts = fts.masked_fill(mask.unsqueeze(1), 0.)
        fts = torch.permute(fts, (0,2,1)) # (N, P, C)

        # Conbination
        fts = torch.cat((direction, fts), dim=2)

        # Edge convolution
        for conv in self.Edge_Conv:
            fts = conv(fts, mask) # (N, C', P)

        # Global average pooling
        fts = torch.sum(fts, dim=2) / (64. - torch.sum(mask, dim=1).unsqueeze(1)).clamp(min=1e-8) # (N, C')

        # Fully connected layer
        fts = self.FC(fts) # (N, C'')

        # Output layer to 4 classes
        fts = self.output_layer(fts)

        return self.make_probability(fts) if self.is_inference else fts
