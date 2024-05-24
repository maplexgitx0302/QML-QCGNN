# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def to_ptrapphie(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # (Px, Py, Pz, E) --> (Pt, Y, Phi, E)

    # Input x : (N, 4, L)
    # Output  : 4 * (N, 1, L)

    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1) # 4*(N, 1, L)

    pt       = torch.sqrt(px**2 + py**2)
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi      = torch.atan2(py, px)

    return pt, rapidity, phi, energy

@torch.jit.script
def delta_phi(phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    return ((phi2 - phi1) + torch.pi) % (2 * torch.pi) - torch.pi

@torch.jit.script
def prepare_interaction(xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:

    pt_i, rapidity_i, phi_i, energy_i = to_ptrapphie(xi)
    pt_j, rapidity_j, phi_j, energy_j = to_ptrapphie(xj)

    delta = torch.sqrt( (rapidity_i - rapidity_j)**2 + delta_phi(phi_i, phi_j)**2 ).clamp(min=1e-8)
    ptmin = torch.minimum(pt_i, pt_j)
    kt = (ptmin * delta).clamp(min=1e-8)
    z  = (ptmin / (pt_i + pt_j).clamp(min=1e-8)).clamp(min=1e-8)

    xij = xi + xj
    m2 = (xij[:,3]**2 - torch.sum(xij[:, 0:3]**2, dim=1)).unsqueeze(1).clamp(min=1e-8)

    return torch.log(torch.cat((delta, kt, z, m2), dim=1))

class ParticleFeatureEmbedding(nn.Module):

    def __init__(self, input_dim: int, embedding_dims: list) -> None:

        super().__init__()


        self.BatchNorm1d = nn.BatchNorm1d(input_dim)

        embedding_chain = []

        _input_dim = input_dim

        for _embedding_dim in embedding_dims:

            embedding_chain.extend([
                nn.LayerNorm(_input_dim),
                nn.Linear(_input_dim, _embedding_dim),
                nn.GELU()
            ])

            _input_dim = _embedding_dim

        self.embedding = nn.Sequential(*embedding_chain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input x : [N, L, C]
        # Output x : [N, L, C']

        x = x.permute(0, 2, 1).contiguous() # (N, C, L)
        x = self.BatchNorm1d(x)
        x = x.permute(0, 2, 1).contiguous() # (N, L, C')

        return self.embedding(x)

class InteractionMatrixEmbedding(nn.Module):

    def __init__(self, input_dim: int, embedding_dims: list) -> None:

        super().__init__()

        embedding_chain = [nn.BatchNorm1d(input_dim)]

        _input_dim = input_dim

        for _embedding_dim in embedding_dims:

            embedding_chain.extend([
                nn.Conv1d(_input_dim, _embedding_dim, 1),
                nn.BatchNorm1d(_embedding_dim),
                nn.GELU()
            ])

            _input_dim = _embedding_dim

        self.embedding = nn.Sequential(*embedding_chain)

        self.out_dim = embedding_dims[-1]


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input x : [N, L, 4]
        # --> (Px, Py, Pz, Energy)

        with torch.no_grad():
            x = x.permute(0, 2, 1).contiguous() # (N, 4, L)

            ## Make symmetric pair indices
            ## i = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, ...]
            ## j = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, ...]
            batch_size, _, seq_len = x.shape

            #i, j = torch.tril_indices(seq_len, seq_len, device=x.device) # (L*(L+1)/2)

            #xi = x[:,:,i] # (N, 4, L*(L+1)/2)
            #xj = x[:,:,j]

            ## Prepare particle pair interaction sequence
            #x = prepare_interaction(xi, xj) # (N, 4, L*(L+1)/2)

            x = prepare_interaction(x.unsqueeze(-1), x.unsqueeze(-2))
            x = x.view(-1, x.size(1), seq_len * seq_len)

        # Embedding by using Conv1d
        x = self.embedding(x) # (N, C, L*(L+1)/2)

        ## Make symmetric interaction matrix
        #y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=x.dtype, device=x.device) # (N, C, L, L)
        #y[:, :, i, j] = x
        #y[:, :, j, i] = x

        x = x.view(-1, self.out_dim, seq_len, seq_len)

        return x

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: int = 0.1, bias: bool = False, isheadscale: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, and values
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Linear projection for the output of attention heads
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        self.gamma = None
        # Head scale from https://arxiv.org/pdf/2110.09456.pdf
        if isheadscale:
            self.gamma = nn.Parameter(torch.ones(num_heads), requires_grad = True)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            atte_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Linear projections
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # Split into multiple heads
        q = q.view(q.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(k.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(v.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        # Apply attention mask
        if atte_mask is not None:
            scores = scores + atte_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        weighted_sum = torch.matmul(attn_weights, v)

        if self.gamma is not None:
            weighted_sum = torch.einsum('bhtd,h->bhtd', weighted_sum, self.gamma)

        # Concatenate attention heads and apply the final linear projection
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).contiguous().view(q.size(0), -1, self.embed_dim)
        output = self.out_linear(weighted_sum)

        return output

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, fc_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.atte = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.pre_atte_layerNorm = nn.LayerNorm(embed_dim)
        self.post_atte_layerNorm = nn.LayerNorm(embed_dim)

        self.pre_fc_layerNorm = nn.LayerNorm(embed_dim)
        self.post_fc_layerNorm = nn.LayerNorm(fc_dim)

        self.fc1 = nn.Linear(embed_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim ,embed_dim)
        self.act = nn.GELU()

        self.act_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.lambda_resid = nn.Parameter(torch.ones(embed_dim), requires_grad = True)

    def forward(self, x, x_clt: Optional[torch.Tensor] = None,
            atte_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Input x : (N, L, E)

        if x_clt is not None:

            # Add mask for class token which is added in the key
            with torch.no_grad():
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat((torch.zeros_like(key_padding_mask[:, :1]), key_padding_mask), dim=1) # (N, 1) + (N, L) = (N, 1+L)

            residual = x_clt

            x = torch.cat((x_clt, x), dim=1) # (N, 1, E) + (N, L, E) = (N, 1+L, E)
            x = self.pre_atte_layerNorm(x)
            x = self.atte(x_clt, x, x, key_padding_mask=key_padding_mask) # (N, 1, E)

        else:

            residual = x

            x = self.pre_atte_layerNorm(x)
            x = self.atte(x, x, x, atte_mask=atte_mask, key_padding_mask=key_padding_mask)

        x = self.post_atte_layerNorm(x)
        x = self.dropout1(x)

        x = x + residual

        residual = x

        x = self.pre_fc_layerNorm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.post_fc_layerNorm(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        # ResScale
        residual = torch.mul(self.lambda_resid, residual)

        x = x + residual

        return x

class ParT(nn.Module):
    def __init__(self, parameters: dict) -> None:
        super().__init__()


        self.par_embedding = ParticleFeatureEmbedding(
                              parameters['ParEmbed']['input_dim'],
                              parameters['ParEmbed']['embed_dim'])
        self.int_embedding = InteractionMatrixEmbedding(
                              parameters['IntEmbed']['input_dim'],
                              parameters['IntEmbed']['embed_dim'] + [parameters['ParAtteBlock']['num_heads']])


        self.par_atte_blocks = nn.ModuleList( [ AttentionBlock(
                                      embed_dim=parameters['ParEmbed']['embed_dim'][-1],
                                      num_heads=parameters['ParAtteBlock']['num_heads'],
                                      fc_dim=parameters['ParAtteBlock']['fc_dim'],
                                      dropout=parameters['ParAtteBlock']['dropout']
                                      ) for _ in range(parameters['num_ParAtteBlock'])
                                      ] )

        self.class_atte_blocks = nn.ModuleList( [ AttentionBlock(
                                      embed_dim=parameters['ParEmbed']['embed_dim'][-1],
                                      num_heads=parameters['ClassAtteBlock']['num_heads'],
                                      fc_dim=parameters['ClassAtteBlock']['fc_dim'],
                                      dropout=parameters['ClassAtteBlock']['dropout']
                                      ) for _ in range(parameters['num_ClassAtteBlock'])
                                      ] )

        # Class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, parameters['ParEmbed']['embed_dim'][-1]), requires_grad=True)
        nn.init.trunc_normal_(self.class_token)


        self.layerNorm = nn.LayerNorm(parameters['ParEmbed']['embed_dim'][-1])

        self.fc = nn.Sequential(nn.Linear(parameters['ParEmbed']['embed_dim'][-1], parameters['ParEmbed']['embed_dim'][-1]), nn.ReLU(), nn.Dropout(0.1))

        self.final_layer = nn.Linear(parameters['ParEmbed']['embed_dim'][-1], 2)

        self.make_probability = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:

        # Input x : (N, P, C)
        #       p4: (N, P, 4)

        # key padding mask
        key_padding_mask = (x[:,:,0] == 0.) # (N, L)

        # Embedding
        x     = self.par_embedding(x)
        x_int = self.int_embedding(p4)

        # Particle Attention Block
        for block in self.par_atte_blocks:
            x = block(x, x_clt=None, atte_mask=x_int, key_padding_mask=key_padding_mask)

        # Class Attention Block
        class_token = self.class_token.expand(x.size(0), -1, -1)  # (N, 1, C)
        for block in self.class_atte_blocks:
            class_token = block(x, x_clt=class_token, key_padding_mask=key_padding_mask)

        class_token = self.layerNorm(class_token).squeeze(1)

        # Fully connected layer
        class_token = self.fc(class_token)
        class_token = self.final_layer(class_token)

        return self.make_probability(class_token)
