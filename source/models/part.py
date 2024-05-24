"""Particle transformer without interaction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleFeatureEmbedding(nn.Module):

    def __init__(self, input_dim: int, embedding_dims: list) -> None:

        super().__init__()

        embedding_chain = []

        _input_dim = input_dim

        for _embedding_dim in embedding_dims:

            embedding_chain += [
                nn.LayerNorm(_input_dim),
                nn.Linear(_input_dim, _embedding_dim),
                nn.GELU(),
            ]

            _input_dim = _embedding_dim

        self.embedding = nn.Sequential(*embedding_chain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.embedding(x)

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


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Linear projections.
        q = self.q_linear(query) # (N, L, E)
        k = self.k_linear(key)   # (N, L, E)
        v = self.v_linear(value) # (N, L, E)

        # Split into multiple heads. (E = H x D)
        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2) # (N, L, E) -> (N, L, H, D) -> (N, H, L, D)
        k = k.view(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2) # (N, L, E) -> (N, L, H, D) -> (N, H, L, D)
        v = v.view(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2) # (N, L, E) -> (N, L, H, D) -> (N, H, L, D)

        # Scaled Dot-Product Attention.
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5) # (N, H, L, D) x (N, H, D, L) = (N, H, L, L)

        # Apply mask on scores.
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(-2).unsqueeze(-2) # (N, 1, 1, L) | (N, 1, 1, 1 + L)
            scores = scores.masked_fill(key_padding_mask, float('-inf')) # (N, H, L, L) | (N, H, 1, 1 + L)

        attn_weights = F.softmax(scores, dim=-1)  # (N, H, L, L)
        attn_weights = self.dropout(attn_weights) # (N, H, L, L)

        # Weighted sum of values.
        weighted_sum = torch.matmul(attn_weights, v) # (N, H, L, L) x (N, H, L, D) = (N, H, L, D)

        if self.gamma is not None:
            weighted_sum = torch.einsum('bhtd,h->bhtd', weighted_sum, self.gamma)

        # Concatenate attention heads and apply the final linear projection.
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).contiguous().view(q.size(0), -1, self.embed_dim) # (N, H, L, D) -> (N, L, H, D) -> (N, L, E)
        output = self.out_linear(weighted_sum) # (N, L, E)

        return output

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, fc_dim: int, dropout: float = 0.1, **kwargs) -> None:
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

        self.lambda_residual = nn.Parameter(torch.ones(embed_dim), requires_grad = True)

    def forward(self, x: torch.Tensor, x_clt: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if x_clt is not None: # Class Attention Block.

            # Add mask for class token which is added in the key.
            with torch.no_grad():
                if key_padding_mask is not None:
                    clt_padding_mask = torch.zeros_like(key_padding_mask[..., 0]).unsqueeze(-1) # (N, 1)
                    key_padding_mask = torch.cat((clt_padding_mask, key_padding_mask), dim=-1) # (N, 1) + (N, L) = (N, 1 + L)

            # First residual values.
            residual = x_clt

            # Multi-head attention (MHA).
            x = torch.cat((x_clt, x), dim=-2) # (N, 1, E) + (N, L, E) = (N, 1 + L, E)
            x = self.pre_atte_layerNorm(x)
            x = self.atte(query=x_clt, key=x, value=x, key_padding_mask=key_padding_mask) # (N, 1, E)

        else: # Particle Attention Block.

            # First residual values.
            residual = x

            # Particle Multi-head attention (P-MHA) without U interaction.
            x = self.pre_atte_layerNorm(x)
            x = self.atte(query=x, key=x, value=x, key_padding_mask=key_padding_mask)

        x = self.post_atte_layerNorm(x)
        x = self.dropout1(x)

        # First residual connection.
        x = x + residual

        # Second residual values.
        residual = x

        # Fully connected layers after first residual connection.
        x = self.pre_fc_layerNorm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.post_fc_layerNorm(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        # Resudual scaling.
        residual = torch.mul(self.lambda_residual, residual)

        # Second residual connection.
        x = x + residual

        return x

class ParticleTransformer(nn.Module):
    def __init__(self, score_dim: int, parameters: dict) -> None:
        """Particle Transformer (without interaction).
        
        Args:
            score_dim : int
                Dimension of final output.
            parameters : dict
                Hyperparameters for the model, see `configs/benchmark.yaml`.
        """
        super().__init__()

        # Particle Embedding.
        self.par_embedding = ParticleFeatureEmbedding(parameters['ParEmbed']['input_dim'], parameters['ParEmbed']['embed_dim'])
        
        # Embedding dimension used for all attention blocks and fully connected layers.
        atte_embed_dim = parameters['ParEmbed']['embed_dim'][-1]

        # Particle Attention Blocks.
        self.par_atte_blocks = nn.ModuleList([
            AttentionBlock(
                embed_dim=atte_embed_dim,
                **parameters['ParAtteBlock']
            ) for _ in range(parameters['num_ParAtteBlock'])
        ])

        # Class Attention Blocks.
        self.class_atte_blocks = nn.ModuleList([
            AttentionBlock(
                embed_dim=atte_embed_dim,
                **parameters['ClassAtteBlock']
            ) for _ in range(parameters['num_ClassAtteBlock'])
        ])

        # Class token (used in Class Attention Blocks).
        self.class_token = nn.Parameter(torch.zeros(1, 1, atte_embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.class_token)

        self.layerNorm = nn.LayerNorm(atte_embed_dim)

        self.fc = nn.Sequential(nn.Linear(atte_embed_dim, atte_embed_dim), nn.ReLU(), nn.Dropout(0.1))

        self.final_layer = nn.Linear(atte_embed_dim, score_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x : torch.Tensor
                    Input tensor of shape (N, L, D), where N is the batch size, 
                    L is the sequence length, and D is the feature dimension.
        """

        with torch.no_grad():
            key_padding_mask = torch.isnan(x[..., 0]) # (N, L)

        # Embedding
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0) # (N, L, D)
        x = self.par_embedding(x) # (N, L, E)

        # Particle Attention Block
        for block in self.par_atte_blocks:
            x = block(x, x_clt=None, key_padding_mask=key_padding_mask) # (N, L, E)

        # Class Attention Block
        class_token = self.class_token.expand(x.size(0), -1, -1) # (N, 1, E)
        for block in self.class_atte_blocks:
            class_token = block(x, x_clt=class_token, key_padding_mask=key_padding_mask) # (N, L, E)

        # Layer normalization
        class_token = self.layerNorm(class_token).squeeze(1)

        # Fully connected layer
        class_token = self.fc(class_token)
        class_token = self.final_layer(class_token)

        return class_token
