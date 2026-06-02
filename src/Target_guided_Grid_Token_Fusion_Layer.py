import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Learnable2DPositionalEmbedding(nn.Module):
    def __init__(self, grid_size: int, d_model: int):
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model

        self.row_embed = nn.Embedding(grid_size, d_model)
        self.col_embed = nn.Embedding(grid_size, d_model)

        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

        rows = torch.arange(grid_size)
        cols = torch.arange(grid_size)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("row_ids", yy.reshape(-1), persistent=False)
        self.register_buffer("col_ids", xx.reshape(-1), persistent=False)

    def forward(self) -> torch.Tensor:
        pos = self.row_embed(self.row_ids) + self.col_embed(self.col_ids)
        return pos.unsqueeze(0)


class DistanceBiasedMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 必须能被 nhead={nhead} 整除")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        distance_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if distance_bias is not None:
            attn_scores = attn_scores + distance_bias

        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(out)
        return out


class DistanceBiasedTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = DistanceBiasedMultiheadSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("activation 只能是 'relu' 或 'gelu'")

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        distance_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h, key_padding_mask=key_padding_mask, distance_bias=distance_bias)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.linear2(self.ffn_dropout(self.activation(self.linear1(h))))
        x = x + self.dropout2(h)

        return x


class RadialBiasedMultiheadCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 必须能被 nhead={nhead} 整除")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        radial_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, lq, _ = query.shape
        _, g, _ = key_value.shape

        q = self.q_proj(query).view(bsz, lq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(bsz, g, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(bsz, g, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if radial_bias is not None:
            attn_scores = attn_scores + radial_bias

        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, lq, self.d_model)
        out = self.out_proj(out)
        return out


class TargetGuidedRadialBiasedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.nhead = nhead

        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_q2 = nn.LayerNorm(d_model)

        self.cross_attn = RadialBiasedMultiheadCrossAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout = nn.Dropout(dropout)


        self.radial_base_penalty = nn.Parameter(torch.ones(nhead))


        self.radial_condition_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, nhead),
        )

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("activation 只能是 'relu' 或 'gelu'")

    def forward(
        self,
        target_token: torch.Tensor,
        grid_tokens: torch.Tensor,
        normalized_radial_distance: torch.Tensor,
        grid_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q1(target_token)



        cond_penalty = F.softplus(
            self.radial_base_penalty.unsqueeze(0) + self.radial_condition_mlp(q.squeeze(1))
        )

        if normalized_radial_distance.dim() == 2:

            radial = normalized_radial_distance.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError("normalized_radial_distance 必须是 [1,G] 或 [B,G]")

        if radial.size(0) == 1:
            radial = radial.expand(q.size(0), -1, -1, -1)

        radial_bias = cond_penalty.unsqueeze(-1).unsqueeze(-1) * radial

        attn_out = self.cross_attn(
            query=q,
            key_value=grid_tokens,
            key_padding_mask=grid_key_padding_mask,
            radial_bias=radial_bias,
        )



        h = self.norm_q2(target_token)
        h = self.linear2(self.ffn_dropout(self.activation(self.linear1(h))))
        target_token = self.dropout1(attn_out) + self.dropout2(h)

        return target_token


class TargetGuidedGridTokenFusionLayer(nn.Module):
    def __init__(
        self,
        grid_size_per_row: int,
        square_size_m: float,
        grid_input_dim: int,
        target_input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_grid_layers: int = 2,
        num_cross_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        distance_penalty_init: float = 1.0,
        distance_power: float = 1.5,
        use_gated_fusion: bool = False,
        use_add_fusion: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.grid_size_per_row = grid_size_per_row
        self.square_size_m = float(square_size_m)
        self.grid_input_dim = grid_input_dim
        self.target_input_dim = target_input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.distance_power = distance_power
        self.use_gated_fusion = use_gated_fusion
        self.eps = eps

        grid_token_num = grid_size_per_row * grid_size_per_row
        self.grid_token_num = grid_token_num


        self.grid_proj = nn.Linear(grid_input_dim, d_model)
        self.target_proj = nn.Linear(target_input_dim, d_model)


        self.grid_pos_embed = Learnable2DPositionalEmbedding(
            grid_size=grid_size_per_row,
            d_model=d_model,
        )


        grid_coords = self._create_grid_coords(
            grid_size_per_row=grid_size_per_row,
            square_size_m=square_size_m,
        )
        self.register_buffer("grid_coords", grid_coords, persistent=False)


        normalized_distance = self._build_normalized_grid_distance(
            grid_coords=grid_coords,
            power=distance_power,
            eps=eps,
        )
        self.register_buffer("normalized_distance", normalized_distance, persistent=False)


        normalized_radial_distance = self._build_normalized_radial_distance(
            grid_coords=grid_coords,
            power=distance_power,
            eps=eps,
        )
        self.register_buffer("normalized_radial_distance", normalized_radial_distance, persistent=False)


        self.distance_penalty = nn.Parameter(
            torch.ones(nhead) * distance_penalty_init
        )

        self.grid_encoder_layers = nn.ModuleList([
            DistanceBiasedTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_grid_layers)
        ])

        self.cross_layers = nn.ModuleList([
            TargetGuidedRadialBiasedCrossAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_cross_layers)
        ])

        self.grid_final_norm = nn.LayerNorm(d_model)
        self.target_final_norm = nn.LayerNorm(d_model)

        if use_gated_fusion:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
        self.use_add_fusion = use_add_fusion


    def _create_grid_coords(self, grid_size_per_row: int, square_size_m: float) -> torch.Tensor:
        half = square_size_m / 2.0
        xs = torch.linspace(-half, half, grid_size_per_row, dtype=torch.float32)
        ys = torch.linspace(-half, half, grid_size_per_row, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        return coords

    def _build_normalized_grid_distance(
        self,
        grid_coords: torch.Tensor,
        power: float,
        eps: float,
    ) -> torch.Tensor:
        diff = grid_coords.unsqueeze(1) - grid_coords.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        max_dist = dist.max().clamp_min(eps)
        norm_dist = torch.log((max_dist.pow(power) + 1.0) / (dist.pow(power) + 1.0 + eps))
        return norm_dist.unsqueeze(0)

    def _build_normalized_radial_distance(
        self,
        grid_coords: torch.Tensor,
        power: float,
        eps: float,
    ) -> torch.Tensor:
        radial = torch.norm(grid_coords, dim=-1)
        max_radial = radial.max().clamp_min(eps)
        norm_radial = torch.log((max_radial.pow(power) + 1.0) / (radial.pow(power) + 1.0 + eps))
        return norm_radial.unsqueeze(0)

    def _build_distance_bias(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = self.normalized_distance.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        bias = self.distance_penalty.view(1, self.nhead, 1, 1) * base.unsqueeze(1)
        return bias

    def forward(
        self,
        grid_features: torch.Tensor,
        target_name_embedding: torch.Tensor,
        grid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if grid_features.dim() != 3:
            raise ValueError("grid_features 必须是 [B,G,D_in]")
        if target_name_embedding.dim() != 2:
            raise ValueError("target_name_embedding 必须是 [B,D_target]")

        batch_size, grid_token_num, _ = grid_features.shape
        if grid_token_num != self.grid_token_num:
            raise ValueError(
                f"grid token 数不匹配: 输入是 {grid_token_num}，模块期望 {self.grid_token_num}"
            )

        if grid_mask is None:
            grid_mask = torch.ones(
                batch_size, grid_token_num,
                dtype=torch.bool,
                device=grid_features.device,
            )
        else:
            grid_mask = grid_mask.bool()

        grid_key_padding_mask = ~grid_mask


        x = self.grid_proj(grid_features)
        x = x + self.grid_pos_embed().to(x.dtype).to(x.device)


        distance_bias = self._build_distance_bias(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )


        for layer in self.grid_encoder_layers:
            x = layer(
                x,
                key_padding_mask=grid_key_padding_mask,
                distance_bias=distance_bias,
            )
            x = x * grid_mask.unsqueeze(-1).to(x.dtype)




        target_sem = self.target_proj(target_name_embedding)
        target_token = target_sem.unsqueeze(1)


        normalized_radial_distance = self.normalized_radial_distance.to(
            device=x.device, dtype=x.dtype
        )

        for layer in self.cross_layers:
            target_token = layer(
                target_token=target_token,
                grid_tokens=x,
                normalized_radial_distance=normalized_radial_distance,
                grid_key_padding_mask=grid_key_padding_mask,
            )

        target_token = self.target_final_norm(target_token).squeeze(1)


        if self.use_gated_fusion:
            gate = self.gate_mlp(torch.cat([target_token, target_sem], dim=-1))
            target_token = gate * target_token + (1.0 - gate) * target_sem
        if self.use_add_fusion:
            target_token = target_token + target_sem

        return target_token
