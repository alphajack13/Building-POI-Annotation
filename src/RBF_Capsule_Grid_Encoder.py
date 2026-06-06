from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBF_Capsule_Grid_Encoder(nn.Module):

    def __init__(
        self,
        train_point_embeddings: torch.Tensor,
        grid_size_per_row: int,
        square_size_m: float,
        coord_order: str = "latlon",
        normalize_weights: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        if not isinstance(train_point_embeddings, torch.Tensor):
            raise TypeError("train_point_embeddings 必须是 torch.Tensor")
        if train_point_embeddings.dim() != 2:
            raise ValueError("train_point_embeddings 必须是 [num_train_pois, dim]")
        if grid_size_per_row <= 0:
            raise ValueError("grid_size_per_row 必须 > 0")
        if square_size_m <= 0:
            raise ValueError("square_size_m 必须 > 0")
        if coord_order not in {"latlon", "lonlat"}:
            raise ValueError("coord_order 必须是 'latlon' 或 'lonlat'")

        num_train_pois, dim = train_point_embeddings.shape

        self.num_train_pois = num_train_pois
        self.dim = dim
        self.grid_size_per_row = grid_size_per_row
        self.square_size_m = float(square_size_m)
        self.coord_order = coord_order
        self.normalize_weights = normalize_weights
        self.eps = eps


        self.register_buffer(
            "frozen_train_embeddings",
            train_point_embeddings.detach().clone()
        )


        self.building_point_embedding = nn.Parameter(
            torch.empty(dim).normal_(mean=0.0, std=0.02)
        )


        self.register_buffer(
            "padding_embedding",
            torch.zeros(dim, dtype=train_point_embeddings.dtype)
        )
        if self.grid_size_per_row == 1:
            grid_step = self.square_size_m
        else:
            grid_step = self.square_size_m / (self.grid_size_per_row - 1)
        self.grid_step = grid_step
        grid_points = self._create_grid_points(
            grid_size_per_row=self.grid_size_per_row,
            square_size_m=self.square_size_m,
            dtype=train_point_embeddings.dtype,
        )
        self.register_buffer("grid_points", grid_points)

    def _create_grid_points(
        self,
        grid_size_per_row: int,
        square_size_m: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        half = square_size_m / 2.0
        xs = torch.linspace(-half, half, grid_size_per_row, dtype=dtype)
        ys = torch.linspace(-half, half, grid_size_per_row, dtype=dtype)


        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        return grid

    def _build_full_embedding_table(self) -> torch.Tensor:
        return torch.cat(
            [
                self.frozen_train_embeddings,
                self.building_point_embedding.unsqueeze(0),
                self.padding_embedding.unsqueeze(0),
            ],
            dim=0,
        )

    def _latlon_to_relative_xy(
        self,
        center_coords: torch.Tensor,
        point_coords: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.coord_order == "latlon":
            center_lat = center_coords[:, 0]
            center_lon = center_coords[:, 1]
            point_lat = point_coords[:, :, 0]
            point_lon = point_coords[:, :, 1]
        else:
            center_lon = center_coords[:, 0]
            center_lat = center_coords[:, 1]
            point_lon = point_coords[:, :, 0]
            point_lat = point_coords[:, :, 1]

        center_lat_exp = center_lat.unsqueeze(1).expand_as(point_lat)
        center_lon_exp = center_lon.unsqueeze(1).expand_as(point_lon)

        dlat = point_lat - center_lat_exp
        dlon = point_lon - center_lon_exp

        center_lat_rad = torch.deg2rad(center_lat_exp)
        cos_lat = torch.cos(center_lat_rad)
        cos_lat = torch.where(
            torch.abs(cos_lat) < 1e-6,
            torch.full_like(cos_lat, 1e-6),
            cos_lat
        )

        dy = dlat * 111000.0
        dx = dlon * 111000.0 * cos_lat

        rel_xy = torch.stack([dx, dy], dim=-1)

        if point_mask.dtype != torch.bool:
            mask_bool = point_mask > 0
        else:
            mask_bool = point_mask

        rel_xy = rel_xy * mask_bool.unsqueeze(-1).to(rel_xy.dtype)
        return rel_xy

    def _gather_point_embeddings(
        self,
        point_ids: torch.Tensor,
    ) -> torch.Tensor:
        full_table = self._build_full_embedding_table()
        return F.embedding(point_ids, full_table)

    def forward(
        self,
        point_ids: torch.Tensor,
        point_mask: torch.Tensor,
        point_coords: torch.Tensor,
        center_coords: Optional[torch.Tensor] = None,
        coords_are_relative: bool = False,
    ) -> torch.Tensor:
        if point_ids.dim() != 2:
            raise ValueError("point_ids 必须是 [B, N]")
        if point_mask.dim() != 2:
            raise ValueError("point_mask 必须是 [B, N]")
        if point_coords.dim() != 3 or point_coords.size(-1) != 2:
            raise ValueError("point_coords 必须是 [B, N, 2]")

        batch_size, max_num_neighbors = point_ids.shape
        if point_mask.shape != (batch_size, max_num_neighbors):
            raise ValueError("point_mask 形状与 point_ids 不一致")
        if point_coords.shape[:2] != (batch_size, max_num_neighbors):
            raise ValueError("point_coords 前两维与 point_ids 不一致")

        device = point_ids.device


        min_id = int(point_ids.min().item())
        max_id = int(point_ids.max().item())
        if min_id < 0 or max_id > self.num_train_pois + 1:
            raise ValueError(
                f"point_ids 超出合法范围 [0, {self.num_train_pois + 1}]，"
                f"当前最小={min_id}, 最大={max_id}"
            )

        if point_mask.dtype != torch.bool:
            mask_bool = point_mask > 0
        else:
            mask_bool = point_mask


        point_embeds = self._gather_point_embeddings(point_ids)


        if coords_are_relative:
            rel_xy = point_coords
            rel_xy = rel_xy * mask_bool.unsqueeze(-1).to(rel_xy.dtype)
        else:
            if center_coords is None:
                raise ValueError(
                    "coords_are_relative=False 时，必须提供 center_coords=[B,2]，"
                    "否则无法以目标 poi 为中心构造网格。"
                )
            if center_coords.dim() != 2 or center_coords.size(-1) != 2:
                raise ValueError("center_coords 必须是 [B, 2]")

            rel_xy = self._latlon_to_relative_xy(
                center_coords=center_coords,
                point_coords=point_coords,
                point_mask=mask_bool,
            )




        grid_points = self.grid_points.to(device=device, dtype=rel_xy.dtype)
        dists = torch.cdist(rel_xy, grid_points.unsqueeze(0).expand(batch_size, -1, -1))



        weights = torch.exp(-dists ** 2 / (2 * 10**2))
        weights = weights * mask_bool.unsqueeze(-1).to(weights.dtype)






        weighted_sum = torch.einsum("bng,bnd->bgd", weights, point_embeds)

        if self.normalize_weights:
            denom = weights.sum(dim=1, keepdim=False).unsqueeze(-1)
            grid_features = weighted_sum / (denom + self.eps)
        else:
            grid_features = weighted_sum


        return grid_features
