import json
import math
from typing import Any

import numpy as np
import torch
from shapely.geometry import box, shape
from shapely.ops import transform
from torch.utils.data import Dataset
from tqdm import tqdm


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_pointset_tensors_from_poi_and_building_data(
    poi_data: list[dict[str, Any]],
    building_data: list[dict[str, Any]],
    square_size_m: float,
    boundary_sample_step_m: float = 2.0,
    coord_order: str = "latlon",
    sort_points_by_distance: bool = True,
    mask_dtype: torch.dtype = torch.int32,
    coord_dtype: torch.dtype = torch.float32,
    id_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if coord_order not in {"latlon", "lonlat"}:
        raise ValueError("coord_order must be 'latlon' or 'lonlat'")
    if square_size_m <= 0:
        raise ValueError("square_size_m must be positive")
    if boundary_sample_step_m <= 0:
        raise ValueError("boundary_sample_step_m must be positive")

    half_size_m = square_size_m / 2.0

    def parse_lat_lon(coords):
        if coord_order == "latlon":
            return float(coords[0]), float(coords[1])
        return float(coords[1]), float(coords[0])

    def format_coords(lat, lon):
        if coord_order == "latlon":
            return [lat, lon]
        return [lon, lat]

    def local_xy_from_latlon(center_lat, center_lon, lat, lon):
        cos_lat = math.cos(math.radians(center_lat))
        if abs(cos_lat) < 1e-6:
            cos_lat = 1e-6
        dx = (lon - center_lon) * 111000.0 * cos_lat
        dy = (lat - center_lat) * 111000.0
        return dx, dy

    def latlon_from_local_xy(center_lat, center_lon, dx, dy):
        cos_lat = math.cos(math.radians(center_lat))
        if abs(cos_lat) < 1e-6:
            cos_lat = 1e-6
        lat = center_lat + dy / 111000.0
        lon = center_lon + dx / (111000.0 * cos_lat)
        return lat, lon

    def make_lonlat_to_local_transformer(center_lat, center_lon):
        cos_lat = math.cos(math.radians(center_lat))
        if abs(cos_lat) < 1e-6:
            cos_lat = 1e-6

        def _transform(x, y, z=None):
            x_arr = np.asarray(x, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
            return (x_arr - center_lon) * 111000.0 * cos_lat, (y_arr - center_lat) * 111000.0

        return _transform

    def sample_linestring_points(line, step_m):
        if line.is_empty:
            return []
        length = float(line.length)
        if length < 1e-9:
            point = line.interpolate(0.0)
            return [(float(point.x), float(point.y))]
        distances = [0.0]
        current = step_m
        while current < length:
            distances.append(current)
            current += step_m
        if abs(distances[-1] - length) > 1e-6:
            distances.append(length)
        return [(float(line.interpolate(d).x), float(line.interpolate(d).y)) for d in distances]

    def collect_boundary_points(building_geom_geojson, center_lat, center_lon):
        geom = shape(building_geom_geojson)
        if geom.is_empty:
            return []
        if not geom.is_valid:
            geom = geom.buffer(0)
        boundary = transform(make_lonlat_to_local_transformer(center_lat, center_lon), geom.boundary)
        clipped = boundary.intersection(box(-half_size_m, -half_size_m, half_size_m, half_size_m))
        if clipped.is_empty:
            return []
        points = []

        def collect(g):
            if g.is_empty:
                return
            if g.geom_type in {"LineString", "LinearRing"}:
                points.extend(sample_linestring_points(g, boundary_sample_step_m))
            elif g.geom_type == "Point":
                points.append((float(g.x), float(g.y)))
            elif hasattr(g, "geoms"):
                for sub_g in g.geoms:
                    collect(sub_g)

        collect(clipped)
        unique_points = []
        seen = set()
        for x, y in points:
            key = (round(x, 6), round(y, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append((x, y))
        return unique_points

    poi_id_to_item = {int(item["poi_id"]): item for item in poi_data}
    building_id_to_geom = {int(item["building_id"]): item["geometry"] for item in building_data}
    train_poi_id_to_embed_idx = {}
    for item in poi_data:
        if item.get("split") == "train":
            train_poi_id_to_embed_idx[int(item["poi_id"])] = len(train_poi_id_to_embed_idx)

    building_point_id = len(train_poi_id_to_embed_idx)
    padding_point_id = building_point_id + 1
    per_poi_points = []

    for target_item in tqdm(poi_data, desc="building point sets", total=len(poi_data)):
        target_building_id = int(target_item["building_id"])
        target_lat, target_lon = parse_lat_lon(target_item["data"]["info"]["coordinates"])
        current_points = []

        for neighbor_poi_id in target_item.get("neighbor_ids", []):
            neighbor_poi_id = int(neighbor_poi_id)
            if neighbor_poi_id not in poi_id_to_item or neighbor_poi_id not in train_poi_id_to_embed_idx:
                continue
            neighbor_item = poi_id_to_item[neighbor_poi_id]
            neighbor_lat, neighbor_lon = parse_lat_lon(neighbor_item["data"]["info"]["coordinates"])
            dx, dy = local_xy_from_latlon(target_lat, target_lon, neighbor_lat, neighbor_lon)
            if abs(dx) <= half_size_m and abs(dy) <= half_size_m:
                current_points.append((train_poi_id_to_embed_idx[neighbor_poi_id], format_coords(neighbor_lat, neighbor_lon), dx * dx + dy * dy))

        if target_building_id not in building_id_to_geom:
            raise ValueError(f"building_id={target_building_id} not found")

        for x_local, y_local in collect_boundary_points(building_id_to_geom[target_building_id], target_lat, target_lon):
            lat, lon = latlon_from_local_xy(target_lat, target_lon, x_local, y_local)
            current_points.append((building_point_id, format_coords(lat, lon), x_local * x_local + y_local * y_local))

        if sort_points_by_distance:
            current_points.sort(key=lambda item: (item[2], item[0]))
        per_poi_points.append(current_points)

    num_pois = len(per_poi_points)
    max_points = max((len(points) for points in per_poi_points), default=0)
    point_ids_np = np.full((num_pois, max_points), padding_point_id, dtype=np.int64)
    point_mask_np = np.zeros((num_pois, max_points), dtype=np.int64)
    point_coords_np = np.zeros((num_pois, max_points, 2), dtype=np.float32)

    for i, point_list in tqdm(enumerate(per_poi_points), desc="padding point sets", total=num_pois):
        if not point_list:
            continue
        length = len(point_list)
        point_ids_np[i, :length] = [point_id for point_id, _, _ in point_list]
        point_mask_np[i, :length] = 1
        point_coords_np[i, :length, :] = np.asarray([coords for _, coords, _ in point_list], dtype=np.float32)

    return (
        torch.as_tensor(point_ids_np, dtype=id_dtype),
        torch.as_tensor(point_mask_np, dtype=mask_dtype),
        torch.as_tensor(point_coords_np, dtype=coord_dtype),
    )


class BuildingPADataset(Dataset):
    def __init__(
        self,
        poi_data_file_path,
        poi_source_data_file_path,
        load_name_embeddings_path,
        attention_mask,
        input_ids,
        building_data_path,
        mode,
    ):
        poi_data = read_json(poi_data_file_path)
        source_data = read_json(poi_source_data_file_path)
        building_data = read_json(building_data_path)
        labels = [int(item["data"]["info"]["class"]) for item in source_data]
        coordinates = [item["data"]["info"]["coordinates"] for item in source_data]
        name_embeddings = torch.load(load_name_embeddings_path, weights_only=False)

        point_ids, point_mask, point_coords = build_pointset_tensors_from_poi_and_building_data(
            poi_data=poi_data,
            building_data=building_data,
            square_size_m=200.0,
            boundary_sample_step_m=2.0,
            coord_order="latlon",
        )

        train_index = int(len(poi_data) * 0.8)
        if mode == "train":
            data_slice = slice(0, train_index)
        elif mode == "test":
            data_slice = slice(train_index, None)
        else:
            raise ValueError("mode must be 'train' or 'test'")

        self.poi_data = poi_data[data_slice]
        self.labels = labels[data_slice]
        self.coordinates = coordinates[data_slice]
        self.name_embeddings = name_embeddings[data_slice]
        self.llm_attention_mask = attention_mask[data_slice]
        self.llm_input_ids = input_ids[data_slice]
        self.point_ids = point_ids[data_slice]
        self.point_mask = point_mask[data_slice]
        self.point_coords = point_coords[data_slice]

    def __len__(self):
        return len(self.poi_data)

    def __getitem__(self, index):
        return {
            "poi_type": torch.tensor(self.labels[index], dtype=torch.long),
            "coordinates": torch.tensor(self.coordinates[index], dtype=torch.float32),
            "name_embeddings": torch.as_tensor(self.name_embeddings[index], dtype=torch.float32),
            "llm_attention_mask": torch.tensor(self.llm_attention_mask[index], dtype=torch.long),
            "llm_input_ids": torch.tensor(self.llm_input_ids[index], dtype=torch.long),
            "point_ids": self.point_ids[index],
            "point_mask": self.point_mask[index],
            "point_coords": self.point_coords[index],
        }


def collate_fn(batch):
    return {
        "poi_type": torch.stack([item["poi_type"] for item in batch]),
        "coordinates": torch.stack([item["coordinates"] for item in batch]),
        "name_embeddings": torch.stack([item["name_embeddings"] for item in batch]),
        "llm_attention_mask": torch.stack([item["llm_attention_mask"] for item in batch]),
        "llm_input_ids": torch.stack([item["llm_input_ids"] for item in batch]),
        "point_ids": torch.stack([item["point_ids"] for item in batch]),
        "point_mask": torch.stack([item["point_mask"] for item in batch]),
        "point_coords": torch.stack([item["point_coords"] for item in batch]),
    }
