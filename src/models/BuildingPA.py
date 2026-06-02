import json
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from LLMFusion import LLMFusion
from RBF_Capsule_Grid_Encoder import RBF_Capsule_Grid_Encoder
from Target_guided_Grid_Token_Fusion_Layer import TargetGuidedGridTokenFusionLayer


def build_train_poi_name_plus_category_tensor(load_source_data_path, name_embedding, num_classes):
    with open(load_source_data_path, "r", encoding="utf-8") as file:
        poi_data = json.load(file)

    if not isinstance(name_embedding, torch.Tensor):
        name_embedding = torch.tensor(name_embedding, dtype=torch.float32)
    else:
        name_embedding = name_embedding.float()

    train_indices = []
    train_class_ids = []
    for index, item in enumerate(poi_data):
        if item.get("split") == "train":
            train_indices.append(index)
            train_class_ids.append(int(item["data"]["info"]["class"]))

    if not train_indices:
        raise ValueError("No training POIs found in source data")

    train_name_embedding = name_embedding[train_indices]
    train_class_tensor = torch.tensor(train_class_ids, dtype=torch.long, device=train_name_embedding.device)
    train_class_onehot = F.one_hot(train_class_tensor, num_classes=num_classes).to(train_name_embedding.dtype)
    return torch.cat([train_name_embedding, train_class_onehot], dim=-1)


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight):
        ctx.save_for_backward(weight)
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]
        return grad_output + grad_output * weight, torch.tensor(1.0, device=grad_output.device)


class SharedConcatAUXIHead(nn.Module):
    def __init__(self, spa_dim, llm_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.spa_dim = spa_dim
        self.llm_dim = llm_dim
        input_dim = spa_dim + llm_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(p=dropout),
            nn.ELU(),
            nn.Linear(input_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def shared_head(self, spa_feat, llm_feat):
        return self.classifier(self.projector(torch.cat((spa_feat, llm_feat), dim=1)))

    def forward(self, spa_feat, llm_feat):
        zero_spa = torch.zeros_like(spa_feat)
        zero_llm = torch.zeros_like(llm_feat)
        return (
            self.shared_head(spa_feat, llm_feat),
            self.shared_head(zero_spa, llm_feat),
            self.shared_head(spa_feat, zero_llm),
        )


class BuildingPA(nn.Module):
    def __init__(
        self,
        num_classes,
        llm_path,
        llm_dmodel,
        llm_tokenizer,
        name_embedding_path,
        load_source_data_path,
        pretrained_llm_fusion_path,
        arl_start_epoch=3,
        arl_gamma=1.0,
        hidden_dim=256,
    ):
        super().__init__()
        if llm_tokenizer is None:
            raise ValueError("llm_tokenizer is required")

        self.num_classes = num_classes
        self.current_epoch = 0
        self.arl_start_epoch = arl_start_epoch
        self.arl_gamma = arl_gamma
        self.llm_weight = 0.5
        self.spa_weight = 0.5
        self.loss_fct = CrossEntropyLoss()

        self.LLMFusion = LLMFusion(
            model_path=llm_path,
            llm_tokenizer=llm_tokenizer,
            pretrained_weights_path=pretrained_llm_fusion_path,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        name_embedding = torch.load(name_embedding_path, weights_only=False)
        if not isinstance(name_embedding, torch.Tensor):
            name_embedding = torch.tensor(name_embedding, dtype=torch.float32)
        else:
            name_embedding = name_embedding.float()

        train_point_embeddings = build_train_poi_name_plus_category_tensor(
            load_source_data_path=load_source_data_path,
            name_embedding=name_embedding,
            num_classes=num_classes,
        )
        name_dim = name_embedding.shape[1]
        grid_input_dim = train_point_embeddings.shape[1]
        self.building_polygon_poi_encoder = RBF_Capsule_Grid_Encoder(
            train_point_embeddings=train_point_embeddings,
            grid_size_per_row=10,
            square_size_m=200.0,
            sigma_m=10,
            coord_order="latlon",
        )
        self.building_polygon_poi_fusion = TargetGuidedGridTokenFusionLayer(
            grid_size_per_row=10,
            square_size_m=200.0,
            grid_input_dim=grid_input_dim,
            target_input_dim=name_dim,
            d_model=512,
            nhead=8,
            num_grid_layers=2,
            num_cross_layers=1,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            distance_penalty_init=0,
            distance_power=0,
            use_gated_fusion=False,
            use_add_fusion=True,
        )
        self.fusion_auxi_head = SharedConcatAUXIHead(
            spa_dim=512,
            llm_dim=llm_dmodel,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=0.5,
        )

    def set_current_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def set_arl_weights(self, llm_weight, spa_weight):
        self.llm_weight = float(llm_weight)
        self.spa_weight = float(spa_weight)

    def forward(
        self,
        labels,
        coordinates,
        name_embeddings,
        point_ids,
        point_mask,
        point_coords,
        llm_attention_mask,
        llm_input_ids,
    ):
        if coordinates.dim() == 1:
            coordinates = coordinates.unsqueeze(0)
        if name_embeddings.dim() == 1:
            name_embeddings = name_embeddings.unsqueeze(0)
        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0)
        if point_ids.dim() == 1:
            point_ids = point_ids.unsqueeze(0)
        if point_mask.dim() == 1:
            point_mask = point_mask.unsqueeze(0)

        grid_features = self.building_polygon_poi_encoder(
            point_ids=point_ids,
            point_mask=point_mask,
            point_coords=point_coords,
            center_coords=coordinates,
            coords_are_relative=False,
        )
        z_spa = self.building_polygon_poi_fusion(
            grid_features=grid_features,
            target_name_embedding=name_embeddings,
        )
        z_llm = self.LLMFusion(
            llm_attention_mask=llm_attention_mask,
            llm_input_ids=llm_input_ids,
        )
        if z_llm.dim() == 1:
            z_llm = z_llm.unsqueeze(0)

        if self.current_epoch > self.arl_start_epoch:
            z_llm = GradScale.apply(z_llm, torch.tensor(self.llm_weight, device=z_llm.device))
            z_spa = GradScale.apply(z_spa, torch.tensor(self.spa_weight, device=z_spa.device))

        out_fused, out_llm, out_spa = self.fusion_auxi_head(z_spa, z_llm)
        loss_fused = self.loss_fct(out_fused.view(-1, self.num_classes), labels.view(-1))
        loss_llm = self.loss_fct(out_llm.view(-1, self.num_classes), labels.view(-1))
        loss_spa = self.loss_fct(out_spa.view(-1, self.num_classes), labels.view(-1))
        total_loss = loss_fused + self.arl_gamma * (loss_llm + loss_spa)
        extra_info = {
            "out_llm": out_llm,
            "out_spa": out_spa,
            "llm_weight": self.llm_weight,
            "spa_weight": self.spa_weight,
        }
        return total_loss, out_fused, loss_fused, extra_info
