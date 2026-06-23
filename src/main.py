import gc
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import rcParams
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader.poi_image_loader import BuildingPADataset, collate_fn
from models.BuildingPA import BuildingPA


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


@torch.no_grad()
def _set_buildingpa_epoch(model, epoch):
    target_model = model.module if hasattr(model, "module") else model
    if hasattr(target_model, "set_current_epoch"):
        target_model.set_current_epoch(epoch)
    else:
        target_model.current_epoch = int(epoch)


@torch.no_grad()
def _update_buildingpa_arl_weights(model, out_llm, out_spa, temperature=8.0, min_entropy_floor=0.3):
    target_model = model.module if hasattr(model, "module") else model
    calculate_llm = torch.mean(torch.abs(out_llm), dim=0).sum().item()
    calculate_spa = torch.mean(torch.abs(out_spa), dim=0).sum().item()
    prob_llm = torch.softmax(out_llm, dim=1)
    prob_spa = torch.softmax(out_spa, dim=1)
    h_llm = torch.sum(-prob_llm * torch.log(prob_llm + 1e-16), dim=1).mean().item()
    h_spa = torch.sum(-prob_spa * torch.log(prob_spa + 1e-16), dim=1).mean().item()
    h_sum = h_llm + h_spa + 1e-12
    h_llm_n = max(h_llm / h_sum, min_entropy_floor)
    h_spa_n = max(h_spa / h_sum, min_entropy_floor)
    llm_factor = 1.0 / h_llm_n
    spa_factor = 1.0 / h_spa_n
    factor_sum = llm_factor + spa_factor + 1e-12
    llm_factor = llm_factor / factor_sum
    spa_factor = spa_factor / factor_sum
    calc_sum = calculate_llm + calculate_spa + 1e-12
    calculate_llm_n = calculate_llm / calc_sum
    calculate_spa_n = calculate_spa / calc_sum
    weight = F.softmax(
        torch.tensor(
            [
                (calculate_spa_n * llm_factor) * temperature,
                (calculate_llm_n * spa_factor) * temperature,
            ],
            device=out_llm.device,
        ),
        dim=0,
    )
    llm_weight = weight[0].item()
    spa_weight = weight[1].item()
    if hasattr(target_model, "set_arl_weights"):
        target_model.set_arl_weights(llm_weight, spa_weight)
    else:
        target_model.llm_weight = llm_weight
        target_model.spa_weight = spa_weight


def build_llm_inputs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    prompts = read_json(args.llm_fusion_prompts_path)
    inputs = tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].detach().numpy().tolist()
    attention_mask = inputs["attention_mask"].detach().numpy().tolist()
    return tokenizer, input_ids, attention_mask


def build_datasets(args, input_ids, attention_mask):
    common_kwargs = dict(
        poi_data_file_path=args.load_data_path,
        poi_source_data_file_path=args.load_source_data_path,
        load_name_embeddings_path=args.load_name_embeddings_path,
        attention_mask=attention_mask,
        input_ids=input_ids,
        building_data_path=args.building_data_path,
    )
    train_dataset = BuildingPADataset(mode="train", **common_kwargs)
    test_dataset = BuildingPADataset(mode="test", **common_kwargs)
    return train_dataset, test_dataset


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def training(args):
    print("Data_path:", args.load_data_path)
    tokenizer, input_ids, attention_mask = build_llm_inputs(args)
    train_dataset, test_dataset = build_datasets(args, input_ids, attention_mask)
    del input_ids, attention_mask
    gc.collect()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    model = BuildingPA(
        num_classes=args.num_classes,
        llm_path=args.llm_path,
        llm_dmodel=args.llm_dmodel,
        llm_tokenizer=tokenizer,
        name_embedding_path=args.load_name_embeddings_path,
        load_source_data_path=args.load_source_data_path,
        pretrained_llm_fusion_path=args.pretrained_llm_fusion_path,
        arl_start_epoch=args.arl_start_epoch,
        arl_gamma=args.arl_gamma,
    ).to(args.device)
    del tokenizer
    gc.collect()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.9)
    loss_history = []
    global_iter = 0

    for epoch in tqdm(range(args.epoch)):
        _set_buildingpa_epoch(model, epoch)
        model.train()
        loop = tqdm(train_loader, leave=True)
        optimizer.zero_grad()
        total_loss = 0.0
        total_typing_loss = 0.0

        for batch_idx, batch in enumerate(loop):
            batch = move_batch_to_device(batch, args.device)
            outputs = model(
                labels=batch["poi_type"],
                coordinates=batch["coordinates"],
                name_embeddings=batch["name_embeddings"],
                point_ids=batch["point_ids"],
                point_mask=batch["point_mask"],
                point_coords=batch["point_coords"],
                llm_attention_mask=batch["llm_attention_mask"],
                llm_input_ids=batch["llm_input_ids"],
            )
            loss = outputs[0]
            typing_loss = outputs[2]
            extra_info = outputs[3]

            if "out_llm" in extra_info and "out_spa" in extra_info:
                _update_buildingpa_arl_weights(
                    model=model,
                    out_llm=extra_info["out_llm"].detach(),
                    out_spa=extra_info["out_spa"].detach(),
                    temperature=args.arl_temperature,
                )

            loss.backward()
            total_loss += loss.item()
            total_typing_loss += typing_loss.item()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix({"typing_loss": total_typing_loss / max(batch_idx + 1, 1)})
            loss_history.append(total_loss / max(batch_idx + 1, 1))

            if global_iter % 2000 == 0:
                plt_batch(args, loss_history)
            global_iter += 1

        scheduler.step()
        test(test_loader, model, args.device, epoch)


def test(test_loader, model, device, epoch):
    model.eval()
    _set_buildingpa_epoch(model, epoch)

    pred_list = []
    label_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=True):
            batch = move_batch_to_device(batch, device)
            outputs = model(
                labels=batch["poi_type"],
                coordinates=batch["coordinates"],
                name_embeddings=batch["name_embeddings"],
                point_ids=batch["point_ids"],
                point_mask=batch["point_mask"],
                point_coords=batch["point_coords"],
                llm_attention_mask=batch["llm_attention_mask"],
                llm_input_ids=batch["llm_input_ids"],
            )
            pred_list.extend(outputs[1].cpu().numpy())
            label_list.extend(batch["poi_type"].cpu().numpy())

    y_testlabel = np.array(label_list)
    predictions_test = np.array(pred_list)
    predictions_test_dim = np.argmax(predictions_test, axis=1)
    accuracy_score_test = accuracy_score(y_testlabel, predictions_test_dim)
    f1_score_test = f1_score(y_testlabel, predictions_test_dim, average="macro")
    mrr_test = compute_mrr(y_testlabel, predictions_test)

    print(
        f"Test epoch {epoch}: "
        f"Accuracy={accuracy_score_test:.6f}, "
        f"F1-score={f1_score_test:.6f}, "
        f"MRR={mrr_test:.6f}"
    )


def compute_mrr(true_labels, machine_preds):
    rr_total = 0.0
    for i in range(len(true_labels)):
        ranklist = list(np.argsort(machine_preds[i])[::-1])
        rank = ranklist.index(true_labels[i]) + 1
        rr_total += 1.0 / rank
    return rr_total / len(true_labels)


def plt_batch(args, values):
    if not values:
        return
    os.makedirs(args.base_result_image_outdir, exist_ok=True)
    rcParams["figure.figsize"] = (16, 6)
    rcParams["figure.dpi"] = 100
    rcParams["font.size"] = 8
    rcParams["font.family"] = "sans-serif"
    rcParams["axes.facecolor"] = "#ffffff"
    rcParams["lines.linewidth"] = 2.0
    plt.figure()
    plt.plot(values, "g", label="train loss")
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train"], loc="upper right")
    plt.savefig(os.path.join(args.base_result_image_outdir, f"{args.l}.png"))
    plt.close()


class Config:
    def __init__(self, id_dataset):
        src_dir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
        self.project_root = os.path.dirname(src_dir).replace("\\", "/")
        self.id_dataset = id_dataset
        self.epoch = 15
        self.batch_size = 64
        self.lr = 0.0001
        self.num_workers = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.task = "gated_fuse_name_category_category_count"
        self.l = 0
        self.llm_dmodel = 2048
        self.arl_start_epoch = 3
        self.arl_temperature = 8.0
        self.arl_gamma = 1.0
        self.num_classes = {
            "NYC": 373,
            "TKY": 371,
            "LDN": 449,
        }[id_dataset]
        dataset_dir = os.path.join(self.project_root, "datasets", id_dataset).replace("\\", "/")
        self.load_data_path = f"{dataset_dir}/pois_with_building_and_neighbors.json"
        self.load_source_data_path = self.load_data_path
        self.load_name_embeddings_path = f"{dataset_dir}/qwen3_name_embedding.pt"
        self.llm_fusion_prompts_path = f"{dataset_dir}/llm_building_attribute_prompts.json"
        self.building_data_path = f"{dataset_dir}/buildings_filtered.json"
        self.llm_path = f"{self.project_root}/src/llm/Qwen3-1.7B"
        self.pretrained_llm_fusion_path = f"{self.project_root}/model_save_path/{id_dataset}/model_weight.pth"
        self.base_result_image_outdir = f"{self.project_root}/result/image/{id_dataset}/"


def seed_torch(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    id_datasets = ["LDN", "NYC", "TKY"]
    for id_dataset in id_datasets:
        seed_torch()
        config = Config(id_dataset=id_dataset)
        config.task = "B3PA"
        print(config.task)
        training(config)
