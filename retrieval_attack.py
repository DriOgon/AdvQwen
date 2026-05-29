from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.metrics import asr_at_ks, recall_at_ks_from_ranks
from utils.patch_utils import clamp_patch, patch_initialization
from utils.qwen import QwenVictimModel, default_model_path, normalize_victim_name


DATASET_ALIASES = {
    "flickr30k": "flickr30k_std",
    "flickr30k_std": "flickr30k_std",
    "coco": "coco_karpathy",
    "mscoco": "coco_karpathy",
    "ms_coco": "coco_karpathy",
    "coco_karpathy": "coco_karpathy",
}


class JsonlRowsDataset(Dataset):
    # 训练补丁时使用的简单 jsonl 数据集封装。
    # 每个样本包含图像路径、对应文本、匹配标签和样本 id。
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return row["image_path"], row["text"], row["labels"], row["id"]


def collate_rows(batch):
    # DataLoader 默认 collate 不适合变长文本和标签列表，
    # 因此这里手动整理成批量 image_paths / texts / labels / ids。
    image_paths = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    ids = [item[3] for item in batch]
    return image_paths, texts, labels, ids


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_dataset_name(dataset: str) -> str:
    # 支持 flickr30k、coco、mscoco 等别名，统一映射到 data_std 目录名。
    normalized = dataset.lower().replace("-", "_")
    return DATASET_ALIASES.get(normalized, dataset)


def resolve_dataset_root(data_std_root: str, dataset: str) -> tuple[str, Path]:
    # 根据数据集名称定位标准数据目录，例如 data_std/flickr30k_std。
    dataset_name = resolve_dataset_name(dataset)
    root = Path(data_std_root)
    if not root.is_absolute():
        root = project_root() / root
    return dataset_name, root / dataset_name


def resolve_output_root(path: str) -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = project_root() / root
    return root


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    # 读取标准化后的 jsonl 文件；limit 用于 smoke test，0 表示读取全部。
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def maybe_load_metadata(dataset_root: Path) -> dict[str, Any]:
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    # 该脚本是 AdvQwen 图文检索攻击主入口：
    # 先在代理数据集上训练通用补丁，再在下游数据集官方路径上评测。
    parser = argparse.ArgumentParser(
        description="Retrieval-oriented zero-shot universal patch attack for standard Flickr30K/MS COCO protocols."
    )
    parser.add_argument("--train_dataset", type=str, default="flickr30k_std", help="Surrogate dataset used to train the patch.")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Downstream dataset used for official-path attack evaluation.")
    parser.add_argument("--data_std_root", type=str, default="data_std")
    parser.add_argument("--clean_cache_root", type=str, default="output/features_std")
    parser.add_argument("--output_root", type=str, default="output/zero_shot_attack")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--attack_method",type=str,default="ours",choices=["rand_patch", "uap", "adv_patch", "advclip", "ours"],help="Qwen-adapted comparison baseline. Presets override loss weights unless --no_method_preset is set.",)
    parser.add_argument("--no_method_preset", action="store_true", help="Use manually supplied loss weights instead of method presets.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--lambda_feat", type=float, default=0.25)
    parser.add_argument("--lambda_cross", type=float, default=0.25)
    parser.add_argument("--lambda_rank", type=float, default=2.0)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)
    parser.add_argument("--rank_margin", type=float, default=0.10)
    parser.add_argument("--noise_percentage", type=float, default=0.03)

    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--test_image_limit", type=int, default=0)
    parser.add_argument("--test_text_limit", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--metric_batch_size", type=int, default=128)
    parser.add_argument("--reuse_clean_cache", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--model_path", type=str, default=default_model_path())
    return parser.parse_args()


def apply_method_preset(args: argparse.Namespace) -> None:
    # 为不同对比方法设置损失权重。注意这些 baseline 都是 Qwen 适配版本，
    # 目的是在相同评测协议下比较不同攻击目标的效果。
    if args.no_method_preset:
        return
    if args.attack_method == "rand_patch":
        args.lambda_feat = 0.0
        args.lambda_cross = 0.0
        args.lambda_rank = 0.0
        args.lambda_reg = 0.0
    elif args.attack_method == "uap":
        # Qwen 适配版 UAP：只优化图像特征偏移，使 adv 图像远离 clean 图像 embedding。
        args.lambda_feat = 1.0
        args.lambda_cross = 0.0
        args.lambda_rank = 0.0
        args.lambda_reg = 1e-4
    elif args.attack_method == "adv_patch":
        # Qwen 适配版 Adv-Patch：只破坏正确图文对的跨模态匹配关系。
        args.lambda_feat = 0.0
        args.lambda_cross = 1.0
        args.lambda_rank = 0.0
        args.lambda_reg = 1e-4
    elif args.attack_method == "advclip":
        # AdvCLIP-style 非排序版本：结合图像特征偏移和跨模态失配，但不使用 ranking loss。
        args.lambda_feat = 0.5
        args.lambda_cross = 0.5
        args.lambda_rank = 0.0
        args.lambda_reg = 1e-4
    elif args.attack_method == "ours":
        args.lambda_feat = 0.25
        args.lambda_cross = 0.25
        args.lambda_rank = 2.0
        args.lambda_reg = 1e-4


def build_train_loader(args: argparse.Namespace, train_dataset_root: Path) -> DataLoader:
    # train_surrogate.jsonl 是补丁训练用的代理图文对。
    # train_limit 可用于快速 smoke test，正式实验可以设为 0 或更大规模。
    train_rows = load_jsonl(train_dataset_root / "train_surrogate.jsonl", limit=args.train_limit)
    return DataLoader(
        JsonlRowsDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_rows,
    )


def clean_cache_path(clean_cache_root: Path, victim_name: str, dataset_name: str, split_name: str, limit: int) -> Path:
    # clean 特征缓存路径，与 clean_retrieval_baseline.py 保持同一组织方式。
    suffix = f"{split_name}.pt" if limit <= 0 else f"{split_name}_limit{limit}.pt"
    return clean_cache_root / victim_name / dataset_name / suffix


def load_clean_feature_cache(path: Path) -> dict[str, Any]:
    # 加载 clean baseline 已经缓存好的官方路径 embedding。
    payload = torch.load(path, map_location="cpu")
    return {
        "ids": payload["ids"],
        "labels": [list(labels) for labels in payload["labels"]],
        "features": payload["features"].float(),
    }


def compute_clean_eval_cache(
    *,
    victim_model: QwenVictimModel,
    image_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    eval_dataset_name: str,
    victim_name: str,
    clean_cache_root: Path,
    image_limit: int,
    text_limit: int,
    reuse_clean_cache: bool,
) -> dict[str, Any]:
    # 构建评测缓存：clean 图像/text embedding 只需要编码一次。
    # 后续每轮 eval 只需要重新编码贴补丁后的 adv 图像特征。
    image_paths = [row["image_path"] for row in image_rows]
    texts = [row["text"] for row in text_rows]
    image_labels = [row["labels"] for row in image_rows]
    text_labels = [row["labels"] for row in text_rows]

    image_cache_path = clean_cache_path(clean_cache_root, victim_name, eval_dataset_name, "test_images", image_limit)
    text_cache_path = clean_cache_path(clean_cache_root, victim_name, eval_dataset_name, "test_texts", text_limit)

    if reuse_clean_cache and image_cache_path.exists() and text_cache_path.exists():
        # 复用 clean_retrieval_baseline.py 产生的特征缓存，可显著减少评测耗时。
        print(f"[clean-cache] loading images: {image_cache_path}", flush=True)
        print(f"[clean-cache] loading texts:  {text_cache_path}", flush=True)
        image_cache = load_clean_feature_cache(image_cache_path)
        text_cache = load_clean_feature_cache(text_cache_path)
        return {
            "image_rows": image_rows,
            "text_rows": text_rows,
            "image_labels": image_cache["labels"],
            "text_labels": text_cache["labels"],
            "clean_image_features": image_cache["features"],
            "clean_text_features": text_cache["features"],
        }

    print("[clean-cache] encoding clean eval features on official path...", flush=True)
    with torch.no_grad():
        # 如果没有缓存，则在官方路径上重新编码 clean 图像和文本特征。
        clean_image_features = victim_model.encode_image(image_paths, mode="official").detach().cpu()
        clean_text_features = victim_model.encode_text(texts).detach().cpu()

    return {
        "image_rows": image_rows,
        "text_rows": text_rows,
        "image_labels": image_labels,
        "text_labels": text_labels,
        "clean_image_features": clean_image_features,
        "clean_text_features": clean_text_features,
    }


def labels_overlap(lhs: list[int], rhs: list[int]) -> bool:
    # labels 有交集表示 query 与 gallery 是同一图文语义匹配。
    return bool(set(lhs) & set(rhs))


def calc_rank_positions_chunked(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_labels: list[list[int]],
    gallery_labels: list[list[int]],
    *,
    metric_batch_size: int,
) -> np.ndarray:
    # 计算每个 query 的正确匹配排名位置。rank=0 表示排第一，
    # rank 越大说明正确匹配越靠后，检索性能越差。
    query = F.normalize(torch.as_tensor(query, dtype=torch.float32), dim=-1)
    gallery = F.normalize(torch.as_tensor(gallery, dtype=torch.float32), dim=-1)
    ranks = np.zeros(query.shape[0], dtype=np.int64)

    for start in range(0, query.shape[0], metric_batch_size):
        end = min(start + metric_batch_size, query.shape[0])
        similarity = torch.matmul(query[start:end], gallery.t())
        # 对 gallery 按相似度从高到低排序，再找到第一个正确匹配的位置。
        order = torch.argsort(similarity, dim=1, descending=True).tolist()
        for local_idx, ranked_indices in enumerate(order):
            labels = query_labels[start + local_idx]
            rank = next(
                (r for r, gallery_idx in enumerate(ranked_indices) if labels_overlap(labels, gallery_labels[gallery_idx])),
                len(ranked_indices),
            )
            ranks[start + local_idx] = rank
    return ranks


def evaluate_zero_shot_attack(
    victim_model: QwenVictimModel,
    eval_cache: dict[str, Any],
    patch: torch.Tensor,
    noise_percentage: float,
    metric_batch_size: int,
) -> dict[str, Any]:
    # 每次评测都回到官方路径：补丁先贴到图像上，再调用 Qwen 官方 process()。
    # 这样得到的 ASR 才能说明补丁是否真正迁移到了真实推理流程。
    image_rows = eval_cache["image_rows"]
    image_paths = [row["image_path"] for row in image_rows]
    image_labels = eval_cache["image_labels"]
    text_labels = eval_cache["text_labels"]
    clean_image_features = eval_cache["clean_image_features"]
    clean_text_features = eval_cache["clean_text_features"]

    clean_i2t_ranks = calc_rank_positions_chunked(
        clean_image_features,
        clean_text_features,
        image_labels,
        text_labels,
        metric_batch_size=metric_batch_size,
    )
    clean_t2i_ranks = calc_rank_positions_chunked(
        clean_text_features,
        clean_image_features,
        text_labels,
        image_labels,
        metric_batch_size=metric_batch_size,
    )

    with torch.no_grad():
        # 只对图像加补丁；文本 embedding 保持 clean，用于评估图像补丁攻击。
        adv_image_features = victim_model.encode_image_paths_with_patch_official(
            image_paths,
            patch.detach(),
            noise_percentage=noise_percentage,
        ).detach().cpu()

    adv_i2t_ranks = calc_rank_positions_chunked(
        adv_image_features,
        clean_text_features,
        image_labels,
        text_labels,
        metric_batch_size=metric_batch_size,
    )
    adv_t2i_ranks = calc_rank_positions_chunked(
        clean_text_features,
        adv_image_features,
        text_labels,
        image_labels,
        metric_batch_size=metric_batch_size,
    )

    clean_i2t = recall_at_ks_from_ranks(clean_i2t_ranks)
    clean_t2i = recall_at_ks_from_ranks(clean_t2i_ranks)
    adv_i2t = recall_at_ks_from_ranks(adv_i2t_ranks)
    adv_t2i = recall_at_ks_from_ranks(adv_t2i_ranks)
    i2t_asr = asr_at_ks(clean_i2t_ranks, adv_i2t_ranks)
    t2i_asr = asr_at_ks(clean_t2i_ranks, adv_t2i_ranks)

    mean_clean = sum(clean_i2t + clean_t2i) / 6.0
    mean_adv = sum(adv_i2t + adv_t2i) / 6.0
    mean_drop = mean_clean - mean_adv
    mean_asr = sum(i2t_asr + t2i_asr) / 6.0

    return {
        "clean_i2t": clean_i2t,
        "clean_t2i": clean_t2i,
        "adv_i2t": adv_i2t,
        "adv_t2i": adv_t2i,
        "i2t_asr": i2t_asr,
        "t2i_asr": t2i_asr,
        "mean_clean": mean_clean,
        "mean_adv": mean_adv,
        "mean_drop": mean_drop,
        "mean_asr": mean_asr,
    }


def format_triplet(values) -> str:
    return "{:.2f} / {:.2f} / {:.2f}".format(*values)


def cosine_mean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 两组 embedding 的平均余弦相似度。训练时最小化它，
    # 表示推动 adv embedding 远离对应 clean/text embedding。
    return F.cosine_similarity(x, y, dim=-1).mean()


def hard_negative_ranking_loss(
    adv_image_features: torch.Tensor,
    clean_text_features: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    # 排序破坏损失：不仅降低正确图文对相似度，还显式要求 batch 内
    # 最难负样本的相似度超过真实匹配，从而更贴近 Recall@K 指标。
    adv_image_features = F.normalize(adv_image_features, dim=-1)
    clean_text_features = F.normalize(clean_text_features, dim=-1)
    sim = adv_image_features @ clean_text_features.t()
    pos = sim.diag()
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    masked = sim.masked_fill(eye, -1e4)
    # I2T 方向：对每张图像，找到最像它的错误文本。
    hardest_text_neg = masked.max(dim=1).values
    # T2I 方向：对每条文本，找到最像它的错误图像。
    hardest_image_neg = masked.max(dim=0).values
    i2t_rank = F.relu(pos - hardest_text_neg + margin).mean()
    t2i_rank = F.relu(pos - hardest_image_neg + margin).mean()
    return i2t_rank + t2i_rank


def train_zero_shot_attack(args, train_loader, eval_cache, victim_model, device):
    # 训练过程中唯一被优化的是通用图像补丁；Qwen 模型本身保持冻结。
    # 这对应论文中的输入空间攻击设定，而不是微调受害模型。
    init_patch = torch.from_numpy(patch_initialization(args)).to(device)
    patch = nn.Parameter(init_patch)
    optimizer = torch.optim.Adam([patch], lr=args.lr)
    victim_model.eval()

    best_metrics = None
    best_patch = None

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_feat = 0.0
        epoch_cross = 0.0
        epoch_rank = 0.0
        epoch_reg = 0.0
        num_steps = 0

        for step, (image_paths, texts, labels, ids) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            with torch.no_grad():
                # 官方路径得到 clean 锚点：这些 embedding 表示原始图文对
                # 在 Qwen 嵌入空间中的位置，训练时不更新它们。
                clean_image_features = victim_model.encode_image(image_paths, mode="official").detach()
                clean_text_features = victim_model.encode_text(texts).detach()

            # 可微代理路径：贴补丁后的 tensor 图像通过近似官方的张量预处理
            # 进入 Qwen，因此梯度可以一路回传到 `patch`。
            adv_image_features = victim_model.encode_image_paths_with_patch_proxy(
                image_paths,
                patch,
                noise_percentage=args.noise_percentage,
            )

            # 优化目标：降低对抗图像与 clean 图像、正确文本的相似度，
            # 同时推动 hardest negative 在排序中超过真实匹配。
            feat_loss = cosine_mean(adv_image_features, clean_image_features)
            cross_loss = cosine_mean(adv_image_features, clean_text_features)
            rank_loss = hard_negative_ranking_loss(
                adv_image_features,
                clean_text_features,
                margin=args.rank_margin,
            )
            reg_loss = patch.pow(2).mean()

            # 总损失越低，表示补丁越能偏移视觉语义、破坏图文对齐，
            # 并使真实匹配在检索排序中更靠后。
            loss = (
                args.lambda_feat * feat_loss
                + args.lambda_cross * cross_loss
                + args.lambda_rank * rank_loss
                + args.lambda_reg * reg_loss
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # 每次优化器更新后裁剪像素值，确保补丁仍是合法 RGB 图像。
                patch.data = clamp_patch(args, patch.data)

            epoch_loss += float(loss.item())
            epoch_feat += float(feat_loss.item())
            epoch_cross += float(cross_loss.item())
            epoch_rank += float(rank_loss.item())
            epoch_reg += float(reg_loss.item())
            num_steps += 1

            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(
                    f"[epoch {epoch + 1}/{args.num_epochs}] step {step}/{len(train_loader)} | "
                    f"loss={float(loss.item()):.4f} | feat={float(feat_loss.item()):.4f} | "
                    f"cross={float(cross_loss.item()):.4f} | rank={float(rank_loss.item()):.4f} | "
                    f"reg={float(reg_loss.item()):.6f}",
                    flush=True,
                )

        if num_steps > 0:
            print(
                f"[epoch {epoch + 1}/{args.num_epochs}] train_avg | "
                f"loss={epoch_loss / num_steps:.4f} | feat={epoch_feat / num_steps:.4f} | "
                f"cross={epoch_cross / num_steps:.4f} | rank={epoch_rank / num_steps:.4f} | "
                f"reg={epoch_reg / num_steps:.6f}",
                flush=True,
            )

        if (epoch + 1) % args.eval_every != 0:
            continue

        # 每隔 eval_every 轮，在官方评测路径上计算一次 ASR 和 Recall 下降。
        metrics = evaluate_zero_shot_attack(
            victim_model,
            eval_cache,
            patch.detach(),
            args.noise_percentage,
            args.metric_batch_size,
        )
        print("######################## Zero-shot Attack Eval ########################", flush=True)
        print(f"Clean I2T R@1/5/10: {format_triplet(metrics['clean_i2t'])}", flush=True)
        print(f"Clean T2I R@1/5/10: {format_triplet(metrics['clean_t2i'])}", flush=True)
        print(f"Adv   I2T R@1/5/10: {format_triplet(metrics['adv_i2t'])}", flush=True)
        print(f"Adv   T2I R@1/5/10: {format_triplet(metrics['adv_t2i'])}", flush=True)
        print(f"I2T ASR@1/5/10: {format_triplet(metrics['i2t_asr'])}", flush=True)
        print(f"T2I ASR@1/5/10: {format_triplet(metrics['t2i_asr'])}", flush=True)
        print(
            "Mean Recall Clean/Adv/Drop | Mean ASR: {:.2f} / {:.2f} / {:.2f} | {:.2f}".format(
                metrics["mean_clean"],
                metrics["mean_adv"],
                metrics["mean_drop"],
                metrics["mean_asr"],
            ),
            flush=True,
        )
        print("#####################################################################", flush=True)

        # 没有早停机制：训练会跑满 num_epochs，但保存 mean_drop 最大的补丁。
        if best_metrics is None or metrics["mean_drop"] > best_metrics["mean_drop"]:
            best_metrics = metrics
            best_patch = patch.detach().cpu().clone()

    return best_metrics, best_patch


def evaluate_random_patch(args, eval_cache, victim_model, device):
    # Rand-Patch 不训练，只随机初始化补丁并直接在官方路径上评测。
    patch = torch.from_numpy(patch_initialization(args)).to(device)
    metrics = evaluate_zero_shot_attack(
        victim_model,
        eval_cache,
        patch,
        args.noise_percentage,
        args.metric_batch_size,
    )
    return metrics, patch.detach().cpu()


def to_jsonable(obj):
    # 将 tuple / numpy array 等对象转换为 JSON 可保存格式。
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    return obj


def main() -> None:
    args = parse_args()
    if args.eval_dataset is None:
        # 如果不显式指定下游评测数据集，则默认在代理数据集自身上评测。
        args.eval_dataset = args.train_dataset
    apply_method_preset(args)

    # 固定随机种子，减少补丁初始化和数据 shuffle 带来的结果波动。
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    train_dataset_name, train_dataset_root = resolve_dataset_root(args.data_std_root, args.train_dataset)
    eval_dataset_name, eval_dataset_root = resolve_dataset_root(args.data_std_root, args.eval_dataset)
    clean_cache_root = resolve_output_root(args.clean_cache_root)
    output_root = resolve_output_root(args.output_root)
    victim_name = normalize_victim_name("Qwen3-VL-Embedding-2B")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Rand-Patch 不需要训练 loader；其他方法都需要代理数据集训练补丁。
    train_loader = None if args.attack_method == "rand_patch" else build_train_loader(args, train_dataset_root)
    # 下游评测集：图像作为 I2T query，文本作为 T2I query / I2T gallery。
    test_image_rows = load_jsonl(eval_dataset_root / "test_images.jsonl", limit=args.test_image_limit)
    test_text_rows = load_jsonl(eval_dataset_root / "test_texts.jsonl", limit=args.test_text_limit)

    print(
        f"[datasets] train={train_dataset_name} eval={eval_dataset_name} | "
        f"train_batches={0 if train_loader is None else len(train_loader)} "
        f"test_images={len(test_image_rows)} test_texts={len(test_text_rows)}",
        flush=True,
    )
    print(
        f"[method] {args.attack_method} | "
        f"lambda_feat={args.lambda_feat} lambda_cross={args.lambda_cross} "
        f"lambda_rank={args.lambda_rank} lambda_reg={args.lambda_reg}",
        flush=True,
    )
    if args.train_limit or args.test_image_limit or args.test_text_limit:
        print("[mode] limited/smoke mode; metrics are not formal full-test results", flush=True)

    # 加载 Qwen3-VL-Embedding-2B 受害模型。后续训练只更新 patch，不更新模型。
    victim_model = QwenVictimModel(model_path=args.model_path)
    victim_model.eval()

    # 先准备 clean 评测缓存，避免每轮 evaluation 都重复编码 clean 图文特征。
    eval_cache = compute_clean_eval_cache(
        victim_model=victim_model,
        image_rows=test_image_rows,
        text_rows=test_text_rows,
        eval_dataset_name=eval_dataset_name,
        victim_name=victim_name,
        clean_cache_root=clean_cache_root,
        image_limit=args.test_image_limit,
        text_limit=args.test_text_limit,
        reuse_clean_cache=args.reuse_clean_cache,
    )
    print("[clean-cache] ready.", flush=True)

    if args.attack_method == "rand_patch":
        # 随机补丁没有训练阶段，直接评测一次。
        best_metrics, best_patch = evaluate_random_patch(args, eval_cache, victim_model, device)
    else:
        if train_loader is None:
            raise RuntimeError("train_loader should not be None for optimized attack methods.")
        # 主训练过程：代理路径训练补丁，官方路径周期性评测并选择最佳补丁。
        best_metrics, best_patch = train_zero_shot_attack(args, train_loader, eval_cache, victim_model, device)
    if best_metrics is None:
        print("No evaluation was run. Check --num_epochs and --eval_every.", flush=True)
        return

    print("Best zero-shot attack result:", flush=True)
    print(f"Clean I2T R@1/5/10: {format_triplet(best_metrics['clean_i2t'])}", flush=True)
    print(f"Clean T2I R@1/5/10: {format_triplet(best_metrics['clean_t2i'])}", flush=True)
    print(f"Adv   I2T R@1/5/10: {format_triplet(best_metrics['adv_i2t'])}", flush=True)
    print(f"Adv   T2I R@1/5/10: {format_triplet(best_metrics['adv_t2i'])}", flush=True)
    print(f"I2T ASR@1/5/10: {format_triplet(best_metrics['i2t_asr'])}", flush=True)
    print(f"T2I ASR@1/5/10: {format_triplet(best_metrics['t2i_asr'])}", flush=True)
    print(
        "Mean Recall Clean/Adv/Drop | Mean ASR: {:.2f} / {:.2f} / {:.2f} | {:.2f}".format(
            best_metrics["mean_clean"],
            best_metrics["mean_adv"],
            best_metrics["mean_drop"],
            best_metrics["mean_asr"],
        ),
        flush=True,
    )

    if args.save and best_patch is not None:
        run_name = args.run_name
        if run_name is None:
            mode = "full" if not (args.train_limit or args.test_image_limit or args.test_text_limit) else "limited"
            run_name = f"{args.attack_method}_{train_dataset_name}_to_{eval_dataset_name}_{mode}_np{args.noise_percentage:.4f}"
        save_root = output_root / victim_name / run_name
        save_root.mkdir(parents=True, exist_ok=True)
        # 保存最佳补丁张量，后续可以直接用于检索复评或分类迁移评测。
        torch.save(best_patch, save_root / "best_patch.pt")
        # 同时保存完整参数和指标，方便论文表格追溯具体实验设置。
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "train_dataset": train_dataset_name,
            "eval_dataset": eval_dataset_name,
            "train_dataset_root": str(train_dataset_root),
            "eval_dataset_root": str(eval_dataset_root),
            "train_metadata": maybe_load_metadata(train_dataset_root),
            "eval_metadata": maybe_load_metadata(eval_dataset_root),
            "args": vars(args),
            "attack_method": args.attack_method,
            "metrics": best_metrics,
        }
        with (save_root / "best_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)
        print(f"Saved best patch and metrics to {save_root}", flush=True)


if __name__ == "__main__":
    main()
