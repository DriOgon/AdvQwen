from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from utils.qwen import default_model_path


# 允许用户用不同写法指定同一个标准数据集目录，避免命令行参数太死板。
DATASET_ALIASES = {
    "flickr30k": "flickr30k_std",
    "flickr30k_std": "flickr30k_std",
    "coco": "coco_karpathy",
    "mscoco": "coco_karpathy",
    "ms_coco": "coco_karpathy",
    "coco_karpathy": "coco_karpathy",
}


def parse_args() -> argparse.Namespace:
    # 该脚本只做 clean 图文检索基线，不训练补丁，也不加载对抗样本。
    parser = argparse.ArgumentParser(
        description=(
            "Run a clean zero-shot image-text retrieval baseline on a standard "
            "caption-level dataset such as Flickr30K or MS COCO Karpathy."
        )
    )
    parser.add_argument("--dataset", type=str, default="coco_karpathy", help="Dataset alias or data_std directory name.")
    parser.add_argument("--data_std_root", type=str, default="data_std", help="Root directory containing standard datasets.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Optional explicit dataset directory.")
    parser.add_argument("--cache_root", type=str, default="output/features_std", help="Feature cache root.")
    parser.add_argument("--result_root", type=str, default="output/clean_baseline", help="Metric JSON output root.")
    parser.add_argument("--victim", type=str, default="Qwen3-VL-Embedding-2B")
    parser.add_argument("--model_path", type=str, default=default_model_path())
    parser.add_argument("--image_batch_size", type=int, default=8)
    parser.add_argument("--text_batch_size", type=int, default=32)
    parser.add_argument("--metric_batch_size", type=int, default=256, help="Query chunk size used during Recall@K computation.")
    parser.add_argument("--image_limit", type=int, default=0, help="Smoke-test limit for image queries; 0 means full split.")
    parser.add_argument("--text_limit", type=int, default=0, help="Smoke-test limit for text gallery; 0 means full split.")
    parser.add_argument("--reuse_cache", action="store_true", help="Reuse existing feature caches if present.")
    parser.add_argument("--no_save_cache", action="store_true", help="Do not save newly encoded features.")
    parser.add_argument("--log_every", type=int, default=20, help="Print encoding progress every N batches.")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_dataset_name(dataset: str) -> str:
    # 将 coco、mscoco、flickr30k 等别名统一映射到 data_std 下的目录名。
    normalized = dataset.lower().replace("-", "_")
    return DATASET_ALIASES.get(normalized, dataset)


def normalize_victim_name(victim: str) -> str:
    if victim.lower() in {"qwen3-vl-embedding-2b", "qwen", "qwen3"}:
        return "Qwen3VL2B"
    return victim.replace("/", "_")


def resolve_dataset_root(args: argparse.Namespace) -> tuple[str, Path]:
    # 默认从 data_std/{dataset_name} 读取标准检索数据集；
    # 如果用户显式指定 dataset_root，则优先使用用户给定路径。
    dataset_name = resolve_dataset_name(args.dataset)
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        data_std_root = Path(args.data_std_root)
        if not data_std_root.is_absolute():
            data_std_root = project_root() / data_std_root
        dataset_root = data_std_root / dataset_name
    return dataset_name, dataset_root


def resolve_output_root(path: str) -> Path:
    output_root = Path(path)
    if not output_root.is_absolute():
        output_root = project_root() / output_root
    return output_root


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    # 标准检索数据集以 jsonl 存储：每行包含 id、labels、image_path 或 text。
    # limit 用于 smoke test，正式实验时为 0，表示读取完整测试集。
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


def cache_path(cache_root: Path, victim_name: str, dataset_name: str, split_name: str, limit: int) -> Path:
    # 特征缓存按 victim / dataset / split 组织，避免重复编码 Qwen embedding。
    suffix = f"{split_name}.pt" if limit <= 0 else f"{split_name}_limit{limit}.pt"
    return cache_root / victim_name / dataset_name / suffix


def encode_records(
    *,
    rows: list[dict[str, Any]],
    split_name: str,
    modality: str,
    victim_model: QwenVictimModel,
    batch_size: int,
    log_every: int,
) -> dict[str, Any]:
    # 将图像或文本通过 Qwen 官方路径编码成 embedding。
    # clean baseline 必须走官方路径，这样得到的指标才是正式评测结果。
    ids: list[str] = []
    labels: list[list[int]] = []
    features: list[torch.Tensor] = []

    total_batches = (len(rows) + batch_size - 1) // batch_size
    print(f"[encode:{split_name}] modality={modality} rows={len(rows)} batches={total_batches}", flush=True)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_idx = start // batch_size + 1
        ids.extend(str(row["id"]) for row in batch_rows)
        labels.extend([list(row["labels"]) for row in batch_rows])

        with torch.no_grad():
            if modality == "image":
                image_paths = [row["image_path"] for row in batch_rows]
                # 图像查询：官方 process() 路径编码 image embedding。
                batch_features = victim_model.encode_image(image_paths, mode="official")
            elif modality == "text":
                texts = [row["text"] for row in batch_rows]
                # 文本候选：官方文本编码路径编码 text embedding。
                batch_features = victim_model.encode_text(texts)
            else:
                raise ValueError(f"Unsupported modality: {modality}")

        features.append(batch_features.detach().cpu().float())
        if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
            print(f"[encode:{split_name}] batch {batch_idx}/{total_batches} rows={len(ids)}", flush=True)

    return {
        "split": split_name,
        "modality": modality,
        "ids": ids,
        "labels": labels,
        "features": torch.cat(features, dim=0),
    }


def load_or_encode(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    split_name: str,
    modality: str,
    victim_model: QwenVictimModel,
    batch_size: int,
    log_every: int,
    reuse_cache: bool,
    save_cache: bool,
) -> dict[str, Any]:
    if reuse_cache and path.exists():
        # 如果已有缓存，可直接加载，避免重复跑 Qwen3-VL-Embedding。
        print(f"[cache] loading {path}", flush=True)
        return torch.load(path, map_location="cpu")

    # 没有缓存或不复用缓存时，重新编码并按需保存。
    payload = encode_records(
        rows=rows,
        split_name=split_name,
        modality=modality,
        victim_model=victim_model,
        batch_size=batch_size,
        log_every=log_every,
    )
    if save_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"[cache] saved {path}", flush=True)
    return payload


def labels_overlap(lhs: list[int], rhs: list[int]) -> bool:
    # 检索数据集中，一个图像通常对应多条 caption。
    # labels 有交集就表示 query 与 gallery 是正确匹配。
    return bool(set(lhs) & set(rhs))


def calc_recall_at_ks(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_labels: list[list[int]],
    gallery_labels: list[list[int]],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    metric_batch_size: int = 256,
) -> tuple[float, ...]:
    # 基于归一化 embedding 的余弦相似度计算 Recall@K。
    # I2T 时 query 是图像、gallery 是文本；T2I 时二者反过来。
    query = torch.nn.functional.normalize(torch.as_tensor(query, dtype=torch.float32), dim=-1)
    gallery = torch.nn.functional.normalize(torch.as_tensor(gallery, dtype=torch.float32), dim=-1)
    max_k = min(max(ks), gallery.shape[0])
    hits = {k: 0 for k in ks}
    total = query.shape[0]

    for start in range(0, total, metric_batch_size):
        end = min(start + metric_batch_size, total)
        similarity = torch.matmul(query[start:end], gallery.t())
        # 对每个 query 取相似度最高的 Top-K 候选，再判断其中是否包含正确匹配。
        top_indices = torch.topk(similarity, k=max_k, dim=1, largest=True, sorted=True).indices.tolist()
        for local_idx, ranked_indices in enumerate(top_indices):
            labels = query_labels[start + local_idx]
            for k in ks:
                topk = ranked_indices[: min(k, len(ranked_indices))]
                if any(labels_overlap(labels, gallery_labels[index]) for index in topk):
                    hits[k] += 1

    return tuple(100.0 * hits[k] / total for k in ks)


def compute_metrics(image_cache: dict[str, Any], text_cache: dict[str, Any], metric_batch_size: int) -> dict[str, Any]:
    # 分别计算图像到文本检索 I2T 和文本到图像检索 T2I。
    i2t = calc_recall_at_ks(
        image_cache["features"],
        text_cache["features"],
        image_cache["labels"],
        text_cache["labels"],
        metric_batch_size=metric_batch_size,
    )
    t2i = calc_recall_at_ks(
        text_cache["features"],
        image_cache["features"],
        text_cache["labels"],
        image_cache["labels"],
        metric_batch_size=metric_batch_size,
    )
    return {
        "i2t": {"R@1": i2t[0], "R@5": i2t[1], "R@10": i2t[2]},
        "t2i": {"R@1": t2i[0], "R@5": t2i[1], "R@10": t2i[2]},
        "mean_recall": float(sum(i2t + t2i) / 6.0),
    }


def save_metrics(result_path: Path, payload: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[result] saved {result_path}", flush=True)


def main() -> None:
    from utils.qwen import QwenVictimModel

    args = parse_args()
    dataset_name, dataset_root = resolve_dataset_root(args)
    cache_root = resolve_output_root(args.cache_root)
    result_root = resolve_output_root(args.result_root)
    victim_name = normalize_victim_name(args.victim)

    image_limit = max(args.image_limit, 0)
    text_limit = max(args.text_limit, 0)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    # 标准检索数据集已经预处理成两个测试 split：
    # test_images 作为图像查询，test_texts 作为文本查询/候选。
    test_images = load_jsonl(dataset_root / "test_images.jsonl", limit=image_limit)
    test_texts = load_jsonl(dataset_root / "test_texts.jsonl", limit=text_limit)
    metadata = maybe_load_metadata(dataset_root)

    print(
        f"[dataset] name={dataset_name} root={dataset_root} images={len(test_images)} texts={len(test_texts)}",
        flush=True,
    )
    if image_limit or text_limit:
        print("[dataset] running in smoke/limited mode; metrics are not formal full-test results", flush=True)

    victim_model = QwenVictimModel(model_path=args.model_path)
    victim_model.image_batch_size = args.image_batch_size
    victim_model.text_batch_size = args.text_batch_size
    victim_model.eval()

    # 编码或加载 clean 图像特征。
    image_cache = load_or_encode(
        path=cache_path(cache_root, victim_name, dataset_name, "test_images", image_limit),
        rows=test_images,
        split_name="test_images",
        modality="image",
        victim_model=victim_model,
        batch_size=args.image_batch_size,
        log_every=args.log_every,
        reuse_cache=args.reuse_cache,
        save_cache=not args.no_save_cache,
    )
    # 编码或加载 clean 文本特征。
    text_cache = load_or_encode(
        path=cache_path(cache_root, victim_name, dataset_name, "test_texts", text_limit),
        rows=test_texts,
        split_name="test_texts",
        modality="text",
        victim_model=victim_model,
        batch_size=args.text_batch_size,
        log_every=args.log_every,
        reuse_cache=args.reuse_cache,
        save_cache=not args.no_save_cache,
    )

    # 在 clean embedding 上计算正式的 Recall@1/5/10。
    metrics = compute_metrics(image_cache, text_cache, metric_batch_size=args.metric_batch_size)
    print(
        "Clean {} I2T R@1/R@5/R@10: {:.2f} / {:.2f} / {:.2f}".format(
            dataset_name,
            metrics["i2t"]["R@1"],
            metrics["i2t"]["R@5"],
            metrics["i2t"]["R@10"],
        ),
        flush=True,
    )
    print(
        "Clean {} T2I R@1/R@5/R@10: {:.2f} / {:.2f} / {:.2f}".format(
            dataset_name,
            metrics["t2i"]["R@1"],
            metrics["t2i"]["R@5"],
            metrics["t2i"]["R@10"],
        ),
        flush=True,
    )
    print("Clean {} Mean Recall: {:.2f}".format(dataset_name, metrics["mean_recall"]), flush=True)

    mode = "full" if image_limit == 0 and text_limit == 0 else f"limit_i{len(test_images)}_t{len(test_texts)}"
    # 保存完整实验配置、缓存路径和指标，方便论文结果复现。
    result_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "victim": args.victim,
        "victim_name": victim_name,
        "model_path": args.model_path,
        "mode": mode,
        "counts": {"test_images": len(test_images), "test_texts": len(test_texts)},
        "cache_paths": {
            "test_images": str(cache_path(cache_root, victim_name, dataset_name, "test_images", image_limit)),
            "test_texts": str(cache_path(cache_root, victim_name, dataset_name, "test_texts", text_limit)),
        },
        "metadata": metadata,
        "metrics": metrics,
    }
    save_metrics(result_root / victim_name / dataset_name / f"clean_retrieval_{mode}.json", result_payload)


if __name__ == "__main__":
    main()
