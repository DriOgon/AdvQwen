from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from retrieval_attack import (
    compute_clean_eval_cache,
    evaluate_zero_shot_attack,
    format_triplet,
    load_jsonl,
    resolve_dataset_root,
    resolve_output_root,
    to_jsonable,
)
from utils.qwen import QwenVictimModel, default_model_path, normalize_victim_name


def parse_args() -> argparse.Namespace:
    # 该脚本只做“已有补丁复评”，不训练补丁，适合答辩现场演示。
    parser = argparse.ArgumentParser(description="Evaluate an existing universal patch on retrieval datasets.")
    parser.add_argument("--eval_dataset", type=str, default="flickr30k_std", help="Downstream retrieval dataset.")
    parser.add_argument("--patch_path", type=str, required=True, help="Path to an existing best_patch.pt.")
    parser.add_argument("--patch_name", type=str, default=None, help="Name used in saved result directory.")
    parser.add_argument("--data_std_root", type=str, default="data_std")
    parser.add_argument("--clean_cache_root", type=str, default="output/features_std")
    parser.add_argument("--output_root", type=str, default="output/patch_eval")
    parser.add_argument("--model_path", type=str, default=default_model_path())
    parser.add_argument("--victim", type=str, default="Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--noise_percentage", type=float, default=0.03)
    parser.add_argument("--test_image_limit", type=int, default=0)
    parser.add_argument("--test_text_limit", type=int, default=0)
    parser.add_argument("--metric_batch_size", type=int, default=128)
    parser.add_argument("--reuse_clean_cache", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    return p


def main() -> None:
    args = parse_args()
    eval_dataset_name, eval_dataset_root = resolve_dataset_root(args.data_std_root, args.eval_dataset)
    clean_cache_root = resolve_output_root(args.clean_cache_root)
    output_root = resolve_output_root(args.output_root)
    patch_path = resolve_path(args.patch_path)
    patch_name = args.patch_name or patch_path.parent.name
    victim_name = normalize_victim_name(args.victim)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patch = torch.load(patch_path, map_location=device).float()
    print(f"[patch] loaded {patch_path} shape={tuple(patch.shape)}", flush=True)

    test_image_rows = load_jsonl(eval_dataset_root / "test_images.jsonl", limit=args.test_image_limit)
    test_text_rows = load_jsonl(eval_dataset_root / "test_texts.jsonl", limit=args.test_text_limit)
    print(
        f"[dataset] eval={eval_dataset_name} images={len(test_image_rows)} texts={len(test_text_rows)}",
        flush=True,
    )
    if args.test_image_limit or args.test_text_limit:
        print("[mode] limited/smoke mode; metrics are not formal full-test results", flush=True)

    victim_model = QwenVictimModel(model_path=args.model_path)
    victim_model.eval()

    # 先准备 clean 特征。若传入 --reuse_clean_cache，会优先复用 clean baseline 缓存。
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

    # 评测阶段走官方路径：补丁贴到图像后转回 PIL，再调用 Qwen 官方 process()。
    metrics = evaluate_zero_shot_attack(
        victim_model=victim_model,
        eval_cache=eval_cache,
        patch=patch,
        noise_percentage=args.noise_percentage,
        metric_batch_size=args.metric_batch_size,
    )

    print("################ Existing Patch Retrieval Eval ################", flush=True)
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
    print("###############################################################", flush=True)

    if args.save:
        mode = "full" if not (args.test_image_limit or args.test_text_limit) else f"limit_i{len(test_image_rows)}_t{len(test_text_rows)}"
        save_root = output_root / victim_name / eval_dataset_name / f"{patch_name}_{mode}"
        save_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "eval_dataset": eval_dataset_name,
            "eval_dataset_root": str(eval_dataset_root),
            "victim": args.victim,
            "victim_name": victim_name,
            "model_path": args.model_path,
            "patch_path": str(patch_path),
            "patch_name": patch_name,
            "noise_percentage": args.noise_percentage,
            "test_image_limit": args.test_image_limit,
            "test_text_limit": args.test_text_limit,
            "metrics": metrics,
        }
        with (save_root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)
        print(f"[result] saved {save_root / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
