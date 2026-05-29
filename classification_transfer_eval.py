from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms import functional as TF

from utils.patch_utils import apply_patch_single_image
from utils.qwen import QwenVictimModel, default_model_path, normalize_victim_name


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    # 该脚本用于“迁移评测”：把图文检索任务训练得到的通用补丁，
    # 直接贴到 CIFAR-10 / ImageNet 图像上，测试其分类攻击效果。
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval-trained universal patches on zero-shot image classification."
    )
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--data_root", type=str, default="data_std/cifar10")
    parser.add_argument("--patch_path", type=str, required=True)
    parser.add_argument("--patch_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="output/classification_eval")
    parser.add_argument("--clean_cache_root", type=str, default="output/classification_features")
    parser.add_argument("--reuse_clean_cache", action="store_true")
    parser.add_argument("--save", action="store_true")

    parser.add_argument("--model_path", type=str, default=default_model_path())
    parser.add_argument("--victim", type=str, default="Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Use a subset for smoke tests; 0 means full test set.")
    parser.add_argument("--noise_percentage", type=float, default=0.03)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a photo of {}.",
        help="Class prompt template. The class name is inserted with format().",
    )
    parser.add_argument(
        "--text_instruction",
        type=str,
        default=None,
        help="Optional Qwen text embedding instruction. Use None by default for class-name prompts.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    # 支持相对路径和绝对路径；相对路径默认相对于 AdvQwen-new 项目根目录。
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    return p


def load_cifar10(data_root: Path, limit: int) -> tuple[list[Image.Image | str], list[int], list[str]]:
    # CIFAR-10 由 torchvision 读取，返回 PIL 图像、类别标签和样本 id。
    dataset = CIFAR10(root=str(data_root), train=False, download=False)
    images: list[Image.Image] = []
    labels: list[int] = []
    ids: list[str] = []
    count = len(dataset) if limit <= 0 else min(limit, len(dataset))
    for idx in range(count):
        image, label = dataset[idx]
        images.append(image.convert("RGB"))
        labels.append(int(label))
        ids.append(f"cifar10_test_{idx:05d}")
    return images, labels, ids


def load_imagenet(data_root: Path, limit: int) -> tuple[list[Image.Image | str], list[int], list[str], list[str]]:
    # ImageNet 使用官方验证集目录结构读取。这里返回图像路径，
    # 避免一次性把 50000 张图片全部加载到内存。
    dataset = ImageNet(root=str(data_root), split="val")
    count = len(dataset.samples) if limit <= 0 else min(limit, len(dataset.samples))
    image_refs = [dataset.samples[idx][0] for idx in range(count)]
    labels = [int(dataset.samples[idx][1]) for idx in range(count)]
    ids = [Path(path).stem for path in image_refs]
    class_names = [classes[0] if isinstance(classes, tuple) else str(classes) for classes in dataset.classes]
    return image_refs, labels, ids, class_names


def load_dataset(args: argparse.Namespace) -> tuple[list[Image.Image | str], list[int], list[str], list[str]]:
    # 统一 CIFAR-10 和 ImageNet 的输出格式，后续评测逻辑可以复用。
    if args.dataset == "cifar10":
        images, labels, ids = load_cifar10(resolve_path(args.data_root), args.limit)
        return images, labels, ids, CIFAR10_CLASSES
    if args.dataset == "imagenet":
        return load_imagenet(resolve_path(args.data_root), args.limit)
    raise ValueError(f"Unsupported classification dataset: {args.dataset}")


def cache_path(root: Path, victim_name: str, dataset_name: str, limit: int) -> Path:
    # clean 图像特征缓存路径。分类迁移评测中 clean 特征可复用，
    # 因为不同补丁只会影响 adv 特征，不会改变 clean 图像 embedding。
    suffix = "test_images.pt" if limit <= 0 else f"test_images_limit{limit}.pt"
    return root / victim_name / dataset_name / suffix


def image_to_pil(image_ref: Image.Image | str) -> Image.Image:
    # CIFAR-10 传入 PIL 对象，ImageNet 传入图片路径，这里统一转成 RGB PIL。
    if isinstance(image_ref, Image.Image):
        return image_ref.convert("RGB")
    return Image.open(image_ref).convert("RGB")


def encode_clean_images(
    victim_model: QwenVictimModel,
    images: list[Image.Image | str],
    *,
    batch_size: int,
    log_every: int,
) -> torch.Tensor:
    # 编码干净图像特征，作为 CA 和 ASR 计算的 clean 参考。
    # 这里使用官方 Qwen 路径，不涉及可微代理训练。
    features: list[torch.Tensor] = []
    total_batches = (len(images) + batch_size - 1) // batch_size
    print(f"[encode-clean] images={len(images)} batches={total_batches}", flush=True)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = [image_to_pil(image_ref) for image_ref in images[start : start + batch_size]]
            batch_idx = start // batch_size + 1
            feats = victim_model.encoder.encode_images_official(batch, batch_size=min(len(batch), batch_size))
            features.append(feats.detach().cpu().float())
            if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                print(f"[encode-clean] batch {batch_idx}/{total_batches}", flush=True)
    return torch.cat(features, dim=0)


def apply_patch_to_pil(image: Image.Image, patch: torch.Tensor, noise_percentage: float) -> Image.Image:
    # 分类评测阶段不再训练补丁，只把已训练好的 patch 贴到图像上，
    # 然后转回 PIL，走官方 process() 路径评测真实迁移效果。
    image_tensor = TF.pil_to_tensor(image.convert("RGB")).float() / 255.0
    adv_tensor = apply_patch_single_image(image_tensor, patch.cpu(), noise_percentage=noise_percentage)
    return TF.to_pil_image(adv_tensor.cpu())


def encode_adv_images(
    victim_model: QwenVictimModel,
    images: list[Image.Image | str],
    patch: torch.Tensor,
    *,
    noise_percentage: float,
    batch_size: int,
    log_every: int,
) -> torch.Tensor:
    # 编码贴补丁后的图像特征。注意这里同样走官方路径，
    # 因为最终分类迁移结果应反映补丁在真实推理流程中的效果。
    features: list[torch.Tensor] = []
    total_batches = (len(images) + batch_size - 1) // batch_size
    print(f"[encode-adv] images={len(images)} batches={total_batches}", flush=True)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = [image_to_pil(image_ref) for image_ref in images[start : start + batch_size]]
            batch_idx = start // batch_size + 1
            patched = [apply_patch_to_pil(image, patch, noise_percentage) for image in batch_images]
            feats = victim_model.encoder.encode_images_official(patched, batch_size=min(len(patched), batch_size))
            features.append(feats.detach().cpu().float())
            if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                print(f"[encode-adv] batch {batch_idx}/{total_batches}", flush=True)
    return torch.cat(features, dim=0)


def encode_class_texts(
    victim_model: QwenVictimModel,
    class_names: list[str],
    *,
    prompt_template: str,
    text_instruction: str | None,
) -> tuple[torch.Tensor, list[str]]:
    # zero-shot 分类：把每个类别名写成文本提示词，再编码成类别 embedding。
    # 图像类别由 image embedding 与 class text embedding 的相似度决定。
    prompts = [prompt_template.format(name.replace("_", " ")) for name in class_names]
    preview = prompts[:10]
    print(f"[class-prompts] total={len(prompts)} preview={preview}", flush=True)
    with torch.no_grad():
        text_features = victim_model.encode_text(prompts, instruction=text_instruction).detach().cpu().float()
    return text_features, prompts


def predict(image_features: torch.Tensor, class_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 归一化后计算图像与类别文本的余弦相似度，取相似度最高的类别作为预测。
    image_features = F.normalize(image_features.float(), dim=-1)
    class_features = F.normalize(class_features.float(), dim=-1)
    logits = image_features @ class_features.t()
    top5 = torch.topk(logits, k=min(5, class_features.shape[0]), dim=1).indices
    top1 = top5[:, 0]
    return top1, top5


def compute_metrics(clean_features: torch.Tensor, adv_features: torch.Tensor, class_features: torch.Tensor, labels: list[int]) -> dict[str, Any]:
    # CA: clean 图像分类准确率。
    # ASR: clean 分类正确、贴补丁后分类错误的样本比例。
    # FR: 贴补丁后所有被错误分类样本占总样本比例。
    label_tensor = torch.tensor(labels, dtype=torch.long)
    clean_top1, clean_top5 = predict(clean_features, class_features)
    adv_top1, adv_top5 = predict(adv_features, class_features)

    clean_correct = clean_top1.eq(label_tensor)
    adv_correct = adv_top1.eq(label_tensor)
    clean_top5_correct = (clean_top5 == label_tensor.unsqueeze(1)).any(dim=1)
    adv_top5_correct = (adv_top5 == label_tensor.unsqueeze(1)).any(dim=1)

    total = int(label_tensor.numel())
    clean_correct_count = int(clean_correct.sum().item())
    adv_wrong = ~adv_correct
    attack_success = clean_correct & adv_wrong

    ca_top1 = 100.0 * clean_correct.float().mean().item()
    ca_top5 = 100.0 * clean_top5_correct.float().mean().item()
    adv_acc_top1 = 100.0 * adv_correct.float().mean().item()
    adv_acc_top5 = 100.0 * adv_top5_correct.float().mean().item()
    fr = 100.0 * adv_wrong.float().mean().item()
    asr = 0.0 if clean_correct_count == 0 else 100.0 * int(attack_success.sum().item()) / clean_correct_count

    return {
        "num_samples": total,
        "clean_correct": clean_correct_count,
        "ca_top1": ca_top1,
        "ca_top5": ca_top5,
        "adv_acc_top1": adv_acc_top1,
        "adv_acc_top5": adv_acc_top5,
        "asr": asr,
        "fr": fr,
    }


def main() -> None:
    args = parse_args()
    victim_name = normalize_victim_name(args.victim)
    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)
    clean_cache_root = resolve_path(args.clean_cache_root)
    patch_path = resolve_path(args.patch_path)
    patch_name = args.patch_name or patch_path.parent.name

    images, labels, ids, class_names = load_dataset(args)
    print(f"[dataset] {args.dataset} root={data_root} images={len(images)} classes={len(class_names)}", flush=True)
    if args.limit > 0:
        print("[mode] limited/smoke mode; metrics are not formal full-test results", flush=True)

    patch = torch.load(patch_path, map_location="cpu").float()
    print(f"[patch] loaded {patch_path} shape={tuple(patch.shape)}", flush=True)

    # 加载受害模型。分类任务不训练模型，只调用 Qwen embedding 做 zero-shot 预测。
    victim_model = QwenVictimModel(model_path=args.model_path)
    victim_model.eval()

    # 将类别名转换为文本提示词并编码，得到分类用的类别原型向量。
    class_features, prompts = encode_class_texts(
        victim_model,
        class_names,
        prompt_template=args.prompt_template,
        text_instruction=args.text_instruction,
    )

    clean_path = cache_path(clean_cache_root, victim_name, args.dataset, args.limit)
    if args.reuse_clean_cache and clean_path.exists():
        # 同一数据集的 clean 图像 embedding 可以复用，节省大量 Qwen 编码时间。
        print(f"[cache] loading clean image features: {clean_path}", flush=True)
        clean_payload = torch.load(clean_path, map_location="cpu")
        clean_features = clean_payload["features"].float()
    else:
        # 首次运行或不复用缓存时，重新编码 clean 图像并保存。
        clean_features = encode_clean_images(victim_model, images, batch_size=args.batch_size, log_every=args.log_every)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": ids, "labels": labels, "features": clean_features}, clean_path)
        print(f"[cache] saved clean image features: {clean_path}", flush=True)

    # 对同一批图像贴上指定补丁，重新走官方路径编码 adv 图像特征。
    adv_features = encode_adv_images(
        victim_model,
        images,
        patch,
        noise_percentage=args.noise_percentage,
        batch_size=args.batch_size,
        log_every=args.log_every,
    )
    # 基于 clean / adv 图像特征和类别文本特征计算 CA、ASR、FR。
    metrics = compute_metrics(clean_features, adv_features, class_features, labels)

    print("################ Classification Transfer Eval ################", flush=True)
    print("CA Top-1/Top-5: {:.2f} / {:.2f}".format(metrics["ca_top1"], metrics["ca_top5"]), flush=True)
    print("Adv Acc Top-1/Top-5: {:.2f} / {:.2f}".format(metrics["adv_acc_top1"], metrics["adv_acc_top5"]), flush=True)
    print("ASR / FR: {:.2f} / {:.2f}".format(metrics["asr"], metrics["fr"]), flush=True)
    print("###############################################################", flush=True)

    if args.save:
        mode = "full" if args.limit <= 0 else f"limit{args.limit}"
        save_root = output_root / victim_name / args.dataset / f"{patch_name}_{mode}"
        save_root.mkdir(parents=True, exist_ok=True)
        # 保存配置和指标，保证后续写论文或复现实验时能追溯补丁来源。
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "data_root": str(data_root),
            "victim": args.victim,
            "victim_name": victim_name,
            "model_path": args.model_path,
            "patch_path": str(patch_path),
            "patch_name": patch_name,
            "noise_percentage": args.noise_percentage,
            "prompt_template": args.prompt_template,
            "text_instruction": args.text_instruction,
            "class_names": class_names,
            "class_prompts": prompts,
            "limit": args.limit,
            "metrics": metrics,
        }
        with (save_root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"[result] saved {save_root / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
