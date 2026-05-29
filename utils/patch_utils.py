from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def patch_initialization(args, patch_type: str = "rectangle"):
    noise_percentage = args.noise_percentage
    image_size = (3, 224, 224)
    if patch_type != "rectangle":
        raise ValueError("Only rectangle patch is currently supported.")
    # 初始化通用补丁：补丁面积约为 224x224 参考图像面积的指定比例。
    # 后续真正贴到不同图像时，会根据每张图的实际尺寸重新缩放。
    mask_length = int((noise_percentage * image_size[1] * image_size[2]) ** 0.5)
    return np.random.rand(image_size[0], mask_length, mask_length).astype(np.float32)


def mask_generation(args, patch):
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size, dtype=np.float32)
    x_location = image_size[1] - 14 - patch.shape[1]
    y_location = image_size[2] - 14 - patch.shape[2]
    applied_patch[:, x_location : x_location + patch.shape[1], y_location : y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch, x_location, y_location


def clamp_patch(args, patch: torch.Tensor) -> torch.Tensor:
    # 每次优化后将补丁裁剪到合法 RGB 图像范围 [0, 1]。
    return torch.clamp(patch, min=0.0, max=1.0)


def apply_patch(images: torch.Tensor, patch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=images.device, dtype=images.dtype)
    patch = patch.to(device=images.device, dtype=images.dtype)
    if patch.dim() == 4 and patch.shape[0] == 1:
        patch = patch.squeeze(0)
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)
    return torch.clamp(mask.unsqueeze(0) * patch + (1 - mask.unsqueeze(0)) * images, 0.0, 1.0)


def apply_patch_single_image(
    image: torch.Tensor,
    patch: torch.Tensor,
    noise_percentage: float,
    margin_ratio: float = 14.0 / 224.0,
) -> torch.Tensor:
    """将通用补丁贴到单张图像上，并保持梯度可回传。

    输入图像是取值范围为 [0, 1] 的 CHW tensor。训练时 `patch` 是
    `nn.Parameter`，因此贴加区域仍然和攻击损失保持计算图连接。
    """

    if image.dim() != 3:
        raise ValueError(f"Expected single image tensor in CHW format, got shape={tuple(image.shape)}")
    if patch.dim() == 4 and patch.shape[0] == 1:
        patch = patch.squeeze(0)
    if patch.dim() != 3:
        raise ValueError(f"Expected patch in CHW format, got shape={tuple(patch.shape)}")

    image = image.float()
    patch = patch.to(device=image.device, dtype=image.dtype)
    _, height, width = image.shape
    # 使用固定面积比例，而不是固定像素大小，使补丁在不同分辨率图像上
    # 具有相近的视觉强度。
    target_area = max(1.0, noise_percentage * height * width)
    patch_side = max(1, int(target_area ** 0.5))
    # 可微缩放：来自图像 embedding 的梯度可以从缩放后的补丁
    # 回传到原始可学习补丁张量。
    patch_resized = F.interpolate(
        patch.unsqueeze(0),
        size=(patch_side, patch_side),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    patch_resized = torch.clamp(patch_resized, 0.0, 1.0)

    margin_h = int(round(height * margin_ratio))
    margin_w = int(round(width * margin_ratio))
    # 参考 AdvCLIP 的设置，将补丁贴在右下角，尽量减少对主体内容的直接遮挡。
    x0 = max(0, height - margin_h - patch_side)
    y0 = max(0, width - margin_w - patch_side)
    x1 = min(height, x0 + patch_side)
    y1 = min(width, y0 + patch_side)

    adv = image.clone()
    # 该赋值使输出图像依赖 `patch_resized`；反向传播时，
    # autograd 会通过这块贴加区域更新补丁。
    adv[:, x0:x1, y0:y1] = patch_resized[:, : x1 - x0, : y1 - y0]
    return torch.clamp(adv, 0.0, 1.0)
