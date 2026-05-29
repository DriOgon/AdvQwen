from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QWEN_REPO_ROOT = Path(os.environ.get("QWEN_REPO_ROOT", PROJECT_ROOT.parent / "Qwen3-VL-Embedding")).expanduser()
if str(QWEN_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_REPO_ROOT))

try:
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - import-time environment guard
    raise ModuleNotFoundError(
        "Could not import `Qwen3VLEmbedder`. Set `QWEN_REPO_ROOT` to your local "
        "`Qwen3-VL-Embedding` repository path before running this project."
    ) from exc
from utils.patch_utils import apply_patch_single_image  # noqa: E402


RETRIEVAL_INSTRUCTION = "Retrieve images or text relevant to the user's query."
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    # 对齐 Qwen 视觉 token 的网格约束：图像高宽需要是
    # patch_size * merge_size 的整数倍，同时不能超出官方像素范围。
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class OfficialLikeQwenImagePreprocessor:
    """Qwen 官方图像预处理流程的可微近似版本。

    官方 `process()` 路径适合做 clean baseline 和最终评测，但其中包含
    PIL 转换以及高层 processor 封装，不方便把梯度传回对抗补丁。
    因此这里用 PyTorch tensor 操作重建图像侧关键步骤，使攻击损失
    可以反向传播到补丁像素。
    """

    def __init__(
        self,
        *,
        min_pixels: int,
        max_pixels: int,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        # Qwen 图像处理器会在 merge 后组织视觉 patch，因此 resize 后的
        # 高宽必须能被该因子整除。
        self.factor = patch_size * merge_size
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)

    def _resize_batch(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = images.shape[-2:]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        if (resized_height, resized_width) != (height, width):
            # 使用 `F.interpolate` 保持 resize 操作可导；如果转回 PIL 再 resize，
            # 梯度链路就会被破坏。
            images = F.interpolate(
                images,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return images, (resized_height, resized_width)

    def preprocess(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.dim() != 4:
            raise ValueError(f"Expected images in BCHW format, got shape={tuple(images.shape)}")
        images = images.float()
        images, (resized_height, resized_width) = self._resize_batch(images)
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        # 使用官方 processor 的均值和方差进行归一化。
        patches = (images - mean) / std
        # Qwen3-VL 使用兼容视频输入的视觉格式。单张图片会先被看作
        # 一个时间帧，再进入 patch 化流程。
        patches = patches.unsqueeze(1)

        if patches.shape[1] % self.temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=1)

        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        # 复现 Qwen 期望的视觉 patch 排列方式。最终得到的 `pixel_values`
        # 和 `image_grid_thw` 会在 `encode_image_tensors_proxy` 中替换掉
        # dummy image 生成的视觉输入。
        patches = patches.view(
            batch_size,
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * batch_size, dtype=torch.long, device=images.device)
        pixel_values = flatten_patches.reshape(
            batch_size * grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return pixel_values, image_grid_thw


class QwenTrainableImageEncoder:
    def __init__(self, model_path: str | None = None, attack_image_size: int = 224):
        self.model_path = model_path or str(QWEN_REPO_ROOT / "models" / "Qwen3-VL-Embedding-2B")
        self.embedder = Qwen3VLEmbedder(model_name_or_path=self.model_path)
        image_processor = self.embedder.processor.image_processor
        self.attack_image_size = attack_image_size
        # 代理预处理器直接读取官方 Qwen processor 的关键参数，
        # 使可微训练路径尽量贴近最终评测路径。
        self.proxy_preprocessor = OfficialLikeQwenImagePreprocessor(
            min_pixels=self.embedder.min_pixels,
            max_pixels=self.embedder.max_pixels,
            patch_size=image_processor.patch_size,
            temporal_patch_size=image_processor.temporal_patch_size,
            merge_size=image_processor.merge_size,
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
        )

    @property
    def device(self) -> torch.device:
        return self.embedder.model.device

    @property
    def name(self) -> str:
        return "qwen3_vl_embedding_2b_official_clean_proxy_attack"

    @property
    def feature_dim(self) -> int:
        return 2048

    def load_image_tensors(self, image_paths: Sequence[str]) -> torch.Tensor:
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image = TF.resize(
                image,
                [self.attack_image_size, self.attack_image_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            tensor = TF.pil_to_tensor(image).float() / 255.0
            images.append(tensor)
        return torch.stack(images, dim=0)

    def load_original_image_tensor(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        # 转为 [C, H, W] 格式、取值范围为 [0, 1] 的 tensor。
        # 训练攻击时保持 tensor 形式，梯度才能从模型输出回传到补丁像素。
        return TF.pil_to_tensor(image).float() / 255.0

    def load_original_pil_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int,
        instruction: str | None = RETRIEVAL_INSTRUCTION,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_inputs = [{"text": text, "instruction": instruction} for text in batch_texts]
            emb = self.embedder.process(batch_inputs).float().cpu()
            outputs.append(emb)
        return torch.cat(outputs, dim=0)

    def _build_image_only_conversations(self, batch_size: int, instruction: str | None) -> list[list[dict]]:
        prompt = instruction or self.embedder.default_instruction
        conversations = []
        for _ in range(batch_size):
            conversations.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": prompt}]},
                    {"role": "user", "content": [{"type": "image", "image": "file:///dev/null"}]},
                ]
            )
        return conversations

    def _tokenize_image_only_prompts(
        self,
        *,
        batch_size: int,
        image_size: tuple[int, int],
        instruction: str | None,
    ) -> dict[str, torch.Tensor]:
        conversations = self._build_image_only_conversations(batch_size, instruction)
        text = self.embedder.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        # dummy image 只用于借助官方 processor 生成正确的文本 token、
        # attention mask 和图像占位结构，不参与真正的攻击图像编码。
        dummy_image = Image.new("RGB", (image_size[1], image_size[0]), color=0)
        inputs = self.embedder.processor(
            text=text,
            images=[dummy_image for _ in range(batch_size)],
            truncation=True,
            max_length=self.embedder.max_length,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def encode_image_tensors_proxy(
        self,
        image_tensors: torch.Tensor,
        *,
        requires_grad: bool = False,
        instruction: str | None = None,
    ) -> torch.Tensor:
        image_tensors = image_tensors.to(self.device)
        # 可微图像侧路径：tensor 图像 -> 近似官方格式的
        # pixel_values/image_grid_thw。梯度可以通过这里继续回传。
        pixel_values, image_grid_thw = self.proxy_preprocessor.preprocess(image_tensors)
        resized_height = int(image_grid_thw[0, 1].item() * self.proxy_preprocessor.patch_size)
        resized_width = int(image_grid_thw[0, 2].item() * self.proxy_preprocessor.patch_size)
        inputs = self._tokenize_image_only_prompts(
            batch_size=image_tensors.shape[0],
            image_size=(resized_height, resized_width),
            instruction=instruction,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 用可微视觉张量替换 dummy image 产生的视觉输入。
        # 文本和多模态输入结构仍来自官方 processor，图像像素部分保持可训练。
        inputs["pixel_values"] = pixel_values
        inputs["image_grid_thw"] = image_grid_thw

        if not requires_grad:
            with torch.no_grad():
                outputs = self.embedder.model(**inputs)
                embeddings = self.embedder._pooling_last(outputs.last_hidden_state, inputs["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings.float().cpu()

        # 攻击训练会进入这里。这里不使用 `torch.no_grad()`，
        # 因此损失可以经过 Qwen 前向过程继续反向传播到补丁。
        outputs = self.embedder.model(**inputs)
        embeddings = self.embedder._pooling_last(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.float()

    def encode_images_official(
        self,
        images: Sequence[str | Image.Image],
        batch_size: int,
        instruction: str | None = None,
    ) -> torch.Tensor:
        # 官方路径：用于 clean baseline 和最终攻击评测。
        # 这条路径最贴近 Qwen 公开 API，但不适合直接训练补丁，
        # 因为它不方便进行像素级梯度反传。
        outputs = []
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch_inputs = [{"image": image_obj, "instruction": instruction} if instruction else {"image": image_obj} for image_obj in batch_images]
            emb = self.embedder.process(batch_inputs).float().cpu()
            outputs.append(emb)
        return torch.cat(outputs, dim=0)

    def encode_images_proxy(
        self,
        image_paths: Sequence[str],
        batch_size: int,
        instruction: str | None = None,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            image_tensors = self.load_image_tensors(batch_paths)
            outputs.append(self.encode_image_tensors_proxy(image_tensors, requires_grad=False, instruction=instruction))
        return torch.cat(outputs, dim=0)

    def encode_image_paths_proxy_with_patch(
        self,
        image_paths: Sequence[str],
        patch: torch.Tensor,
        *,
        noise_percentage: float,
        instruction: str | None = None,
    ) -> torch.Tensor:
        outputs = []
        for image_path in image_paths:
            image_tensor = self.load_original_image_tensor(image_path).to(self.device)
            # 先贴补丁，再进入代理预处理路径。因此最终 embedding
            # 会直接依赖可学习的补丁参数。
            adv_tensor = apply_patch_single_image(image_tensor, patch, noise_percentage=noise_percentage)
            adv_feature = self.encode_image_tensors_proxy(
                adv_tensor.unsqueeze(0),
                requires_grad=patch.requires_grad,
                instruction=instruction,
            )
            if adv_feature.dim() == 2 and adv_feature.shape[0] == 1:
                outputs.append(adv_feature)
            else:
                outputs.append(adv_feature.unsqueeze(0))
        return torch.cat(outputs, dim=0)

    def encode_image_paths_official_with_patch(
        self,
        image_paths: Sequence[str],
        patch: torch.Tensor,
        *,
        noise_percentage: float,
        instruction: str | None = None,
        batch_size: int = 8,
    ) -> torch.Tensor:
        patched_images: list[Image.Image] = []
        for image_path in image_paths:
            image_tensor = self.load_original_image_tensor(image_path)
            adv_tensor = apply_patch_single_image(image_tensor, patch, noise_percentage=noise_percentage)
            # 评测时刻意转回 PIL 并调用官方 process()，
            # 这样报告的指标才代表补丁迁移到真实推理路径后的效果。
            patched_images.append(TF.to_pil_image(adv_tensor.cpu()))
        return self.encode_images_official(patched_images, batch_size=batch_size, instruction=instruction)
