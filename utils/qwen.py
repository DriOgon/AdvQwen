from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from victims.qwen_trainable import QwenTrainableImageEncoder, RETRIEVAL_INSTRUCTION


class QwenVictimModel(nn.Module):
    """Minimal CLIP-like wrapper around the Qwen trainable image/text encoders."""

    def __init__(self, model_path: str | None = None, image_size: int = 224):
        super().__init__()
        self.encoder = QwenTrainableImageEncoder(model_path=model_path, attack_image_size=image_size)
        self.model_path = model_path
        self.image_size = image_size
        self.name = "Qwen3-VL-Embedding-2B"
        self.feature_dim = self.encoder.feature_dim
        self.image_batch_size = 8
        self.text_batch_size = 32

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def to(self, device: torch.device | str):
        # The underlying Qwen embedder already manages its own device placement.
        return self

    def train(self, mode: bool = True):
        self.encoder.embedder.model.train(mode)
        return self

    def eval(self):
        self.encoder.embedder.model.eval()
        return self

    def load_image_tensors(self, image_paths: Sequence[str]) -> torch.Tensor:
        return self.encoder.load_image_tensors(image_paths)

    def encode_image(self, images: torch.Tensor | Sequence[str], mode: str = "auto") -> torch.Tensor:
        if mode not in {"auto", "official", "proxy"}:
            raise ValueError(f"Unsupported image encoding mode: {mode}")
        if isinstance(images, torch.Tensor):
            if mode == "official":
                raise ValueError("Official image encoding expects file paths, not tensors.")
            feats = self.encoder.encode_image_tensors_proxy(images, requires_grad=images.requires_grad)
            return feats if feats.device == self.device else feats.to(self.device)
        if mode == "proxy":
            feats = self.encoder.encode_images_proxy(images, batch_size=min(len(images), self.image_batch_size))
        else:
            feats = self.encoder.encode_images_official(images, batch_size=min(len(images), self.image_batch_size))
        return feats.to(self.device)

    def encode_text(self, texts: Sequence[str], instruction: str | None = RETRIEVAL_INSTRUCTION) -> torch.Tensor:
        feats = self.encoder.encode_texts(texts, batch_size=min(len(texts), self.text_batch_size), instruction=instruction)
        return feats.to(self.device)

    def encode_image_paths_with_patch_proxy(self, image_paths: Sequence[str], patch: torch.Tensor, noise_percentage: float) -> torch.Tensor:
        feats = self.encoder.encode_image_paths_proxy_with_patch(
            image_paths,
            patch,
            noise_percentage=noise_percentage,
        )
        return feats if feats.device == self.device else feats.to(self.device)

    def encode_image_paths_with_patch_official(self, image_paths: Sequence[str], patch: torch.Tensor, noise_percentage: float) -> torch.Tensor:
        feats = self.encoder.encode_image_paths_official_with_patch(
            image_paths,
            patch,
            noise_percentage=noise_percentage,
            batch_size=min(len(image_paths), self.image_batch_size),
        )
        return feats if feats.device == self.device else feats.to(self.device)


def normalize_victim_name(victim: str) -> str:
    if victim.lower() in {"qwen3-vl-embedding-2b", "qwen", "qwen3"}:
        return "Qwen3VL2B"
    return victim.replace("/", "_")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_qwen_repo_root() -> Path:
    env_value = os.environ.get("QWEN_REPO_ROOT")
    if env_value:
        return Path(env_value).expanduser()
    return project_root().parent / "Qwen3-VL-Embedding"


def default_model_path() -> str:
    env_value = os.environ.get("QWEN_MODEL_PATH")
    if env_value:
        return str(Path(env_value).expanduser())
    return str(default_qwen_repo_root() / "models" / "Qwen3-VL-Embedding-2B")
