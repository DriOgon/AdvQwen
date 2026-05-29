# AdvQwen-new

[中文](README.md) | English

`AdvQwen-new` is the public thesis-project repository for studying
transferable adversarial patch attacks against `Qwen3-VL-Embedding-2B`.

This repository keeps the main experimental pipeline only:

1. Clean image-text retrieval baselines on standard datasets
2. Universal patch training on Flickr30K or MS COCO
3. Official-path retrieval attack evaluation
4. Transfer evaluation on zero-shot classification for CIFAR-10 and ImageNet

## Environment

This project is intended to run inside the official
[`Qwen3-VL-Embedding`](https://github.com/QwenLM/Qwen3-VL-Embedding/)
environment rather than a fully standalone one.

Recommended setup:

```bash
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
cd Qwen3-VL-Embedding
bash scripts/setup_environment.sh
source .venv/bin/activate
```

Then set:

```bash
export QWEN_REPO_ROOT=/path/to/Qwen3-VL-Embedding
export QWEN_MODEL_PATH=/path/to/Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B
```

## Main Files

- `clean_retrieval_baseline.py`
- `retrieval_attack.py`
- `retrieval_patch_eval.py`
- `classification_transfer_eval.py`
- `utils/`
- `victims/`

## Data

Datasets are expected under `data_std/` and are not included in this repository.
See `DATASETS.md` for the Chinese dataset notes and `README_EXPERIMENTS.md`
for detailed command examples.
