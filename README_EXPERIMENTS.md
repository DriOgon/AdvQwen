# AdvQwen-new 实验命令说明

下面的命令默认假设：

- 当前工作目录是仓库根目录
- 当前 Python 环境已经基于官方 `Qwen3-VL-Embedding` 仓库正确配置
- 已正确设置 `QWEN_REPO_ROOT` 和 `QWEN_MODEL_PATH`

## 1. 干净图文检索基线

使用 `clean_retrieval_baseline.py` 在标准 caption-level 图文检索数据集上运行零样本基线。

支持的数据集：

- `data_std/flickr30k_std`
- `data_std/coco_karpathy`

快速 smoke test：

```bash
python clean_retrieval_baseline.py \
  --dataset coco_karpathy \
  --image_limit 20 \
  --text_limit 100 \
  --reuse_cache
```

完整 MS COCO Karpathy 基线：

```bash
python clean_retrieval_baseline.py \
  --dataset coco_karpathy \
  --image_batch_size 8 \
  --text_batch_size 32 \
  --metric_batch_size 256 \
  --reuse_cache
```

完整 Flickr30K 基线：

```bash
python clean_retrieval_baseline.py \
  --dataset flickr30k_std \
  --image_batch_size 8 \
  --text_batch_size 32 \
  --metric_batch_size 256 \
  --reuse_cache
```

特征缓存默认保存在：

```text
output/features_std/Qwen3VL2B/<dataset>/
```

指标结果默认保存在：

```text
output/clean_baseline/Qwen3VL2B/<dataset>/
```

## 2. 零样本对抗补丁攻击

使用 `retrieval_attack.py` 运行当前主实验中的图文检索补丁攻击实现。

COCO smoke attack：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python retrieval_attack.py \
  --train_dataset coco_karpathy \
  --eval_dataset coco_karpathy \
  --num_epochs 1 \
  --eval_every 1 \
  --batch_size 4 \
  --train_limit 64 \
  --test_image_limit 100 \
  --test_text_limit 500 \
  --reuse_clean_cache \
  --save \
  --run_name coco_to_coco_smoke
```

完整 Flickr30K 到 Flickr30K 攻击：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python retrieval_attack.py \
  --train_dataset flickr30k_std \
  --eval_dataset flickr30k_std \
  --num_epochs 10 \
  --eval_every 1 \
  --batch_size 4 \
  --train_limit 2048 \
  --reuse_clean_cache \
  --save \
  --run_name flickr_to_flickr_full
```

完整 COCO 到 COCO 攻击：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python retrieval_attack.py \
  --train_dataset coco_karpathy \
  --eval_dataset coco_karpathy \
  --num_epochs 10 \
  --eval_every 1 \
  --batch_size 4 \
  --train_limit 2048 \
  --reuse_clean_cache \
  --save \
  --run_name coco_to_coco_full
```

跨数据集迁移示例：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python retrieval_attack.py \
  --train_dataset flickr30k_std \
  --eval_dataset coco_karpathy \
  --num_epochs 10 \
  --eval_every 1 \
  --batch_size 4 \
  --train_limit 2048 \
  --reuse_clean_cache \
  --save \
  --run_name flickr_to_coco_full
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python retrieval_attack.py \
  --train_dataset coco_karpathy \
  --eval_dataset flickr30k_std \
  --num_epochs 10 \
  --eval_every 1 \
  --batch_size 4 \
  --train_limit 2048 \
  --reuse_clean_cache \
  --save \
  --run_name coco_to_flickr_full
```

攻击输出默认保存在：

```text
output/zero_shot_attack/Qwen3VL2B/<run_name>/
```

## 3. 分类迁移评测

使用 `classification_transfer_eval.py` 评测检索任务训练得到的补丁在零样本分类任务上的迁移效果。

CIFAR-10 示例：

```bash
python classification_transfer_eval.py \
  --dataset cifar10 \
  --data_root data_std/cifar10 \
  --patch_path /path/to/patch.pt \
  --reuse_clean_cache \
  --save
```

ImageNet 示例：

```bash
python classification_transfer_eval.py \
  --dataset imagenet \
  --data_root data_std/imagenet \
  --patch_path /path/to/patch.pt \
  --reuse_clean_cache \
  --save
```

输出默认保存在：

```text
output/classification_eval/
output/classification_features/
```
