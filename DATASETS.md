# AdvQwen 数据集说明

本项目主要使用两类数据集：

- 图文检索数据集，用于干净基线、补丁训练和攻击评测
- 图像分类数据集，用于迁移评测

数据集默认放在：

```text
data_std/
```

本仓库**不直接包含**这些数据集文件。

## 数据集总览

| 数据集 | 本地目录 | 任务类型 | 主要用途 |
| --- | --- | --- | --- |
| Flickr30K | `data_std/flickr30k_std` | 图文检索 | 干净基线、补丁训练、攻击评测 |
| MS COCO Karpathy | `data_std/coco_karpathy` | 图文检索 | 干净基线、补丁训练、攻击评测 |
| CIFAR-10 | `data_std/cifar10` | 图像分类 | 迁移评测 |
| ImageNet | `data_std/imagenet` | 图像分类 | 迁移评测 |

## 图文检索数据集

Flickr30K 和 MS COCO Karpathy 都被整理成统一的 `jsonl` 格式。
每张图像分配一个 label id，与之匹配的 caption 共享同一个 label。

常见文件包括：

- `train_surrogate.jsonl`：补丁训练用代理数据
- `train_downstream.jsonl`：预留的下游训练划分
- `train_full.jsonl`：完整 caption-level 训练数据
- `val_images.jsonl`
- `val_texts.jsonl`
- `test_images.jsonl`
- `test_texts.jsonl`

评测协议如下：

- image-to-text：图像作为 query，文本作为 gallery
- text-to-image：文本作为 query，图像作为 gallery
- 当 query 与候选项的 `labels` 有交集时，认为二者匹配

常见 `jsonl` 字段说明：

| 字段 | 含义 |
| --- | --- |
| `id` | 标准化后的样本编号 |
| `image_rel` | 图像相对路径 |
| `image_path` | 实验代码使用的本地图像路径 |
| `text` | 当前文本描述 |
| `captions` | 当前图像对应的全部 caption |
| `caption_index` | 当前 caption 在全部 caption 中的位置；图像行通常为 `null` |
| `labels` | 图像与文本共享的匹配标签 |
| `split` | 划分名称 |
| `modality` | `image` 或 `text` |

## 图像分类数据集

`classification_transfer_eval.py` 默认使用：

- `data_std/cifar10` 进行 CIFAR-10 零样本分类迁移评测
- `data_std/imagenet` 进行 ImageNet 验证集迁移评测

这两类数据集不参与补丁训练，只用于衡量在图文检索任务上训练出的补丁是否具有跨任务迁移性。

## 公开仓库建议

- Git 中只保留代码和文档
- 不要提交数据集本体、缓存文件和实验输出
- 如果你在本地重新构建标准化数据集，注意检查其中的绝对路径是否适合当前环境
