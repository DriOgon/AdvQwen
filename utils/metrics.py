from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def KL(P, Q, mask=None):
    eps = 1e-7
    d = (P + eps).log() - (Q + eps).log()
    d = P * d
    if mask is not None:
        d = d * mask
    return torch.sum(d)


def CE(P, Q, mask=None):
    return KL(P, Q, mask) + KL(1 - P, 1 - Q, mask)


def umap(output_net, target_net, eps=1e-7):
    n, _ = output_net.shape
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0
    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    model_distance = 1 - model_similarity
    model_distance[range(n), range(n)] = 3
    model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
    model_distance[range(n), range(n)] = 0
    model_similarity = 1 - model_distance
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
    target_distance = 1 - target_similarity
    target_distance[range(n), range(n)] = 3
    target_distance = target_distance - torch.min(target_distance, dim=1)[0].view(-1, 1)
    target_distance[range(n), range(n)] = 0
    target_similarity = 1 - target_distance
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
    return CE(target_similarity, model_similarity)


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(X, norm)


def calcdist(img, txt):
    dist = img.unsqueeze(1) - txt.unsqueeze(0)
    dist = torch.sum(torch.pow(dist, 2), dim=2)
    return torch.sqrt(dist)


def _labels_overlap(lhs: Iterable[int], rhs: Iterable[int]) -> bool:
    return bool(set(lhs) & set(rhs))


def _match_matrix_from_labels(labels):
    if isinstance(labels, torch.Tensor):
        if labels.dim() == 1:
            return labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
        if labels.dim() == 2:
            return (labels.float() @ labels.float().t() > 0).float()
    rows = list(labels)
    n = len(rows)
    match = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            if _labels_overlap(rows[i], rows[j]):
                match[i, j] = 1.0
    return match


def average_precision(relevance_flags: list[bool], num_relevant: int) -> float:
    if num_relevant == 0:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for rank, flag in enumerate(relevance_flags, start=1):
        if flag:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / num_relevant if hit_count > 0 else 0.0


def fx_calc_map_label(query, gallery, query_labels, gallery_labels, k=0, dist_method="COS"):
    query = torch.as_tensor(query, dtype=torch.float32)
    gallery = torch.as_tensor(gallery, dtype=torch.float32)
    if dist_method == "COS":
        query = torch.nn.functional.normalize(query, dim=-1)
        gallery = torch.nn.functional.normalize(gallery, dim=-1)
        similarity = torch.matmul(query, gallery.t())
        order = torch.argsort(similarity, dim=1, descending=True)
    else:
        distance = torch.cdist(query, gallery, p=2)
        order = torch.argsort(distance, dim=1, descending=False)
    numcases = order.shape[0]
    if k == 0:
        k = order.shape[1]
    res = []
    for i in range(numcases):
        ranked = order[i][:k].tolist()
        relevant_total = sum(1 for labels in gallery_labels if _labels_overlap(query_labels[i], labels))
        flags = [_labels_overlap(query_labels[i], gallery_labels[idx]) for idx in ranked]
        res.append(average_precision(flags, relevant_total))
    return round(float(np.mean(res)) if res else 0.0, 4)


def fx_calc_recall(query, gallery, query_labels, gallery_labels, dist_method="COS"):
    query = torch.as_tensor(query, dtype=torch.float32)
    gallery = torch.as_tensor(gallery, dtype=torch.float32)
    if dist_method == "COS":
        query = torch.nn.functional.normalize(query, dim=-1)
        gallery = torch.nn.functional.normalize(gallery, dim=-1)
        similarity = torch.matmul(query, gallery.t())
        order = torch.argsort(similarity, dim=1, descending=True)
    else:
        distance = torch.cdist(query, gallery, p=2)
        order = torch.argsort(distance, dim=1, descending=False)
    ranks = np.zeros(query.shape[0])
    for i in range(query.shape[0]):
        ranked = order[i].tolist()
        rank = next((r for r, idx in enumerate(ranked) if _labels_overlap(query_labels[i], gallery_labels[idx])), len(ranked))
        ranks[i] = rank
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return r1, r5, r10


def calc_rank_positions(query, gallery, query_labels, gallery_labels, dist_method="COS"):
    query = torch.as_tensor(query, dtype=torch.float32)
    gallery = torch.as_tensor(gallery, dtype=torch.float32)
    if dist_method == "COS":
        query = torch.nn.functional.normalize(query, dim=-1)
        gallery = torch.nn.functional.normalize(gallery, dim=-1)
        similarity = torch.matmul(query, gallery.t())
        order = torch.argsort(similarity, dim=1, descending=True)
    else:
        distance = torch.cdist(query, gallery, p=2)
        order = torch.argsort(distance, dim=1, descending=False)
    ranks = np.zeros(query.shape[0], dtype=np.int64)
    for i in range(query.shape[0]):
        ranked = order[i].tolist()
        rank = next((r for r, idx in enumerate(ranked) if _labels_overlap(query_labels[i], gallery_labels[idx])), len(ranked))
        ranks[i] = rank
    return ranks


def recall_at_ks_from_ranks(ranks, ks=(1, 5, 10)):
    ranks = np.asarray(ranks)
    recalls = []
    for k in ks:
        recalls.append(100.0 * float(np.mean(ranks < k)))
    return tuple(recalls)


def asr_at_ks(clean_ranks, adv_ranks, ks=(1, 5, 10)):
    clean_ranks = np.asarray(clean_ranks)
    adv_ranks = np.asarray(adv_ranks)
    values = []
    for k in ks:
        clean_success = clean_ranks < k
        denom = int(clean_success.sum())
        if denom == 0:
            values.append(0.0)
            continue
        adv_failure = adv_ranks >= k
        values.append(100.0 * float(np.sum(clean_success & adv_failure)) / float(denom))
    return tuple(values)


def Contrastive_Loss(img, txt, label, margin=0.2):
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = _match_matrix_from_labels(label).to(device=img.device, dtype=img.dtype)
    pos = torch.mul(dist, match)
    neg = margin - torch.mul(dist, 1 - match)
    neg = torch.clamp(neg, 0)
    loss = torch.sum(pos) + torch.sum(neg)
    return loss / batch
