"""Prototypical network baseline - the algorithm-oriented comparison.

Trains a task-specific embedding with episodic prototypical loss (Snell et al.,
2017) on the project's own crops, and compares it head to head against the
frozen CLIP embedding on identical data.

The comparison is threshold-free, and that is the important design decision.
An earlier version scored both embeddings at a mean-similarity threshold and
got recall pinned at 1.000 for every configuration - the numbers were
measuring the threshold rule, not the embedding. Ranking metrics (average
precision, ROC-AUC, best achievable F1 over all thresholds) measure whether
the embedding separates the classes at all, which is the actual question.
Under class imbalance, AP is the headline: the random-ranker baseline is the
positive prevalence, so an AP near prevalence means the embedding is useless
regardless of how good the raw number looks.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = ["ProtoEmbedNet", "ProtoTrainer", "ranking_metrics", "compare_embeddings"]


def _torch():
    import torch
    return torch


class ProtoEmbedNet:
    """ResNet18 backbone with a linear head, L2-normalised output.

    Wrapped in a plain class rather than exposed as a bare ``nn.Module`` so
    importing this file does not require torch - the rest of the package is
    usable without it.
    """

    def __new__(cls, emb_dim: int = 128):
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.models as tvm

        class _Net(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
                backbone.fc = nn.Linear(backbone.fc.in_features, dim)
                self.backbone = backbone

            def forward(self, x):
                return F.normalize(self.backbone(x), p=2, dim=-1)

        return _Net(emb_dim)


class ProtoTrainer:
    """Episodic trainer for the two-way (target vs non-target) task.

    Args:
        train_crops: PIL crops from the **train** split only.
        train_labels: 1 for target, 0 otherwise.
        image_size: crops are resized to this square. 96 px is a deliberate
            compromise - the objects are tiny, so upsampling much beyond their
            native resolution buys nothing but compute.
    """

    def __init__(self, train_crops: Sequence, train_labels: np.ndarray,
                 emb_dim: int = 128, image_size: int = 96, seed: int = 42,
                 device: str | None = None):
        import random

        import torch
        import torchvision.transforms as T

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_size = image_size
        self.net = ProtoEmbedNet(emb_dim).to(self.device)

        labels = np.asarray(train_labels)
        positive = np.where(labels == 1)[0]
        negative = np.where(labels == 0)[0]
        print(f"Prototypical training pool -> target: {len(positive)}, "
              f"non-target: {len(negative)}")
        if len(positive) < 2 or len(negative) < 2:
            raise RuntimeError(
                f"Need at least 2 crops of each class (target={len(positive)}, "
                f"non-target={len(negative)}). Check that the train split "
                "actually contains target labels.")

        self.pos_tensors = self.crops_to_tensor([train_crops[i] for i in positive])
        self.neg_tensors = self.crops_to_tensor([train_crops[i] for i in negative])

    def crops_to_tensor(self, pil_crops: Sequence):
        torch = _torch()
        if not len(pil_crops):
            return torch.empty(0, 3, self.image_size, self.image_size)
        return torch.stack([self.transform(c.convert("RGB")) for c in pil_crops])

    def sample_episode(self, n_support: int = 5, n_query: int = 5):
        """One two-way episode: support and query tensors plus query labels."""
        torch = _torch()

        def pick(tensors, n):
            n = min(n, len(tensors))
            return tensors[torch.randperm(len(tensors))[:n]]

        s_pos, q_pos = pick(self.pos_tensors, n_support), pick(self.pos_tensors, n_query)
        s_neg, q_neg = pick(self.neg_tensors, n_support), pick(self.neg_tensors, n_query)

        support = torch.cat([s_pos, s_neg], 0)
        query = torch.cat([q_pos, q_neg], 0)
        query_labels = torch.cat([torch.ones(len(q_pos)), torch.zeros(len(q_neg))]).long()
        return support, (len(s_pos), len(s_neg)), query, query_labels

    def prototypical_loss(self, support, split, query, query_labels):
        """Cross-entropy over negative distances to the two class prototypes."""
        import torch
        import torch.nn.functional as F

        support_emb = self.net(support.to(self.device))
        query_emb = self.net(query.to(self.device))

        n_pos, n_neg = split
        proto_pos = support_emb[:n_pos].mean(0, keepdim=True)
        proto_neg = support_emb[n_pos:n_pos + n_neg].mean(0, keepdim=True)
        prototypes = torch.cat([proto_neg, proto_pos], 0)   # index 0=non-target

        logits = -torch.cdist(query_emb, prototypes)
        loss = F.cross_entropy(logits, query_labels.to(self.device))
        accuracy = (logits.argmax(1).cpu() == query_labels).float().mean().item()
        return loss, accuracy

    def train(self, episodes: int = 300, lr: float = 1e-4,
              n_support: int = 5, n_query: int = 5, log_every: int = 50):
        import torch

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        print(f"\nTraining prototypical embedding for {episodes} episodes...")

        for episode in range(1, episodes + 1):
            support, split, query, query_labels = self.sample_episode(n_support, n_query)
            loss, accuracy = self.prototypical_loss(support, split, query, query_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if episode % log_every == 0:
                print(f"  episode {episode:4d}/{episodes}  loss={loss.item():.4f}  "
                      f"episode_acc={accuracy:.3f}")

        self.net.eval()
        print("Prototypical embedding trained.")
        return self.net

    def embed(self, pil_crops: Sequence, batch_size: int = 64):
        import torch

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(pil_crops), batch_size):
                batch = self.crops_to_tensor(pil_crops[i:i + batch_size]).to(self.device)
                embeddings.append(self.net(batch).cpu())
        return torch.cat(embeddings, 0) if embeddings else torch.empty(0, 128)

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        print(f"Saved embedding weights: {path}")
        return path


def ranking_metrics(support_crops: Sequence, embed_fn: Callable, query_emb,
                    query_labels: np.ndarray, prevalence: float, k: int) -> dict:
    """Threshold-free quality of a K-shot prototype ranking.

    ``ap_lift_over_random`` is the number to read: AP divided by prevalence.
    A lift near 1.0 means the embedding ranks no better than chance, however
    respectable the raw AP looks on an imbalanced set.
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    def l2(x):
        return F.normalize(torch.as_tensor(np.asarray(x), dtype=torch.float32),
                           p=2, dim=-1)

    k_eff = min(k, len(support_crops))
    support_emb = l2(embed_fn(list(support_crops)[:k_eff]))
    prototype = F.normalize(support_emb.mean(dim=0, keepdim=True), p=2, dim=-1)
    similarities = (l2(query_emb) @ prototype.T).squeeze(1).numpy()

    average_precision = average_precision_score(query_labels, similarities)
    auc = roc_auc_score(query_labels, similarities)

    precision, recall, _ = precision_recall_curve(query_labels, similarities)
    f1_curve = np.divide(2 * precision * recall, precision + recall,
                         out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.argmax(f1_curve))

    return {
        "num_support": k_eff,
        "average_precision": float(average_precision),
        "ap_lift_over_random": float(average_precision / prevalence) if prevalence else None,
        "roc_auc": float(auc),
        "best_f1": float(f1_curve[best]),
        "precision_at_best_f1": float(precision[best]),
        "recall_at_best_f1": float(recall[best]),
    }


def compare_embeddings(support_crops: Sequence, query_crops: Sequence,
                       query_labels: np.ndarray, clip_embed_fn: Callable,
                       proto_embed_fn: Callable, k_values: Sequence[int] = (1, 2, 5, 10, 20)):
    """Score the frozen-CLIP and task-trained embeddings on identical data.

    Restricting the query set to crops from real photographs is essential and
    is the caller's responsibility: synthetic crops descend from the same 44
    cutouts the support set is drawn from, so a task-trained embedding can
    score highly by recognising cutout provenance rather than the species.
    """
    import pandas as pd

    query_labels = np.asarray(query_labels)
    n_positive = int((query_labels == 1).sum())
    prevalence = n_positive / max(len(query_labels), 1)
    if n_positive == 0 or n_positive == len(query_labels):
        raise RuntimeError("Query set is single-class; cannot evaluate ranking.")

    print(f"Query crops: {len(query_crops)} (target={n_positive})")
    print(f"Positive prevalence = {prevalence:.4f}  (the random-baseline AP)")

    clip_query = clip_embed_fn(list(query_crops))
    proto_query = proto_embed_fn(list(query_crops))

    rows = []
    for k in k_values:
        clip_row = ranking_metrics(support_crops, clip_embed_fn, clip_query,
                                   query_labels, prevalence, k)
        proto_row = ranking_metrics(support_crops, proto_embed_fn, proto_query,
                                    query_labels, prevalence, k)
        rows.append({"embedding": "CLIP (frozen foundation model)", "K": k, **clip_row})
        rows.append({"embedding": "Prototypical (task-trained)", "K": k, **proto_row})
        print(f"K={k:>2}  CLIP AP={clip_row['average_precision']:.4f} "
              f"AUC={clip_row['roc_auc']:.3f}  |  "
              f"Proto AP={proto_row['average_precision']:.4f} "
              f"AUC={proto_row['roc_auc']:.3f}")

    return pd.DataFrame(rows), prevalence
