import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
import matplotlib.pyplot as plt
from tqdm import tqdm
import faiss


USER_CATEGORICAL = [
    "user_id",
    "user_age_le",
    "user_gender_le",
    "user_country_le",
    "user_device_brand_le",
    "user_device_price_le",
    "fans_num_le",
    "follow_num_le",
    "accu_watch_live_cnt_le",
    "accu_watch_live_duration_le",
]

USER_NUMERIC = [
    "is_live_streamer",
    "is_photo_author",
    "user_onehot_feat0",
    "user_onehot_feat1",
    "user_onehot_feat2",
    "user_onehot_feat3",
    "user_onehot_feat4",
    "user_onehot_feat5",
    "user_onehot_feat6",
    "user_account_age",
    "user_watch_live_age",
]

ITEM_CATEGORICAL = [
    "streamer_id",
    "live_type",
    "live_content_category_le",
    "live_start_year",
    "live_start_month",
    "live_start_day",
    "live_start_hour",
    "streamer_age_le",
    "streamer_country_le",
    "streamer_device_brand_le",
    "streamer_device_price_le",
    "live_operation_tag_le",
    "fans_user_num_le",
    "fans_group_fans_num_le",
    "follow_user_num_le",
    "accu_live_cnt_le",
    "accu_live_duration_le",
    "accu_play_cnt_le",
    "accu_play_duration_le",
]

ITEM_NUMERIC = [
    "streamer_gender_le",
    "live_is_weekend",
    "title_emb_missing",
    "time_since_live_start",
    "streamer_onehot_feat0",
    "streamer_onehot_feat1",
    "streamer_onehot_feat2",
    "streamer_onehot_feat3",
    "streamer_onehot_feat4",
    "streamer_onehot_feat5",
    "streamer_onehot_feat6",
] + [f"title_emb_{i}" for i in range(128)]


@dataclass
class FeatureMeta:
    user_cat_sizes: Dict[str, int]
    item_cat_sizes: Dict[str, int]


class TwoTowerDataset(Dataset):
    # Dataset wrapper that returns user/item features and label tensors for each row.
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        user_cat: List[str],
        user_num: List[str],
        item_cat: List[str],
        item_num: List[str],
        sample_weight: np.ndarray | None = None,
    ) -> None:
        self.label = torch.tensor(df[label_col].values, dtype=torch.float32)
        self.user_cat = {
            col: torch.tensor(df[col].values, dtype=torch.long) for col in user_cat
        }
        self.user_num = torch.tensor(df[user_num].values, dtype=torch.float32)
        self.item_cat = {
            col: torch.tensor(df[col].values, dtype=torch.long) for col in item_cat
        }
        self.item_num = torch.tensor(df[item_num].values, dtype=torch.float32)
        if sample_weight is None:
            sample_weight = np.ones(len(df), dtype=np.float32)
        self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int):
        user_cat = {k: v[idx] for k, v in self.user_cat.items()}
        item_cat = {k: v[idx] for k, v in self.item_cat.items()}
        return (
            user_cat,
            self.user_num[idx],
            item_cat,
            self.item_num[idx],
            self.label[idx],
            self.sample_weight[idx],
        )


class UserBatchSampler(Sampler[List[int]]):
    # Batch sampler that groups by user and limits max positives per user per batch.
    def __init__(
        self,
        user_ids: np.ndarray,
        batch_size: int,
        max_pos_per_user: int,
        seed: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.max_pos_per_user = max_pos_per_user
        self.rng = np.random.default_rng(seed)
        self.user_to_indices: Dict[int, List[int]] = {}
        for idx, uid in enumerate(user_ids):
            self.user_to_indices.setdefault(int(uid), []).append(idx)
        self.users = list(self.user_to_indices.keys())

    def __iter__(self):
        self.rng.shuffle(self.users)
        batch: List[int] = []
        for uid in self.users:
            indices = self.user_to_indices[uid]
            if len(indices) > self.max_pos_per_user:
                indices = self.rng.choice(indices, size=self.max_pos_per_user, replace=False).tolist()
            for idx in indices:
                batch.append(idx)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        # Approximate number of batches
        total = sum(min(len(v), self.max_pos_per_user) for v in self.user_to_indices.values())
        return max(1, (total + self.batch_size - 1) // self.batch_size)

    def dropped_count(self) -> int:
        # How many positives are dropped due to max_pos_per_user.
        return sum(max(0, len(v) - self.max_pos_per_user) for v in self.user_to_indices.values())


class MLP(nn.Module):
    # Simple feed-forward network used for user/item towers.
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTowerModel(nn.Module):
    # Two-tower model that embeds categorical features and combines with numeric features.
    def __init__(
        self,
        user_cat_sizes: Dict[str, int],
        item_cat_sizes: Dict[str, int],
        user_num_dim: int,
        item_num_dim: int,
        emb_dim: int = 32,
        tower_hidden: List[int] = None,
        debug_shapes: bool = False,
        dropout: float = 0.1,
        normalize_emb: bool = False,
    ) -> None:
        super().__init__()
        if tower_hidden is None:
            tower_hidden = [256, 128]

        self.debug_shapes = debug_shapes
        self.normalize_emb = normalize_emb
        self.user_embeddings = nn.ModuleDict(
            {k: nn.Embedding(v, emb_dim) for k, v in user_cat_sizes.items()}
        )
        self.item_embeddings = nn.ModuleDict(
            {k: nn.Embedding(v, emb_dim) for k, v in item_cat_sizes.items()}
        )

        user_in = emb_dim * len(user_cat_sizes) + user_num_dim
        item_in = emb_dim * len(item_cat_sizes) + item_num_dim

        self.user_tower = MLP(user_in, tower_hidden, emb_dim, dropout)
        self.item_tower = MLP(item_in, tower_hidden, emb_dim, dropout)

    def forward(
        self,
        user_cat: Dict[str, torch.Tensor],
        user_num: torch.Tensor,
        item_cat: Dict[str, torch.Tensor],
        item_num: torch.Tensor,
    ) -> torch.Tensor:
        # user_cat: dict of [B] int tensors, user_num: [B, user_num_dim]
        # item_cat: dict of [B] int tensors, item_num: [B, item_num_dim]
        user_vec, item_vec = self.encode(user_cat, user_num, item_cat, item_num)
        return (user_vec * item_vec).sum(dim=1)
 
    def encode(
        self,
        user_cat: Dict[str, torch.Tensor],
        user_num: torch.Tensor,
        item_cat: Dict[str, torch.Tensor],
        item_num: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return user/item embeddings for contrastive or triplet losses.
        user_vec = self.encode_user(user_cat, user_num)
        item_vec = self.encode_item(item_cat, item_num)
        return user_vec, item_vec

    def encode_user(
        self,
        user_cat: Dict[str, torch.Tensor],
        user_num: torch.Tensor,
    ) -> torch.Tensor:
        # Encode only user tower (for retrieval inference).
        user_embs = [self.user_embeddings[k](user_cat[k]) for k in self.user_embeddings]
        if self.debug_shapes:
            print("user_cat keys:", list(user_cat.keys()))
            print("user_embs shapes:", [tuple(t.shape) for t in user_embs])
            print("user_num shape:", tuple(user_num.shape))
        user_x = torch.cat(user_embs + [user_num], dim=1)
        if self.debug_shapes:
            print("user_x shape:", tuple(user_x.shape))
        user_vec = self.user_tower(user_x)
        if self.debug_shapes:
            print("user_vec shape:", tuple(user_vec.shape))
        if self.normalize_emb:
            user_vec = nn.functional.normalize(user_vec, dim=1)
        return user_vec

    def encode_item(
        self,
        item_cat: Dict[str, torch.Tensor],
        item_num: torch.Tensor,
    ) -> torch.Tensor:
        # Encode only item tower (for retrieval inference).
        item_embs = [self.item_embeddings[k](item_cat[k]) for k in self.item_embeddings]
        if self.debug_shapes:
            print("item_cat keys:", list(item_cat.keys()))
            print("item_embs shapes:", [tuple(t.shape) for t in item_embs])
            print("item_num shape:", tuple(item_num.shape))
        item_x = torch.cat(item_embs + [item_num], dim=1)
        if self.debug_shapes:
            print("item_x shape:", tuple(item_x.shape))
        item_vec = self.item_tower(item_x)
        if self.debug_shapes:
            print("item_vec shape:", tuple(item_vec.shape))
        if self.normalize_emb:
            item_vec = nn.functional.normalize(item_vec, dim=1)
        return item_vec


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Convert specified columns to numeric, coercing invalid values to NaN.
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _sanitize_categorical(
    df: pd.DataFrame,
    cols: List[str],
    max_values: Dict[str, int] | None = None,
) -> pd.DataFrame:
    # Ensure categorical ids are integer and non-negative, optionally clip to max_values.
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
        df[col] = df[col].clip(lower=0)
        if max_values is not None and col in max_values:
            df[col] = df[col].clip(upper=max_values[col])
    return df


def load_data(
    path: Path,
    label_col: str,
    meta: FeatureMeta | None = None,
) -> Tuple[pd.DataFrame, FeatureMeta]:
    # Load a dataset CSV and build (or reuse) categorical vocab sizes for embeddings.
    df = pd.read_csv(path)

    required_cols = set(USER_CATEGORICAL + USER_NUMERIC + ITEM_CATEGORICAL + ITEM_NUMERIC + [label_col])
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df = _coerce_numeric(df, USER_NUMERIC + ITEM_NUMERIC)

    if meta is None:
        df = _sanitize_categorical(df, USER_CATEGORICAL + ITEM_CATEGORICAL)
        df[USER_NUMERIC + ITEM_NUMERIC] = df[USER_NUMERIC + ITEM_NUMERIC].fillna(0)
        user_cat_sizes = {col: int(df[col].max() + 1) for col in USER_CATEGORICAL}
        item_cat_sizes = {col: int(df[col].max() + 1) for col in ITEM_CATEGORICAL}
        meta = FeatureMeta(user_cat_sizes=user_cat_sizes, item_cat_sizes=item_cat_sizes)
        return df, meta

    max_values = {
        **{k: v - 1 for k, v in meta.user_cat_sizes.items()},
        **{k: v - 1 for k, v in meta.item_cat_sizes.items()},
    }
    df = _sanitize_categorical(df, USER_CATEGORICAL + ITEM_CATEGORICAL, max_values=max_values)
    df[USER_NUMERIC + ITEM_NUMERIC] = df[USER_NUMERIC + ITEM_NUMERIC].fillna(0)
    return df, meta


def build_meta_from_dfs(dfs: List[pd.DataFrame]) -> FeatureMeta:
    # Build embedding vocab sizes from multiple splits combined.
    user_cat_sizes = {}
    item_cat_sizes = {}
    for col in USER_CATEGORICAL:
        max_val = max(int(pd.to_numeric(df[col], errors="coerce").fillna(0).max()) for df in dfs)
        user_cat_sizes[col] = max_val + 1
    for col in ITEM_CATEGORICAL:
        max_val = max(int(pd.to_numeric(df[col], errors="coerce").fillna(0).max()) for df in dfs)
        item_cat_sizes[col] = max_val + 1
    return FeatureMeta(user_cat_sizes=user_cat_sizes, item_cat_sizes=item_cat_sizes)


def build_dataset(
    df: pd.DataFrame,
    label_col: str,
    sample_weight: np.ndarray | None = None,
) -> TwoTowerDataset:
    # Build torch Dataset from a dataframe.
    return TwoTowerDataset(
        df,
        label_col=label_col,
        user_cat=USER_CATEGORICAL,
        user_num=USER_NUMERIC,
        item_cat=ITEM_CATEGORICAL,
        item_num=ITEM_NUMERIC,
        sample_weight=sample_weight,
    )


@torch.no_grad()
def evaluate(
    model: TwoTowerModel,
    loader: DataLoader,
    device: str,
) -> float:
    # Evaluate loss over a dataloader without gradient updates.
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    count = 0
    for user_cat, user_num, item_cat, item_num, y in loader:
        user_cat = {k: v.to(device) for k, v in user_cat.items()}
        item_cat = {k: v.to(device) for k, v in item_cat.items()}
        user_num = user_num.to(device)
        item_num = item_num.to(device)
        y = y.to(device)
        logits = model(user_cat, user_num, item_cat, item_num)
        loss = loss_fn(logits, y)
        total += float(loss.item()) * y.size(0)
        count += y.size(0)
    model.train()
    return total / max(count, 1)


def _build_item_pool(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build a unique item pool using live_id + streamer_id to deduplicate rows.
    # Returns (pool_df, key_df) where key_df has live_id/streamer_id for mapping.
    key_cols = ["live_id", "streamer_id"]
    key_df = df[key_cols].copy()
    item_cat_cols = [c for c in ITEM_CATEGORICAL if c not in key_cols]
    pool_df = (
        df[key_cols + item_cat_cols + ITEM_NUMERIC]
        .drop_duplicates(key_cols)
        .reset_index(drop=True)
    )
    return pool_df, key_df


def _make_item_key(df: pd.DataFrame) -> pd.Series:
    # Build a stable key for (live_id, streamer_id) by forcing int string format.
    key = pd.DataFrame(
        {
            "live_id": pd.to_numeric(df["live_id"], errors="coerce"),
            "streamer_id": pd.to_numeric(df["streamer_id"], errors="coerce"),
        }
    )
    key = key.fillna(-1).astype(np.int64)
    return key.astype(str).agg("_".join, axis=1)


def _encode_items(
    model: TwoTowerModel,
    pool_df: pd.DataFrame,
    device: str,
    batch_size: int,
) -> np.ndarray:
    # Encode item vectors for the entire pool in batches.
    model.eval()
    item_cat = {
        col: torch.tensor(pool_df[col].values, dtype=torch.long, device=device).reshape(-1)
        for col in ITEM_CATEGORICAL
    }
    item_num = torch.tensor(pool_df[ITEM_NUMERIC].values, dtype=torch.float32, device=device)
    if item_num.ndim == 1:
        item_num = item_num.view(-1, 1)

    item_vecs = []
    with torch.no_grad():
        for i in range(0, len(pool_df), batch_size):
            batch_cat = {k: v[i:i + batch_size] for k, v in item_cat.items()}
            batch_num = item_num[i:i + batch_size]
            if batch_cat:
                first_len = next(iter(batch_cat.values())).size(0)
                if batch_num.size(0) != first_len:
                    min_len = min(first_len, batch_num.size(0))
                    batch_cat = {k: v[:min_len] for k, v in batch_cat.items()}
                    batch_num = batch_num[:min_len]
            item_vec = model.encode_item(batch_cat, batch_num)
            item_vecs.append(item_vec.detach().cpu().numpy())
    return np.vstack(item_vecs)


def _encode_users(
    model: TwoTowerModel,
    user_df: pd.DataFrame,
    device: str,
    batch_size: int,
) -> np.ndarray:
    # Encode user vectors for the given user dataframe in batches.
    model.eval()
    user_cat = {
        col: torch.tensor(user_df[col].values, dtype=torch.long, device=device).reshape(-1)
        for col in USER_CATEGORICAL
    }
    user_num = torch.tensor(user_df[USER_NUMERIC].values, dtype=torch.float32, device=device)
    if user_num.ndim == 1:
        user_num = user_num.view(-1, 1)

    user_vecs = []
    with torch.no_grad():
        for i in range(0, len(user_df), batch_size):
            batch_cat = {k: v[i:i + batch_size] for k, v in user_cat.items()}
            batch_num = user_num[i:i + batch_size]
            if batch_cat:
                first_len = next(iter(batch_cat.values())).size(0)
                if batch_num.size(0) != first_len:
                    min_len = min(first_len, batch_num.size(0))
                    batch_cat = {k: v[:min_len] for k, v in batch_cat.items()}
                    batch_num = batch_num[:min_len]
            user_vec = model.encode_user(batch_cat, batch_num)
            user_vecs.append(user_vec.detach().cpu().numpy())
    return np.vstack(user_vecs)


def _compute_recall_at_k(
    user_ids: np.ndarray,
    topk_indices: np.ndarray,
    gt_map: Dict[int, set],
    ks: List[int],
) -> Dict[str, float]:
    # Compute Recall@K averaged over users with at least 1 ground-truth item.
    # For each user:
    #   numerator = number of relevant items retrieved in top-K
    #   denominator = total number of relevant (ground-truth) items for that user
    # Recall@K is then averaged across users with at least one ground-truth item.
    results = {f"recall@{k}": 0.0 for k in ks}
    valid_users = 0
    for i, uid in enumerate(user_ids):
        gt_items = gt_map.get(int(uid), set())
        if not gt_items:
            continue
        valid_users += 1
        preds = topk_indices[i]
        for k in ks:
            topk = preds[:k]
            num_hit = sum(1 for idx in topk if idx in gt_items)
            results[f"recall@{k}"] += num_hit / max(len(gt_items), 1)

    if valid_users == 0:
        return {k: 0.0 for k in results}
    return {k: v / valid_users for k, v in results.items()}


def _random_recall_at_k(
    user_ids: np.ndarray,
    gt_map: Dict[int, set],
    pool_size: int,
    ks: List[int],
    seed: int = 42,
) -> Dict[str, float]:
    # Random baseline: uniform random top-K from the item pool.
    rng = np.random.default_rng(seed)
    results = {f"recall@{k}": 0.0 for k in ks}
    valid_users = 0
    for uid in user_ids:
        gt_items = gt_map.get(int(uid), set())
        if not gt_items:
            continue
        valid_users += 1
        for k in ks:
            if pool_size == 0:
                continue
            topk = rng.choice(pool_size, size=min(k, pool_size), replace=False)
            num_hit = sum(1 for idx in topk if idx in gt_items)
            results[f"recall@{k}"] += num_hit / max(len(gt_items), 1)
    if valid_users == 0:
        return {k: 0.0 for k in results}
    return {k: v / valid_users for k, v in results.items()}


def evaluate_retrieval(
    model: TwoTowerModel,
    df: pd.DataFrame,
    label_col: str,
    device: str,
    batch_size: int,
    ks: List[int],
    debug: bool,
    multi_interest_k: int,
) -> Dict[str, float]:
    # Retrieval evaluation using ANN (FAISS) over all items in the split.
    # Ground truth: all positive items (label_col == 1) per user.
    pool_df, key_df = _build_item_pool(df)
    pool_df = pool_df.reset_index(drop=True)
    key_df = key_df.reset_index(drop=True)

    # map item key -> pool index
    pool_key = _make_item_key(pool_df)
    pool_index = {k: i for i, k in enumerate(pool_key)}

    pos_df = df[df[label_col] == 1].copy()
    pos_keys = _make_item_key(pos_df)
    pos_df["pool_idx"] = pos_keys.map(pool_index)

    # Build ground-truth map: user_id -> set(pool_idx)
    gt_map: Dict[int, set] = {}
    for uid, idx in zip(pos_df["user_id"].values, pos_df["pool_idx"].values):
        if pd.isna(idx):
            continue
        gt_map.setdefault(int(uid), set()).add(int(idx))

    # Unique users for evaluation
    user_df = df.drop_duplicates(["user_id"]).reset_index(drop=True)

    if debug:
        intersect = len(set(pos_keys) & set(pool_key))
        print("retrieval_debug:", {
            "rows": len(df),
            "positives": int((df[label_col] == 1).sum()),
            "unique_users": int(df["user_id"].nunique()),
            "users_with_positives": int(pos_df["user_id"].nunique()),
            "pool_size": int(pool_df.shape[0]),
            "gt_users": int(len(gt_map)),
            "pos_pool_intersect": int(intersect),
            "pos_pool_match_rate": float((~pos_df["pool_idx"].isna()).mean()) if len(pos_df) else 0.0,
        })
        rand_metrics = _random_recall_at_k(
            user_ids=user_df["user_id"].values,
            gt_map=gt_map,
            pool_size=pool_df.shape[0],
            ks=ks,
        )
        print("random_baseline:", rand_metrics)
        if intersect == 0 and len(pos_keys) > 0 and len(pool_key) > 0:
            print("debug_sample_pos_keys:", pos_keys.head(5).tolist())
            print("debug_sample_pool_keys:", pool_key.head(5).tolist())

    # Encode embeddings
    item_vecs = _encode_items(model, pool_df, device=device, batch_size=batch_size)
    user_vecs = _encode_users(model, user_df, device=device, batch_size=batch_size)

    # Build FAISS index (inner product) and search top-K
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(item_vecs.astype(np.float32))

    max_k = max(ks)
    scores, indices = index.search(user_vecs.astype(np.float32), max_k)

    # Optional multi-interest retrieval (k=2): use most recent positive item
    # as an additional interest vector per user and merge top-K results.
    if multi_interest_k == 2:
        pos_recent = pos_df.copy()
        if "imp_timestamp" in pos_recent.columns:
            pos_recent["imp_timestamp"] = pd.to_datetime(
                pos_recent["imp_timestamp"], errors="coerce"
            )
            pos_recent = pos_recent.sort_values("imp_timestamp")
        pos_recent = pos_recent.dropna(subset=["pool_idx"])
        recent_map = (
            pos_recent.groupby("user_id")["pool_idx"]
            .last()
            .to_dict()
        )
        has_second = np.array(
            [int(uid) in recent_map for uid in user_df["user_id"].values],
            dtype=bool,
        )
        second_vecs = np.zeros_like(user_vecs, dtype=np.float32)
        for i, uid in enumerate(user_df["user_id"].values):
            if int(uid) in recent_map:
                second_vecs[i] = item_vecs[int(recent_map[int(uid)])]
        scores2, indices2 = index.search(second_vecs.astype(np.float32), max_k)

        # Merge two lists by max score per item, then take top-K.
        merged_indices = []
        for i in range(len(user_df)):
            if not has_second[i]:
                merged_indices.append(indices[i])
                continue
            score_map: Dict[int, float] = {}
            for idx, sc in zip(indices[i], scores[i]):
                score_map[int(idx)] = max(score_map.get(int(idx), -1e9), float(sc))
            for idx, sc in zip(indices2[i], scores2[i]):
                score_map[int(idx)] = max(score_map.get(int(idx), -1e9), float(sc))
            top_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:max_k]
            merged_indices.append(np.array([i for i, _ in top_items], dtype=np.int64))
        indices = np.vstack(merged_indices)

    metrics = _compute_recall_at_k(
        user_ids=user_df["user_id"].values,
        topk_indices=indices,
        gt_map=gt_map,
        ks=ks,
    )

    if debug and len(gt_map) > 0:
        # Print a single-user sanity check: gt items vs top-K retrieval
        first_uid = next(iter(gt_map.keys()))
        user_idx = int(np.where(user_df["user_id"].values == first_uid)[0][0])
        gt_items = gt_map[first_uid]
        topk = indices[user_idx][: max(ks)]
        hit_count = sum(1 for idx in topk if idx in gt_items)
        # Map pool indices back to (live_id, streamer_id) for readability.
        pool_pairs = list(
            zip(
                pool_df["live_id"].astype(str).tolist(),
                pool_df["streamer_id"].astype(str).tolist(),
            )
        )
        topk_pairs = [pool_pairs[i] for i in topk if i < len(pool_pairs)]
        gt_pairs = [pool_pairs[i] for i in gt_items if i < len(pool_pairs)]
        print("single_user_debug:", {
            "user_id": int(first_uid),
            "gt_item_count": int(len(gt_items)),
            "topk": topk.tolist(),
            "topk_pairs": topk_pairs,
            "gt_pairs": gt_pairs,
            "hits_in_topk": int(hit_count),
            "recall_at_max_k": float(hit_count / max(len(gt_items), 1)),
        })
    return metrics


def compute_loss(
    model: TwoTowerModel,
    user_cat: Dict[str, torch.Tensor],
    user_num: torch.Tensor,
    item_cat: Dict[str, torch.Tensor],
    item_num: torch.Tensor,
    y: torch.Tensor,
    user_ids: torch.Tensor,
    sample_weight: torch.Tensor | None,
    loss_name: str,
    temperature: float,
    triplet_margin: float,
    debug_sim: bool,
) -> torch.Tensor:
    # Flexible loss function: bce, contrastive (in-batch InfoNCE), or triplet.
    logits = model(user_cat, user_num, item_cat, item_num)
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()(logits, y)

    user_vec, item_vec = model.encode(user_cat, user_num, item_cat, item_num)
    y_pos = y > 0.5

    if loss_name == "contrastive":
        if y_pos.sum() == 0:
            return nn.BCEWithLogitsLoss()(logits, y)
        user_norm = nn.functional.normalize(user_vec, dim=1)
        item_norm = nn.functional.normalize(item_vec, dim=1)
        # Multi-positive InfoNCE: for each user i, all items in the batch that
        # belong to the same user are treated as positives (not negatives).
        # L_i = -log( sum_{p in P_i} exp(sim(i,p)/tau) / sum_j exp(sim(i,j)/tau) )
        sim = user_norm @ item_norm.t() / temperature
        if debug_sim:
            with torch.no_grad():
                print("contrastive sim matrix (batch x batch):")
                print(sim.detach().cpu())
        # Positives mask: same user_id -> positive set P_i.
        pos_mask = user_ids.view(-1, 1) == user_ids.view(1, -1)
        neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)
        pos_sim = torch.where(pos_mask, sim, neg_inf)
        numerator = torch.logsumexp(pos_sim, dim=1)
        denominator = torch.logsumexp(sim, dim=1)
        loss_vec = denominator - numerator
        if sample_weight is not None:
            # Freshness weighting: newer interactions can contribute more to the loss.
            loss_vec = loss_vec * sample_weight
        return loss_vec.mean()

    if loss_name == "triplet":
        if y_pos.sum() == 0:
            return nn.BCEWithLogitsLoss()(logits, y)
        user_norm = nn.functional.normalize(user_vec, dim=1)
        item_norm = nn.functional.normalize(item_vec, dim=1)

        neg_mask = ~y_pos
        if neg_mask.sum() == 0:
            return nn.BCEWithLogitsLoss()(logits, y)

        # cosine distance = 1 - cosine similarity
        sim = user_norm @ item_norm.t()
        pos_idx = torch.where(y_pos)[0]
        neg_idx = torch.where(neg_mask)[0]

        anchor = user_norm[pos_idx]
        positive = item_norm[pos_idx]
        neg_sim = sim[pos_idx][:, neg_idx]
        hard_neg_idx = neg_sim.argmax(dim=1)
        negative = item_norm[neg_idx[hard_neg_idx]]

        pos_dist = 1.0 - (anchor * positive).sum(dim=1)
        neg_dist = 1.0 - (anchor * negative).sum(dim=1)
        loss = nn.functional.relu(triplet_margin + pos_dist - neg_dist).mean()
        return loss

    raise ValueError(f"Unknown loss: {loss_name}")


def train(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    label_col: str,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    loss_name: str,
    temperature: float,
    triplet_margin: float,
    emb_dim: int,
    tower_hidden: List[int],
    dropout: float,
    normalize_emb: bool,
    num_workers: int,
    plot_path: Path,
    debug_sim: bool,
    eval_ks: List[int],
    debug_retrieval: bool,
    max_pos_per_user: int,
    freshness_weighting: bool,
    freshness_half_life_hours: float,
    early_stopping: bool,
    early_patience: int,
    early_k: int,
    checkpoint_path: Path,
    multi_interest_k: int,
) -> None:
    # Train on train split, track val loss per epoch, report test loss at the end.
    train_raw = pd.read_csv(train_path)
    val_raw = pd.read_csv(val_path)
    test_raw = pd.read_csv(test_path)

    meta = build_meta_from_dfs([train_raw, val_raw, test_raw])

    train_df, _ = load_data(train_path, label_col, meta=meta)
    val_df, _ = load_data(val_path, label_col, meta=meta)
    test_df, _ = load_data(test_path, label_col, meta=meta)

    # Summary stats: average positives/negatives per user in training data.
    if label_col in train_df.columns:
        user_counts = train_df.groupby("user_id")[label_col].agg(
            positives=lambda s: (s == 1).sum(),
            negatives=lambda s: (s == 0).sum(),
        )
        avg_pos = user_counts["positives"].mean()
        avg_neg = user_counts["negatives"].mean()
        print(
            "train_user_stats:",
            {
                "users": int(user_counts.shape[0]),
                "avg_positives_per_user": float(avg_pos),
                "avg_negatives_per_user": float(avg_neg),
            },
        )

    # Summary stats: validation and test (positives and total interactions per user).
    for name, df in [("val", val_df), ("test", test_df)]:
        if label_col in df.columns:
            user_stats = df.groupby("user_id")[label_col].agg(
                positives=lambda s: (s == 1).sum(),
                total="count",
            )
            print(
                f"{name}_user_stats:",
                {
                    "users": int(user_stats.shape[0]),
                    "avg_positives_per_user": float(user_stats["positives"].mean()),
                    "std_positives_per_user": float(user_stats["positives"].std(ddof=0)),
                    "avg_total_per_user": float(user_stats["total"].mean()),
                    "std_total_per_user": float(user_stats["total"].std(ddof=0)),
                },
            )

    # For contrastive retrieval training, keep only positive pairs.
    if loss_name == "contrastive":
        train_df = train_df[train_df[label_col] == 1].reset_index(drop=True)

    # Drop weak positives: watch time < 30s.
    if "watch_live_time" in train_df.columns:
        before = len(train_df)
        train_df = train_df[
            (train_df["watch_live_time"].astype(float) >= 30) | (train_df[label_col] == 0)
        ].reset_index(drop=True)
        dropped = before - len(train_df)
        if dropped > 0:
            print(f"note: dropped {dropped} weak positives (watch_live_time < 30s)")

    # Optional freshness weighting: newer interactions get higher weight.
    sample_weight = None
    if freshness_weighting and "imp_timestamp" in train_df.columns:
        ts = pd.to_datetime(train_df["imp_timestamp"], errors="coerce")
        max_ts = ts.max()
        age_hours = (max_ts - ts).dt.total_seconds() / 3600.0
        # Exponential decay: weight halves every `freshness_half_life_hours`.
        weight = np.exp(-np.log(2) * age_hours / max(freshness_half_life_hours, 1e-6))
        sample_weight = weight.fillna(0).to_numpy(dtype=np.float32)

    train_ds = build_dataset(train_df, label_col, sample_weight=sample_weight)
    if loss_name == "contrastive":
        # Group by user and cap positives per user in a batch to reduce dominance
        # of heavy users and improve batch diversity.
        sampler = UserBatchSampler(
            user_ids=train_df["user_id"].to_numpy(),
            batch_size=batch_size,
            max_pos_per_user=max_pos_per_user,
        )
        dropped = sampler.dropped_count()
        if dropped > 0:
            print(
                f"note: dropping {dropped} positives due to max_pos_per_user={max_pos_per_user} "
                f"({dropped / max(len(train_df), 1):.2%} of training positives)"
            )
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if loss_name == "contrastive":
        normalize_emb = True

    model = TwoTowerModel(
        user_cat_sizes=meta.user_cat_sizes,
        item_cat_sizes=meta.item_cat_sizes,
        user_num_dim=len(USER_NUMERIC),
        item_num_dim=len(ITEM_NUMERIC),
        emb_dim=emb_dim,
        tower_hidden=tower_hidden,
        dropout=dropout,
        normalize_emb=normalize_emb,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    train_losses = []
    best_metric = -float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, epochs + 1):
        total = 0.0
        for user_cat, user_num, item_cat, item_num, y, sample_weight in tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            total=len(train_loader),
        ):
            user_cat = {k: v.to(device) for k, v in user_cat.items()}
            item_cat = {k: v.to(device) for k, v in item_cat.items()}
            user_num = user_num.to(device)
            item_num = item_num.to(device)
            y = y.to(device)

            loss = compute_loss(
                model,
                user_cat,
                user_num,
                item_cat,
                item_num,
                y,
                user_ids=user_cat["user_id"],
                sample_weight=sample_weight.to(device) if sample_weight is not None else None,
                loss_name=loss_name,
                temperature=temperature,
                triplet_margin=triplet_margin,
                debug_sim=debug_sim,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * y.size(0)

        avg_loss = total / len(train_ds)
        val_metrics = evaluate_retrieval(
            model,
            val_df,
            label_col=label_col,
            device=device,
            batch_size=batch_size,
            ks=eval_ks,
            debug=debug_retrieval,
            multi_interest_k=multi_interest_k,
        )
        train_losses.append(avg_loss)
        print(f"epoch={epoch} train_loss={avg_loss:.6f} val_metrics={val_metrics}")

        if early_stopping:
            key = f"recall@{early_k}"
            current = float(val_metrics.get(key, 0.0))
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                no_improve = 0
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "config": {
                                "emb_dim": emb_dim,
                                "tower_hidden": tower_hidden,
                                "dropout": dropout,
                                "normalize_emb": normalize_emb,
                            },
                            "best_metric": best_metric,
                            "metric_key": key,
                        },
                        checkpoint_path,
                    )
                    print(f"saved best checkpoint -> {checkpoint_path}")
            else:
                no_improve += 1
                if no_improve >= early_patience:
                    print(
                        f"early_stop: no improvement in {early_patience} epochs "
                        f"(best {key}={best_metric:.6f} at epoch {best_epoch})"
                    )
                    break

    test_metrics = evaluate_retrieval(
        model,
        test_df,
        label_col=label_col,
        device=device,
        batch_size=batch_size,
        ks=eval_ks,
        debug=debug_retrieval,
        multi_interest_k=multi_interest_k,
    )
    print(f"test_metrics={test_metrics}")

    if plot_path is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training Progress")
        plt.legend()
        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        print(f"saved plot -> {plot_path}")


def parse_args() -> argparse.Namespace:
    # Parse CLI arguments for training.
    # Tunable hyperparameters:
    # - batch_size, epochs, lr, device, num_workers
    # - loss (bce/contrastive/triplet), temperature (contrastive), triplet_margin (triplet)
    # - emb_dim, tower_hidden, dropout, normalize_emb
    # - max_pos_per_user, freshness_weighting, freshness_half_life_hours
    # - label_col, train_path, val_path, test_path
    parser = argparse.ArgumentParser(description="Two-tower training on draft_sample.csv")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "TTNN_train.csv",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "TTNN_val.csv",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "TTNN_test.csv",
    )
    parser.add_argument("--label-col", type=str, default="is_click")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--loss",
        type=str,
        default="bce",
        choices=["bce", "contrastive", "triplet"],
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--emb-dim", type=int, default=512)
    parser.add_argument(
        "--tower-hidden",
        type=str,
        default="1024,512",
        help="Comma-separated hidden sizes for each tower, e.g. 256,128",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalize-emb", action="store_true")
    parser.add_argument(
        "--debug-sim",
        action="store_true",
        help="Print contrastive similarity matrix for each batch",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs" / "train_progress.png",
    )
    parser.add_argument(
        "--eval-ks",
        type=str,
        default="10,50,100",
        help="Comma-separated K values for retrieval metrics (e.g. 10,50,100)",
    )
    parser.add_argument(
        "--max-pos-per-user",
        type=int,
        default=30,
        help="Max positives per user per batch (contrastive only)",
    )
    parser.add_argument(
        "--freshness-weighting",
        action="store_true",
        help="Enable freshness weighting on contrastive loss",
    )
    parser.add_argument(
        "--freshness-half-life-hours",
        type=float,
        default=24.0,
        help="Half-life in hours for freshness weighting",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping based on recall@K on validation set",
    )
    parser.add_argument(
        "--early-patience",
        type=int,
        default=3,
        help="Stop after N epochs without improvement",
    )
    parser.add_argument(
        "--early-k",
        type=int,
        default=100,
        help="Use recall@K for early stopping",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "best_tower.pt",
        help="Path to save the best model checkpoint",
    )
    parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Print retrieval debug stats (positives/users/pool size)",
    )
    parser.add_argument(
        "--multi-interest-k",
        type=int,
        default=2,
        choices=[1, 2],
        help="Use 2 interest vectors per user at retrieval time (k=2)",
    )
    return parser.parse_args()


def main() -> None:
    # Entry point.
    args = parse_args()
    tower_hidden = [int(x) for x in args.tower_hidden.split(",") if x.strip()]
    if args.loss == "contrastive" and not args.normalize_emb:
        print("note: forcing normalize_emb=True for contrastive loss")
        args.normalize_emb = True
    print(
        "config:",
        {
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "test_path": str(args.test_path),
            "label_col": args.label_col,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "lr": args.lr,
            "device": args.device,
            "loss": args.loss,
            "temperature": args.temperature,
            "triplet_margin": args.triplet_margin,
            "emb_dim": args.emb_dim,
            "tower_hidden": tower_hidden,
            "dropout": args.dropout,
            "normalize_emb": args.normalize_emb,
            "eval_ks": args.eval_ks,
            "debug_retrieval": args.debug_retrieval,
            "max_pos_per_user": args.max_pos_per_user,
            "freshness_weighting": args.freshness_weighting,
            "freshness_half_life_hours": args.freshness_half_life_hours,
            "early_stopping": args.early_stopping,
            "early_patience": args.early_patience,
            "early_k": args.early_k,
            "checkpoint_path": str(args.checkpoint_path),
            "multi_interest_k": args.multi_interest_k,
        },
    )
    eval_ks = [int(x) for x in args.eval_ks.split(",") if x.strip()]
    train(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        label_col=args.label_col,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        loss_name=args.loss,
        temperature=args.temperature,
        triplet_margin=args.triplet_margin,
        emb_dim=args.emb_dim,
        tower_hidden=tower_hidden,
        dropout=args.dropout,
        normalize_emb=args.normalize_emb,
        num_workers=args.num_workers,
        plot_path=args.plot_path,
        debug_sim=args.debug_sim,
        eval_ks=eval_ks,
        debug_retrieval=args.debug_retrieval,
        max_pos_per_user=args.max_pos_per_user,
        freshness_weighting=args.freshness_weighting,
        freshness_half_life_hours=args.freshness_half_life_hours,
        early_stopping=args.early_stopping,
        early_patience=args.early_patience,
        early_k=args.early_k,
        checkpoint_path=args.checkpoint_path,
        multi_interest_k=args.multi_interest_k,
    )


if __name__ == "__main__":
    main()
