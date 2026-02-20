from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch

from ..config import ITEM_CATEGORICAL, ITEM_NUMERIC, USER_CATEGORICAL, USER_NUMERIC
from ..models import TwoTowerModel


def build_item_pool(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["live_id", "streamer_id"]
    key_df = df[key_cols].copy()
    item_cat_cols = [c for c in ITEM_CATEGORICAL if c not in key_cols]
    pool_df = (
        df[key_cols + item_cat_cols + ITEM_NUMERIC]
        .drop_duplicates(key_cols)
        .reset_index(drop=True)
    )
    return pool_df, key_df


def make_item_key(df: pd.DataFrame) -> pd.Series:
    key = pd.DataFrame(
        {
            "live_id": pd.to_numeric(df["live_id"], errors="coerce"),
            "streamer_id": pd.to_numeric(df["streamer_id"], errors="coerce"),
        }
    )
    key = key.fillna(-1).astype(np.int64)
    return key.astype(str).agg("_".join, axis=1)


def encode_items(
    model: TwoTowerModel,
    pool_df: pd.DataFrame,
    device: str,
    batch_size: int,
) -> np.ndarray:
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


def encode_users(
    model: TwoTowerModel,
    user_df: pd.DataFrame,
    device: str,
    batch_size: int,
) -> np.ndarray:
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


def ann_topk(
    item_vecs: np.ndarray,
    user_vecs: np.ndarray,
    ks: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(item_vecs.astype(np.float32))
    max_k = max(ks)
    return index.search(user_vecs.astype(np.float32), max_k)


def merge_multi_interest_topk(
    scores: np.ndarray,
    indices: np.ndarray,
    scores2: np.ndarray,
    indices2: np.ndarray,
    has_second: np.ndarray,
    max_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    merged_indices = []
    merged_scores = []
    for i in range(len(indices)):
        if not has_second[i]:
            merged_indices.append(indices[i][:max_k])
            merged_scores.append(scores[i][:max_k])
            continue
        score_map: Dict[int, float] = {}
        for idx, sc in zip(indices[i], scores[i]):
            idx = int(idx)
            if idx < 0:
                continue
            score_map[idx] = max(score_map.get(idx, -1e9), float(sc))
        for idx, sc in zip(indices2[i], scores2[i]):
            idx = int(idx)
            if idx < 0:
                continue
            score_map[idx] = max(score_map.get(idx, -1e9), float(sc))
        top_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:max_k]
        idx_arr = np.array([item_idx for item_idx, _ in top_items], dtype=np.int64)
        score_arr = np.array([score for _, score in top_items], dtype=np.float32)
        if len(top_items) < max_k:
            pad_n = max_k - len(top_items)
            idx_arr = np.pad(idx_arr, (0, pad_n), mode="constant", constant_values=-1)
            score_arr = np.pad(score_arr, (0, pad_n), mode="constant", constant_values=-1e9)
        merged_indices.append(idx_arr)
        merged_scores.append(score_arr)
    return np.vstack(merged_scores), np.vstack(merged_indices)
