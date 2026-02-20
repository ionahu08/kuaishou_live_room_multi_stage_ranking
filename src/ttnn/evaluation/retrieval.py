from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models import TwoTowerModel
from .ann import (
    ann_topk,
    build_item_pool,
    encode_items,
    encode_users,
    make_item_key,
    merge_multi_interest_topk,
)
from .metrics import compute_recall_at_k, random_recall_at_k


@torch.no_grad()
def evaluate(
    model: TwoTowerModel,
    loader: DataLoader,
    device: str,
) -> float:
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
    pool_df, user_df, _, indices = _retrieve_core(
        model=model,
        df=df,
        device=device,
        batch_size=batch_size,
        topk=max(ks),
        multi_interest_k=multi_interest_k,
        multi_interest_source_df=df[df[label_col] == 1].copy(),
    )

    pos_df = df[df[label_col] == 1].copy()
    pool_key = make_item_key(pool_df)
    pool_index = {k: i for i, k in enumerate(pool_key)}
    pos_keys = make_item_key(pos_df)
    pos_df["pool_idx"] = pos_keys.map(pool_index)

    gt_map: Dict[int, Set[int]] = {}
    for uid, idx in zip(pos_df["user_id"].values, pos_df["pool_idx"].values):
        if pd.isna(idx):
            continue
        gt_map.setdefault(int(uid), set()).add(int(idx))

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
        rand_metrics = random_recall_at_k(
            user_ids=user_df["user_id"].values,
            gt_map=gt_map,
            pool_size=pool_df.shape[0],
            ks=ks,
        )
        print("random_baseline:", rand_metrics)
        if intersect == 0 and len(pos_keys) > 0 and len(pool_key) > 0:
            print("debug_sample_pos_keys:", pos_keys.head(5).tolist())
            print("debug_sample_pool_keys:", pool_key.head(5).tolist())

    max_k = max(ks)

    metrics = compute_recall_at_k(
        user_ids=user_df["user_id"].values,
        topk_indices=indices,
        gt_map=gt_map,
        ks=ks,
    )

    if debug and len(gt_map) > 0:
        first_uid = next(iter(gt_map.keys()))
        user_idx = int(np.where(user_df["user_id"].values == first_uid)[0][0])
        gt_items = gt_map[first_uid]
        topk = indices[user_idx][:max_k]
        hit_count = sum(1 for idx in topk if idx in gt_items)
        pool_pairs = list(
            zip(
                pool_df["live_id"].astype(str).tolist(),
                pool_df["streamer_id"].astype(str).tolist(),
            )
        )
        topk_pairs = [pool_pairs[i] for i in topk if 0 <= i < len(pool_pairs)]
        gt_pairs = [pool_pairs[i] for i in gt_items if 0 <= i < len(pool_pairs)]
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


def _retrieve_core(
    model: TwoTowerModel,
    df: pd.DataFrame,
    device: str,
    batch_size: int,
    topk: int,
    multi_interest_k: int,
    multi_interest_source_df: pd.DataFrame | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    pool_df, _ = build_item_pool(df)
    pool_df = pool_df.reset_index(drop=True)
    user_df = df.drop_duplicates(["user_id"]).reset_index(drop=True)

    item_vecs = encode_items(model, pool_df, device=device, batch_size=batch_size)
    user_vecs = encode_users(model, user_df, device=device, batch_size=batch_size)

    scores, indices = ann_topk(item_vecs=item_vecs, user_vecs=user_vecs, ks=[topk])

    if multi_interest_k == 2:
        source_df = multi_interest_source_df if multi_interest_source_df is not None else df.copy()
        source_df = source_df.copy()
        source_keys = make_item_key(source_df)
        pool_keys = make_item_key(pool_df)
        pool_index = {k: i for i, k in enumerate(pool_keys)}
        source_df["pool_idx"] = source_keys.map(pool_index)

        if "imp_timestamp" in source_df.columns:
            source_df["imp_timestamp"] = pd.to_datetime(source_df["imp_timestamp"], errors="coerce")
            source_df = source_df.sort_values("imp_timestamp")
        source_df = source_df.dropna(subset=["pool_idx"])
        recent_map = source_df.groupby("user_id")["pool_idx"].last().to_dict()

        has_second = np.array([int(uid) in recent_map for uid in user_df["user_id"].values], dtype=bool)
        second_vecs = np.zeros_like(user_vecs, dtype=np.float32)
        for i, uid in enumerate(user_df["user_id"].values):
            if int(uid) in recent_map:
                second_vecs[i] = item_vecs[int(recent_map[int(uid)])]

        scores2, indices2 = ann_topk(item_vecs=item_vecs, user_vecs=second_vecs, ks=[topk])
        scores, indices = merge_multi_interest_topk(
            scores=scores,
            indices=indices,
            scores2=scores2,
            indices2=indices2,
            has_second=has_second,
            max_k=topk,
        )

    return pool_df, user_df, scores, indices


def retrieve_topk_items(
    model: TwoTowerModel,
    df: pd.DataFrame,
    device: str,
    batch_size: int,
    topk: int,
    multi_interest_k: int,
) -> pd.DataFrame:
    pool_df, user_df, scores, indices = _retrieve_core(
        model=model,
        df=df,
        device=device,
        batch_size=batch_size,
        topk=topk,
        multi_interest_k=multi_interest_k,
        multi_interest_source_df=df,
    )

    rows = []
    topk = min(topk, indices.shape[1])
    for i, uid in enumerate(user_df["user_id"].values):
        for rank in range(topk):
            pool_idx = int(indices[i, rank])
            if pool_idx < 0 or pool_idx >= len(pool_df):
                continue
            rows.append(
                {
                    "user_id": int(uid),
                    "rank": rank + 1,
                    "pool_idx": pool_idx,
                    "live_id": pool_df.iloc[pool_idx]["live_id"],
                    "streamer_id": pool_df.iloc[pool_idx]["streamer_id"],
                    "score": float(scores[i, rank]) if scores.size else 0.0,
                }
            )
    return pd.DataFrame(rows)
