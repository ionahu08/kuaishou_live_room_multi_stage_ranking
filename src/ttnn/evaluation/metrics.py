from typing import Dict, List, Set

import numpy as np


def compute_recall_at_k(
    user_ids: np.ndarray,
    topk_indices: np.ndarray,
    gt_map: Dict[int, Set[int]],
    ks: List[int],
) -> Dict[str, float]:
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


def random_recall_at_k(
    user_ids: np.ndarray,
    gt_map: Dict[int, Set[int]],
    pool_size: int,
    ks: List[int],
    seed: int = 42,
) -> Dict[str, float]:
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
