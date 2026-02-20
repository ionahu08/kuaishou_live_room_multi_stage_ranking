from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from ..config import ITEM_CATEGORICAL, ITEM_NUMERIC, USER_CATEGORICAL, USER_NUMERIC


class TwoTowerDataset(Dataset):
    """
    Torch dataset for two-tower training samples.

    Each row in `df` is converted to tensors for:
    - user categorical features (`Dict[str, LongTensor]`)
    - user numeric features (`FloatTensor`)
    - item categorical features (`Dict[str, LongTensor]`)
    - item numeric features (`FloatTensor`)
    - label (`FloatTensor`)
    - optional per-sample weight (`FloatTensor`, defaults to 1.0)

    `__getitem__` returns one training tuple:
    `(user_cat, user_num, item_cat, item_num, label, sample_weight)`.
    """
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
    """
    Batch sampler that limits per-user sample count in each epoch.

    Purpose:
    - prevent heavy users from dominating contrastive training
    - improve user diversity within batches

    Behavior:
    - groups row indices by `user_id`
    - shuffles user order each epoch
    - for each user, keeps at most `max_pos_per_user` indices
      (randomly sampled without replacement if more)
    - emits flat index batches of size `batch_size`

    Notes:
    - `__len__` is based on capped per-user counts, so it is an estimate of
      number of yielded batches.
    - `dropped_count()` reports how many samples are excluded by the cap.
    """
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
        total = sum(min(len(v), self.max_pos_per_user) for v in self.user_to_indices.values())
        return max(1, (total + self.batch_size - 1) // self.batch_size)

    def dropped_count(self) -> int:
        return sum(max(0, len(v) - self.max_pos_per_user) for v in self.user_to_indices.values())


def build_dataset(
    df: pd.DataFrame,
    label_col: str,
    sample_weight: np.ndarray | None = None,
) -> TwoTowerDataset:
    """
    Create a `TwoTowerDataset` from a preprocessed dataframe using the
    project-wide feature schema (user/item categorical + numeric columns).

    Optionally attaches per-row `sample_weight` used by weighted losses.
    """
    return TwoTowerDataset(
        df,
        label_col=label_col,
        user_cat=USER_CATEGORICAL,
        user_num=USER_NUMERIC,
        item_cat=ITEM_CATEGORICAL,
        item_num=ITEM_NUMERIC,
        sample_weight=sample_weight,
    )

