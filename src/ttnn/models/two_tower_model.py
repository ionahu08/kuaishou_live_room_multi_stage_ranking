from typing import Dict, List, Tuple

import torch
from torch import nn


class MLP(nn.Module):
    """
    Generic feed-forward projection network used as a tower backbone.

    Architecture:
    - input dimension: `in_dim`
    - repeated hidden blocks: Linear -> ReLU -> Dropout for each size in `hidden_dims`
    - output projection: Linear(last_hidden, `out_dim`)

    In this project, both user and item towers use this MLP to map concatenated
    sparse+dense features into a shared embedding space.
    """
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
    def __init__(
        self,
        user_cat_sizes: Dict[str, int],
        item_cat_sizes: Dict[str, int],
        user_num_dim: int,
        item_num_dim: int,
        emb_dim: int = 32,
        tower_hidden: List[int] | None = None,
        debug_shapes: bool = False,
        dropout: float = 0.1,
        normalize_emb: bool = False,
    ) -> None:
        super().__init__()
        if tower_hidden is None:
            tower_hidden = [1024, 512]

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
        user_vec, item_vec = self.encode(user_cat, user_num, item_cat, item_num)
        return (user_vec * item_vec).sum(dim=1)

    def encode(
        self,
        user_cat: Dict[str, torch.Tensor],
        user_num: torch.Tensor,
        item_cat: Dict[str, torch.Tensor],
        item_num: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_vec = self.encode_user(user_cat, user_num)
        item_vec = self.encode_item(item_cat, item_num)
        return user_vec, item_vec

    def encode_user(
        self,
        user_cat: Dict[str, torch.Tensor],
        user_num: torch.Tensor,
    ) -> torch.Tensor:
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

