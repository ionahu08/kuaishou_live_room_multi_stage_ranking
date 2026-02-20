from typing import Dict

import torch
from torch import nn

from ..models import TwoTowerModel


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
    bce_pos_weight: float | None = None,
) -> torch.Tensor:
    logits = model(user_cat, user_num, item_cat, item_num)
    if loss_name == "bce":
        pos_weight = None
        if bce_pos_weight is not None:
            pos_weight = torch.tensor(float(bce_pos_weight), device=logits.device, dtype=logits.dtype)
        loss_vec = nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none",
            pos_weight=pos_weight,
        )
        if sample_weight is not None:
            loss_vec = loss_vec * sample_weight
        return loss_vec.mean()

    user_vec, item_vec = model.encode(user_cat, user_num, item_cat, item_num)
    y_pos = y > 0.5

    if loss_name == "contrastive":
        if y_pos.sum() == 0:
            return nn.BCEWithLogitsLoss()(logits, y)
        user_norm = nn.functional.normalize(user_vec, dim=1)
        item_norm = nn.functional.normalize(item_vec, dim=1)
        sim = user_norm @ item_norm.t() / temperature
        if debug_sim:
            with torch.no_grad():
                print("contrastive sim matrix (batch x batch):")
                print(sim.detach().cpu())
        pos_mask = user_ids.view(-1, 1) == user_ids.view(1, -1)
        neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)
        pos_sim = torch.where(pos_mask, sim, neg_inf)
        numerator = torch.logsumexp(pos_sim, dim=1)
        denominator = torch.logsumexp(sim, dim=1)
        loss_vec = denominator - numerator
        if sample_weight is not None:
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
        return nn.functional.relu(triplet_margin + pos_dist - neg_dist).mean()

    raise ValueError(f"Unknown loss: {loss_name}")
