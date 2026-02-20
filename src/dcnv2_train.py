from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


TASKS_BIN = ["is_click", "watch_greater_30s", "is_like"]
TASK_REG = "watch_live_time_log"
ALL_TASKS = TASKS_BIN + [TASK_REG]
FORCE_NUMERIC_BINARY = {"user_gender_le", "streamer_gender_le"}
COMPACT_METRIC_KEYS = [
    "loss",
    "auc_is_click",
    "auc_watch_greater_30s",
    "auc_is_like",
    "auc_mean",
    "precision@10_is_click",
    "precision@10_is_like",
    f"rmse_{TASK_REG}",
]
DEFAULT_DROP_COLUMNS = {
    "amount_gift_live_15min",
    "amount_gift_live_1hr",
    "amount_gift_live_3hr",
    "comment_trend_room",
    "gift_trend_room",
    "like_trend_room",
    "num_comment_live_15min",
    "num_comment_live_1hr",
    "num_comment_live_3hr",
    "num_gift_live_15min",
    "num_gift_live_1hr",
    "num_gift_live_3hr",
    "num_like_live_15min",
}


@dataclass
class FeaturePack:
    cat_cols: List[str]
    num_cols: List[str]
    cat_maps: Dict[str, Dict[str, int]]
    cat_cardinalities: List[int]


class RankingDataset(Dataset):
    def __init__(self, cat_x: np.ndarray, num_x: np.ndarray, ys: Dict[str, np.ndarray], user_ids: np.ndarray) -> None:
        self.cat_x = cat_x.astype(np.int64)
        self.num_x = num_x.astype(np.float32)
        self.ys = {k: v.astype(np.float32) for k, v in ys.items()}
        self.user_ids = user_ids.astype(np.int64)

    def __len__(self) -> int:
        return self.cat_x.shape[0]

    def __getitem__(self, idx: int):
        labels = {k: self.ys[k][idx] for k in ALL_TASKS}
        return self.cat_x[idx], self.num_x[idx], labels, self.user_ids[idx]


class CrossNetV2(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, low_rank: int = 32) -> None:
        super().__init__()
        self.ws = nn.ModuleList([nn.Linear(input_dim, low_rank, bias=False) for _ in range(num_layers)])
        self.vs = nn.ModuleList([nn.Linear(low_rank, input_dim, bias=False) for _ in range(num_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, v, b in zip(self.ws, self.vs, self.bs):
            x = x0 * (v(w(x)) + b) + x
        return x


class MultiTaskDCNv2(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        num_features: int,
        num_cross_layers: int = 3,
        deep_layers: Sequence[int] = (512, 256),
        dropout: float = 0.1,
        cross_low_rank: int = 32,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.emb_dims: List[int] = []
        for card in cat_cardinalities:
            emb_dim = min(64, max(4, (card + 1) // 2))
            self.emb_dims.append(emb_dim)
            self.embeddings.append(nn.Embedding(card, emb_dim))

        input_dim = int(sum(self.emb_dims) + num_features)
        self.cross = CrossNetV2(input_dim=input_dim, num_layers=num_cross_layers, low_rank=cross_low_rank)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in deep_layers:
            layers.extend([nn.Linear(prev_dim, int(h)), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = int(h)
        self.deep = nn.Sequential(*layers)
        deep_out_dim = int(deep_layers[-1]) if len(deep_layers) > 0 else 0

        shared_dim = input_dim + deep_out_dim
        self.heads = nn.ModuleDict({task: nn.Linear(shared_dim, 1) for task in ALL_TASKS})

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb_out = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(emb_out, dim=1) if len(emb_out) > 0 else torch.empty((num_x.size(0), 0), device=num_x.device)
        x0 = torch.cat([x_cat, num_x], dim=1)
        cross_out = self.cross(x0)
        deep_out = self.deep(x0) if len(self.deep) > 0 else torch.empty((x0.size(0), 0), device=num_x.device)
        shared = torch.cat([cross_out, deep_out], dim=1)
        return {task: self.heads[task](shared).squeeze(1) for task in ALL_TASKS}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train multitask DCNv2 for CTR/watch>30s/like.")
    parser.add_argument("--train-path", type=Path, default=root / "data" / "dcnv2_train.csv")
    parser.add_argument("--val-path", type=Path, default=root / "data" / "dcnv2_val.csv")
    parser.add_argument("--test-path", type=Path, default=root / "data" / "dcnv2_test.csv")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-cross-layers", type=int, default=3)
    parser.add_argument("--cross-low-rank", type=int, default=32)
    parser.add_argument("--deep-layers", type=str, default="1024,512,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable-id-features", action="store_true", help="Exclude user_id/live_id/streamer_id from categorical features for leakage ablation.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Global grad norm clip. Set <=0 to disable.")
    parser.add_argument("--w-ctr", type=float, default=1.0)
    parser.add_argument("--w-watch30", type=float, default=1.0)
    parser.add_argument("--w-like", type=float, default=1.0)
    parser.add_argument("--w-watch-time", type=float, default=1.0, help="Loss weight for watch_live_time_log MSE task.")
    parser.add_argument(
        "--reg-loss",
        type=str,
        default="huber",
        choices=["mse", "huber"],
        help="Regression loss for watch_live_time_log task.",
    )
    parser.add_argument(
        "--watch-log-train-clip",
        type=float,
        default=8.0,
        help="Clip absolute regression logits to this value for loss computation (<=0 disables).",
    )
    parser.add_argument("--watch-log-standardize", action=argparse.BooleanOptionalAction, default=True, help="Standardize watch_live_time_log target using train split stats.")
    parser.add_argument("--watch-log-export-min", type=float, default=-5.0, help="Min clamp before expm1 when exporting watch time.")
    parser.add_argument("--watch-log-export-max", type=float, default=12.0, help="Max clamp before expm1 when exporting watch time.")
    parser.add_argument("--checkpoint-path", type=Path, default=root / "models" / "dcnv2_multitask_best.pt")
    parser.add_argument("--save-every-epochs", type=int, default=0)
    parser.add_argument("--periodic-checkpoint-dir", type=Path, default=root / "models")
    parser.add_argument("--pred-output-path", type=Path, default=root / "outputs" / "dcnv2_test_predictions.csv")
    parser.add_argument(
        "--precision-ks",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values for per-user precision@K on binary tasks.",
    )
    return parser.parse_args()


def _auc_fallback(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(np.float64)
    y_score = y_score.astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    ranks = y_score.argsort().argsort().astype(np.float64) + 1
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum())
    return float(auc)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return _auc_fallback(y_true, y_score)


def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_click"] = pd.to_numeric(out.get("is_click"), errors="coerce").fillna(0).astype(np.float32)
    out["is_like"] = pd.to_numeric(out.get("is_like"), errors="coerce").fillna(0).astype(np.float32)
    if "watch_greater_30s" not in out.columns:
        watch = pd.to_numeric(out.get("watch_live_time"), errors="coerce").fillna(0)
        out["watch_greater_30s"] = (watch >= 30).astype(np.float32)
    else:
        out["watch_greater_30s"] = pd.to_numeric(out["watch_greater_30s"], errors="coerce").fillna(0).astype(np.float32)
    watch_time = pd.to_numeric(out.get("watch_live_time"), errors="coerce").fillna(0).clip(lower=0)
    out["watch_live_time_log"] = np.log1p(watch_time).astype(np.float32)
    return out


def build_feature_columns(df: pd.DataFrame, disable_id_features: bool = False) -> Tuple[List[str], List[str]]:
    base_cat = [
        "user_id",
        "streamer_id",
        "imp_year",
        "imp_month",
        "imp_day",
        "imp_hour",
        "imp_is_weekend",
    ]
    suffix_cat = [c for c in df.columns if c.endswith("_le") and c not in FORCE_NUMERIC_BINARY]
    extra_cat = [
        "live_type",
        "live_start_year",
        "live_start_month",
        "live_start_day",
        "live_start_hour",
    ]
    cat_cols = [c for c in base_cat + suffix_cat + extra_cat if c in df.columns and c not in DEFAULT_DROP_COLUMNS]
    if disable_id_features:
        cat_cols = [c for c in cat_cols if c not in {"user_id", "live_id", "streamer_id"}]

    exclude = set(cat_cols + ALL_TASKS + ["watch_live_time", "live_id"])
    exclude.update(DEFAULT_DROP_COLUMNS)
    for c in df.columns:
        lc = c.lower()
        if "timestamp" in lc or lc.endswith("_ts") or c in {"imp_timestamp", "p_date"}:
            exclude.add(c)
    non_feature = {"user_reg_timestamp", "first_watch_live_timestamp", "start_timestamp", "end_timestamp", "streamer_reg_timestamp", "first_live_timestamp"}
    exclude.update(non_feature)

    num_cols = [c for c in df.columns if c not in exclude]
    for c in FORCE_NUMERIC_BINARY:
        if c in df.columns and c not in num_cols:
            num_cols.append(c)
    keep_num = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            keep_num.append(c)
    return cat_cols, keep_num


def build_cat_maps(dfs: Sequence[pd.DataFrame], cat_cols: Sequence[str]) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    all_df = pd.concat([df[list(cat_cols)] for df in dfs], axis=0, ignore_index=True)
    for c in cat_cols:
        vals = all_df[c].astype(str).unique().tolist()
        maps[c] = {v: i + 1 for i, v in enumerate(vals)}
    return maps


def encode_cat(df: pd.DataFrame, cat_cols: Sequence[str], cat_maps: Dict[str, Dict[str, int]]) -> np.ndarray:
    arrs = []
    for c in cat_cols:
        arrs.append(df[c].astype(str).map(cat_maps[c]).fillna(0).astype(np.int64).to_numpy())
    return np.stack(arrs, axis=1) if len(arrs) > 0 else np.empty((len(df), 0), dtype=np.int64)


def encode_num(df: pd.DataFrame, num_cols: Sequence[str]) -> np.ndarray:
    x = df[list(num_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    return x.to_numpy()


def prepare_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    disable_id_features: bool = False,
) -> Tuple[FeaturePack, RankingDataset, RankingDataset, RankingDataset]:
    cat_cols, num_cols = build_feature_columns(train_df, disable_id_features=disable_id_features)
    cat_maps = build_cat_maps([train_df, val_df, test_df], cat_cols)
    card = [len(cat_maps[c]) + 1 for c in cat_cols]
    feature_pack = FeaturePack(cat_cols=cat_cols, num_cols=num_cols, cat_maps=cat_maps, cat_cardinalities=card)

    def make_ds(df: pd.DataFrame) -> RankingDataset:
        cat_x = encode_cat(df, cat_cols, cat_maps)
        num_x = encode_num(df, num_cols)
        ys = {k: pd.to_numeric(df[k], errors="coerce").fillna(0).to_numpy(dtype=np.float32) for k in ALL_TASKS}
        user_ids = pd.to_numeric(df.get("user_id"), errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        return RankingDataset(cat_x, num_x, ys, user_ids)

    return feature_pack, make_ds(train_df), make_ds(val_df), make_ds(test_df)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    task_weights: Dict[str, float],
    grad_clip_norm: float = 0.0,
    reg_loss_name: str = "huber",
    watch_log_train_clip: float = 0.0,
) -> float:
    model.train()
    bce = nn.BCEWithLogitsLoss()
    reg_loss_fn: nn.Module = nn.MSELoss() if reg_loss_name == "mse" else nn.SmoothL1Loss(beta=1.0)
    losses = []
    skipped_nonfinite = 0
    for cat_x, num_x, labels, _user_ids in tqdm(loader, desc="train", leave=False):
        cat_x = cat_x.to(device)
        num_x = num_x.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        logits = model(cat_x, num_x)
        reg_pred = logits[TASK_REG]
        if watch_log_train_clip > 0:
            reg_pred = reg_pred.clamp(min=-watch_log_train_clip, max=watch_log_train_clip)
        loss_bin = sum(task_weights[k] * bce(logits[k], labels[k]) for k in TASKS_BIN)
        loss_reg = task_weights[TASK_REG] * reg_loss_fn(reg_pred, labels[TASK_REG])
        loss = loss_bin + loss_reg
        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            continue
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    if skipped_nonfinite > 0:
        print(f"warning: skipped {skipped_nonfinite} non-finite batches in train_epoch")
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def _precision_at_k_by_user(
    user_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: List[int],
) -> Dict[str, float]:
    results = {f"precision@{k}": 0.0 for k in ks}
    if len(user_ids) == 0:
        return {k: float("nan") for k in results}

    uniq = np.unique(user_ids)
    valid_users = 0
    for uid in uniq:
        idx = np.where(user_ids == uid)[0]
        if len(idx) == 0:
            continue
        valid_users += 1
        local_scores = y_score[idx]
        local_true = y_true[idx]
        order = np.argsort(-local_scores)
        ranked_true = local_true[order]
        for k in ks:
            top = ranked_true[:k]
            denom = max(len(top), 1)
            results[f"precision@{k}"] += float(top.sum()) / float(denom)

    if valid_users == 0:
        return {k: float("nan") for k in results}
    return {k: v / valid_users for k, v in results.items()}


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    task_weights: Dict[str, float],
    precision_ks: List[int],
    reg_loss_name: str = "huber",
    watch_log_train_clip: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    reg_loss_fn: nn.Module = nn.MSELoss() if reg_loss_name == "mse" else nn.SmoothL1Loss(beta=1.0)
    losses = []
    probs: Dict[str, List[np.ndarray]] = {k: [] for k in TASKS_BIN}
    ys: Dict[str, List[np.ndarray]] = {k: [] for k in TASKS_BIN}
    user_ids_all: List[np.ndarray] = []
    reg_preds: List[np.ndarray] = []
    reg_true: List[np.ndarray] = []
    for cat_x, num_x, labels, user_ids in tqdm(loader, desc="eval", leave=False):
        cat_x = cat_x.to(device)
        num_x = num_x.to(device)
        labels_t = {k: v.to(device) for k, v in labels.items()}
        logits = model(cat_x, num_x)
        reg_pred = logits[TASK_REG]
        if watch_log_train_clip > 0:
            reg_pred = reg_pred.clamp(min=-watch_log_train_clip, max=watch_log_train_clip)
        loss_bin = sum(task_weights[k] * bce(logits[k], labels_t[k]) for k in TASKS_BIN)
        loss_reg = task_weights[TASK_REG] * reg_loss_fn(reg_pred, labels_t[TASK_REG])
        loss = loss_bin + loss_reg
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
        for k in TASKS_BIN:
            probs[k].append(torch.sigmoid(logits[k]).detach().cpu().numpy())
            ys[k].append(labels_t[k].detach().cpu().numpy())
        user_ids_all.append(user_ids.detach().cpu().numpy())
        reg_preds.append(reg_pred.detach().cpu().numpy())
        reg_true.append(labels_t[TASK_REG].detach().cpu().numpy())

    metrics: Dict[str, float] = {"loss": float(np.mean(losses)) if losses else 0.0}
    auc_values = []
    for k in TASKS_BIN:
        y_true = np.concatenate(ys[k]) if ys[k] else np.array([], dtype=np.float32)
        y_prob = np.concatenate(probs[k]) if probs[k] else np.array([], dtype=np.float32)
        auc = compute_auc(y_true, y_prob) if len(y_true) > 0 else None
        if auc is None:
            metrics[f"auc_{k}"] = float("nan")
        else:
            metrics[f"auc_{k}"] = float(auc)
            auc_values.append(float(auc))
        if len(y_true) > 0 and user_ids_all:
            p_at_k = _precision_at_k_by_user(
                user_ids=np.concatenate(user_ids_all),
                y_true=y_true,
                y_score=y_prob,
                ks=precision_ks,
            )
            for pkey, pval in p_at_k.items():
                metrics[f"{pkey}_{k}"] = float(pval)
    metrics["auc_mean"] = float(np.mean(auc_values)) if auc_values else float("nan")
    if reg_true and reg_preds:
        y_t = np.concatenate(reg_true).astype(np.float64, copy=False)
        y_p = np.concatenate(reg_preds).astype(np.float64, copy=False)
        diff = np.clip(y_t - y_p, -1e6, 1e6)
        mse_val = float(np.mean(diff * diff))
        metrics[f"mse_{TASK_REG}"] = mse_val
        metrics[f"rmse_{TASK_REG}"] = float(np.sqrt(mse_val))
    return metrics


def compact_metrics(metrics: Dict[str, float], keys: Sequence[str] = COMPACT_METRIC_KEYS) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        if k in metrics:
            out[k] = float(metrics[k])
    return out


def format_metrics_vertical(title: str, metrics: Dict[str, float]) -> str:
    lines = [title]
    for k, v in metrics.items():
        if np.isnan(v):
            lines.append(f"  {k}: nan")
        else:
            lines.append(f"  {k}: {v:.6f}")
    return "\n".join(lines)


def _safe_tag(x: float) -> str:
    s = f"{x:g}"
    return re.sub(r"[^0-9a-zA-Z]+", "", s)


def resolve_run_specific_checkpoint_path(args: argparse.Namespace, project_root: Path) -> Path:
    default_path = project_root / "models" / "dcnv2_multitask_best.pt"
    if args.checkpoint_path != default_path:
        return args.checkpoint_path
    name = (
        "dcnv2_run_specific_best"
        f"_lr{_safe_tag(args.lr)}"
        f"_wt{_safe_tag(args.w_watch_time)}"
        f"_gc{_safe_tag(args.grad_clip_norm)}"
        f"_clip{_safe_tag(args.watch_log_train_clip)}"
        f"_drop{_safe_tag(args.dropout)}.pt"
    )
    return project_root / "models" / name


@torch.no_grad()
def predict_test(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    preds = {k: [] for k in ALL_TASKS}
    for cat_x, num_x, _, _user_ids in tqdm(loader, desc="predict", leave=False):
        cat_x = cat_x.to(device)
        num_x = num_x.to(device)
        logits = model(cat_x, num_x)
        for k in TASKS_BIN:
            preds[k].append(torch.sigmoid(logits[k]).detach().cpu().numpy())
        preds[TASK_REG].append(logits[TASK_REG].detach().cpu().numpy())
    return {k: np.concatenate(v) if len(v) > 0 else np.array([], dtype=np.float32) for k, v in preds.items()}


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.checkpoint_path = resolve_run_specific_checkpoint_path(args, project_root)
    deep_layers = tuple(int(x) for x in args.deep_layers.split(",") if x.strip())
    precision_ks = [int(x) for x in args.precision_ks.split(",") if x.strip()]
    device = args.device
    task_weights = {
        "is_click": args.w_ctr,
        "watch_greater_30s": args.w_watch30,
        "is_like": args.w_like,
        "watch_live_time_log": args.w_watch_time,
    }

    print("loading csvs...")
    train_df = ensure_labels(pd.read_csv(args.train_path, low_memory=False))
    val_df = ensure_labels(pd.read_csv(args.val_path, low_memory=False))
    test_df = ensure_labels(pd.read_csv(args.test_path, low_memory=False))
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    watch_log_mean = float(train_df[TASK_REG].mean())
    watch_log_std = float(train_df[TASK_REG].std(ddof=0))
    if watch_log_std <= 0:
        watch_log_std = 1.0
    if args.watch_log_standardize:
        for df in [train_df, val_df, test_df]:
            df[TASK_REG] = (df[TASK_REG] - watch_log_mean) / watch_log_std
        print(
            "watch_log_standardize:",
            {"enabled": True, "train_mean": watch_log_mean, "train_std": watch_log_std},
        )
    else:
        print("watch_log_standardize:", {"enabled": False})

    feature_pack, train_ds, val_ds, test_ds = prepare_data(
        train_df,
        val_df,
        test_df,
        disable_id_features=args.disable_id_features,
    )
    print(
        f"features: categorical={len(feature_pack.cat_cols)} numeric={len(feature_pack.num_cols)} "
        f"disable_id_features={args.disable_id_features}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultiTaskDCNv2(
        cat_cardinalities=feature_pack.cat_cardinalities,
        num_features=len(feature_pack.num_cols),
        num_cross_layers=args.num_cross_layers,
        deep_layers=deep_layers,
        dropout=args.dropout,
        cross_low_rank=args.cross_low_rank,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -float("inf")
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    args.periodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            task_weights,
            grad_clip_norm=args.grad_clip_norm,
            reg_loss_name=args.reg_loss,
            watch_log_train_clip=args.watch_log_train_clip,
        )
        val_metrics = eval_epoch(
            model,
            val_loader,
            device,
            task_weights,
            precision_ks,
            reg_loss_name=args.reg_loss,
            watch_log_train_clip=args.watch_log_train_clip,
        )
        val_metrics_compact = compact_metrics(val_metrics)
        print(f"epoch={epoch}")
        print(f"  train_loss: {train_loss:.6f}")
        print(format_metrics_vertical("  val_metrics:", val_metrics_compact))

        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            periodic_path = args.periodic_checkpoint_dir / f"dcnv2_multitask_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "feature_pack": {
                        "cat_cols": feature_pack.cat_cols,
                        "num_cols": feature_pack.num_cols,
                        "cat_maps": feature_pack.cat_maps,
                    },
                    "val_metrics": val_metrics,
                    "config": {
                        "deep_layers": deep_layers,
                        "num_cross_layers": args.num_cross_layers,
                        "cross_low_rank": args.cross_low_rank,
                        "dropout": args.dropout,
                    },
                },
                periodic_path,
            )
            print(f"saved periodic checkpoint -> {periodic_path}")

        auc_mean = float(val_metrics.get("auc_mean", float("nan")))
        if np.isfinite(auc_mean) and auc_mean > best_auc:
            best_auc = auc_mean
            payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_auc_mean": best_auc,
                "feature_pack": {
                    "cat_cols": feature_pack.cat_cols,
                    "num_cols": feature_pack.num_cols,
                    "cat_maps": feature_pack.cat_maps,
                },
                "config": {
                    "deep_layers": deep_layers,
                    "num_cross_layers": args.num_cross_layers,
                    "cross_low_rank": args.cross_low_rank,
                    "dropout": args.dropout,
                },
            }
            torch.save(payload, args.checkpoint_path)
            print(f"updated best checkpoint -> {args.checkpoint_path}")

    if args.checkpoint_path.exists():
        best_payload = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(best_payload["model_state"])
        best_epoch = best_payload.get("epoch")
        print(f"loaded best checkpoint for test -> {args.checkpoint_path} epoch={best_epoch}")
    else:
        print(f"warning: best checkpoint not found at {args.checkpoint_path}; using last-epoch model for test")

    test_metrics = eval_epoch(
        model,
        test_loader,
        device,
        task_weights,
        precision_ks,
        reg_loss_name=args.reg_loss,
        watch_log_train_clip=args.watch_log_train_clip,
    )
    print(format_metrics_vertical("test_metrics:", compact_metrics(test_metrics)))

    preds = predict_test(model, test_loader, device)
    id_cols = [c for c in ["user_id", "live_id", "streamer_id"] if c in test_df.columns]
    out = test_df[id_cols].copy()
    out["score_ctr"] = preds["is_click"]
    out["score_watch_greater_30s"] = preds["watch_greater_30s"]
    out["score_like"] = preds["is_like"]
    out["score_watch_live_time_log"] = preds["watch_live_time_log"]
    if args.watch_log_standardize:
        out["score_watch_live_time_log_unscaled"] = out["score_watch_live_time_log"] * watch_log_std + watch_log_mean
        export_log = out["score_watch_live_time_log_unscaled"].to_numpy()
    else:
        export_log = out["score_watch_live_time_log"].to_numpy()
    export_log = np.clip(export_log, a_min=args.watch_log_export_min, a_max=args.watch_log_export_max)
    out["score_watch_live_time_pred"] = np.expm1(export_log)
    out["score_multitask_mean"] = (out["score_ctr"] + out["score_watch_greater_30s"] + out["score_like"]) / 3.0
    args.pred_output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.pred_output_path, index=False)
    print(f"saved test predictions -> {args.pred_output_path} rows={len(out)}")


if __name__ == "__main__":
    main()
