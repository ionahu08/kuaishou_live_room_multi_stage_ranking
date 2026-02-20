from pathlib import Path
import hashlib
import pickle
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from ..config import ITEM_NUMERIC, USER_NUMERIC
from ..data import UserBatchSampler, build_dataset, build_meta_from_dfs, load_inference_data, preprocess_df
from ..evaluation import evaluate_retrieval, retrieve_topk_items
from ..models import TwoTowerModel
from .losses import compute_loss


def train(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    label_col: str,
    cache_dir: Path | None,
    rebuild_cache: bool,
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
    save_every_epochs: int,
    periodic_checkpoint_dir: Path,
    multi_interest_k: int,
    test_topk_k: int,
    test_topk_output_path: Path,
    unseen_path: Path | None,
    unseen_topk_k: int,
    unseen_topk_output_path: Path | None,
    bce_pos_weight: float | None,
    bce_weighted_sampler: bool,
    weak_positive_weight: float,
) -> None:
    start_ts = time.perf_counter()
    best_event_log_path = train_path.parents[1] / "logs" / "best_tower_events.log"

    def log_stage(msg: str) -> None:
        elapsed = time.perf_counter() - start_ts
        print(f"[train][{elapsed:8.1f}s] {msg}", flush=True)

    def _cache_key() -> str:
        h = hashlib.sha256()
        h.update(str(label_col).encode("utf-8"))
        for p in [train_path, val_path, test_path]:
            st = p.stat()
            h.update(str(p.resolve()).encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
            h.update(str(st.st_mtime_ns).encode("utf-8"))
        return h.hexdigest()[:20]

    cache_path: Path | None = None
    train_df = val_df = test_df = meta = None

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key()
        cache_path = cache_dir / f"ttnn_preprocessed_{key}.pkl"
        if cache_path.exists() and not rebuild_cache:
            log_stage(f"cache hit: loading preprocessed data from {cache_path}")
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            train_df = payload["train_df"]
            val_df = payload["val_df"]
            test_df = payload["test_df"]
            meta = payload["meta"]
            log_stage(
                "cache loaded: "
                f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
            )
        elif cache_path.exists() and rebuild_cache:
            log_stage(f"rebuild_cache=True: ignoring existing cache {cache_path}")
        else:
            log_stage("cache miss: preprocessing from raw csv files")

    if train_df is None or val_df is None or test_df is None or meta is None:
        log_stage(f"reading train csv: {train_path}")
        train_raw = pd.read_csv(train_path, low_memory=False, memory_map=True)
        log_stage(f"train loaded: rows={len(train_raw)} cols={train_raw.shape[1]}")

        log_stage(f"reading val csv: {val_path}")
        val_raw = pd.read_csv(val_path, low_memory=False, memory_map=True)
        log_stage(f"val loaded: rows={len(val_raw)} cols={val_raw.shape[1]}")

        log_stage(f"reading test csv: {test_path}")
        test_raw = pd.read_csv(test_path, low_memory=False, memory_map=True)
        log_stage(f"test loaded: rows={len(test_raw)} cols={test_raw.shape[1]}")

        # Build categorical feature metadata from multiple data splits.
        log_stage("building categorical metadata from train/val/test")
        meta = build_meta_from_dfs([train_raw, val_raw, test_raw])
        log_stage("categorical metadata ready")

        log_stage("preprocessing train split")
        train_df, _ = preprocess_df(train_raw, label_col, meta=meta, source_name=train_path.name)
        log_stage(f"train preprocessed: rows={len(train_df)}")
        log_stage("preprocessing val split")
        val_df, _ = preprocess_df(val_raw, label_col, meta=meta, source_name=val_path.name)
        log_stage(f"val preprocessed: rows={len(val_df)}")
        log_stage("preprocessing test split")
        test_df, _ = preprocess_df(test_raw, label_col, meta=meta, source_name=test_path.name)
        log_stage(f"test preprocessed: rows={len(test_df)}")

        if cache_path is not None:
            payload = {
                "train_df": train_df,
                "val_df": val_df,
                "test_df": test_df,
                "meta": meta,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            log_stage(f"saved preprocessing cache -> {cache_path}")

    if label_col in train_df.columns:
        user_counts = train_df.groupby("user_id")[label_col].agg(
            positives=lambda s: (s == 1).sum(),
            negatives=lambda s: (s == 0).sum(),
        )
        print(
            "train_user_stats:",
            {
                "users": int(user_counts.shape[0]),
                "avg_positives_per_user": float(user_counts["positives"].mean()),
                "avg_negatives_per_user": float(user_counts["negatives"].mean()),
            },
        )

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

    if loss_name == "contrastive":
        train_df = train_df[train_df[label_col] == 1].reset_index(drop=True)

    # Build sample weights from optional freshness weighting and BCE soft-label policy.
    sample_weight = np.ones(len(train_df), dtype=np.float32)

    if freshness_weighting and "imp_timestamp" in train_df.columns:
        ts = pd.to_datetime(train_df["imp_timestamp"], errors="coerce")
        max_ts = ts.max()
        age_hours = (max_ts - ts).dt.total_seconds() / 3600.0
        freshness = np.exp(-np.log(2) * age_hours / max(freshness_half_life_hours, 1e-6))
        sample_weight *= freshness.fillna(0).to_numpy(dtype=np.float32)

    if "watch_live_time" in train_df.columns:
        weak_pos_mask = (
            (pd.to_numeric(train_df["watch_live_time"], errors="coerce") < 30)
            & (train_df[label_col] == 1)
        )
        weak_count = int(weak_pos_mask.sum())
        if weak_count > 0:
            if loss_name == "bce":
                # For BCE we keep weak positives as positives but down-weight their contribution.
                weak_positive_weight = float(np.clip(weak_positive_weight, 0.0, 1.0))
                sample_weight[weak_pos_mask.to_numpy()] *= weak_positive_weight
                print(
                    "note: down-weighted weak positives",
                    {
                        "count": weak_count,
                        "watch_live_time_threshold": 30,
                        "weak_positive_weight": weak_positive_weight,
                    },
                )
            else:
                # Keep existing behavior for non-BCE objectives.
                before = len(train_df)
                keep_mask = ~weak_pos_mask.to_numpy()
                train_df = train_df[keep_mask].reset_index(drop=True)
                sample_weight = sample_weight[keep_mask]
                dropped = before - len(train_df)
                if dropped > 0:
                    print(f"note: dropped {dropped} weak positives (watch_live_time < 30s)")

    train_ds = build_dataset(train_df, label_col, sample_weight=sample_weight)
    log_stage(f"dataset ready: train_ds={len(train_ds)}")
    if loss_name == "contrastive":
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
    elif loss_name == "bce" and bce_weighted_sampler:
        y_np = train_df[label_col].to_numpy()
        pos_count = int((y_np == 1).sum())
        neg_count = int((y_np == 0).sum())
        pos_sample_w = float(neg_count / max(pos_count, 1))
        per_sample_w = np.where(y_np == 1, pos_sample_w, 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(per_sample_w, dtype=torch.double),
            num_samples=len(per_sample_w),
            replacement=True,
        )
        print(
            "note: enabled BCE weighted sampler",
            {
                "pos_count": pos_count,
                "neg_count": neg_count,
                "positive_sample_weight": pos_sample_w,
            },
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    log_stage(f"dataloader ready: batches={len(train_loader)} batch_size={batch_size}")

    if loss_name in {"contrastive", "bce"}:
        if not normalize_emb:
            print(f"note: forcing normalize_emb=True for {loss_name} retrieval stability")
        normalize_emb = True

    effective_bce_pos_weight = bce_pos_weight
    if loss_name == "bce" and effective_bce_pos_weight is None:
        pos_count = int((train_df[label_col] == 1).sum())
        neg_count = int((train_df[label_col] == 0).sum())
        effective_bce_pos_weight = float(neg_count / max(pos_count, 1))
        print(
            "note: auto bce_pos_weight from train labels",
            {
                "pos_count": pos_count,
                "neg_count": neg_count,
                "bce_pos_weight": effective_bce_pos_weight,
            },
        )

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
    log_stage(f"model initialized on device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    train_losses = []
    best_metric = -float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, epochs + 1):
        log_stage(f"starting epoch {epoch}/{epochs}")
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
                bce_pos_weight=effective_bce_pos_weight,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * y.size(0)

        avg_loss = total / len(train_ds)
        log_stage(f"finished epoch {epoch}/{epochs} train_loss={avg_loss:.6f}; running val retrieval")
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

        if save_every_epochs > 0 and (epoch % save_every_epochs == 0):
            periodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            periodic_path = periodic_checkpoint_dir / f"tower_epoch_{epoch:03d}.pt"
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
                    "val_metrics": val_metrics,
                },
                periodic_path,
            )
            print(f"saved periodic checkpoint -> {periodic_path}")

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
                    best_named_path = checkpoint_path.parent / f"best_tower_{key.replace('@', '_')}.pt"
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
                        best_named_path,
                    )
                    print(f"updated named best checkpoint -> {best_named_path}")
                    elapsed = time.perf_counter() - start_ts
                    best_event_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(best_event_log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"[train][{elapsed:8.1f}s] finished epoch {epoch}/{epochs} "
                            f"train_loss={avg_loss:.6f}; running val retrieval\n"
                        )
                        f.write(
                            f"epoch={epoch} train_loss={avg_loss:.6f} val_metrics={val_metrics}\n"
                        )
                        f.write(
                            f"best_update metric_key={key} metric_value={best_metric:.6f} "
                            f"checkpoint={checkpoint_path} named_checkpoint={best_named_path}\n"
                        )
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
    log_stage("test retrieval done; exporting test top-k")

    test_topk_df = retrieve_topk_items(
        model=model,
        df=test_df,
        device=device,
        batch_size=batch_size,
        topk=test_topk_k,
        multi_interest_k=multi_interest_k,
    )
    test_topk_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_topk_df.to_csv(test_topk_output_path, index=False)
    print(f"saved test topk -> {test_topk_output_path} rows={len(test_topk_df)}")

    if unseen_path is not None:
        unseen_df = load_inference_data(unseen_path, meta=meta)
        unseen_topk_df = retrieve_topk_items(
            model=model,
            df=unseen_df,
            device=device,
            batch_size=batch_size,
            topk=unseen_topk_k,
            multi_interest_k=multi_interest_k,
        )
        output_path = unseen_topk_output_path
        if output_path is None:
            output_path = unseen_path.with_name(f"{unseen_path.stem}_topk_recommendations.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        unseen_topk_df.to_csv(output_path, index=False)
        print(f"saved unseen topk -> {output_path} rows={len(unseen_topk_df)}")

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
    log_stage("training pipeline finished")
