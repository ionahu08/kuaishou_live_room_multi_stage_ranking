from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dcnv2_train import (
    ALL_TASKS,
    TASK_REG,
    MultiTaskDCNv2,
    RankingDataset,
    compact_metrics,
    encode_cat,
    encode_num,
    ensure_labels,
    eval_epoch,
    format_metrics_vertical,
    predict_test,
)
from ttnn.config import FeatureMeta, ITEM_CATEGORICAL, USER_CATEGORICAL
from ttnn.data import preprocess_df
from ttnn.models import TwoTowerModel


@dataclass
class TtnnEvalBundle:
    df: pd.DataFrame
    topk_df: pd.DataFrame
    recall_by_user: pd.DataFrame
    recall_mean: float
    users_with_positive: int


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="End-to-end 2-stage eval: TTNN ANN top-K retrieval -> DCNv2 rerank/eval."
    )
    parser.add_argument(
        "--ttnn-val-path",
        type=Path,
        default=root / "data" / "TTNN_full_test.csv",
        help="TTNN evaluation split used for ANN candidate retrieval (test-only by default).",
    )
    parser.add_argument(
        "--ttnn-checkpoint-path",
        type=Path,
        default=root / "models" / "best_tower.pt",
        help="Trained TTNN checkpoint.",
    )
    parser.add_argument(
        "--ttnn-label-col",
        type=str,
        default="is_click",
        help="TTNN positive label column used for recall@K.",
    )
    parser.add_argument("--ttnn-topk", type=int, default=100, help="ANN top-K per user.")
    parser.add_argument("--ttnn-multi-interest-k", type=int, default=2, choices=[1, 2])

    parser.add_argument(
        "--dcn-data-path",
        type=Path,
        default=root / "data" / "dcnv2_full_retry_test.csv",
        help="DCNv2 split used for rerank scoring/evaluation (test-only by default).",
    )
    parser.add_argument(
        "--dcn-train-path",
        type=Path,
        default=root / "data" / "dcnv2_full_retry_train.csv",
        help="DCNv2 train split for watch_log standardization stats.",
    )
    parser.add_argument(
        "--allow-non-test-eval-paths",
        action="store_true",
        help="Allow non-test eval paths (disabled by default to prevent accidental train/val evaluation).",
    )
    parser.add_argument(
        "--dcn-checkpoint-path",
        type=Path,
        default=root / "models" / "dcnv2_rerun_with_id_v1.pt",
        help="Trained DCNv2 checkpoint.",
    )
    parser.add_argument("--dcn-batch-size", type=int, default=4096)
    parser.add_argument("--dcn-num-workers", type=int, default=0)
    parser.add_argument("--dcn-reg-loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--dcn-watch-log-train-clip", type=float, default=8.0)
    parser.add_argument("--dcn-precision-ks", type=str, default="1,3,5,10")
    parser.add_argument(
        "--dcn-ndcg-ks",
        type=str,
        default="10,50,100",
        help="Comma-separated K values for candidate-set NDCG@K.",
    )
    parser.add_argument("--dcn-w-ctr", type=float, default=1.0)
    parser.add_argument("--dcn-w-watch30", type=float, default=1.0)
    parser.add_argument("--dcn-w-like", type=float, default=1.0)
    parser.add_argument("--dcn-w-watch-time", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="pipeline_eval",
        help="Prefix for outputs under outputs/.",
    )
    return parser.parse_args()


def _build_ttnn_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> tuple[TwoTowerModel, FeatureMeta]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload.get("config", {})
    state = payload["model_state"]

    user_cat_sizes = {}
    item_cat_sizes = {}
    for col in USER_CATEGORICAL:
        k = f"user_embeddings.{col}.weight"
        if k not in state:
            raise KeyError(f"Missing {k} in TTNN checkpoint.")
        user_cat_sizes[col] = int(state[k].shape[0])
    for col in ITEM_CATEGORICAL:
        k = f"item_embeddings.{col}.weight"
        if k not in state:
            raise KeyError(f"Missing {k} in TTNN checkpoint.")
        item_cat_sizes[col] = int(state[k].shape[0])

    emb_dim = int(config.get("emb_dim", 512))
    tower_hidden = list(config.get("tower_hidden", [1024, 512]))
    dropout = float(config.get("dropout", 0.1))
    normalize_emb = bool(config.get("normalize_emb", True))

    model = TwoTowerModel(
        user_cat_sizes=user_cat_sizes,
        item_cat_sizes=item_cat_sizes,
        user_num_dim=11,
        item_num_dim=139,
        emb_dim=emb_dim,
        tower_hidden=tower_hidden,
        dropout=dropout,
        normalize_emb=normalize_emb,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    meta = FeatureMeta(user_cat_sizes=user_cat_sizes, item_cat_sizes=item_cat_sizes)
    return model, meta


def _compute_recall_at_k_from_pairs(
    source_df: pd.DataFrame,
    pred_topk_df: pd.DataFrame,
    label_col: str,
    k: int,
) -> tuple[pd.DataFrame, float]:
    gt = (
        source_df[source_df[label_col] == 1][["user_id", "live_id", "streamer_id"]]
        .drop_duplicates()
        .copy()
    )
    pred = (
        pred_topk_df[pred_topk_df["rank"] <= k][["user_id", "live_id", "streamer_id"]]
        .drop_duplicates()
        .copy()
    )

    gt_count = gt.groupby("user_id").size().rename("gt_count")
    if gt_count.empty:
        return pd.DataFrame(columns=["user_id", "gt_count", "hit_count", f"recall@{k}"]), float("nan")

    hits = gt.merge(pred, on=["user_id", "live_id", "streamer_id"], how="inner")
    hit_count = hits.groupby("user_id").size().rename("hit_count")

    recall_df = gt_count.to_frame().join(hit_count, how="left").fillna({"hit_count": 0}).reset_index()
    recall_df["hit_count"] = recall_df["hit_count"].astype(int)
    recall_col = f"recall@{k}"
    recall_df[recall_col] = recall_df["hit_count"] / recall_df["gt_count"].clip(lower=1)
    mean_recall = float(recall_df[recall_col].mean()) if len(recall_df) > 0 else float("nan")
    return recall_df, mean_recall


def run_ttnn_stage(args: argparse.Namespace, output_dir: Path) -> TtnnEvalBundle:
    try:
        from ttnn.evaluation import retrieve_topk_items
    except ModuleNotFoundError as exc:
        if exc.name == "faiss":
            raise ModuleNotFoundError(
                "faiss is required for TTNN ANN retrieval. Install faiss-cpu/faiss-gpu in this environment."
            ) from exc
        raise

    model, meta = _build_ttnn_model_from_checkpoint(args.ttnn_checkpoint_path, args.device)
    raw_df = pd.read_csv(args.ttnn_val_path, low_memory=False)
    ttnn_df, _ = preprocess_df(
        df=raw_df,
        label_col=args.ttnn_label_col,
        meta=meta,
        source_name=args.ttnn_val_path.name,
    )

    unique_users = int(ttnn_df["user_id"].nunique())
    unique_rooms = int(ttnn_df["live_id"].nunique())
    print(
        f"ttnn_stage: rows={len(ttnn_df)} unique_users={unique_users} unique_live_rooms={unique_rooms}"
    )

    topk_df = retrieve_topk_items(
        model=model,
        df=ttnn_df,
        device=args.device,
        batch_size=args.dcn_batch_size,
        topk=args.ttnn_topk,
        multi_interest_k=args.ttnn_multi_interest_k,
    )
    topk_path = output_dir / f"{args.output_prefix}_ttnn_top{args.ttnn_topk}.csv"
    topk_df.to_csv(topk_path, index=False)
    print(f"saved TTNN top-k -> {topk_path} rows={len(topk_df)}")

    recall_by_user, recall_mean = _compute_recall_at_k_from_pairs(
        source_df=ttnn_df,
        pred_topk_df=topk_df,
        label_col=args.ttnn_label_col,
        k=args.ttnn_topk,
    )
    recall_path = output_dir / f"{args.output_prefix}_ttnn_recall_at_{args.ttnn_topk}_by_user.csv"
    recall_by_user.to_csv(recall_path, index=False)
    print(f"saved TTNN recall-by-user -> {recall_path} rows={len(recall_by_user)}")
    print(f"ttnn_stage: mean recall@{args.ttnn_topk}={recall_mean:.6f}")

    return TtnnEvalBundle(
        df=ttnn_df,
        topk_df=topk_df,
        recall_by_user=recall_by_user,
        recall_mean=recall_mean,
        users_with_positive=len(recall_by_user),
    )


def _deduplicate_dcn_rows(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["user_id", "live_id", "streamer_id"]
    if "imp_timestamp" in df.columns:
        out = df.copy()
        out["__imp_ts"] = pd.to_datetime(out["imp_timestamp"], errors="coerce")
        out = out.sort_values("__imp_ts", ascending=False).drop(columns=["__imp_ts"])
        return out.drop_duplicates(key_cols, keep="first").reset_index(drop=True)
    return df.drop_duplicates(key_cols, keep="first").reset_index(drop=True)


def _prepare_dcn_candidate_df(
    dcn_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    topk: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_cols = ["user_id", "live_id", "streamer_id", "rank", "score"]
    cand_pairs = topk_df[topk_df["rank"] <= topk][pair_cols].copy()
    cand_pairs = cand_pairs.rename(columns={"score": "ttnn_score", "rank": "ttnn_rank"})

    dcn_dedup = _deduplicate_dcn_rows(dcn_df)
    joined = cand_pairs.merge(
        dcn_dedup,
        on=["user_id", "live_id", "streamer_id"],
        how="inner",
    )
    return cand_pairs, dcn_dedup, joined


def _compute_overlap_diagnostics(
    cand_pairs: pd.DataFrame,
    dcn_dedup: pd.DataFrame,
) -> Dict[str, object]:
    cand_pairs_u = cand_pairs[["user_id", "live_id", "streamer_id"]].drop_duplicates()
    dcn_pairs_u = dcn_dedup[["user_id", "live_id", "streamer_id"]].drop_duplicates()

    cand_users = set(pd.to_numeric(cand_pairs_u["user_id"], errors="coerce").fillna(-1).astype(np.int64).tolist())
    dcn_users = set(pd.to_numeric(dcn_pairs_u["user_id"], errors="coerce").fillna(-1).astype(np.int64).tolist())
    cand_lives = set(pd.to_numeric(cand_pairs_u["live_id"], errors="coerce").fillna(-1).astype(np.int64).tolist())
    dcn_lives = set(pd.to_numeric(dcn_pairs_u["live_id"], errors="coerce").fillna(-1).astype(np.int64).tolist())
    cand_streamers = set(pd.to_numeric(cand_pairs_u["streamer_id"], errors="coerce").fillna(-1).astype(np.int64).tolist())
    dcn_streamers = set(pd.to_numeric(dcn_pairs_u["streamer_id"], errors="coerce").fillna(-1).astype(np.int64).tolist())

    user_overlap = cand_users & dcn_users
    live_overlap = cand_lives & dcn_lives
    streamer_overlap = cand_streamers & dcn_streamers

    pair_merge = cand_pairs_u.merge(
        dcn_pairs_u,
        on=["user_id", "live_id", "streamer_id"],
        how="left",
        indicator=True,
    )
    pair_overlap_count = int((pair_merge["_merge"] == "both").sum())
    missing_pairs = pair_merge[pair_merge["_merge"] == "left_only"][["user_id", "live_id", "streamer_id"]].head(10)

    return {
        "candidate_unique_counts": {
            "users": int(len(cand_users)),
            "live_ids": int(len(cand_lives)),
            "streamers": int(len(cand_streamers)),
            "pairs": int(len(cand_pairs_u)),
        },
        "dcn_unique_counts": {
            "users": int(len(dcn_users)),
            "live_ids": int(len(dcn_lives)),
            "streamers": int(len(dcn_streamers)),
            "pairs": int(len(dcn_pairs_u)),
        },
        "overlap_counts": {
            "users": int(len(user_overlap)),
            "live_ids": int(len(live_overlap)),
            "streamers": int(len(streamer_overlap)),
            "pairs": int(pair_overlap_count),
        },
        "overlap_rates_wrt_candidates": {
            "users": float(len(user_overlap) / max(len(cand_users), 1)),
            "live_ids": float(len(live_overlap) / max(len(cand_lives), 1)),
            "streamers": float(len(streamer_overlap) / max(len(cand_streamers), 1)),
            "pairs": float(pair_overlap_count / max(len(cand_pairs_u), 1)),
        },
        "missing_examples": {
            "users_not_in_dcn": [int(x) for x in sorted(cand_users - dcn_users)[:10]],
            "live_ids_not_in_dcn": [int(x) for x in sorted(cand_lives - dcn_lives)[:10]],
            "streamers_not_in_dcn": [int(x) for x in sorted(cand_streamers - dcn_streamers)[:10]],
            "pairs_not_in_dcn": missing_pairs.to_dict(orient="records"),
        },
    }


def _build_dcn_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> tuple[MultiTaskDCNv2, Dict[str, object], Dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload["model_state"]
    feature_pack = payload["feature_pack"]
    config = payload["config"]

    cat_cols: List[str] = list(feature_pack["cat_cols"])
    cat_maps: Dict[str, Dict[str, int]] = dict(feature_pack["cat_maps"])
    cat_cardinalities = [len(cat_maps[c]) + 1 for c in cat_cols]
    num_cols: List[str] = list(feature_pack["num_cols"])

    model = MultiTaskDCNv2(
        cat_cardinalities=cat_cardinalities,
        num_features=len(num_cols),
        num_cross_layers=int(config["num_cross_layers"]),
        deep_layers=tuple(int(x) for x in config["deep_layers"]),
        dropout=float(config["dropout"]),
        cross_low_rank=int(config["cross_low_rank"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, feature_pack, config


def _build_dcn_dataset(
    df: pd.DataFrame,
    feature_pack: Dict[str, object],
) -> RankingDataset:
    cat_cols: Sequence[str] = feature_pack["cat_cols"]  # type: ignore[assignment]
    num_cols: Sequence[str] = feature_pack["num_cols"]  # type: ignore[assignment]
    cat_maps: Dict[str, Dict[str, int]] = feature_pack["cat_maps"]  # type: ignore[assignment]

    cat_x = encode_cat(df, cat_cols, cat_maps)
    num_x = encode_num(df, num_cols)
    ys = {k: pd.to_numeric(df[k], errors="coerce").fillna(0).to_numpy(dtype=np.float32) for k in ALL_TASKS}
    user_ids = pd.to_numeric(df["user_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    return RankingDataset(cat_x=cat_x, num_x=num_x, ys=ys, user_ids=user_ids)


def _ndcg_at_k_by_user(
    user_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, float]:
    metrics = {f"ndcg@{k}": float("nan") for k in ks}
    if len(user_ids) == 0:
        return metrics

    uniq = np.unique(user_ids)
    acc = {k: 0.0 for k in ks}
    valid_users = {k: 0 for k in ks}

    for uid in uniq:
        idx = np.where(user_ids == uid)[0]
        if len(idx) == 0:
            continue
        rel = y_true[idx].astype(np.float64)
        score = y_score[idx].astype(np.float64)
        order = np.argsort(-score)
        ranked_rel = rel[order]
        ideal_rel = np.sort(rel)[::-1]
        for k in ks:
            top_rel = ranked_rel[:k]
            ideal_top = ideal_rel[:k]
            discounts = 1.0 / np.log2(np.arange(2, len(top_rel) + 2))
            dcg = float((top_rel * discounts).sum())
            idcg = float((ideal_top * discounts).sum())
            if idcg <= 0:
                continue
            acc[k] += dcg / idcg
            valid_users[k] += 1

    for k in ks:
        if valid_users[k] > 0:
            metrics[f"ndcg@{k}"] = acc[k] / valid_users[k]
    return metrics


def _apply_watch_log_standardization(df: pd.DataFrame, dcn_train_path: Path) -> tuple[pd.DataFrame, float, float]:
    train_df = ensure_labels(pd.read_csv(dcn_train_path, low_memory=False))
    mean = float(train_df[TASK_REG].mean())
    std = float(train_df[TASK_REG].std(ddof=0))
    if std <= 0:
        std = 1.0
    out = df.copy()
    out[TASK_REG] = (out[TASK_REG] - mean) / std
    return out, mean, std


def run_dcn_stage(
    args: argparse.Namespace,
    output_dir: Path,
    topk_df: pd.DataFrame,
) -> tuple[Dict[str, float], Dict[str, object]]:
    dcn_raw = ensure_labels(pd.read_csv(args.dcn_data_path, low_memory=False))
    cand_pairs, dcn_dedup, candidate_df = _prepare_dcn_candidate_df(dcn_raw, topk_df, args.ttnn_topk)
    diagnostics = _compute_overlap_diagnostics(cand_pairs, dcn_dedup)

    overlap_rates = diagnostics["overlap_rates_wrt_candidates"]  # type: ignore[index]
    print(
        "dcn_stage_overlap:",
        {
            "user_overlap_rate": f"{overlap_rates['users']:.2%}",
            "live_overlap_rate": f"{overlap_rates['live_ids']:.2%}",
            "streamer_overlap_rate": f"{overlap_rates['streamers']:.2%}",
            "pair_overlap_rate": f"{overlap_rates['pairs']:.2%}",
        },
    )

    diag_path = output_dir / f"{args.output_prefix}_overlap_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"saved overlap diagnostics -> {diag_path}")

    matched_pairs = int(candidate_df[["user_id", "live_id", "streamer_id"]].drop_duplicates().shape[0])
    requested_pairs = int(topk_df[topk_df["rank"] <= args.ttnn_topk][["user_id", "live_id", "streamer_id"]].drop_duplicates().shape[0])
    print(
        f"dcn_stage: matched candidate pairs={matched_pairs}/{requested_pairs} "
        f"({(matched_pairs / max(requested_pairs, 1)):.2%})"
    )
    if len(candidate_df) == 0:
        raise ValueError("No candidate pairs matched DCNv2 dataset. Check split alignment or IDs.")

    candidate_df, train_mean, train_std = _apply_watch_log_standardization(candidate_df, args.dcn_train_path)
    print(
        "dcn_stage: watch_log_standardize",
        {"enabled": True, "train_mean": train_mean, "train_std": train_std},
    )

    model, feature_pack, _ = _build_dcn_model_from_checkpoint(args.dcn_checkpoint_path, args.device)
    ds = _build_dcn_dataset(candidate_df, feature_pack)
    loader = DataLoader(ds, batch_size=args.dcn_batch_size, shuffle=False, num_workers=args.dcn_num_workers)
    precision_ks = [int(x) for x in args.dcn_precision_ks.split(",") if x.strip()]
    task_weights = {
        "is_click": args.dcn_w_ctr,
        "watch_greater_30s": args.dcn_w_watch30,
        "is_like": args.dcn_w_like,
        TASK_REG: args.dcn_w_watch_time,
    }

    metrics = eval_epoch(
        model=model,
        loader=loader,
        device=args.device,
        task_weights=task_weights,
        precision_ks=precision_ks,
        reg_loss_name=args.dcn_reg_loss,
        watch_log_train_clip=args.dcn_watch_log_train_clip,
    )
    print(format_metrics_vertical("dcn_stage_metrics:", compact_metrics(metrics)))

    preds = predict_test(model=model, loader=loader, device=args.device)
    ndcg_ks = [int(x) for x in args.dcn_ndcg_ks.split(",") if x.strip()]
    user_ids = pd.to_numeric(candidate_df["user_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    for task in ["is_click", "watch_greater_30s", "is_like"]:
        y_true = pd.to_numeric(candidate_df[task], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        y_score = preds[task]
        ndcg_vals = _ndcg_at_k_by_user(user_ids=user_ids, y_true=y_true, y_score=y_score, ks=ndcg_ks)
        for k in ndcg_ks:
            key = f"ndcg@{k}_{task}"
            metrics[key] = float(ndcg_vals[f"ndcg@{k}"])

    ndcg_view = {k: v for k, v in metrics.items() if k.startswith("ndcg@")}
    print(format_metrics_vertical("dcn_stage_ndcg_metrics:", ndcg_view))

    scored = candidate_df[["user_id", "live_id", "streamer_id", "ttnn_rank", "ttnn_score"]].copy()
    scored["score_ctr"] = preds["is_click"]
    scored["score_watch_greater_30s"] = preds["watch_greater_30s"]
    scored["score_like"] = preds["is_like"]
    scored["score_watch_live_time_log"] = preds[TASK_REG]
    scored["score_multitask_mean"] = (
        scored["score_ctr"] + scored["score_watch_greater_30s"] + scored["score_like"]
    ) / 3.0
    scored_path = output_dir / f"{args.output_prefix}_dcn_scored_candidates.csv"
    scored.to_csv(scored_path, index=False)
    print(f"saved DCNv2 scored candidates -> {scored_path} rows={len(scored)}")

    return metrics, diagnostics


def main() -> None:
    args = parse_args()
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    expected_ttnn_test = (root / "data" / "TTNN_full_test.csv").resolve()
    expected_dcn_test = (root / "data" / "dcnv2_full_retry_test.csv").resolve()

    if not args.allow_non_test_eval_paths:
        if args.ttnn_val_path.resolve() != expected_ttnn_test:
            raise ValueError(
                f"TTNN eval path must be test set: {expected_ttnn_test}. "
                "Use --allow-non-test-eval-paths to override."
            )
        if args.dcn_data_path.resolve() != expected_dcn_test:
            raise ValueError(
                f"DCNv2 eval path must be test set: {expected_dcn_test}. "
                "Use --allow-non-test-eval-paths to override."
            )

    if not args.ttnn_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing TTNN checkpoint: {args.ttnn_checkpoint_path}")
    if not args.dcn_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing DCNv2 checkpoint: {args.dcn_checkpoint_path}")

    print("pipeline_eval: starting end-to-end 2-stage evaluation")
    ttnn_bundle = run_ttnn_stage(args, output_dir)
    dcn_metrics, overlap_diagnostics = run_dcn_stage(args, output_dir, ttnn_bundle.topk_df)

    summary = {
        "ttnn_recall_at_k": args.ttnn_topk,
        "ttnn_recall_mean": ttnn_bundle.recall_mean,
        "ttnn_users_with_positive": ttnn_bundle.users_with_positive,
        "overlap_diagnostics": overlap_diagnostics,
        "dcn_metrics": {k: float(v) if np.isfinite(v) else None for k, v in dcn_metrics.items()},
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved pipeline summary -> {summary_path}")
    print("pipeline_eval: completed")


if __name__ == "__main__":
    main()
