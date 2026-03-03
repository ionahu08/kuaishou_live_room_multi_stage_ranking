from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from dcnv2_train import ensure_labels


TASKS = ["is_click", "watch_greater_30s", "is_like"]
PAIR_KEYS = ["user_id", "live_id", "streamer_id"]
ITEM_KEYS = ["live_id", "streamer_id"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="MMR rerank evaluation on DCNv2 scored candidates.")
    parser.add_argument(
        "--scored-candidates-path",
        type=Path,
        default=root / "outputs" / "pipeline_eval_v1_dcn_scored_candidates.csv",
        help="Input candidate scores produced by pipeline_eval.py",
    )
    parser.add_argument(
        "--label-feature-path",
        type=Path,
        default=root / "data" / "dcnv2_full_retry_test.csv",
        help="Dataset containing labels + item features for evaluation/similarity (test-only by default).",
    )
    parser.add_argument(
        "--allow-non-test-eval-paths",
        action="store_true",
        help="Allow non-test eval paths (disabled by default to prevent accidental train/val evaluation).",
    )
    parser.add_argument("--output-prefix", type=str, default="rerank_eval_v1")

    parser.add_argument("--lambda-mmr", type=float, default=0.7, help="MMR relevance weight in [0,1].")
    parser.add_argument("--prefilter-k", type=int, default=100, help="Max candidates per user before MMR.")
    parser.add_argument("--rerank-topn", type=int, default=20, help="Final MMR selection size per user.")

    parser.add_argument("--w-ctr", type=float, default=1.0)
    parser.add_argument("--w-watch30", type=float, default=1.0)
    parser.add_argument("--w-like", type=float, default=1.0)

    parser.add_argument(
        "--sim-cat-cols",
        type=str,
        default="streamer_id,live_type,live_content_category_le,live_start_day,live_start_hour",
        help="Comma-separated item categorical columns for similarity.",
    )
    parser.add_argument(
        "--sim-num-cols",
        type=str,
        default="",
        help="Comma-separated item numeric columns for cosine similarity.",
    )
    parser.add_argument("--sim-cat-weight", type=float, default=0.7)
    parser.add_argument("--sim-num-weight", type=float, default=0.3)

    parser.add_argument("--metric-ks", type=str, default="5,10,20")
    parser.add_argument(
        "--freshness-col",
        type=str,
        default="time_since_live_start",
        help="Freshness age column (smaller means fresher).",
    )
    parser.add_argument(
        "--freshness-half-life-hours",
        type=float,
        default=6.0,
        help="Half-life for freshness score exp(-age/tau) using tau=half_life/ln(2).",
    )
    parser.add_argument(
        "--popularity-col",
        type=str,
        default="accu_play_cnt_le",
        help="Popularity column to aggregate in top-N.",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _minmax01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.full_like(x, 0.0, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((x.shape[0], x.shape[0]), dtype=np.float64)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    x_norm = x / denom
    sim = x_norm @ x_norm.T
    sim = np.clip(sim, -1.0, 1.0)
    return (sim + 1.0) / 2.0


def _cat_sim_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    n = len(df)
    if n == 0 or len(cols) == 0:
        return np.zeros((n, n), dtype=np.float64)
    mats = []
    for c in cols:
        vals = df[c].astype(str).to_numpy()
        mats.append((vals[:, None] == vals[None, :]).astype(np.float64))
    return np.mean(np.stack(mats, axis=0), axis=0)


def _build_similarity_matrix(
    cand_df: pd.DataFrame,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    cat_weight: float,
    num_weight: float,
) -> np.ndarray:
    n = len(cand_df)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    active_cat = [c for c in cat_cols if c in cand_df.columns]
    active_num = [c for c in num_cols if c in cand_df.columns]

    sim_cat = _cat_sim_matrix(cand_df, active_cat) if active_cat else np.zeros((n, n), dtype=np.float64)
    if active_num:
        num_x = cand_df[active_num].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        sim_num = _cosine_sim_matrix(num_x)
    else:
        sim_num = np.zeros((n, n), dtype=np.float64)

    total_w = 0.0
    out = np.zeros((n, n), dtype=np.float64)
    if active_cat and cat_weight > 0:
        out += cat_weight * sim_cat
        total_w += cat_weight
    if active_num and num_weight > 0:
        out += num_weight * sim_num
        total_w += num_weight
    if total_w > 0:
        out /= total_w
    out = np.clip(out, 0.0, 1.0)
    return out


def _mmr_select(
    rel: np.ndarray,
    sim: np.ndarray,
    lambda_mmr: float,
    topn: int,
) -> List[int]:
    n = rel.shape[0]
    if n == 0:
        return []
    topn = min(topn, n)
    selected: List[int] = []
    remaining = set(range(n))
    while len(selected) < topn and remaining:
        best_i = None
        best_score = -1e18
        for i in remaining:
            if not selected:
                penalty = 0.0
            else:
                penalty = float(np.max(sim[i, selected]))
            score = lambda_mmr * float(rel[i]) - (1.0 - lambda_mmr) * penalty
            if score > best_score:
                best_score = score
                best_i = i
        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def _ndcg_precision_recall_at_k(
    user_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    uniq = np.unique(user_ids)
    for k in ks:
        ndcg_sum, p_sum, r_sum = 0.0, 0.0, 0.0
        ndcg_n, p_n, r_n = 0, 0, 0
        for uid in uniq:
            idx = np.where(user_ids == uid)[0]
            if len(idx) == 0:
                continue
            yt = y_true[idx].astype(np.float64)
            ys = y_score[idx].astype(np.float64)
            order = np.argsort(-ys)
            top_rel = yt[order][:k]

            p_sum += float(np.mean(top_rel)) if len(top_rel) > 0 else 0.0
            p_n += 1

            denom = float(np.sum(yt))
            if denom > 0:
                r_sum += float(np.sum(top_rel) / denom)
                r_n += 1

            ideal = np.sort(yt)[::-1][:k]
            disc = 1.0 / np.log2(np.arange(2, len(top_rel) + 2))
            dcg = float(np.sum(top_rel * disc))
            idcg = float(np.sum(ideal * disc))
            if idcg > 0:
                ndcg_sum += dcg / idcg
                ndcg_n += 1

        out[f"ndcg@{k}"] = ndcg_sum / ndcg_n if ndcg_n > 0 else float("nan")
        out[f"precision@{k}"] = p_sum / p_n if p_n > 0 else float("nan")
        out[f"recall@{k}"] = r_sum / r_n if r_n > 0 else float("nan")
    return out


def _evaluate_ranked_df(
    df: pd.DataFrame,
    score_col: str,
    tasks: Sequence[str],
    ks: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    uids = pd.to_numeric(df["user_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    rel_for_sort = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    for task in tasks:
        y_true = pd.to_numeric(df[task], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        m = _ndcg_precision_recall_at_k(uids, y_true, rel_for_sort, ks)
        for k, v in m.items():
            metrics[f"{k}_{task}"] = float(v)
    return metrics


def _compute_diversity_metrics(
    ranked_df: pd.DataFrame,
    topn: int,
) -> Dict[str, float]:
    x = ranked_df[ranked_df["rank"] <= topn].copy()
    if len(x) == 0:
        return {
            "diversity_unique_streamers_mean@N": float("nan"),
            "diversity_unique_live_types_mean@N": float("nan"),
            "diversity_streamer_coverage@N": float("nan"),
            "diversity_live_type_coverage@N": float("nan"),
        }
    users = max(x["user_id"].nunique(), 1)

    streamer_per_user = x.groupby("user_id")["streamer_id"].nunique()
    live_type_col = "live_type" if "live_type" in x.columns else None
    if live_type_col is not None:
        ltype_per_user = x.groupby("user_id")[live_type_col].nunique()
        ltype_cov = float(x[live_type_col].nunique() / max(ranked_df[live_type_col].nunique(), 1))
        ltype_mean = float(ltype_per_user.mean())
    else:
        ltype_cov = float("nan")
        ltype_mean = float("nan")

    streamer_cov = float(x["streamer_id"].nunique() / max(ranked_df["streamer_id"].nunique(), 1))
    return {
        "diversity_unique_streamers_mean@N": float(streamer_per_user.mean()) if users > 0 else float("nan"),
        "diversity_unique_live_types_mean@N": ltype_mean,
        "diversity_streamer_coverage@N": streamer_cov,
        "diversity_live_type_coverage@N": ltype_cov,
    }


def _compute_freshness_metrics(
    ranked_df: pd.DataFrame,
    topn: int,
    freshness_col: str,
    half_life_hours: float,
) -> Dict[str, float]:
    x = ranked_df[ranked_df["rank"] <= topn].copy()
    if freshness_col not in x.columns or len(x) == 0:
        return {
            "freshness_avg_age@N": float("nan"),
            "freshness_score_mean@N": float("nan"),
        }
    age = pd.to_numeric(x[freshness_col], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float64)
    tau = max(half_life_hours, 1e-6) / np.log(2.0)
    freshness_score = np.exp(-np.clip(age, 0.0, None) / tau)
    return {
        "freshness_avg_age@N": float(np.nanmean(age)) if np.isfinite(np.nanmean(age)) else float("nan"),
        "freshness_score_mean@N": float(np.nanmean(freshness_score)) if np.isfinite(np.nanmean(freshness_score)) else float("nan"),
    }


def _compute_popularity_metrics(
    ranked_df: pd.DataFrame,
    topn: int,
    popularity_col: str,
) -> Dict[str, float]:
    x = ranked_df[ranked_df["rank"] <= topn].copy()
    if popularity_col not in x.columns or len(x) == 0:
        return {
            "popularity_mean_raw@N": float("nan"),
            "popularity_mean_norm@N": float("nan"),
        }
    pop = pd.to_numeric(x[popularity_col], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float64)
    all_pop = pd.to_numeric(ranked_df[popularity_col], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float64)
    lo = float(np.nanmin(all_pop)) if np.isfinite(np.nanmin(all_pop)) else 0.0
    hi = float(np.nanmax(all_pop)) if np.isfinite(np.nanmax(all_pop)) else 0.0
    if hi > lo:
        pop_norm = (pop - lo) / (hi - lo)
    else:
        pop_norm = np.full_like(pop, np.nan, dtype=np.float64)
    return {
        "popularity_mean_raw@N": float(np.nanmean(pop)) if np.isfinite(np.nanmean(pop)) else float("nan"),
        "popularity_mean_norm@N": float(np.nanmean(pop_norm)) if np.isfinite(np.nanmean(pop_norm)) else float("nan"),
    }


def _compute_extra_metrics(
    ranked_df: pd.DataFrame,
    topn: int,
    freshness_col: str,
    freshness_half_life_hours: float,
    popularity_col: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(_compute_diversity_metrics(ranked_df, topn))
    out.update(_compute_freshness_metrics(ranked_df, topn, freshness_col, freshness_half_life_hours))
    out.update(_compute_popularity_metrics(ranked_df, topn, popularity_col))
    return out


def main() -> None:
    args = parse_args()
    args.lambda_mmr = float(np.clip(args.lambda_mmr, 0.0, 1.0))
    root = Path(__file__).resolve().parents[1]
    expected_dcn_test = (root / "data" / "dcnv2_full_retry_test.csv").resolve()
    if not args.allow_non_test_eval_paths and args.label_feature_path.resolve() != expected_dcn_test:
        raise ValueError(
            f"label-feature-path must be test set: {expected_dcn_test}. "
            "Use --allow-non-test-eval-paths to override."
        )

    metric_ks = [int(x) for x in _parse_csv_list(args.metric_ks)]
    sim_cat_cols = _parse_csv_list(args.sim_cat_cols)
    sim_num_cols = _parse_csv_list(args.sim_num_cols)

    scored = pd.read_csv(args.scored_candidates_path, low_memory=False)
    labels_src = ensure_labels(pd.read_csv(args.label_feature_path, low_memory=False))

    required_cols = set(PAIR_KEYS + ["score_ctr", "score_watch_greater_30s", "score_like"])
    missing = sorted(required_cols - set(scored.columns))
    if missing:
        raise ValueError(f"scored-candidates missing columns: {missing}")

    labels_cols = PAIR_KEYS + TASKS
    labels = labels_src[labels_cols].drop_duplicates(PAIR_KEYS, keep="first")
    merged = scored.merge(labels, on=PAIR_KEYS, how="inner")
    if len(merged) == 0:
        raise ValueError("No rows after joining scored candidates with labels on user/live/streamer.")

    item_feature_cols = [c for c in set(sim_cat_cols + sim_num_cols + ITEM_KEYS) if c in labels_src.columns]
    item_features = labels_src[item_feature_cols].drop_duplicates(ITEM_KEYS, keep="first")
    merged = merged.merge(item_features, on=ITEM_KEYS, how="left")

    rel_raw = (
        args.w_ctr * pd.to_numeric(merged["score_ctr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        + args.w_watch30
        * pd.to_numeric(merged["score_watch_greater_30s"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        + args.w_like * pd.to_numeric(merged["score_like"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    )
    merged["relevance_score"] = rel_raw

    baseline_rows: List[pd.DataFrame] = []
    reranked_rows: List[pd.DataFrame] = []
    for uid, g in merged.groupby("user_id", sort=False):
        local = g.sort_values("relevance_score", ascending=False).head(args.prefilter_k).copy()
        if len(local) == 0:
            continue
        baseline = local.head(args.rerank_topn).copy()
        baseline["rank"] = np.arange(1, len(baseline) + 1)
        baseline["baseline_score"] = baseline["relevance_score"].to_numpy(dtype=np.float64)
        baseline_rows.append(baseline)

        rel = _minmax01(local["relevance_score"].to_numpy(dtype=np.float64))
        sim = _build_similarity_matrix(
            local,
            cat_cols=sim_cat_cols,
            num_cols=sim_num_cols,
            cat_weight=args.sim_cat_weight,
            num_weight=args.sim_num_weight,
        )
        selected_idx = _mmr_select(rel=rel, sim=sim, lambda_mmr=args.lambda_mmr, topn=args.rerank_topn)
        selected = local.iloc[selected_idx].copy()
        selected["rank"] = np.arange(1, len(selected) + 1)
        selected["mmr_score"] = [
            args.lambda_mmr * float(rel[i])
            - (1.0 - args.lambda_mmr) * (float(np.max(sim[i, selected_idx[:j]])) if j > 0 else 0.0)
            for j, i in enumerate(selected_idx)
        ]
        reranked_rows.append(selected)

    baseline_df = pd.concat(baseline_rows, axis=0, ignore_index=True) if baseline_rows else pd.DataFrame()
    reranked = pd.concat(reranked_rows, axis=0, ignore_index=True) if reranked_rows else pd.DataFrame()
    if len(reranked) == 0 or len(baseline_df) == 0:
        raise ValueError("No baseline/reranked rows produced.")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / f"{args.output_prefix}_baseline_top{args.rerank_topn}.csv"
    rerank_path = out_dir / f"{args.output_prefix}_mmr_top{args.rerank_topn}.csv"
    baseline_df.to_csv(baseline_path, index=False)
    reranked.to_csv(rerank_path, index=False)

    baseline_metrics = _evaluate_ranked_df(
        df=baseline_df,
        score_col="baseline_score",
        tasks=TASKS,
        ks=metric_ks,
    )
    reranked_metrics = _evaluate_ranked_df(
        df=reranked,
        score_col="mmr_score",
        tasks=TASKS,
        ks=metric_ks,
    )
    metric_delta: Dict[str, float] = {}
    for k, v_after in reranked_metrics.items():
        v_before = baseline_metrics.get(k, float("nan"))
        if np.isfinite(v_after) and np.isfinite(v_before):
            metric_delta[k] = float(v_after - v_before)
        else:
            metric_delta[k] = float("nan")

    baseline_extra = _compute_extra_metrics(
        ranked_df=baseline_df,
        topn=args.rerank_topn,
        freshness_col=args.freshness_col,
        freshness_half_life_hours=args.freshness_half_life_hours,
        popularity_col=args.popularity_col,
    )
    reranked_extra = _compute_extra_metrics(
        ranked_df=reranked,
        topn=args.rerank_topn,
        freshness_col=args.freshness_col,
        freshness_half_life_hours=args.freshness_half_life_hours,
        popularity_col=args.popularity_col,
    )
    extra_delta: Dict[str, float] = {}
    for k, v_after in reranked_extra.items():
        v_before = baseline_extra.get(k, float("nan"))
        if np.isfinite(v_after) and np.isfinite(v_before):
            extra_delta[k] = float(v_after - v_before)
        else:
            extra_delta[k] = float("nan")

    summary = {
        "rows_scored_input": int(len(scored)),
        "rows_joined_for_eval": int(len(merged)),
        "rows_baseline_output": int(len(baseline_df)),
        "rows_reranked_output": int(len(reranked)),
        "users_reranked": int(reranked["user_id"].nunique()),
        "lambda_mmr": float(args.lambda_mmr),
        "prefilter_k": int(args.prefilter_k),
        "rerank_topn": int(args.rerank_topn),
        "relevance_weights": {
            "w_ctr": float(args.w_ctr),
            "w_watch30": float(args.w_watch30),
            "w_like": float(args.w_like),
        },
        "similarity": {
            "sim_cat_cols": sim_cat_cols,
            "sim_num_cols": sim_num_cols,
            "sim_cat_weight": float(args.sim_cat_weight),
            "sim_num_weight": float(args.sim_num_weight),
        },
        "candidate_conditioned_metrics": {
            "before_rerank": baseline_metrics,
            "after_rerank": reranked_metrics,
            "delta_after_minus_before": metric_delta,
        },
        "rerank_quality_metrics": {
            "before_rerank": baseline_extra,
            "after_rerank": reranked_extra,
            "delta_after_minus_before": extra_delta,
        },
    }
    summary_path = out_dir / f"{args.output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved baseline output -> {baseline_path} rows={len(baseline_df)}")
    print(f"saved reranked output -> {rerank_path} rows={len(reranked)}")
    print(f"saved summary -> {summary_path}")
    print("metrics (before -> after, delta):")
    for k in sorted(reranked_metrics.keys()):
        b = baseline_metrics.get(k, float("nan"))
        a = reranked_metrics.get(k, float("nan"))
        d = metric_delta.get(k, float("nan"))
        b_str = "nan" if np.isnan(b) else f"{b:.6f}"
        a_str = "nan" if np.isnan(a) else f"{a:.6f}"
        d_str = "nan" if np.isnan(d) else f"{d:+.6f}"
        print(f"  {k}: {b_str} -> {a_str} ({d_str})")

    print("extra metrics: diversity/freshness/popularity (before -> after, delta):")
    for k in sorted(reranked_extra.keys()):
        b = baseline_extra.get(k, float("nan"))
        a = reranked_extra.get(k, float("nan"))
        d = extra_delta.get(k, float("nan"))
        b_str = "nan" if np.isnan(b) else f"{b:.6f}"
        a_str = "nan" if np.isnan(a) else f"{a:.6f}"
        d_str = "nan" if np.isnan(d) else f"{d:+.6f}"
        print(f"  {k}: {b_str} -> {a_str} ({d_str})")


if __name__ == "__main__":
    main()
