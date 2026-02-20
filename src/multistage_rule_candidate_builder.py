from __future__ import annotations

"""
multistage_rule_candidate_builder.py

Purpose
-------
Build stage-1 rule-based user-item candidates from full features, then evaluate
candidate coverage with recall@K on the test split.

Step-by-step workflow
---------------------
1) Load input CSV
   - Reads: data/full_sample_all_features.csv
   - Uses chunked reading to handle large files.

2) Keep only test rows
   - Constructs `live_start_date` from:
     live_start_year, live_start_month, live_start_day
   - Keeps rows where:
     live_start_date >= pd.to_datetime("2025-05-23").date()
   - Date can be changed via --test-live-start-date.

3) Build unique user and item lists
   - Unique users from `user_id`
   - Unique items from `live_id` (plus item attributes)

4) Build deduplicated ground-truth user-item labels
   - Dedup key: (user_id, streamer_id, live_id)
   - Label rule:
     if a user clicked at least once in duplicates -> positive (1),
     otherwise negative (0)
   - For user-item pairs not appearing in test data, they are treated as
     negative by default when evaluating recall.

5) Pair users with items conceptually (candidate generation)
   - For each user, start from the filtered global item pool.
   - This is pairwise generation per user over candidate items.

6) Apply initial rule filters (editable starter rules)
   - Rule A: Keep only recent lives (`--max-live-age-days`, default 14)
   - Rule B: Keep expected live_type values {1,2,3} (if numeric)
   - Rule C: Remove self recommendations (streamer_id != user_id)
   - Rule D: Country match by default
            (user_country_le == streamer_country_le)
            Disable with --allow-cross-country
   - Then compute a simple rule score from popularity/CTR proxies and keep top-K
     per user (`--topk`, default 300).

7) Save outputs under data/
   - <prefix>_candidates_topK.csv
     Row-level candidate table per user-item with rank and rule score.
   - <prefix>_user_item_list_topK.csv
     One row per user with a pipe-separated item list after filtering.
   - <prefix>_recall_at_K_by_user.csv
     Per-user recall@K using the deduped positive labels.

8) Report recall@K
   - Prints mean recall@K across users with at least one positive.

Notes
-----
- This script is a stage-1 rule-based candidate builder for multi-stage ranking.
- Rule logic is intentionally simple and intended to be expanded.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Build rule-filtered user-item candidates from full_sample_all_features.csv "
            "and compute recall@300 per user on test split."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "full_sample_all_features.csv",
        help="Input full feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data",
        help="Output directory for candidate and recall files.",
    )
    parser.add_argument(
        "--test-live-start-date",
        type=str,
        default="2025-05-23",
        help="Keep only rows with live_start_date >= this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=300,
        help="Top-K candidate size per user for recall computation.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Read chunk size for large CSV.",
    )
    parser.add_argument(
        "--max-live-age-days",
        type=int,
        default=14,
        help="Rule: filter out lives older than this many days from test_live_start_date.",
    )
    parser.add_argument(
        "--allow-cross-country",
        action="store_true",
        help="If set, disable user_country == streamer_country rule.",
    )
    parser.add_argument(
        "--max-age-bucket-gap",
        type=int,
        default=3,
        help="Rule: keep candidate only if |user_age_le - streamer_age_le| <= this gap (negative disables).",
    )
    parser.add_argument(
        "--max-device-price-gap",
        type=int,
        default=3,
        help="Rule: keep candidate only if |user_device_price_le - streamer_device_price_le| <= this gap (negative disables).",
    )
    parser.add_argument(
        "--strict-device-brand",
        action="store_true",
        help="Rule: require user_device_brand_le == streamer_device_brand_le (unknown values are allowed).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="stage1_rule",
        help="Prefix for output files.",
    )
    return parser.parse_args()


def required_columns() -> List[str]:
    # Keep only needed columns to reduce memory pressure.
    cols = [
        "user_id",
        "live_id",
        "streamer_id",
        "is_click",
        "live_start_year",
        "live_start_month",
        "live_start_day",
        "live_type",
        "live_content_category_le",
        "live_operation_tag_le",
        "user_country_le",
        "user_age_le",
        "user_device_brand_le",
        "user_device_price_le",
        "streamer_country_le",
        "streamer_age_le",
        "streamer_device_brand_le",
        "streamer_device_price_le",
        "fans_user_num_le",
        "follow_user_num_le",
        "accu_live_cnt_le",
        "accu_live_duration_le",
        "accu_play_duration_le",
        "time_since_live_start",
        "title_emb_missing",
        # optional popularity signals
        "num_click_room_1d",
        "num_imp_room_1d",
        "ctr_room_12hr",
        "ctr_room_2hr",
        "accu_play_cnt_le",
    ]
    return cols


def coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype(np.int64)


def build_live_start_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        dict(
            year=pd.to_numeric(df.get("live_start_year"), errors="coerce"),
            month=pd.to_numeric(df.get("live_start_month"), errors="coerce"),
            day=pd.to_numeric(df.get("live_start_day"), errors="coerce"),
        ),
        errors="coerce",
    ).dt.date


def normalize01(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mn = float(x.min())
    mx = float(x.max())
    if mx <= mn:
        return pd.Series(np.zeros(len(x), dtype=np.float64), index=x.index)
    return (x - mn) / (mx - mn)


def compute_item_score(items: pd.DataFrame) -> pd.Series:
    # Rule/retrieval bootstrap scoring: popularity + short-term ctr proxies.
    score = pd.Series(np.zeros(len(items), dtype=np.float64), index=items.index)

    if "num_click_room_1d" in items.columns:
        clicks = pd.to_numeric(items["num_click_room_1d"], errors="coerce").fillna(0.0).clip(lower=0.0)
        score += 0.45 * normalize01(np.log1p(clicks))

    if "num_imp_room_1d" in items.columns:
        imps = pd.to_numeric(items["num_imp_room_1d"], errors="coerce").fillna(0.0).clip(lower=0.0)
        score += 0.10 * normalize01(np.log1p(imps))

    if "ctr_room_12hr" in items.columns:
        score += 0.25 * normalize01(items["ctr_room_12hr"])

    if "ctr_room_2hr" in items.columns:
        score += 0.15 * normalize01(items["ctr_room_2hr"])

    if "accu_play_cnt_le" in items.columns:
        score += 0.05 * normalize01(items["accu_play_cnt_le"])

    if "fans_user_num_le" in items.columns:
        score += 0.05 * normalize01(items["fans_user_num_le"])

    if "follow_user_num_le" in items.columns:
        score += 0.03 * normalize01(items["follow_user_num_le"])

    if "accu_live_cnt_le" in items.columns:
        score += 0.03 * normalize01(items["accu_live_cnt_le"])

    if "accu_play_duration_le" in items.columns:
        score += 0.03 * normalize01(items["accu_play_duration_le"])

    if "time_since_live_start" in items.columns:
        t = pd.to_numeric(items["time_since_live_start"], errors="coerce").fillna(0.0).clip(lower=0.0)
        score += 0.08 * (1.0 - normalize01(t))

    if "title_emb_missing" in items.columns:
        miss = pd.to_numeric(items["title_emb_missing"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        score -= 0.05 * miss

    return score


def read_and_filter_test_rows(args: argparse.Namespace) -> pd.DataFrame:
    test_date = pd.to_datetime(args.test_live_start_date).date()
    usecols = required_columns()

    chunks: List[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0

    for chunk in pd.read_csv(args.input, usecols=lambda c: c in set(usecols), chunksize=args.chunksize, low_memory=False):
        total_rows += len(chunk)
        chunk["live_start_date"] = build_live_start_date(chunk)
        mask = chunk["live_start_date"] >= test_date
        kept = chunk.loc[mask].copy()
        kept_rows += len(kept)
        if len(kept) > 0:
            chunks.append(kept)

    if not chunks:
        raise ValueError("No rows matched test filter. Check --test-live-start-date and input data.")

    out = pd.concat(chunks, ignore_index=True)
    out["user_id"] = coerce_int_series(out["user_id"])
    out["live_id"] = coerce_int_series(out["live_id"])
    out["streamer_id"] = coerce_int_series(out["streamer_id"])
    out["is_click"] = pd.to_numeric(out.get("is_click"), errors="coerce").fillna(0).astype(np.int8)

    print(f"[info] total rows scanned: {total_rows}")
    print(f"[info] test rows kept (live_start_date >= {test_date}): {kept_rows}")
    print(f"[info] test dataframe rows after concat: {len(out)}")
    return out


def dedup_positive_pairs(test_df: pd.DataFrame) -> pd.DataFrame:
    # If a pair has any click=1, treat pair as positive and ignore negative duplicates.
    keys = ["user_id", "streamer_id", "live_id"]
    dedup = (
        test_df.groupby(keys, as_index=False)["is_click"]
        .max()
        .rename(columns={"is_click": "is_positive"})
    )
    dedup["is_positive"] = dedup["is_positive"].astype(np.int8)
    return dedup


def build_user_profile(test_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "user_id",
            "user_country_le",
            "user_age_le",
            "user_device_brand_le",
            "user_device_price_le",
            "user_watch_live_age",
        ]
        if c in test_df.columns
    ]
    u = test_df[cols].drop_duplicates(subset=["user_id"]).copy()
    u["user_id"] = coerce_int_series(u["user_id"])
    return u


def build_item_profile(test_df: pd.DataFrame) -> pd.DataFrame:
    # Keep one row per live_id (latest known row by live_start_date not required here).
    item_cols = [
        "live_id",
        "streamer_id",
        "live_type",
        "live_content_category_le",
        "live_operation_tag_le",
        "streamer_country_le",
        "streamer_age_le",
        "streamer_device_brand_le",
        "streamer_device_price_le",
        "live_start_date",
        "fans_user_num_le",
        "follow_user_num_le",
        "accu_live_cnt_le",
        "accu_live_duration_le",
        "accu_play_duration_le",
        "time_since_live_start",
        "title_emb_missing",
        "num_click_room_1d",
        "num_imp_room_1d",
        "ctr_room_12hr",
        "ctr_room_2hr",
        "accu_play_cnt_le",
    ]
    item_cols = [c for c in item_cols if c in test_df.columns]
    items = test_df[item_cols].drop_duplicates(subset=["live_id"]).copy()
    items["live_id"] = coerce_int_series(items["live_id"])
    items["streamer_id"] = coerce_int_series(items["streamer_id"])
    return items


def apply_global_item_rules(items: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    ref_date = pd.to_datetime(args.test_live_start_date).date()
    out = items.copy()

    # Rule A: keep only recent lives.
    if "live_start_date" in out.columns:
        min_date = (pd.Timestamp(ref_date) - pd.Timedelta(days=args.max_live_age_days)).date()
        out = out[(out["live_start_date"] >= min_date) & (out["live_start_date"] <= ref_date)].copy()

    # Rule B: optionally keep only expected live types (example: 1/2/3 if numeric).
    if "live_type" in out.columns:
        live_type_numeric = pd.to_numeric(out["live_type"], errors="coerce")
        keep_mask = live_type_numeric.isna() | live_type_numeric.isin([1, 2, 3])
        out = out[keep_mask].copy()

    out["rule_score"] = compute_item_score(out)
    out = out.sort_values("rule_score", ascending=False).reset_index(drop=True)
    return out


def build_user_candidate_topk(
    users: pd.DataFrame,
    items: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    # Conceptual pairwise user-item generation with rule filtering done per user.
    out_rows: List[pd.DataFrame] = []
    topk = int(args.topk)

    base_cols = ["live_id", "streamer_id", "rule_score"]
    if "streamer_country_le" in items.columns:
        base_cols.append("streamer_country_le")
    if "streamer_age_le" in items.columns:
        base_cols.append("streamer_age_le")
    if "streamer_device_brand_le" in items.columns:
        base_cols.append("streamer_device_brand_le")
    if "streamer_device_price_le" in items.columns:
        base_cols.append("streamer_device_price_le")
    item_base = items[base_cols].copy()

    for row in users.itertuples(index=False):
        uid = int(getattr(row, "user_id"))
        user_country = getattr(row, "user_country_le", np.nan)
        user_age = getattr(row, "user_age_le", np.nan)
        user_device_brand = getattr(row, "user_device_brand_le", np.nan)
        user_device_price = getattr(row, "user_device_price_le", np.nan)

        cand = item_base.copy()

        # Rule C: avoid self-recommendation (if id-space overlaps).
        cand = cand[cand["streamer_id"] != uid]

        # Rule D: country match (or unknown), unless disabled.
        if (not args.allow_cross_country) and ("streamer_country_le" in cand.columns):
            if pd.notna(user_country):
                sc = pd.to_numeric(cand["streamer_country_le"], errors="coerce")
                uc = pd.to_numeric(pd.Series([user_country]), errors="coerce").iloc[0]
                cand = cand[(sc.isna()) | (sc == uc)]

        # Rule E: age-bucket compatibility.
        if args.max_age_bucket_gap >= 0 and ("streamer_age_le" in cand.columns) and pd.notna(user_age):
            sa = pd.to_numeric(cand["streamer_age_le"], errors="coerce")
            ua = pd.to_numeric(pd.Series([user_age]), errors="coerce").iloc[0]
            cand = cand[(sa.isna()) | (np.abs(sa - ua) <= args.max_age_bucket_gap)]

        # Rule F: device-brand compatibility (optional strict mode).
        if args.strict_device_brand and ("streamer_device_brand_le" in cand.columns) and pd.notna(user_device_brand):
            sb = pd.to_numeric(cand["streamer_device_brand_le"], errors="coerce")
            ub = pd.to_numeric(pd.Series([user_device_brand]), errors="coerce").iloc[0]
            cand = cand[(sb.isna()) | (sb == ub)]

        # Rule G: device-price bucket compatibility.
        if args.max_device_price_gap >= 0 and ("streamer_device_price_le" in cand.columns) and pd.notna(user_device_price):
            sp = pd.to_numeric(cand["streamer_device_price_le"], errors="coerce")
            up = pd.to_numeric(pd.Series([user_device_price]), errors="coerce").iloc[0]
            cand = cand[(sp.isna()) | (np.abs(sp - up) <= args.max_device_price_gap)]

        cand = cand.sort_values("rule_score", ascending=False).head(topk).copy()
        cand["user_id"] = uid
        cand["rank"] = np.arange(1, len(cand) + 1, dtype=np.int32)
        out_rows.append(cand[["user_id", "live_id", "streamer_id", "rank", "rule_score"]])

    if not out_rows:
        return pd.DataFrame(columns=["user_id", "live_id", "streamer_id", "rank", "rule_score"])
    return pd.concat(out_rows, ignore_index=True)


def compute_recall_at_k_by_user(
    candidates: pd.DataFrame,
    pair_labels: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    positives = pair_labels[pair_labels["is_positive"] == 1][["user_id", "live_id"]].copy()
    pos_cnt = positives.groupby("user_id")["live_id"].nunique().rename("num_pos")

    pred = candidates[candidates["rank"] <= k][["user_id", "live_id"]].drop_duplicates()
    hit = pred.merge(positives, on=["user_id", "live_id"], how="inner")
    hit_cnt = hit.groupby("user_id")["live_id"].nunique().rename("num_hit")

    recall = pd.concat([pos_cnt, hit_cnt], axis=1).fillna(0).reset_index()
    recall["num_pos"] = recall["num_pos"].astype(np.int64)
    recall["num_hit"] = recall["num_hit"].astype(np.int64)
    recall[f"recall@{k}"] = np.where(
        recall["num_pos"] > 0,
        recall["num_hit"] / recall["num_pos"],
        np.nan,
    )
    return recall.sort_values("user_id").reset_index(drop=True)


def build_user_item_list_csv(candidates: pd.DataFrame, topk: int) -> pd.DataFrame:
    c = candidates[candidates["rank"] <= topk].sort_values(["user_id", "rank"]).copy()
    agg = (
        c.groupby("user_id", as_index=False)
        .agg(
            candidate_count=("live_id", "nunique"),
            item_list=("live_id", lambda x: "|".join(str(int(v)) for v in x)),
        )
    )
    return agg


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_df = read_and_filter_test_rows(args)
    pair_labels = dedup_positive_pairs(test_df)
    users = build_user_profile(test_df)
    items = build_item_profile(test_df)

    print(f"[info] unique users (test): {users['user_id'].nunique()}")
    print(f"[info] unique items/live_id (test): {items['live_id'].nunique()}")
    print(f"[info] dedup user-streamer-live pairs: {len(pair_labels)}")
    print(f"[info] positive pairs after dedup: {(pair_labels['is_positive'] == 1).sum()}")

    items_filtered = apply_global_item_rules(items, args)
    print(f"[info] items after global rules: {items_filtered['live_id'].nunique()}")

    candidates = build_user_candidate_topk(users, items_filtered, args)
    candidates = candidates.drop_duplicates(subset=["user_id", "live_id"]).copy()
    print(f"[info] generated user-item candidates: {len(candidates)}")

    # mark candidate positives for easier debugging
    cand_eval = candidates.merge(
        pair_labels[["user_id", "live_id", "is_positive"]],
        on=["user_id", "live_id"],
        how="left",
    )
    cand_eval["is_positive"] = cand_eval["is_positive"].fillna(0).astype(np.int8)

    # user-level item list after filtering (requested output)
    user_item_list = build_user_item_list_csv(candidates, args.topk)

    recall_df = compute_recall_at_k_by_user(candidates, pair_labels, args.topk)
    mean_recall = float(recall_df[f"recall@{args.topk}"].dropna().mean()) if len(recall_df) > 0 else float("nan")
    print(f"[result] mean recall@{args.topk} across users with positives: {mean_recall:.6f}")

    cand_path = args.output_dir / f"{args.output_prefix}_candidates_top{args.topk}.csv"
    user_list_path = args.output_dir / f"{args.output_prefix}_user_item_list_top{args.topk}.csv"
    recall_path = args.output_dir / f"{args.output_prefix}_recall_at_{args.topk}_by_user.csv"

    cand_eval.to_csv(cand_path, index=False)
    user_item_list.to_csv(user_list_path, index=False)
    recall_df.to_csv(recall_path, index=False)

    print(f"[saved] candidates: {cand_path}")
    print(f"[saved] user item list: {user_list_path}")
    print(f"[saved] recall by user: {recall_path}")


if __name__ == "__main__":
    main()
