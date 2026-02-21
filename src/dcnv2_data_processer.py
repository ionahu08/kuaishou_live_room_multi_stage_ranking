from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from TTNN_data_processer import deduplicate, split_by_date
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _NoOpTqdm:
        def update(self, n: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NoOpTqdm()

# Requested feature set (from provided spec/screenshot), plus required labels.
USER_CATEGORICAL = [
    "user_id",
    "user_age_le",
    "user_gender_le",
    "user_country_le",
    "user_device_brand_le",
    "user_device_price_le",
    "fans_num_le",
    "follow_num_le",
    "accu_watch_live_cnt_le",
    "accu_watch_live_duration_le",
    "is_live_streamer",
    "is_photo_author",
    "user_onehot_feat0",
    "user_onehot_feat1",
    "user_onehot_feat2",
    "user_onehot_feat3",
    "user_onehot_feat4",
    "user_onehot_feat5",
    "user_onehot_feat6",
]

USER_NUMERIC = [
    "user_account_age",
    "user_watch_live_age",
    "ctr_user_15min",
    "ctr_user_3hr",
    "ctr_user_1d",
    "ctr_user_7d",
    "num_imp_user_10min",
    "num_imp_user_30min",
    "num_imp_user_2hr",
    "num_imp_user_12hr",
    "num_imp_user_1d",
    "num_imp_user_7d",
    "num_click_user_15min",
    "num_click_user_3hr",
    "num_click_user_1d",
    "num_click_user_7d",
    "click_trend_user",
    "time_since_last_impression_user",
    "time_since_last_click_user",
    "tsli_missing",
    "tslc_missing",
    "consecutive_skips_user",
    "avg_watch_time_user",
    "avg_watch_time_user_missing",
    "median_watch_time_user",
    "median_watch_time_user_missing",
    "pct_long_watch_user_30s",
    "comment_rate_user",
    "has_comment_user_24h",
    "num_comment_user_24h",
    "like_rate_user",
    "has_like_user_24h",
    "num_like_user_24h",
    "has_gift_user_7d",
    "num_gift_user_7d",
    "amount_gift_user_7d",
]

ROOM_STREAMER_CATEGORICAL = [
    "live_id",
    "streamer_id",
    "live_type",
    "live_content_category_le",
    "live_start_year",
    "live_start_month",
    "live_start_day",
    "live_start_hour",
    "streamer_age_le",
    "streamer_country_le",
    "streamer_device_brand_le",
    "streamer_device_price_le",
    "live_operation_tag_le",
    "fans_user_num_le",
    "fans_group_fans_num_le",
    "follow_user_num_le",
    "accu_live_cnt_le",
    "accu_live_duration_le",
    "accu_play_cnt_le",
    "accu_play_duration_le",
    "streamer_gender_le",
    "live_is_weekend",
    "title_emb_missing",
] + [f"title_emb_{i}" for i in range(128)] + [f"streamer_onehot_feat{i}" for i in range(7)]

ROOM_STREAMER_NUMERIC = [
    "time_since_live_start",
    "ctr_room_10min",
    "ctr_room_30min",
    "ctr_room_2hr",
    "ctr_room_12hr",
    "num_imp_room_10min",
    "num_imp_room_30min",
    "num_imp_room_2hr",
    "num_imp_room_12hr",
    "num_imp_room_1d",
    "num_click_room_10min",
    "num_click_room_30min",
    "num_click_room_2hr",
    "num_click_room_12hr",
    "num_click_room_1d",
    "ctr_trend_room",
    "time_since_start_live",
    "time_since_start_live_bucket",
    "avg_watch_time_live",
    "median_watch_time_live",
    "avg_watch_time_live_30min",
    "median_watch_time_live_30min",
    "watch_time_live_30min_missing",
    "watch_time_live_missing",
    "pct_long_watch_live_60s_30min",
    "comment_rate_live",
    "comment_rate_live_15min",
    "comment_rate_live_1hr",
    "comment_rate_live_3hr",
    "num_comment_live",
    "num_comment_live_15min",
    "num_comment_live_1hr",
    "num_comment_live_3hr",
    "comment_trend_room",
    "like_rate_live",
    "like_rate_live_15min",
    "like_rate_live_1hr",
    "like_rate_live_3hr",
    "num_like_live",
    "num_like_live_15min",
    "num_like_live_1hr",
    "num_like_live_3hr",
    "like_trend_room",
    "gift_rate_live",
    "gift_rate_live_15min",
    "gift_rate_live_1hr",
    "gift_rate_live_3hr",
    "num_gift_live",
    "num_gift_live_15min",
    "num_gift_live_1hr",
    "num_gift_live_3hr",
    "amount_gift_live",
    "amount_gift_live_15min",
    "amount_gift_live_1hr",
    "amount_gift_live_3hr",
    "gift_trend_room",
    "streamer_account_age",
    "streamer_live_age",
    "ctr_streamer_1d",
    "ctr_streamer_7d",
    "num_imp_streamer_7d",
    "num_click_streamer_7d",
    "num_lives_streamer_7d",
    "avg_watch_time_streamer",
    "median_watch_time_streamer",
    "pct_long_watch_streamer_30s",
    "watch_time_streamer_missing",
    "num_comment_streamer_7d",
    "num_like_streamer_7d",
    "amount_gift_streamer_7d",
    "ctr_user_streamer_7d",
    "num_click_user_streamer_7d",
    "num_imp_user_streamer_7d",
    "time_since_last_impression_user_streamer",
    "time_since_last_click_user_streamer",
    "tsli_user_streamer_missing",
    "tslc_user_streamer_missing",
    "ctr_user_category_7d",
    "num_click_user_category_7d",
    "num_imp_user_category_7d",
]

INTERACTION_CATEGORICAL = [
    "imp_year",
    "imp_month",
    "imp_day",
    "imp_hour",
    "imp_is_weekend",
]

BINARY_LABEL_COLUMNS = ["is_click", "watch_greater_30s", "is_like"]
REGRESSION_LABEL_COLUMNS = ["watch_live_time"]
LABEL_COLUMNS = BINARY_LABEL_COLUMNS + REGRESSION_LABEL_COLUMNS

# Name reconciliation: spec name -> actual CSV column
ALIAS_MAP = {
    "time_since_live_start [ms]": "time_since_live_start",
    "ctr_trend_room": "ctr_trend_room",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare DCNv2 train/val/test splits from draft_sample.csv using TTNN date rules."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "draft_sample.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data",
        help="Output directory.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dcnv2",
        help="Output filename prefix. Creates <prefix>_train.csv, <prefix>_val.csv, <prefix>_test.csv.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Enable deduplication before splitting (disabled by default for ranking stage).",
    )
    parser.add_argument(
        "--read-chunksize",
        type=int,
        default=1_000_000,
        help="CSV read chunksize for progress visualization. Set <=0 to disable chunked reading.",
    )
    return parser.parse_args()


def read_csv_with_progress(path: Path, chunksize: int) -> pd.DataFrame:
    if chunksize <= 0:
        return pd.read_csv(path, low_memory=False)

    chunks: List[pd.DataFrame] = []
    for chunk in tqdm(
        pd.read_csv(path, low_memory=False, chunksize=chunksize),
        desc="read_csv",
        unit="chunk",
    ):
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def add_multitask_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_click" in out.columns:
        out["is_click"] = pd.to_numeric(out["is_click"], errors="coerce").fillna(0).astype("int64")
    if "is_like" in out.columns:
        out["is_like"] = pd.to_numeric(out["is_like"], errors="coerce").fillna(0).astype("int64")
    out["watch_live_time"] = pd.to_numeric(out.get("watch_live_time"), errors="coerce").fillna(0).clip(lower=0)
    if "watch_greater_30s" not in out.columns:
        watch = out["watch_live_time"]
        out["watch_greater_30s"] = (watch >= 30).astype("int64")
    else:
        out["watch_greater_30s"] = pd.to_numeric(out["watch_greater_30s"], errors="coerce").fillna(0).astype("int64")
    return out


def _print_split_summary(name: str, df: pd.DataFrame) -> None:
    label_cols = [c for c in LABEL_COLUMNS if c in df.columns]
    summary: Dict[str, object] = {
        "rows": int(len(df)),
        "total_columns": int(df.shape[1]),
        "label_columns": label_cols,
        "num_labels": int(len(label_cols)),
        "num_features": int(df.shape[1] - len(label_cols)),
    }
    if "imp_timestamp" in df.columns:
        summary["min_ts"] = str(df["imp_timestamp"].min())
        summary["max_ts"] = str(df["imp_timestamp"].max())
    for col in BINARY_LABEL_COLUMNS:
        if col in df.columns and len(df) > 0:
            summary[f"{col}_rate"] = float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())
    if "watch_live_time" in df.columns and len(df) > 0:
        s = pd.to_numeric(df["watch_live_time"], errors="coerce").fillna(0)
        summary["watch_live_time_mean"] = float(s.mean())
        summary["watch_live_time_std"] = float(s.std(ddof=0))
    for col in REGRESSION_LABEL_COLUMNS:
        if len(df) > 0:
            # keep explicit mean reporting for regression labels as task-level sanity check
            if col in df.columns:
                summary[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())
    print(f"{name}: {summary}")


def _filter_to_requested_columns(df: pd.DataFrame) -> pd.DataFrame:
    requested = (
        USER_CATEGORICAL
        + USER_NUMERIC
        + ROOM_STREAMER_CATEGORICAL
        + ROOM_STREAMER_NUMERIC
        + INTERACTION_CATEGORICAL
        + LABEL_COLUMNS
    )
    resolved = [ALIAS_MAP.get(c, c) for c in requested]
    keep_cols = [c for c in resolved if c in df.columns]
    missing = [c for c in resolved if c not in df.columns]
    if missing:
        print(f"[dcnv2_data_processer] warning: {len(missing)} requested columns missing")
        print(f"[dcnv2_data_processer] missing sample: {missing[:20]}")
    # preserve order, drop duplicates
    seen = set()
    ordered = []
    for c in keep_cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return df[ordered].copy()


def main() -> None:
    args = parse_args()
    stage_bar = tqdm(total=6, desc="dcnv2_data_processer", unit="stage")

    print(f"[dcnv2_data_processer] reading: {args.input}")
    df = read_csv_with_progress(args.input, chunksize=args.read_chunksize)
    if "imp_timestamp" not in df.columns:
        raise ValueError("`imp_timestamp` column is required.")
    df["imp_timestamp"] = pd.to_datetime(df["imp_timestamp"], errors="coerce")
    print(f"[dcnv2_data_processer] raw rows={len(df)} cols={df.shape[1]}")
    stage_bar.update(1)

    if args.dedup:
        base_df = deduplicate(df)
        print(f"[dcnv2_data_processer] dedup rows={len(base_df)} removed={len(df) - len(base_df)}")
    else:
        base_df = df
        print("[dcnv2_data_processer] dedup skipped (default)")
    stage_bar.update(1)

    base_df = add_multitask_labels(base_df)
    stage_bar.update(1)
    splits = split_by_date(base_df)
    stage_bar.update(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_map = {
        "train": args.output_dir / f"{args.output_prefix}_train.csv",
        "val": args.output_dir / f"{args.output_prefix}_val.csv",
        "test": args.output_dir / f"{args.output_prefix}_test.csv",
    }
    for split_name, split_df in tqdm(splits.items(), desc="save_splits", unit="split"):
        filtered = _filter_to_requested_columns(split_df)
        _print_split_summary(split_name, filtered)
        out_path = output_map[split_name]
        filtered.to_csv(out_path, index=False)
        print(f"[dcnv2_data_processer] saved {split_name} -> {out_path}")
    stage_bar.update(2)
    stage_bar.close()


if __name__ == "__main__":
    main()
