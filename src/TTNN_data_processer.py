"""
TTNN data processor:
- Load draft_sample.csv and parse imp_timestamp
- Deduplicate by (live_id, streamer_id, user_id) keeping first click if any, else latest impression
- Split by date windows into train/val/test (date-only, inclusive)
- Report counts and write TTNN_train/val/test CSVs under data/
"""

import argparse
from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "draft_sample.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test using TTNN rules.")
    parser.add_argument("--input", type=Path, default=DATA_PATH, help="Input CSV path.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="TTNN",
        help="Output filename prefix; files are <prefix>_train.csv, <prefix>_val.csv, <prefix>_test.csv.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["imp_timestamp"] = pd.to_datetime(df["imp_timestamp"], errors="coerce")
    print("initial rows:", len(df))
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["live_id", "streamer_id", "user_id"]

    df = df.sort_values(["live_id", "streamer_id", "user_id", "imp_timestamp"])

    # mark whether group has any click
    df["_has_click"] = df.groupby(keys)["is_click"].transform("max")

    # priority:
    # 1) clicked rows (if exist)
    # 2) otherwise latest impression
    df["_priority"] = (
        df["_has_click"] * 2 +
        df["is_click"]
    )

    dedup = (
        df.sort_values(["_priority", "imp_timestamp"], ascending=[False, True])
          .drop_duplicates(keys, keep="first")
          .drop(columns=["_has_click", "_priority"])
    )

    return dedup



def split_by_date(df: pd.DataFrame) -> dict:
    df = df.sort_values("imp_timestamp")

    df["imp_date"] = df["imp_timestamp"].dt.date

    train_mask = (
        (df["imp_date"] >= pd.to_datetime("2025-05-04").date())
        & (df["imp_date"] <= pd.to_datetime("2025-05-18").date())
    )
    val_mask = (
        (df["imp_date"] >= pd.to_datetime("2025-05-19").date())
        & (df["imp_date"] <= pd.to_datetime("2025-05-22").date())
    )
    test_mask = (
        (df["imp_date"] >= pd.to_datetime("2025-05-23").date())
    )

    covered_mask = train_mask | val_mask | test_mask
    outside = df[~covered_mask]
    if len(outside) > 0:
        min_ts = outside["imp_timestamp"].min()
        max_ts = outside["imp_timestamp"].max()
        print("rows outside split windows:", len(outside), "min_ts:", min_ts, "max_ts:", max_ts)

    return {
        "train": df.loc[train_mask].drop(columns=["imp_date"]).copy(),
        "val": df.loc[val_mask].drop(columns=["imp_date"]).copy(),
        "test": df.loc[test_mask].drop(columns=["imp_date"]).copy(),
    }


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    print("raw rows:", len(df))
    print(df["imp_timestamp"].dt.date.value_counts().sort_index().tail(10))


    dedup = deduplicate(df)
    print("dedup rows:", len(dedup))
    print("rows removed by dedup:", len(df) - len(dedup))
    dup_check = dedup.duplicated(subset=["live_id", "streamer_id", "user_id"]).sum()
    print("duplicates after dedup:", dup_check)

    splits = split_by_date(dedup)
    split_sizes = {name: len(sdf) for name, sdf in splits.items()}
    print("split sizes:", split_sizes)
    print("sum(rows removed + splits):", (len(df) - len(dedup)) + sum(split_sizes.values()))
    for name, sdf in splits.items():
        min_ts = sdf["imp_timestamp"].min()
        max_ts = sdf["imp_timestamp"].max()
        print(f"{name} rows:", len(sdf), "min_ts:", min_ts, "max_ts:", max_ts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_map = {
        "train": args.output_dir / f"{args.output_prefix}_train.csv",
        "val": args.output_dir / f"{args.output_prefix}_val.csv",
        "test": args.output_dir / f"{args.output_prefix}_test.csv",
    }
    for name, sdf in splits.items():
        out_path = output_map[name]
        sdf.to_csv(out_path, index=False)
        print(f"saved {name} -> {out_path}")
        print("finish!!!!")


if __name__ == "__main__":
    main()
