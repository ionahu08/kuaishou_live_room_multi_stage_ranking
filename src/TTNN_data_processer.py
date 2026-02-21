"""
TTNN data processor (memory-safe):
- Stream full_data.csv in chunks
- Deduplicate by (live_id, streamer_id, user_id):
  - keep earliest clicked row if group has any click
  - otherwise keep latest impression row
- Split by date windows into train/val/test (inclusive)
- Report counts and write <prefix>_train/val/test.csv under output dir
"""

import argparse
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "full_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
KEYS = ["live_id", "streamer_id", "user_id"]
REQUIRED_COLS = KEYS + ["is_click", "imp_timestamp"]


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
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per chunk while streaming CSV.",
    )
    return parser.parse_args()


def _pick_chunk_best(chunk: pd.DataFrame, start_row: int) -> pd.DataFrame:
    # Normalize required fields for consistent comparisons.
    work = chunk[REQUIRED_COLS].copy()
    work["is_click"] = pd.to_numeric(work["is_click"], errors="coerce").fillna(0).astype("int8")
    work["_ts"] = pd.to_datetime(work["imp_timestamp"], errors="coerce")
    work["_row_num"] = range(start_row, start_row + len(work))

    clicked = work[work["is_click"] == 1]
    if not clicked.empty:
        clicked = (
            clicked.sort_values(KEYS + ["_ts", "_row_num"], ascending=[True, True, True, True, True], na_position="last")
            .drop_duplicates(KEYS, keep="first")
            .copy()
        )

    non_clicked = work[work["is_click"] != 1]
    if not non_clicked.empty:
        non_clicked = (
            non_clicked.sort_values(
                KEYS + ["_ts", "_row_num"],
                ascending=[True, True, True, False, True],
                na_position="last",
            )
            .drop_duplicates(KEYS, keep="first")
            .copy()
        )

    if clicked.empty:
        cand = non_clicked
    elif non_clicked.empty:
        cand = clicked
    else:
        clicked_keys = set(map(tuple, clicked[KEYS].itertuples(index=False, name=None)))
        non_clicked = non_clicked[
            ~non_clicked[KEYS].apply(tuple, axis=1).isin(clicked_keys)
        ]
        cand = pd.concat([clicked, non_clicked], ignore_index=True)

    ts_int = cand["_ts"].view("int64")
    cand["ts_int"] = ts_int.where(cand["_ts"].notna(), pd.NA).astype("Int64")
    cand["has_click"] = cand["is_click"].astype("int8")

    return cand[KEYS + ["has_click", "ts_int", "_row_num"]]


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS best (
            live_id TEXT NOT NULL,
            streamer_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            has_click INTEGER NOT NULL,
            ts_int INTEGER,
            row_num INTEGER NOT NULL,
            PRIMARY KEY (live_id, streamer_id, user_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_best_row_num ON best(row_num)")


def _upsert_best(conn: sqlite3.Connection, cand: pd.DataFrame) -> None:
    rows = [
        (
            str(live_id),
            str(streamer_id),
            str(user_id),
            int(has_click),
            None if pd.isna(ts_int) else int(ts_int),
            int(row_num),
        )
        for live_id, streamer_id, user_id, has_click, ts_int, row_num in cand.itertuples(index=False, name=None)
    ]

    conn.executemany(
        """
        INSERT INTO best (live_id, streamer_id, user_id, has_click, ts_int, row_num)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(live_id, streamer_id, user_id) DO UPDATE SET
            has_click = CASE
                WHEN excluded.has_click > best.has_click THEN excluded.has_click
                ELSE best.has_click
            END,
            ts_int = CASE
                WHEN excluded.has_click > best.has_click THEN excluded.ts_int
                WHEN excluded.has_click = 1 AND best.has_click = 1 THEN
                    CASE
                        WHEN best.ts_int IS NULL AND excluded.ts_int IS NOT NULL THEN excluded.ts_int
                        WHEN best.ts_int IS NOT NULL AND excluded.ts_int IS NOT NULL AND excluded.ts_int < best.ts_int THEN excluded.ts_int
                        WHEN best.ts_int = excluded.ts_int AND excluded.row_num < best.row_num THEN excluded.ts_int
                        ELSE best.ts_int
                    END
                WHEN excluded.has_click = 0 AND best.has_click = 0 THEN
                    CASE
                        WHEN best.ts_int IS NULL AND excluded.ts_int IS NOT NULL THEN excluded.ts_int
                        WHEN best.ts_int IS NOT NULL AND excluded.ts_int IS NOT NULL AND excluded.ts_int > best.ts_int THEN excluded.ts_int
                        WHEN best.ts_int = excluded.ts_int AND excluded.row_num < best.row_num THEN excluded.ts_int
                        ELSE best.ts_int
                    END
                ELSE best.ts_int
            END,
            row_num = CASE
                WHEN excluded.has_click > best.has_click THEN excluded.row_num
                WHEN excluded.has_click = 1 AND best.has_click = 1 THEN
                    CASE
                        WHEN best.ts_int IS NULL AND excluded.ts_int IS NOT NULL THEN excluded.row_num
                        WHEN best.ts_int IS NOT NULL AND excluded.ts_int IS NOT NULL AND excluded.ts_int < best.ts_int THEN excluded.row_num
                        WHEN best.ts_int = excluded.ts_int AND excluded.row_num < best.row_num THEN excluded.row_num
                        ELSE best.row_num
                    END
                WHEN excluded.has_click = 0 AND best.has_click = 0 THEN
                    CASE
                        WHEN best.ts_int IS NULL AND excluded.ts_int IS NOT NULL THEN excluded.row_num
                        WHEN best.ts_int IS NOT NULL AND excluded.ts_int IS NOT NULL AND excluded.ts_int > best.ts_int THEN excluded.row_num
                        WHEN best.ts_int = excluded.ts_int AND excluded.row_num < best.row_num THEN excluded.row_num
                        ELSE best.row_num
                    END
                ELSE best.row_num
            END
        """,
        rows,
    )


def build_best_index(input_path: Path, db_path: Path, chunksize: int):
    date_counts: Counter = Counter()
    total_rows = 0

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    _init_db(conn)

    try:
        for chunk_idx, chunk in enumerate(
            pd.read_csv(input_path, usecols=REQUIRED_COLS, chunksize=chunksize, low_memory=False)
        ):
            start_row = total_rows
            total_rows += len(chunk)

            ts = pd.to_datetime(chunk["imp_timestamp"], errors="coerce")
            date_counts.update(ts.dt.date.dropna().astype(str).tolist())

            cand = _pick_chunk_best(chunk, start_row)
            _upsert_best(conn, cand)

            if (chunk_idx + 1) % 10 == 0:
                print(f"processed chunks: {chunk_idx + 1}, rows: {total_rows}")
                conn.commit()

        conn.commit()
        dedup_rows = conn.execute("SELECT COUNT(*) FROM best").fetchone()[0]
    finally:
        conn.close()

    return total_rows, dedup_rows, date_counts


def _resolve_split_name(ts: pd.Timestamp) -> Optional[str]:
    if pd.isna(ts):
        return None

    d = ts.date()
    if pd.to_datetime("2025-05-04").date() <= d <= pd.to_datetime("2025-05-18").date():
        return "train"
    if pd.to_datetime("2025-05-19").date() <= d <= pd.to_datetime("2025-05-22").date():
        return "val"
    if d >= pd.to_datetime("2025-05-23").date():
        return "test"
    return None


def write_splits_from_index(
    input_path: Path,
    db_path: Path,
    output_dir: Path,
    output_prefix: str,
    chunksize: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_map = {
        "train": output_dir / f"{output_prefix}_train.csv",
        "val": output_dir / f"{output_prefix}_val.csv",
        "test": output_dir / f"{output_prefix}_test.csv",
    }

    # Ensure fresh outputs.
    for p in output_map.values():
        if p.exists():
            p.unlink()

    split_sizes = {"train": 0, "val": 0, "test": 0}
    split_bounds = {
        "train": [None, None],
        "val": [None, None],
        "test": [None, None],
    }
    outside_rows = 0

    conn = sqlite3.connect(db_path)
    try:
        header_written = {"train": False, "val": False, "test": False}
        start_row = 0

        for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize, low_memory=False)):
            end_row = start_row + len(chunk) - 1

            rows = conn.execute(
                "SELECT row_num FROM best WHERE row_num BETWEEN ? AND ?",
                (start_row, end_row),
            ).fetchall()
            selected = sorted(r[0] for r in rows)

            if selected:
                idx = [r - start_row for r in selected]
                picked = chunk.iloc[idx].copy()
                picked["imp_timestamp"] = pd.to_datetime(picked["imp_timestamp"], errors="coerce")

                for name in ["train", "val", "test"]:
                    mask = picked["imp_timestamp"].apply(_resolve_split_name) == name
                    part = picked.loc[mask]
                    if part.empty:
                        continue

                    out_path = output_map[name]
                    part.to_csv(out_path, mode="a", index=False, header=not header_written[name])
                    header_written[name] = True
                    split_sizes[name] += len(part)

                    min_ts = part["imp_timestamp"].min()
                    max_ts = part["imp_timestamp"].max()
                    cur_min, cur_max = split_bounds[name]
                    split_bounds[name][0] = min_ts if cur_min is None else min(cur_min, min_ts)
                    split_bounds[name][1] = max_ts if cur_max is None else max(cur_max, max_ts)

                outside_rows += picked["imp_timestamp"].apply(_resolve_split_name).isna().sum()

            start_row = end_row + 1
            if (chunk_idx + 1) % 10 == 0:
                print(f"write pass chunks: {chunk_idx + 1}, processed rows: {start_row}")

    finally:
        conn.close()

    bounds = {k: (v[0], v[1]) for k, v in split_bounds.items()}
    return split_sizes, bounds, int(outside_rows)


def main() -> None:
    args = parse_args()
    db_path = args.output_dir / f".{args.output_prefix}_dedup.sqlite"

    print(f"input: {args.input}")
    total_rows, dedup_rows, date_counts = build_best_index(args.input, db_path, args.chunksize)

    print("initial rows:", total_rows)
    print("raw rows:", total_rows)
    if date_counts:
        date_series = pd.Series(date_counts).sort_index()
        print("imp_timestamp")
        print(date_series.tail(10))

    print("dedup rows:", dedup_rows)
    print("rows removed by dedup:", total_rows - dedup_rows)
    print("duplicates after dedup:", 0)

    split_sizes, split_bounds, outside_rows = write_splits_from_index(
        args.input,
        db_path,
        args.output_dir,
        args.output_prefix,
        args.chunksize,
    )

    print("split sizes:", split_sizes)
    print("sum(rows removed + splits):", (total_rows - dedup_rows) + sum(split_sizes.values()))
    for name in ["train", "val", "test"]:
        min_ts, max_ts = split_bounds[name]
        print(f"{name} rows:", split_sizes[name], "min_ts:", min_ts, "max_ts:", max_ts)

    if outside_rows > 0:
        print("rows outside split windows:", outside_rows)

    for name in ["train", "val", "test"]:
        out_path = args.output_dir / f"{args.output_prefix}_{name}.csv"
        print(f"saved {name} -> {out_path}")

    if db_path.exists():
        db_path.unlink()
    print("finish!!!!")


if __name__ == "__main__":
    main()
