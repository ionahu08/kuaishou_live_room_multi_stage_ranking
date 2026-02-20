from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


def log(msg: str) -> None:
    print(f"[data_processing] {msg}")


def load_raw_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    log(f"Loading raw files from: {data_dir}")
    dataframes: Dict[str, pd.DataFrame] = {}

    paths = sorted(data_dir.iterdir())
    for path in tqdm(paths, desc="Load raw files", unit="file"):
        if path.suffix.lower() == ".csv":
            dataframes[path.stem] = pd.read_csv(path)
        elif path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=True)
            if arr.ndim == 1:
                dataframes[path.stem] = pd.DataFrame(arr, columns=[path.stem])
            else:
                dataframes[path.stem] = pd.DataFrame(arr)

    log(f"Loaded {len(dataframes)} tables")
    return dataframes


def build_interactions(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    log("Building interaction table (click + like/comment/gift + negative)")

    df_click = dataframes.get("click")
    if df_click is not None:
        df_click = df_click.copy()
        if "timestamp" in df_click.columns:
            df_click = df_click.rename(columns={"timestamp": "imp_timestamp"})
            df_click["imp_timestamp"] = pd.to_datetime(df_click["imp_timestamp"], unit="ms", errors="coerce")
        df_click["is_click"] = 1

    df_like = dataframes.get("like")
    if df_like is not None:
        df_like = df_like.copy()
        if "timestamp" in df_like.columns:
            df_like = df_like.rename(columns={"timestamp": "like_timestamp"})
            df_like["like_timestamp"] = pd.to_datetime(df_like["like_timestamp"], unit="ms", errors="coerce")
        df_like["is_like"] = 1

    key_cols = ["user_id", "live_id", "streamer_id"]
    for c in key_cols:
        df_click[c] = pd.to_numeric(df_click[c], errors="coerce").astype("int64")
        df_like[c] = pd.to_numeric(df_like[c], errors="coerce").astype("int64")

    df_click["imp_timestamp"] = pd.to_datetime(df_click["imp_timestamp"], errors="coerce")
    df_like["like_timestamp"] = pd.to_datetime(df_like["like_timestamp"], errors="coerce")

    df_click_m = df_click.dropna(subset=key_cols + ["imp_timestamp"]).copy()
    df_like_m = df_like.dropna(subset=key_cols + ["like_timestamp"]).copy()

    df_click_m = df_click_m.sort_values(key_cols + ["imp_timestamp"], kind="mergesort").reset_index(drop=True)
    df_click_m["click_id"] = df_click_m.index

    click_t = df_click_m.rename(columns={"imp_timestamp": "event_time"})
    like_t = df_like_m.rename(columns={"like_timestamp": "event_time"})
    click_t["_is_click"] = 1
    like_t["_is_like"] = 1

    combined = pd.concat([click_t, like_t], ignore_index=True, sort=False)
    combined = combined.sort_values(key_cols + ["event_time"], kind="mergesort").reset_index(drop=True)
    combined["click_id"] = combined.groupby(key_cols, sort=False)["click_id"].ffill()

    like_map = combined[combined["_is_like"].eq(1)].dropna(subset=["click_id"])[["click_id", "event_time"]]
    like_agg = like_map.groupby("click_id").agg(latest_like_ts=("event_time", "max")).reset_index()

    df_click_with_like = df_click_m.merge(like_agg, on="click_id", how="left")
    df_click_with_like["is_like"] = df_click_with_like["latest_like_ts"].notna().astype("int64")
    df_click_with_like = df_click_with_like.drop(columns=["click_id"])

    df_comment = dataframes.get("comment")
    if df_comment is not None:
        df_comment = df_comment.copy()
        if "timestamp" in df_comment.columns:
            df_comment = df_comment.rename(columns={"timestamp": "comment_timestamp"})
            df_comment["comment_timestamp"] = pd.to_datetime(df_comment["comment_timestamp"], unit="ms", errors="coerce")
        df_comment["is_comment"] = 1

    for c in key_cols:
        df_click_with_like[c] = pd.to_numeric(df_click_with_like[c], errors="coerce").astype("int64")
        df_comment[c] = pd.to_numeric(df_comment[c], errors="coerce").astype("int64")

    df_click_with_like["imp_timestamp"] = pd.to_datetime(df_click_with_like["imp_timestamp"], errors="coerce")
    df_comment["comment_timestamp"] = pd.to_datetime(df_comment["comment_timestamp"], errors="coerce")

    df_click_m = df_click_with_like.dropna(subset=key_cols + ["imp_timestamp"]).copy()
    df_comment_m = df_comment.dropna(subset=key_cols + ["comment_timestamp"]).copy()

    df_click_m = df_click_m.sort_values(key_cols + ["imp_timestamp"], kind="mergesort").reset_index(drop=True)
    df_click_m["click_id"] = df_click_m.index

    click_t = df_click_m.rename(columns={"imp_timestamp": "event_time"})
    comment_t = df_comment_m.rename(columns={"comment_timestamp": "event_time"})
    click_t["_is_click"] = 1
    comment_t["_is_comment"] = 1

    combined = pd.concat([click_t, comment_t], ignore_index=True, sort=False)
    combined = combined.sort_values(key_cols + ["event_time"], kind="mergesort").reset_index(drop=True)
    combined["click_id"] = combined.groupby(key_cols, sort=False)["click_id"].ffill()

    comment_map = combined[combined["_is_comment"].eq(1)].dropna(subset=["click_id"])[["click_id", "event_time"]]
    comment_agg = comment_map.groupby("click_id").agg(latest_comment_ts=("event_time", "max")).reset_index()

    df_click_with_like_comment = df_click_m.merge(comment_agg, on="click_id", how="left")
    df_click_with_like_comment["is_comment"] = df_click_with_like_comment["latest_comment_ts"].notna().astype("int64")
    df_click_with_like_comment = df_click_with_like_comment.drop(columns=["click_id"])

    df_gift = dataframes.get("gift")
    if df_gift is not None:
        df_gift = df_gift.copy()
        if "timestamp" in df_gift.columns:
            df_gift = df_gift.rename(columns={"timestamp": "gift_timestamp"})
            df_gift["gift_timestamp"] = pd.to_datetime(df_gift["gift_timestamp"], unit="ms", errors="coerce")
        df_gift["is_gift"] = 1

    for c in key_cols:
        df_click_with_like_comment[c] = pd.to_numeric(df_click_with_like_comment[c], errors="coerce").astype("int64")
        df_gift[c] = pd.to_numeric(df_gift[c], errors="coerce").astype("int64")

    df_click_with_like_comment["imp_timestamp"] = pd.to_datetime(df_click_with_like_comment["imp_timestamp"], errors="coerce")
    df_gift["gift_timestamp"] = pd.to_datetime(df_gift["gift_timestamp"], errors="coerce")

    df_click_m = df_click_with_like_comment.dropna(subset=key_cols + ["imp_timestamp"]).copy()
    df_gift_m = df_gift.dropna(subset=key_cols + ["gift_timestamp"]).copy()

    df_click_m = df_click_m.sort_values(key_cols + ["imp_timestamp"], kind="mergesort").reset_index(drop=True)
    df_click_m["click_id"] = df_click_m.index

    click_t = df_click_m.rename(columns={"imp_timestamp": "event_time"})
    gift_t = df_gift_m.rename(columns={"gift_timestamp": "event_time"})
    click_t["_is_click"] = 1
    gift_t["_is_gift"] = 1

    combined = pd.concat([click_t, gift_t], ignore_index=True, sort=False)
    combined = combined.sort_values(key_cols + ["event_time"], kind="mergesort").reset_index(drop=True)
    combined["click_id"] = combined.groupby(key_cols, sort=False)["click_id"].ffill()

    gift_map = combined[combined["_is_gift"].eq(1)].dropna(subset=["click_id"])[["click_id", "event_time", "gift_price"]]
    gift_map = gift_map.sort_values(["click_id", "event_time"], kind="mergesort")
    gift_agg = gift_map.groupby("click_id", as_index=False).last()
    gift_agg = gift_agg.rename(columns={"event_time": "latest_gift_ts"})

    df_click_with_like_comment_gift = df_click_m.merge(gift_agg, on="click_id", how="left")
    df_click_with_like_comment_gift["is_gift"] = df_click_with_like_comment_gift["latest_gift_ts"].notna().astype("int64")
    df_click_with_like_comment_gift = df_click_with_like_comment_gift.drop(columns=["click_id"])

    df_negative = dataframes.get("negative")
    if df_negative is not None:
        df_negative = df_negative.copy()
        if "timestamp" in df_negative.columns:
            df_negative = df_negative.rename(columns={"timestamp": "imp_timestamp"})
            df_negative["imp_timestamp"] = pd.to_datetime(df_negative["imp_timestamp"], unit="ms", errors="coerce")

        df_negative["is_click"] = 0
        df_negative["watch_live_time"] = 0
        df_negative["is_like"] = 0
        df_negative["is_comment"] = 0
        df_negative["is_gift"] = 0
        df_negative["gift_price"] = 0
        df_negative["latest_like_ts"] = 0
        df_negative["latest_comment_ts"] = 0
        df_negative["latest_gift_ts"] = 0

    df_interactions = pd.concat([df_click_with_like_comment_gift, df_negative], ignore_index=True, sort=False)
    df_interactions = df_interactions.sort_values(
        ["imp_timestamp", "user_id", "live_id", "streamer_id"], kind="mergesort"
    ).reset_index(drop=True)

    df_interactions["imp_year"] = df_interactions["imp_timestamp"].dt.year
    df_interactions["imp_month"] = df_interactions["imp_timestamp"].dt.month
    df_interactions["imp_day"] = df_interactions["imp_timestamp"].dt.day
    df_interactions["imp_hour"] = df_interactions["imp_timestamp"].dt.hour
    df_interactions["imp_is_weekend"] = df_interactions["imp_timestamp"].dt.weekday.ge(5).astype("int64")

    new_cols = ["imp_year", "imp_month", "imp_day", "imp_hour", "imp_is_weekend"]
    cols = df_interactions.columns.tolist()
    imp_idx = cols.index("imp_timestamp")
    for c in new_cols:
        cols.remove(c)
    cols = cols[: imp_idx + 1] + new_cols + cols[imp_idx + 1 :]
    df_interactions = df_interactions[cols]

    df_interactions["gift_price"] = df_interactions["gift_price"].fillna(0)
    log(f"Interactions shape: {df_interactions.shape}")
    return df_interactions


def build_user_features(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    log("Preparing user table")
    df_user = dataframes.get("user")
    df_user = df_user.copy()

    df_user["reg_timestamp"] = pd.to_datetime(df_user["reg_timestamp"], errors="coerce")
    df_user["first_watch_live_timestamp"] = pd.to_datetime(df_user["first_watch_live_timestamp"], errors="coerce")

    rename_map = {
        "age": "user_age",
        "gender": "user_gender",
        "country": "user_country",
        "device_brand": "user_device_brand",
        "device_price": "user_device_price",
        "reg_timestamp": "user_reg_timestamp",
    }
    rename_map.update({f"onehot_feat{i}": f"user_onehot_feat{i}" for i in range(7)})
    df_user = df_user.rename(columns=rename_map)

    le_cols = [
        "user_age",
        "user_gender",
        "user_country",
        "user_device_brand",
        "user_device_price",
        "fans_num",
        "follow_num",
        "accu_watch_live_cnt",
        "accu_watch_live_duration",
    ]
    for c in le_cols:
        codes, _ = pd.factorize(df_user[c], sort=True)
        df_user[f"{c}_le"] = codes

    log(f"User shape: {df_user.shape}")
    return df_user


def build_room_features(dataframes: Dict[str, pd.DataFrame], data_dir: Path) -> pd.DataFrame:
    log("Preparing room table + title embeddings")
    df_room = dataframes.get("room")
    df_room = df_room.copy()

    df_room["p_date"] = pd.to_datetime(df_room["p_date"].astype(str), format="%Y%m%d", errors="coerce")
    df_room["start_timestamp"] = pd.to_datetime(df_room["start_timestamp"], unit="ms", errors="coerce")
    df_room["end_timestamp"] = pd.to_datetime(df_room["end_timestamp"], unit="ms", errors="coerce")

    codes, _ = pd.factorize(df_room["live_content_category"], sort=True)
    df_room["live_content_category_le"] = codes

    df_room["live_start_year"] = df_room["start_timestamp"].dt.year
    df_room["live_start_month"] = df_room["start_timestamp"].dt.month
    df_room["live_start_day"] = df_room["start_timestamp"].dt.day
    df_room["live_start_hour"] = df_room["start_timestamp"].dt.hour
    df_room["live_is_weekend"] = df_room["start_timestamp"].dt.weekday.ge(5).astype("int64")

    emb = np.load(data_dir / "title_embeddings.npy")
    live_ids = df_room["live_name_id"].dropna().astype("int64").sort_values().unique()
    live_ids_for_emb = live_ids[live_ids != -1]
    assert len(live_ids_for_emb) == emb.shape[0]

    df_title_embedding = pd.DataFrame(emb, columns=[f"title_emb_{i}" for i in range(emb.shape[1])])
    df_title_embedding["live_name_id"] = live_ids_for_emb

    df_room = df_room.merge(df_title_embedding, on="live_name_id", how="left")
    emb_cols = [c for c in df_room.columns if c.startswith("title_emb_")]
    df_room["title_emb_missing"] = df_room[emb_cols].isna().any(axis=1).astype("int64")
    df_room[emb_cols] = df_room[emb_cols].fillna(0)

    tmp = df_room[["live_id", "live_name_id", "start_timestamp"]].dropna(subset=["live_id", "live_name_id"])
    counts = tmp.groupby(["live_id", "live_name_id"], as_index=False).agg(
        freq=("live_name_id", "size"), latest_ts=("start_timestamp", "max")
    )
    best = (
        counts.sort_values(["live_id", "freq", "latest_ts"], ascending=[True, False, False])
        .drop_duplicates(subset=["live_id"], keep="first")
        .rename(columns={"live_name_id": "live_name_id_mostfreq"})[["live_id", "live_name_id_mostfreq"]]
    )

    df_room = df_room.merge(best, on="live_id", how="left")
    df_room["live_name_id"] = df_room["live_name_id_mostfreq"]
    df_room = df_room.sort_values(["live_id", "start_timestamp"], ascending=[True, False]).drop_duplicates(
        subset=["live_id"], keep="first"
    )
    df_room = df_room.drop(columns=["live_name_id_mostfreq"], errors="ignore")

    log(f"Room shape: {df_room.shape}")
    return df_room


def build_streamer_features(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    log("Preparing streamer table")
    df_streamer = dataframes.get("streamer")
    df_streamer = df_streamer.copy()

    df_streamer["reg_timestamp"] = pd.to_datetime(df_streamer["reg_timestamp"], errors="coerce")
    df_streamer["first_live_timestamp"] = pd.to_datetime(df_streamer["first_live_timestamp"], errors="coerce")

    rename_map = {
        "age": "streamer_age",
        "gender": "streamer_gender",
        "country": "streamer_country",
        "device_brand": "streamer_device_brand",
        "device_price": "streamer_device_price",
        "reg_timestamp": "streamer_reg_timestamp",
    }
    rename_map.update({f"onehot_feat{i}": f"streamer_onehot_feat{i}" for i in range(7)})
    df_streamer = df_streamer.rename(columns=rename_map)

    le_cols = [
        "streamer_age",
        "streamer_gender",
        "streamer_country",
        "streamer_device_brand",
        "streamer_device_price",
        "live_operation_tag",
        "fans_user_num",
        "fans_group_fans_num",
        "follow_user_num",
        "accu_live_cnt",
        "accu_live_duration",
        "accu_play_cnt",
        "accu_play_duration",
    ]

    for c in le_cols:
        if c in df_streamer.columns:
            codes, _ = pd.factorize(df_streamer[c], sort=True)
            df_streamer[f"{c}_le"] = codes

    log(f"Streamer shape: {df_streamer.shape}")
    return df_streamer


def merge_tables(
    df_interactions: pd.DataFrame,
    df_user: pd.DataFrame,
    df_room: pd.DataFrame,
    df_streamer: pd.DataFrame,
) -> pd.DataFrame:
    log("Merging interaction/user/room/streamer tables")
    df_room_streamer = df_room.merge(df_streamer, on="streamer_id", how="left")
    df_interaction_user = df_interactions.merge(df_user, on="user_id", how="left")
    df_final = df_interaction_user.merge(df_room_streamer, on=["streamer_id", "live_id"], how="left")
    log(f"Merged base table shape: {df_final.shape}")
    return df_final


def engineer_features(df_final: pd.DataFrame, sample_frac: float, random_state: int) -> pd.DataFrame:
    log("Engineering features (sections 6.1 - 6.4)")
    df_final_sample = df_final.sample(frac=sample_frac, random_state=random_state).copy()
    df_final_sample = df_final_sample.reset_index(drop=True)

    sort_state = {"keys": None}

    def ensure_sorted(df: pd.DataFrame, keys):
        key_tuple = tuple(keys)
        if sort_state["keys"] != key_tuple:
            df = df.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
            sort_state["keys"] = key_tuple
        return df

    log(f"Sample shape: {df_final_sample.shape}")

    # 6.1.1
    df_final_sample["user_account_age"] = (
        (df_final_sample["imp_timestamp"] - df_final_sample["user_reg_timestamp"]).dt.total_seconds() / 86400
    )
    df_final_sample["user_watch_live_age"] = (
        (df_final_sample["imp_timestamp"] - df_final_sample["first_watch_live_timestamp"]).dt.total_seconds() / 86400
    )

    # 6.1.2
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "imp_timestamp"])
    g = df_final_sample.groupby("user_id", group_keys=False)
    windows = {
        "15min": "ctr_user_15min",
        "3h": "ctr_user_3hr",
        "1d": "ctr_user_1d",
        "7d": "ctr_user_7d",
    }
    _user_click_roll = {}
    _user_imp_roll = {}
    for w, col in tqdm(list(windows.items()), desc="6.1.2 user ctr", leave=False):
        clicks = g.rolling(w, on="imp_timestamp", closed="left")["is_click"].sum().reset_index(level=0, drop=True)
        imps = g.rolling(w, on="imp_timestamp", closed="left")["is_click"].count().reset_index(level=0, drop=True)
        click_arr = clicks.to_numpy()
        imp_arr = imps.to_numpy()
        _user_click_roll[w] = click_arr
        _user_imp_roll[w] = imp_arr
        df_final_sample[col] = (clicks / imps.replace(0, np.nan)).fillna(0).to_numpy()

    # 6.1.3
    df_final_sample["imp_timestamp"] = pd.to_datetime(df_final_sample["imp_timestamp"], errors="coerce")
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "imp_timestamp"])
    g = df_final_sample.groupby("user_id", group_keys=False)

    imp_windows = {
        "10min": "num_imp_user_10min",
        "30min": "num_imp_user_30min",
        "2h": "num_imp_user_2hr",
        "12h": "num_imp_user_12hr",
        "1d": "num_imp_user_1d",
        "7d": "num_imp_user_7d",
    }
    for w, col in tqdm(list(imp_windows.items()), desc="6.1.3 user imp", leave=False):
        imps = g.rolling(w, on="imp_timestamp", closed="left")["is_click"].count().reset_index(level=0, drop=True)
        df_final_sample[col] = imps.to_numpy()
    df_final_sample[list(imp_windows.values())] = df_final_sample[list(imp_windows.values())].fillna(0)

    # 6.1.4
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "imp_timestamp"])
    click_windows = {
        "15min": "num_click_user_15min",
        "3h": "num_click_user_3hr",
        "1d": "num_click_user_1d",
        "7d": "num_click_user_7d",
    }
    can_reuse = (
        isinstance(_user_click_roll, dict)
        and sort_state["keys"] == ("user_id", "imp_timestamp")
        and all(w in _user_click_roll for w in click_windows)
        and len(next(iter(_user_click_roll.values()))) == len(df_final_sample)
    )
    if can_reuse:
        for w, col in click_windows.items():
            df_final_sample[col] = _user_click_roll[w]
    else:
        g = df_final_sample.groupby("user_id", group_keys=False)
        for w, col in tqdm(list(click_windows.items()), desc="6.1.4 user click", leave=False):
            clicks = g.rolling(w, on="imp_timestamp", closed="left")["is_click"].sum().reset_index(level=0, drop=True)
            df_final_sample[col] = clicks.to_numpy()

    df_final_sample["click_trend_user"] = np.log(df_final_sample["num_click_user_15min"] + 1) - np.log(
        df_final_sample["num_click_user_3hr"] + 1
    )
    df_final_sample[list(click_windows.values()) + ["click_trend_user"]] = df_final_sample[
        list(click_windows.values()) + ["click_trend_user"]
    ].fillna(0)

    # 6.1.5
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "imp_timestamp"])
    g = df_final_sample.groupby("user_id", group_keys=False)

    df_final_sample["time_since_last_impression_user"] = g["imp_timestamp"].diff().dt.total_seconds().mul(1000)
    last_click_time = (
        df_final_sample["imp_timestamp"].where(df_final_sample["is_click"] == 1).groupby(df_final_sample["user_id"]).ffill().shift(1)
    )
    last_click_time = last_click_time.where(last_click_time <= df_final_sample["imp_timestamp"])
    df_final_sample["time_since_last_click_user"] = (
        (df_final_sample["imp_timestamp"] - last_click_time).dt.total_seconds().mul(1000)
    )

    click_group = g["is_click"].cumsum()
    df_final_sample["consecutive_skips_user"] = df_final_sample.groupby(["user_id", click_group]).cumcount()

    df_final_sample["tsli_missing"] = df_final_sample["time_since_last_impression_user"].isna().astype(np.int8)
    df_final_sample["tslc_missing"] = df_final_sample["time_since_last_click_user"].isna().astype(np.int8)

    TSLI_FILL_MS = 7 * 24 * 3600 * 1000
    TSLC_FILL_MS = 30 * 24 * 3600 * 1000
    df_final_sample["time_since_last_impression_user"] = df_final_sample["time_since_last_impression_user"].fillna(TSLI_FILL_MS)
    df_final_sample["time_since_last_click_user"] = df_final_sample["time_since_last_click_user"].fillna(TSLC_FILL_MS)

    # 6.1.6
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "imp_timestamp"])
    df_final_sample["_watch_on_click"] = df_final_sample["watch_live_time"].where(df_final_sample["is_click"] == 1)
    g = df_final_sample.groupby("user_id", group_keys=False)

    df_final_sample["avg_watch_time_user"] = (
        g["_watch_on_click"].expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    df_final_sample["median_watch_time_user"] = (
        g["_watch_on_click"].expanding().median().shift(1).reset_index(level=0, drop=True)
    )

    past_clicks = g["is_click"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    past_long = (
        (df_final_sample["_watch_on_click"] >= 30)
        .groupby(df_final_sample["user_id"])
        .expanding()
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    df_final_sample["pct_long_watch_user_30s"] = (past_long / past_clicks.replace(0, np.nan)).fillna(0)
    df_final_sample = df_final_sample.drop(columns=["_watch_on_click"])

    df_final_sample["avg_watch_time_user_missing"] = df_final_sample["avg_watch_time_user"].isna().astype(np.int8)
    df_final_sample["median_watch_time_user_missing"] = df_final_sample["median_watch_time_user"].isna().astype(np.int8)
    df_final_sample["avg_watch_time_user"] = df_final_sample["avg_watch_time_user"].fillna(0.0)
    df_final_sample["median_watch_time_user"] = df_final_sample["median_watch_time_user"].fillna(0.0)

    # 6.1.7
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "imp_timestamp"])
    g = df_final_sample.groupby("user_id", group_keys=False)

    num_click_user = g["is_click"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    num_comment_user = g["is_comment"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    df_final_sample["comment_rate_user"] = (num_comment_user + 1) / (num_click_user + 1)

    num_comment_user_24h = (
        g.rolling("24h", on="imp_timestamp", closed="left")["is_comment"].sum().reset_index(level=0, drop=True)
    )
    df_final_sample["num_comment_user_24h"] = num_comment_user_24h.to_numpy()
    df_final_sample["has_comment_user_24h"] = (df_final_sample["num_comment_user_24h"] > 0).astype("int64")
    df_final_sample["num_comment_user_24h"] = df_final_sample["num_comment_user_24h"].fillna(0)
    df_final_sample["comment_rate_user"] = df_final_sample["comment_rate_user"].fillna(0)

    # 6.1.8
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "imp_timestamp"])
    g = df_final_sample.groupby("user_id", group_keys=False)

    num_click_user = g["is_click"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    num_like_user = g["is_like"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    df_final_sample["like_rate_user"] = (num_like_user + 1) / (num_click_user + 1)

    num_like_user_24h = (
        g.rolling("24h", on="imp_timestamp", closed="left")["is_like"].sum().reset_index(level=0, drop=True)
    )
    df_final_sample["num_like_user_24h"] = num_like_user_24h.to_numpy()
    df_final_sample["has_like_user_24h"] = (df_final_sample["num_like_user_24h"] > 0).astype("int64")
    df_final_sample["num_like_user_24h"] = df_final_sample["num_like_user_24h"].fillna(0)
    df_final_sample["like_rate_user"] = df_final_sample["like_rate_user"].fillna(0)

    # 6.1.9
    df_final_sample["imp_timestamp"] = pd.to_datetime(df_final_sample["imp_timestamp"], errors="coerce")
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "imp_timestamp"])
    g = df_final_sample.groupby("user_id", group_keys=False)

    num_gift_user_7d = g.rolling("7d", on="imp_timestamp", closed="left")["is_gift"].sum().reset_index(level=0, drop=True)
    amount_gift_user_7d = (
        g.rolling("7d", on="imp_timestamp", closed="left")["gift_price"].sum().reset_index(level=0, drop=True)
    )

    df_final_sample["num_gift_user_7d"] = num_gift_user_7d.to_numpy()
    df_final_sample["amount_gift_user_7d"] = amount_gift_user_7d.to_numpy()
    df_final_sample["has_gift_user_7d"] = (df_final_sample["num_gift_user_7d"] > 0).astype("int64")
    df_final_sample[["num_gift_user_7d", "amount_gift_user_7d"]] = df_final_sample[
        ["num_gift_user_7d", "amount_gift_user_7d"]
    ].fillna(0)

    # 6.2.1
    df_final_sample["time_since_live_start"] = (
        (df_final_sample["imp_timestamp"] - df_final_sample["start_timestamp"]).dt.total_seconds().mul(1000)
    )

    # 6.2.2
    df_final_sample = ensure_sorted(df_final_sample, ["live_id", "imp_timestamp"])
    g = df_final_sample.groupby("live_id", group_keys=False)

    room_windows = {
        "10min": "ctr_room_10min",
        "30min": "ctr_room_30min",
        "2h": "ctr_room_2hr",
        "12h": "ctr_room_12hr",
    }
    _live_click_roll = {}
    _live_imp_roll = {}
    for w, col in tqdm(list(room_windows.items()), desc="6.2.2 room ctr", leave=False):
        clicks = g.rolling(w, on="imp_timestamp", closed="left")["is_click"].sum().reset_index(level=0, drop=True)
        imps = g.rolling(w, on="imp_timestamp", closed="left")["is_click"].count().reset_index(level=0, drop=True)
        click_arr = clicks.to_numpy()
        imp_arr = imps.to_numpy()
        _live_click_roll[w] = click_arr
        _live_imp_roll[w] = imp_arr
        df_final_sample[col] = (clicks / imps.replace(0, np.nan)).fillna(0).to_numpy()

    # 6.2.3
    df_final_sample = ensure_sorted(df_final_sample, ["live_id", "imp_timestamp"])
    g = df_final_sample.groupby("live_id", group_keys=False)

    room_imp_windows = {
        "10min": "num_imp_room_10min",
        "30min": "num_imp_room_30min",
        "2h": "num_imp_room_2hr",
        "12h": "num_imp_room_12hr",
        "1d": "num_imp_room_1d",
    }

    can_reuse = (
        isinstance(_live_imp_roll, dict)
        and sort_state["keys"] == ("live_id", "imp_timestamp")
        and all(w in _live_imp_roll for w in ["10min", "30min", "2h", "12h"])
        and len(_live_imp_roll["10min"]) == len(df_final_sample)
    )

    for w, col in tqdm(list(room_imp_windows.items()), desc="6.2.3 room imp", leave=False):
        if can_reuse and w in _live_imp_roll:
            df_final_sample[col] = _live_imp_roll[w]
        else:
            imps = g.rolling(w, on="imp_timestamp", closed="left")["is_click"].count().reset_index(level=0, drop=True)
            df_final_sample[col] = imps.to_numpy()

    room_imp_cols = [
        "num_imp_room_10min",
        "num_imp_room_30min",
        "num_imp_room_2hr",
        "num_imp_room_12hr",
        "num_imp_room_1d",
    ]
    df_final_sample[room_imp_cols] = df_final_sample[room_imp_cols].fillna(0)

    # 6.2.4
    df_final_sample = ensure_sorted(df_final_sample, ["live_id", "imp_timestamp"])
    g = df_final_sample.groupby("live_id", group_keys=False)

    room_click_windows = {
        "10min": "num_click_room_10min",
        "30min": "num_click_room_30min",
        "2h": "num_click_room_2hr",
        "12h": "num_click_room_12hr",
        "1d": "num_click_room_1d",
    }

    can_reuse = (
        isinstance(_live_click_roll, dict)
        and sort_state["keys"] == ("live_id", "imp_timestamp")
        and all(w in _live_click_roll for w in ["10min", "30min", "2h", "12h"])
        and len(_live_click_roll["10min"]) == len(df_final_sample)
    )

    for w, col in tqdm(list(room_click_windows.items()), desc="6.2.4 room click", leave=False):
        if can_reuse and w in _live_click_roll:
            df_final_sample[col] = _live_click_roll[w]
        else:
            clicks = g.rolling(w, on="imp_timestamp", closed="left")["is_click"].sum().reset_index(level=0, drop=True)
            df_final_sample[col] = clicks.to_numpy()

    df_final_sample["ctr_trend_room"] = np.log(df_final_sample["ctr_room_10min"] + 1e-6) - np.log(
        df_final_sample["ctr_room_2hr"] + 1e-6
    )
    df_final_sample[list(room_click_windows.values()) + ["ctr_trend_room"]] = df_final_sample[
        list(room_click_windows.values()) + ["ctr_trend_room"]
    ].fillna(0)

    # 6.2.5
    df_final_sample["start_timestamp"] = pd.to_datetime(df_final_sample["start_timestamp"], errors="coerce")
    df_final_sample["time_since_start_live"] = (
        (df_final_sample["imp_timestamp"] - df_final_sample["start_timestamp"]).dt.total_seconds().mul(1000)
    )
    mins = df_final_sample["time_since_start_live"] / (60 * 1000)
    df_final_sample["time_since_start_live_bucket"] = pd.cut(
        mins, bins=[-float("inf"), 5, 20, float("inf")], labels=["<5min", "5-20min", ">20min"]
    )
    df_final_sample["time_since_start_live_bucket"] = (
        df_final_sample["time_since_start_live_bucket"].astype("category").cat.codes
    )

    # 6.2.6
    df_final_sample = ensure_sorted(df_final_sample, ["live_id", "imp_timestamp"])
    df_final_sample["_watch_on_click"] = df_final_sample["watch_live_time"].where(df_final_sample["is_click"] == 1)
    df_final_sample["_is_long_watch_60s"] = (df_final_sample["_watch_on_click"] >= 60000).astype("int8")
    g = df_final_sample.groupby("live_id", group_keys=False)

    df_final_sample["avg_watch_time_live"] = (
        g["_watch_on_click"].expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    df_final_sample["median_watch_time_live"] = (
        g["_watch_on_click"].expanding().median().shift(1).reset_index(level=0, drop=True)
    )

    df_final_sample["avg_watch_time_live_30min"] = (
        g.rolling("30min", on="imp_timestamp", closed="left")["_watch_on_click"]
        .mean()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    df_final_sample["median_watch_time_live_30min"] = (
        g.rolling("30min", on="imp_timestamp", closed="left")["_watch_on_click"]
        .median()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    past_long_30 = (
        g.rolling("30min", on="imp_timestamp", closed="left")["_is_long_watch_60s"]
        .sum()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    past_clicks_30 = (
        g.rolling("30min", on="imp_timestamp", closed="left")["_watch_on_click"]
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    df_final_sample["pct_long_watch_live_60s_30min"] = past_long_30 / np.where(past_clicks_30 == 0, np.nan, past_clicks_30)
    df_final_sample["pct_long_watch_live_60s_30min"] = df_final_sample["pct_long_watch_live_60s_30min"].fillna(0)

    df_final_sample["watch_time_live_missing"] = (
        df_final_sample[["avg_watch_time_live", "median_watch_time_live"]].isna().any(axis=1).astype("int64")
    )
    df_final_sample["watch_time_live_30min_missing"] = (
        df_final_sample[["avg_watch_time_live_30min", "median_watch_time_live_30min"]]
        .isna()
        .any(axis=1)
        .astype("int64")
    )

    fill_cols = [
        "avg_watch_time_live",
        "median_watch_time_live",
        "avg_watch_time_live_30min",
        "median_watch_time_live_30min",
    ]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)
    df_final_sample = df_final_sample.drop(columns=["_watch_on_click", "_is_long_watch_60s"])

    # 6.2.7
    df_final_sample = ensure_sorted(df_final_sample, ["live_id", "imp_timestamp"])
    g = df_final_sample.groupby("live_id", group_keys=False)

    num_comment_live = g["is_comment"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    num_imp_live = g["is_comment"].expanding().count().shift(1).reset_index(level=0, drop=True)
    df_final_sample["num_comment_live"] = num_comment_live.to_numpy()
    df_final_sample["comment_rate_live"] = (num_comment_live / num_imp_live.replace(0, np.nan)).fillna(0)

    def _roll_sum_comment(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_comment"].sum().reset_index(level=0, drop=True).to_numpy()

    def _roll_cnt_comment(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_comment"].count().reset_index(level=0, drop=True).to_numpy()

    df_final_sample["num_comment_live_15min"] = _roll_sum_comment("15min")
    df_final_sample["num_comment_live_1hr"] = _roll_sum_comment("1h")
    df_final_sample["num_comment_live_3hr"] = _roll_sum_comment("3h")

    imp_15 = _roll_cnt_comment("15min")
    imp_1h = _roll_cnt_comment("1h")
    imp_3h = _roll_cnt_comment("3h")

    df_final_sample["comment_rate_live_15min"] = df_final_sample["num_comment_live_15min"] / np.where(imp_15 == 0, np.nan, imp_15)
    df_final_sample["comment_rate_live_1hr"] = df_final_sample["num_comment_live_1hr"] / np.where(imp_1h == 0, np.nan, imp_1h)
    df_final_sample["comment_rate_live_3hr"] = df_final_sample["num_comment_live_3hr"] / np.where(imp_3h == 0, np.nan, imp_3h)

    df_final_sample["comment_trend_room"] = np.log(df_final_sample["comment_rate_live_15min"] + 1e-6) - np.log(
        df_final_sample["comment_rate_live_1hr"] + 1e-6
    )

    fill_cols = [
        "num_comment_live",
        "num_comment_live_15min",
        "num_comment_live_1hr",
        "num_comment_live_3hr",
        "comment_rate_live",
        "comment_rate_live_15min",
        "comment_rate_live_1hr",
        "comment_rate_live_3hr",
        "comment_trend_room",
    ]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)

    # 6.2.8
    df_final_sample = ensure_sorted(df_final_sample, ["live_id", "imp_timestamp"])
    g = df_final_sample.groupby("live_id", group_keys=False)

    num_like_live = g["is_like"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    num_imp_live = g["is_like"].expanding().count().shift(1).reset_index(level=0, drop=True)
    df_final_sample["num_like_live"] = num_like_live.to_numpy()
    df_final_sample["like_rate_live"] = (num_like_live / num_imp_live.replace(0, np.nan)).fillna(0)

    def _roll_sum_like(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_like"].sum().reset_index(level=0, drop=True).to_numpy()

    def _roll_cnt_like(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_like"].count().reset_index(level=0, drop=True).to_numpy()

    df_final_sample["num_like_live_15min"] = _roll_sum_like("15min")
    df_final_sample["num_like_live_1hr"] = _roll_sum_like("1h")
    df_final_sample["num_like_live_3hr"] = _roll_sum_like("3h")

    imp_15 = _roll_cnt_like("15min")
    imp_1h = _roll_cnt_like("1h")
    imp_3h = _roll_cnt_like("3h")

    df_final_sample["like_rate_live_15min"] = df_final_sample["num_like_live_15min"] / np.where(imp_15 == 0, np.nan, imp_15)
    df_final_sample["like_rate_live_1hr"] = df_final_sample["num_like_live_1hr"] / np.where(imp_1h == 0, np.nan, imp_1h)
    df_final_sample["like_rate_live_3hr"] = df_final_sample["num_like_live_3hr"] / np.where(imp_3h == 0, np.nan, imp_3h)

    df_final_sample["like_trend_room"] = np.log(df_final_sample["like_rate_live_15min"] + 1e-6) - np.log(
        df_final_sample["like_rate_live_1hr"] + 1e-6
    )

    fill_cols = [
        "num_like_live",
        "num_like_live_15min",
        "num_like_live_1hr",
        "num_like_live_3hr",
        "like_rate_live",
        "like_rate_live_15min",
        "like_rate_live_1hr",
        "like_rate_live_3hr",
        "like_trend_room",
    ]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)

    # 6.2.9
    df_final_sample = ensure_sorted(df_final_sample, ["live_id", "imp_timestamp"])
    g = df_final_sample.groupby("live_id", group_keys=False)

    num_gift_live = g["is_gift"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    amount_gift_live = g["gift_price"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    num_imp_live = g["is_gift"].expanding().count().shift(1).reset_index(level=0, drop=True)

    df_final_sample["num_gift_live"] = num_gift_live.to_numpy()
    df_final_sample["amount_gift_live"] = amount_gift_live.to_numpy()
    df_final_sample["gift_rate_live"] = (num_gift_live / num_imp_live.replace(0, np.nan)).fillna(0)

    def _roll_sum_gift(col, w):
        return g.rolling(w, on="imp_timestamp", closed="left")[col].sum().reset_index(level=0, drop=True).to_numpy()

    def _roll_cnt_gift(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_gift"].count().reset_index(level=0, drop=True).to_numpy()

    df_final_sample["num_gift_live_15min"] = _roll_sum_gift("is_gift", "15min")
    df_final_sample["num_gift_live_1hr"] = _roll_sum_gift("is_gift", "1h")
    df_final_sample["num_gift_live_3hr"] = _roll_sum_gift("is_gift", "3h")

    df_final_sample["amount_gift_live_15min"] = _roll_sum_gift("gift_price", "15min")
    df_final_sample["amount_gift_live_1hr"] = _roll_sum_gift("gift_price", "1h")
    df_final_sample["amount_gift_live_3hr"] = _roll_sum_gift("gift_price", "3h")

    imp_15 = _roll_cnt_gift("15min")
    imp_1h = _roll_cnt_gift("1h")
    imp_3h = _roll_cnt_gift("3h")

    df_final_sample["gift_rate_live_15min"] = df_final_sample["num_gift_live_15min"] / np.where(imp_15 == 0, np.nan, imp_15)
    df_final_sample["gift_rate_live_1hr"] = df_final_sample["num_gift_live_1hr"] / np.where(imp_1h == 0, np.nan, imp_1h)
    df_final_sample["gift_rate_live_3hr"] = df_final_sample["num_gift_live_3hr"] / np.where(imp_3h == 0, np.nan, imp_3h)

    df_final_sample["gift_trend_room"] = np.log(np.log1p(df_final_sample["amount_gift_live_15min"]) + 1) - np.log(
        np.log1p(df_final_sample["amount_gift_live_1hr"]) + 1
    )

    fill_cols = [
        "num_gift_live",
        "amount_gift_live",
        "gift_rate_live",
        "num_gift_live_15min",
        "num_gift_live_1hr",
        "num_gift_live_3hr",
        "amount_gift_live_15min",
        "amount_gift_live_1hr",
        "amount_gift_live_3hr",
        "gift_rate_live_15min",
        "gift_rate_live_1hr",
        "gift_rate_live_3hr",
        "gift_trend_room",
    ]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)

    # 6.3.1
    df_final_sample["streamer_account_age"] = (
        (df_final_sample["imp_timestamp"] - df_final_sample["streamer_reg_timestamp"]).dt.total_seconds() / 86400
    )
    df_final_sample["streamer_live_age"] = (
        (df_final_sample["imp_timestamp"] - df_final_sample["first_live_timestamp"]).dt.total_seconds() / 86400
    )

    # 6.3.2
    df_final_sample = ensure_sorted(df_final_sample, ["streamer_id", "imp_timestamp"])
    g = df_final_sample.groupby("streamer_id", group_keys=False)

    def _roll_cnt_streamer(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_click"].count().reset_index(level=0, drop=True).to_numpy()

    def _roll_sum_streamer(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_click"].sum().reset_index(level=0, drop=True).to_numpy()

    imp_7d = _roll_cnt_streamer("7d")
    click_7d = _roll_sum_streamer("7d")
    df_final_sample["num_imp_streamer_7d"] = imp_7d
    df_final_sample["num_click_streamer_7d"] = click_7d
    df_final_sample["ctr_streamer_7d"] = click_7d / np.where(imp_7d == 0, np.nan, imp_7d)

    imp_1d = _roll_cnt_streamer("1d")
    click_1d = _roll_sum_streamer("1d")
    df_final_sample["ctr_streamer_1d"] = click_1d / np.where(imp_1d == 0, np.nan, imp_1d)

    def _rolling_nunique_7d_closed_left(df, group_col, time_col, value_col):
        out = np.empty(len(df), dtype="float64")
        seven_days = np.timedelta64(7, "D")
        for _, idx in tqdm(df.groupby(group_col, sort=False).groups.items(), desc="6.3.2 nunique", leave=False):
            pos = np.asarray(idx, dtype=np.int64)
            ts = df.iloc[pos][time_col].to_numpy(dtype="datetime64[ns]")
            vals = df.iloc[pos][value_col].to_numpy()

            left = 0
            right = 0
            freq = {}
            uniq = 0
            grp_out = np.empty(len(pos), dtype="float64")

            for i in range(len(pos)):
                t = ts[i]
                while right < len(pos) and ts[right] < t:
                    v = vals[right]
                    c = freq.get(v, 0)
                    if c == 0:
                        uniq += 1
                    freq[v] = c + 1
                    right += 1
                lower = t - seven_days
                while left < right and ts[left] < lower:
                    v = vals[left]
                    c = freq[v] - 1
                    if c == 0:
                        del freq[v]
                        uniq -= 1
                    else:
                        freq[v] = c
                    left += 1
                grp_out[i] = uniq
            out[pos] = grp_out
        return out

    df_final_sample["num_lives_streamer_7d"] = _rolling_nunique_7d_closed_left(
        df_final_sample, group_col="streamer_id", time_col="imp_timestamp", value_col="live_id"
    )

    fill_cols = ["num_imp_streamer_7d", "num_click_streamer_7d", "ctr_streamer_7d", "ctr_streamer_1d", "num_lives_streamer_7d"]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)

    # 6.3.3
    df_final_sample = ensure_sorted(df_final_sample, ["streamer_id", "imp_timestamp"])
    df_final_sample["_watch_on_click"] = df_final_sample["watch_live_time"].where(df_final_sample["is_click"] == 1)
    g = df_final_sample.groupby("streamer_id", group_keys=False)

    df_final_sample["avg_watch_time_streamer"] = (
        g["_watch_on_click"].expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    df_final_sample["median_watch_time_streamer"] = (
        g["_watch_on_click"].expanding().median().shift(1).reset_index(level=0, drop=True)
    )

    past_long = (
        (df_final_sample["_watch_on_click"] >= 30000)
        .groupby(df_final_sample["streamer_id"])
        .expanding()
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    past_clicks = g["_watch_on_click"].expanding().count().shift(1).reset_index(level=0, drop=True)
    df_final_sample["pct_long_watch_streamer_30s"] = past_long / np.where(past_clicks == 0, np.nan, past_clicks)

    df_final_sample["watch_time_streamer_missing"] = (
        df_final_sample[["avg_watch_time_streamer", "median_watch_time_streamer"]].isna().any(axis=1).astype("int64")
    )

    fill_cols = ["avg_watch_time_streamer", "median_watch_time_streamer", "pct_long_watch_streamer_30s"]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)
    df_final_sample = df_final_sample.drop(columns=["_watch_on_click"])

    # 6.3.4
    df_final_sample = ensure_sorted(df_final_sample, ["streamer_id", "imp_timestamp"])
    g = df_final_sample.groupby("streamer_id", group_keys=False)

    def _roll_sum_streamer_inter(col, w):
        return g.rolling(w, on="imp_timestamp", closed="left")[col].sum().reset_index(level=0, drop=True).to_numpy()

    df_final_sample["num_comment_streamer_7d"] = _roll_sum_streamer_inter("is_comment", "7d")
    df_final_sample["num_like_streamer_7d"] = _roll_sum_streamer_inter("is_like", "7d")
    df_final_sample["amount_gift_streamer_7d"] = _roll_sum_streamer_inter("gift_price", "7d")
    fill_cols = ["num_comment_streamer_7d", "num_like_streamer_7d", "amount_gift_streamer_7d"]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)

    # 6.4.1
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", "streamer_id", "imp_timestamp"])
    g = df_final_sample.groupby(["user_id", "streamer_id"], group_keys=False)

    def _roll_cnt_user_streamer(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_click"].count().reset_index(level=[0, 1], drop=True).to_numpy()

    def _roll_sum_user_streamer(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_click"].sum().reset_index(level=[0, 1], drop=True).to_numpy()

    imp_7d = _roll_cnt_user_streamer("7d")
    click_7d = _roll_sum_user_streamer("7d")
    df_final_sample["num_imp_user_streamer_7d"] = imp_7d
    df_final_sample["num_click_user_streamer_7d"] = click_7d
    df_final_sample["ctr_user_streamer_7d"] = click_7d / np.where(imp_7d == 0, np.nan, imp_7d)

    last_imp = g["imp_timestamp"].shift(1)
    df_final_sample["time_since_last_impression_user_streamer"] = (
        (df_final_sample["imp_timestamp"] - last_imp).dt.total_seconds()
    )

    last_click_ts = df_final_sample["imp_timestamp"].where(df_final_sample["is_click"] == 1)
    last_click_ts = last_click_ts.ffill().shift(1)
    df_final_sample["time_since_last_click_user_streamer"] = (
        (df_final_sample["imp_timestamp"] - last_click_ts).dt.total_seconds()
    )

    df_final_sample["tsli_user_streamer_missing"] = df_final_sample["time_since_last_impression_user_streamer"].isna().astype("int64")
    df_final_sample["tslc_user_streamer_missing"] = df_final_sample["time_since_last_click_user_streamer"].isna().astype("int64")

    fill_cols = [
        "num_imp_user_streamer_7d",
        "num_click_user_streamer_7d",
        "ctr_user_streamer_7d",
        "time_since_last_impression_user_streamer",
        "time_since_last_click_user_streamer",
    ]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)

    # 6.4.2
    cat_col = "live_content_category_le"
    df_final_sample = ensure_sorted(df_final_sample, ["user_id", cat_col, "imp_timestamp"])
    g = df_final_sample.groupby(["user_id", cat_col], group_keys=False)

    def _roll_cnt_user_cat(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_click"].count().reset_index(level=[0, 1], drop=True).to_numpy()

    def _roll_sum_user_cat(w):
        return g.rolling(w, on="imp_timestamp", closed="left")["is_click"].sum().reset_index(level=[0, 1], drop=True).to_numpy()

    imp_7d = _roll_cnt_user_cat("7d")
    click_7d = _roll_sum_user_cat("7d")

    df_final_sample["num_imp_user_category_7d"] = imp_7d
    df_final_sample["num_click_user_category_7d"] = click_7d
    df_final_sample["ctr_user_category_7d"] = click_7d / np.where(imp_7d == 0, np.nan, imp_7d)

    fill_cols = ["num_imp_user_category_7d", "num_click_user_category_7d", "ctr_user_category_7d"]
    df_final_sample[fill_cols] = df_final_sample[fill_cols].fillna(0)

    log(f"Feature-engineered sample shape: {df_final_sample.shape}")
    return df_final_sample


def normalize_numeric_feature(df, col, method="zscore", clip_q=None, eps=1e-9):
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column is not numeric: {col}")

    s = df[col].astype(float)
    if clip_q is not None:
        lo = s.quantile(1 - clip_q)
        hi = s.quantile(clip_q)
        s = s.clip(lower=lo, upper=hi)

    if method == "zscore":
        mean = s.mean()
        std = s.std()
        std = std if std > eps else eps
        scaled = (s - mean) / std
        params = {"method": method, "mean": mean, "std": std}
    elif method == "minmax":
        minv = s.min()
        maxv = s.max()
        denom = (maxv - minv) if (maxv - minv) > eps else eps
        scaled = (s - minv) / denom
        params = {"method": method, "min": minv, "max": maxv}
    elif method == "robust":
        med = s.median()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = (q3 - q1) if (q3 - q1) > eps else eps
        scaled = (s - med) / iqr
        params = {"method": method, "median": med, "q1": q1, "q3": q3, "iqr": iqr}
    else:
        raise ValueError("method must be one of: 'zscore', 'minmax', 'robust'")

    return scaled, params


def transform_features(df_final_sample: pd.DataFrame) -> pd.DataFrame:
    log("Applying final feature transformations")
    df_final_sample_transformed = df_final_sample.copy()

    # 6.1.1
    for col in ["user_account_age", "user_watch_live_age"]:
        scaled, _ = normalize_numeric_feature(df_final_sample_transformed, col, method="zscore")
        df_final_sample_transformed[col] = scaled

    # 6.1.2
    for col in ["ctr_user_15min", "ctr_user_3hr", "ctr_user_1d", "ctr_user_7d"]:
        df_final_sample_transformed[col] = df_final_sample_transformed[col].clip(lower=0, upper=1)

    # 6.1.3
    cols = [
        "num_imp_user_10min",
        "num_imp_user_30min",
        "num_imp_user_2hr",
        "num_imp_user_12hr",
        "num_imp_user_1d",
        "num_imp_user_7d",
    ]
    for c in cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        hi = s.quantile(0.99)
        df_final_sample_transformed[c] = s.clip(upper=hi)
    means = df_final_sample_transformed[cols].mean()
    stds = df_final_sample_transformed[cols].std().replace(0, 1.0)
    df_final_sample_transformed[cols] = (df_final_sample_transformed[cols] - means) / stds

    # 6.1.4
    cols = ["num_click_user_15min", "num_click_user_3hr", "num_click_user_1d", "num_click_user_7d"]
    for c in cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        hi = s.quantile(0.99)
        df_final_sample_transformed[c] = s.clip(upper=hi)
    means = df_final_sample_transformed[cols].mean()
    stds = df_final_sample_transformed[cols].std().replace(0, 1.0)
    df_final_sample_transformed[cols] = (df_final_sample_transformed[cols] - means) / stds

    col = "click_trend_user"
    s = df_final_sample_transformed[col]
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lower=lo, upper=hi)
    mean = s.mean()
    std = s.std() or 1.0
    df_final_sample_transformed[col] = (s - mean) / std

    # 6.1.5
    cols = ["time_since_last_impression_user", "time_since_last_click_user", "consecutive_skips_user"]
    for c in cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        hi = s.quantile(0.99)
        df_final_sample_transformed[c] = s.clip(upper=hi)
    means = df_final_sample_transformed[cols].mean()
    stds = df_final_sample_transformed[cols].std().replace(0, 1.0)
    df_final_sample_transformed[cols] = (df_final_sample_transformed[cols] - means) / stds

    # 6.1.6
    cols = ["avg_watch_time_user", "median_watch_time_user"]
    for c in cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        hi = s.quantile(0.99)
        df_final_sample_transformed[c] = s.clip(upper=hi)
    means = df_final_sample_transformed[cols].mean()
    stds = df_final_sample_transformed[cols].std().replace(0, 1.0)
    df_final_sample_transformed[cols] = (df_final_sample_transformed[cols] - means) / stds

    # 6.1.7
    df_final_sample_transformed["comment_rate_user"] = df_final_sample_transformed["comment_rate_user"].clip(0, 1)
    col = "num_comment_user_24h"
    s = df_final_sample_transformed[col].clip(lower=0)
    s = np.log1p(s)
    hi = s.quantile(0.99)
    s = s.clip(upper=hi)
    mean = s.mean()
    std = s.std() or 1.0
    df_final_sample_transformed[col] = (s - mean) / std

    # 6.1.8
    df_final_sample_transformed["like_rate_user"] = df_final_sample_transformed["like_rate_user"].clip(0, 1)
    col = "num_like_user_24h"
    s = df_final_sample_transformed[col].clip(lower=0)
    s = np.log1p(s)
    hi = s.quantile(0.99)
    s = s.clip(upper=hi)
    mean = s.mean()
    std = s.std() or 1.0
    df_final_sample_transformed[col] = (s - mean) / std

    # 6.1.9
    # NOTE: this mirrors notebook behavior exactly (includes original variable-name reuse).
    ols = ["num_gift_user_7d", "amount_gift_user_7d"]
    for c in cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        hi = s.quantile(0.99)
        df_final_sample_transformed[c] = s.clip(upper=hi)
    means = df_final_sample_transformed[cols].mean()
    stds = df_final_sample_transformed[cols].std().replace(0, 1.0)
    df_final_sample_transformed[cols] = (df_final_sample_transformed[cols] - means) / stds

    # 6.2
    log_std_cols = ["time_since_live_start", "time_since_start_live"]
    for c in log_std_cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        mean = s.mean()
        std = s.std() or 1.0
        df_final_sample_transformed[c] = (s - mean) / std

    clip_01_cols = [
        "ctr_room_10min",
        "ctr_room_30min",
        "ctr_room_2hr",
        "ctr_room_12hr",
        "comment_rate_live",
        "comment_rate_live_15min",
        "comment_rate_live_1hr",
        "comment_rate_live_3hr",
        "like_rate_live",
        "like_rate_live_15min",
        "like_rate_live_1hr",
        "like_rate_live_3hr",
        "gift_rate_live",
        "gift_rate_live_15min",
        "gift_rate_live_1hr",
        "gift_rate_live_3hr",
    ]
    for c in clip_01_cols:
        df_final_sample_transformed[c] = df_final_sample_transformed[c].clip(0, 1)

    log_clip_std_cols = [
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
        "num_comment_live",
        "num_comment_live_15min",
        "num_comment_live_1hr",
        "num_comment_live_3hr",
        "num_like_live",
        "num_like_live_15min",
        "num_like_live_1hr",
        "num_like_live_3hr",
        "num_gift_live",
        "num_gift_live_15min",
        "num_gift_live_1hr",
        "num_gift_live_3hr",
        "amount_gift_live",
        "amount_gift_live_15min",
        "amount_gift_live_1hr",
        "amount_gift_live_3hr",
    ]
    for c in log_clip_std_cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        hi = s.quantile(0.99)
        s = s.clip(upper=hi)
        mean = s.mean()
        std = s.std() or 1.0
        df_final_sample_transformed[c] = (s - mean) / std

    trend_cols = ["ctr_trend_room", "comment_trend_room", "like_trend_room", "gift_trend_room"]
    for c in trend_cols:
        s = df_final_sample_transformed[c]
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lower=lo, upper=hi)
        mean = s.mean()
        std = s.std() or 1.0
        df_final_sample_transformed[c] = (s - mean) / std

    std_cols = ["avg_watch_time_live", "median_watch_time_live", "avg_watch_time_live_30min", "median_watch_time_live_30min"]
    for c in std_cols:
        s = df_final_sample_transformed[c]
        mean = s.mean()
        std = s.std() or 1.0
        df_final_sample_transformed[c] = (s - mean) / std

    # 6.3
    log_std_cols = ["streamer_account_age", "streamer_live_age"]
    for c in log_std_cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        mean = s.mean()
        std = s.std() or 1.0
        df_final_sample_transformed[c] = (s - mean) / std

    clip_01_cols = ["ctr_streamer_1d", "ctr_streamer_7d"]
    for c in clip_01_cols:
        df_final_sample_transformed[c] = df_final_sample_transformed[c].clip(0, 1)

    log_clip_std_cols = [
        "num_imp_streamer_7d",
        "num_click_streamer_7d",
        "num_lives_streamer_7d",
        "num_comment_streamer_7d",
        "num_like_streamer_7d",
        "amount_gift_streamer_7d",
    ]
    for c in log_clip_std_cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        hi = s.quantile(0.99)
        s = s.clip(upper=hi)
        mean = s.mean()
        std = s.std() or 1.0
        df_final_sample_transformed[c] = (s - mean) / std

    std_cols = ["avg_watch_time_streamer", "median_watch_time_streamer"]
    for c in std_cols:
        s = df_final_sample_transformed[c]
        mean = s.mean()
        std = s.std() or 1.0
        df_final_sample_transformed[c] = (s - mean) / std

    # 6.4
    clip_01_cols = ["ctr_user_streamer_7d", "ctr_user_category_7d"]
    for c in clip_01_cols:
        df_final_sample_transformed[c] = df_final_sample_transformed[c].clip(0, 1)

    log_clip_std_cols = [
        "num_click_user_streamer_7d",
        "num_imp_user_streamer_7d",
        "num_click_user_category_7d",
        "num_imp_user_category_7d",
        "time_since_last_impression_user_streamer",
        "time_since_last_click_user_streamer",
    ]
    for c in log_clip_std_cols:
        s = df_final_sample_transformed[c].clip(lower=0)
        s = np.log1p(s)
        hi = s.quantile(0.99)
        s = s.clip(upper=hi)
        mean = s.mean()
        std = s.std() or 1.0
        df_final_sample_transformed[c] = (s - mean) / std

    return df_final_sample_transformed


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA_concise-equivalent data processing pipeline.")
    parser.add_argument("--sample-frac", type=float, default=0.02, help="Sampling fraction from df_final.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for sampling.")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path.")
    return parser.parse_args()


def main() -> None:
    start_ts = time.perf_counter()
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    out_path = Path(args.output) if args.output else (data_dir / "draft_sample_concise.csv")

    stages = [
        "load_raw",
        "build_interactions",
        "build_user",
        "build_room",
        "build_streamer",
        "merge_base",
        "feature_engineering",
        "transform",
        "save",
    ]

    bar = tqdm(total=len(stages), desc="Pipeline progress", unit="stage")

    dataframes = load_raw_data(data_dir)
    bar.update(1)

    df_interactions = build_interactions(dataframes)
    bar.update(1)

    df_user = build_user_features(dataframes)
    bar.update(1)

    df_room = build_room_features(dataframes, data_dir)
    bar.update(1)

    df_streamer = build_streamer_features(dataframes)
    bar.update(1)

    df_final = merge_tables(df_interactions, df_user, df_room, df_streamer)
    bar.update(1)

    df_final_sample = engineer_features(df_final, sample_frac=args.sample_frac, random_state=args.random_state)
    bar.update(1)

    df_final_sample_transformed = transform_features(df_final_sample)
    bar.update(1)

    log(f"Saving output to: {out_path}")
    df_final_sample_transformed.to_csv(out_path, index=False)
    bar.update(1)
    bar.close()

    log(f"Done. Output shape: {df_final_sample_transformed.shape}")
    elapsed = time.perf_counter() - start_ts
    log(f"Total runtime: {elapsed:.2f}s ({elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
