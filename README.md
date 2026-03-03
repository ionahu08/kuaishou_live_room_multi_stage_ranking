# Multi-Stage Ranking Pipeline (Kuaishou)

This README explains what each Python file in `src/` does for the requested pipeline:

`data_processing.py --> TTNN_data_processer.py --> two_tower_train.py --> dcnv2_data_processer.py --> dcnv2_train.py --> pipeline_eval.py --> rerank_eval.py`

## 1) Data Sources and Features Used

### Data description (KuaILive)
- Dataset scope:
  - Real behavior logs from users engaging in four interaction types: click, comment, like, and gift.
  - Time range: **May 5, 2025 to May 25, 2025**.
  - Includes both-side profile/context information (for users and streamers) and fine-grained behavioral signals (for example, watch time and gift price).

- Basic statistics (from KuaILive description):

| Dataset | #Users | #Streamers | #Rooms | #Interactions | #Clicks | #Comments | #Likes | #Gifts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiLive | 23,772 | 452,621 | 11,613,708 | 5,357,998 | 4,909,515 | 196,526 | 179,311 | 72,646 |

- Official feature/details page:
  - https://imgkkk574.github.io/KuaiLive/detailed_statistics.html

### Raw data sources (`data/`)
- `click.csv`, `negative.csv`: impression/click supervision.
- `like.csv`, `comment.csv`, `gift.csv`: engagement signals and gift amount.
- `user.csv`: user profile and behavior stats.
- `room.csv`: live-room metadata and timestamps.
- `streamer.csv`: streamer profile and cumulative stats.
- `title_embeddings.npy`: 128-dim title embedding vectors aligned to `live_name_id`.

### Core feature groups produced by `data_processing.py`
- Interaction-time features: `imp_year`, `imp_month`, `imp_day`, `imp_hour`, `imp_is_weekend`.
- User-side features: profile encodings, account age, rolling CTR/count windows, recency, watch/comment/like/gift histories.
- Room/stream features: live-start recency, room CTR/count windows, engagement trends, title embeddings (`title_emb_0..127`).
- Streamer-side features: streamer account/live age, rolling streamer CTR/count/watch/engagement summaries.
- Cross features: user-streamer and user-category rolling interaction features.
- Labels: `is_click`, `watch_live_time`, derived `watch_greater_30s`, `is_like`.

### Feature subsets consumed later
- TTNN (`two_tower_train.py` / `ttnn` package):
  - User categorical + numeric (schema in `src/ttnn/config/feature_schema.py`).
  - Item(stream/streamer) categorical + numeric, including title embeddings.
- DCNv2 (`dcnv2_data_processer.py`, `dcnv2_train.py`):
  - Large mixed set of categorical (`_le`, IDs, time buckets) and numeric behavior features.
  - Multitask labels: `is_click`, `watch_greater_30s`, `is_like`, `watch_live_time`.

### Label and objective definition
- Binary tasks:
  - `is_click`
  - `watch_greater_30s` (derived from `watch_live_time >= 30` when not provided)
  - `is_like`
- Regression task:
  - `watch_live_time_log = log1p(watch_live_time)`
- DCNv2 training objective:
  - Weighted multitask loss = BCE(`is_click`) + BCE(`watch_greater_30s`) + BCE(`is_like`) + regression loss (`watch_live_time_log`)
- Rerank relevance score (`rerank_eval.py`):
  - `relevance = w_ctr*score_ctr + w_watch30*score_watch_greater_30s + w_like*score_like`

## 2) Workflow Across Py Files

### Step 1: `src/data_processing.py`
- Loads raw CSV/NPY tables.
- Reconstructs interaction table by aligning click with like/comment/gift events and appending negative samples.
- Merges interaction + user + room + streamer tables.
- Engineers large rolling-window feature set.
- Applies feature transformation (winsorization/standardization for numeric columns).
- Writes feature table (default: `data/draft_sample_concise.csv`; downstream run used `data/draft_sample.csv`).

### Step 2: `src/TTNN_data_processer.py`
- Input default: `data/full_data.csv`.
- Streams data chunk-by-chunk, deduplicates by `(live_id, streamer_id, user_id)` using click-priority rule:
  - If any clicked row exists in group: keep earliest clicked row.
  - Else: keep latest impression row.
- Splits by date:
  - Train: `2025-05-04` to `2025-05-18`
  - Val: `2025-05-19` to `2025-05-22`
  - Test: `>= 2025-05-23`
- Writes: `data/TTNN_full_train.csv`, `data/TTNN_full_val.csv`, `data/TTNN_full_test.csv`.

#### TTNN dedup/split rule (explicit contract)
- Dedup key: `(live_id, streamer_id, user_id)`.
- Row selection policy:
  - any-click group -> keep earliest clicked row
  - no-click group -> keep latest impression row
- Split by `imp_timestamp` date only:
  - Train: `2025-05-04` to `2025-05-18` (inclusive)
  - Val: `2025-05-19` to `2025-05-22` (inclusive)
  - Test: `>= 2025-05-23`
- Output contract:
  - `TTNN_full_train.csv`
  - `TTNN_full_val.csv`
  - `TTNN_full_test.csv`

### Step 3: `src/two_tower_train.py`
- Thin launcher to `ttnn.main`.
- Trains two-tower retrieval model (`TwoTowerModel`) on TTNN splits.
- Produces TTNN checkpoints (e.g., `models/best_tower.pt`, `models/best_tower_recall_100.pt`).

### Step 4: `src/dcnv2_data_processer.py`
- Input default: `data/draft_sample.csv`.
- Adds/normalizes multitask labels (`watch_greater_30s` derived from watch time when missing).
- Filters to requested DCNv2 feature columns.
- Splits by same date windows as TTNN.
- Writes DCNv2 train/val/test CSVs (e.g., `dcnv2_full_retry_{train,val,test}.csv`).

### Step 5: `src/dcnv2_train.py`
- Trains multitask DCNv2:
  - Binary heads: `is_click`, `watch_greater_30s`, `is_like`.
  - Regression head: `watch_live_time_log`.
- Uses validation `auc_mean` to select best checkpoint.
- Writes checkpoint and test predictions CSV.

### Step 6: `src/pipeline_eval.py`
- End-to-end 2-stage evaluation:
  - Stage A: TTNN ANN retrieval top-K candidates per user.
  - Stage B: Join with DCNv2 test data, score candidates, compute ranking metrics.
- Writes artifacts:
  - `*_ttnn_topK.csv`, `*_ttnn_recall_at_K_by_user.csv`
  - `*_overlap_diagnostics.json`
  - `*_dcn_scored_candidates.csv`
  - `*_summary.json`

### Step 7: `src/rerank_eval.py`
- Takes DCNv2 scored candidates and applies MMR reranking (diversity-aware).
- Evaluates before/after rerank metrics (NDCG/Precision/Recall + diversity/freshness/popularity diagnostics).
- Writes:
  - baseline top-N CSV
  - MMR top-N CSV
  - rerank summary JSON

## Leakage Guardrails

The DCNv2 training code excludes or controls high-risk features that can leak near-future behavior in offline settings.

- Default dropped leakage-sensitive columns (from `DEFAULT_DROP_COLUMNS` in `dcnv2_train.py`):
  - `time_since_last_click_user`
  - `tslc_missing`
  - `consecutive_skips_user`
  - `time_since_last_click_user_streamer`
  - `tslc_user_streamer_missing`
- Additional exclusions:
  - raw timestamps and `*_ts` columns are excluded from model features
  - non-feature timestamp/profile fields are excluded (e.g., registration/start/end timestamps)

These guardrails are designed to reduce optimistic offline lift that may not survive online serving constraints.

## Metrics Dictionary

- `Recall@K`:
  - Fraction of a user's relevant items recovered in top-K.
- `Precision@K`:
  - Fraction of top-K items that are relevant.
- `NDCG@K`:
  - Position-weighted ranking quality at K, normalized by ideal ranking.
- `AUC` (`auc_is_click`, `auc_watch_greater_30s`, `auc_is_like`):
  - Pairwise discrimination between positives and negatives for each binary task.
- `auc_mean`:
  - Mean of available task AUCs.
- `rmse_watch_live_time_log`:
  - RMSE for regression head on `watch_live_time_log`.

Computation notes:
- Pipeline and rerank ranking metrics are computed per user and then averaged across users.
- Candidate-conditioned metrics are evaluated on the matched candidate set after retrieval/ranker join, not on the full test universe.

## 3) Results (Based on Existing Logs and Outputs)

### Data/split sizes found in repository
- `data/draft_sample.csv`: **352,307** rows.
- `data/TTNN_full_train.csv`: **8,726,802** rows.
- `data/TTNN_full_val.csv`: **2,209,953** rows.
- `data/TTNN_full_test.csv`: **1,561,154** rows.
- `data/dcnv2_full_retry_train.csv`: **12,304,625** rows.
- `data/dcnv2_full_retry_val.csv`: **3,089,432** rows.
- `data/dcnv2_full_retry_test.csv`: **2,221,293** rows.

### DCNv2 full test performance (`logs/dcnv2_rerun_with_id_v1.log`)
- Best checkpoint loaded at epoch **69**.
- Test metrics:
  - `auc_is_click`: **0.946381**
  - `auc_watch_greater_30s`: **0.946287**
  - `auc_is_like`: **0.877650**
  - `auc_mean`: **0.923440**
  - `precision@10_is_click`: **0.439414**
  - `precision@10_is_like`: **0.039536**
  - `rmse_watch_live_time_log`: **0.674929**
- Test prediction output rows: **2,221,293** (`outputs/dcnv2_rerun_with_id_v1_test_preds.csv`).

### TTNN test results (retrieval stage)
- Evaluation split: `data/TTNN_full_test.csv`.
- Checkpoint family: `models/best_tower.pt` (and named best variants under `models/`).
- Notes:
  - A separate standalone `two_tower_train` test log snapshot is not present under `logs/`.
  - TTNN retrieval metrics below are taken from saved `pipeline_eval*_summary.json` artifacts on the TTNN test split.

| Run | topK | multi_interest_k | mean recall@K | users with positives | exported topK rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pipeline_eval_grid_v3_topk50_mi1` | 50 | 1 | 0.006754 | 19,921 | 1,136,050 |
| `pipeline_eval_grid_v3_topk50_mi2` | 50 | 2 | 0.068207 | 19,921 | 1,136,050 |
| `pipeline_eval_grid_v3_topk100_mi1` | 100 | 1 | 0.010852 | 19,921 | 2,272,100 |
| `pipeline_eval_test_v1` / `pipeline_eval_grid_v3_topk100_mi2` | 100 | 2 | 0.073070 | 19,921 | 2,272,100 |
| `pipeline_eval_grid_v3_topk200_mi1` | 200 | 1 | 0.018603 | 19,921 | 4,544,200 |

- Key takeaway:
  - `multi_interest_k=2` substantially improves TTNN recall at the same K (for example at `K=100`: **0.073070** vs **0.010852**).

### End-to-end pipeline results (`outputs/pipeline_eval_*_summary.json`)
- `pipeline_eval_test_v1` (`topk=100`, `mi=2`):
  - TTNN recall@100 mean: **0.07307**
  - Candidate pair overlap into DCN test: **1.77%**
  - DCN on matched candidates: `auc_mean` **0.78642**, `precision@10_is_click` **0.22710**, `ndcg@10_is_click` **0.89062**
  - Scored candidates: **40,259** rows

- Grid snapshots (`v3`):
  - `topk=200, mi=1`: strongest candidate-set DCN quality among listed configs
    - `auc_mean` **0.84473**, `rmse_watch_live_time_log` **0.95103**
    - but very low pair overlap (**0.36%**) and TTNN recall@200 mean (**0.01860**)
  - `topk=50, mi=2`: higher TTNN recall/overlap than `mi=1`, but lower DCN candidate metrics than `mi=1`.

### Rerank results (`outputs/rerank_top100_mi2_*_summary.json`)
- Input scored rows: **40,259**; reranked outputs: **40,228**.
- Most lambda settings (`0.5` to `0.9`) slightly reduced click NDCG@10.
- `rerank_top100_mi2_lam08_numsim_v1` (cat+numeric similarity) gave the most balanced behavior:
  - `delta ndcg@10_is_click`: **-0.000244** (very small drop)
  - `delta ndcg@10_is_like`: **+0.000199** (small gain)
  - diversity change ~0, freshness score small positive change.

### Rerank configuration registry

| run | lambda_mmr | similarity features | delta ndcg@10_is_click | delta ndcg@10_is_like | diversity delta (unique streamers mean@N) |
| --- | ---: | --- | ---: | ---: | ---: |
| `rerank_top100_mi2_v1` | 0.7 | categorical only | -0.000557 | -0.001085 | +0.000088 |
| `rerank_top100_mi2_lam09` | 0.9 | categorical only | -0.000119 | +0.000226 | +0.000000 |
| `rerank_top100_mi2_lam08` | 0.8 | categorical only | -0.000421 | -0.000209 | +0.000044 |
| `rerank_top100_mi2_lam06` | 0.6 | categorical only | -0.000836 | -0.002004 | +0.000132 |
| `rerank_top100_mi2_lam05` | 0.5 | categorical only | -0.001273 | -0.002643 | +0.000176 |
| `rerank_top100_mi2_lam08_numsim_v1` | 0.8 | categorical + numeric (`time_since_live_start`, `accu_play_cnt_le`, `num_click_room_1d`, `ctr_room_12hr`) | -0.000244 | +0.000199 | +0.000000 |

## 4) Conclusion

- The project implements a complete retrieval-to-ranking-to-reranking stack with reproducible artifacts at every stage.
- The strongest standalone DCNv2 test performance is high (`auc_mean` ~0.923), but in pipeline mode final quality is constrained by retrieval-to-ranker pair overlap.
- Current best practical improvement direction is to increase TTNN candidate overlap/coverage without losing candidate quality; reranking currently yields only marginal tradeoffs.
- In short: **the bottleneck is candidate handoff (retrieval coverage/alignment), not DCNv2 model capacity.**
