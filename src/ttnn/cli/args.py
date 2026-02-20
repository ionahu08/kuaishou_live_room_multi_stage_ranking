import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Two-tower training on draft_sample.csv")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=project_root / "data" / "TTNN_train.csv",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=project_root / "data" / "TTNN_val.csv",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=project_root / "data" / "TTNN_test.csv",
    )
    parser.add_argument("--label-col", type=str, default="is_click")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache preprocessed train/val/test tables for faster repeated runs.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild preprocessing cache even if an existing cache entry is found.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--loss",
        type=str,
        default="bce",
        choices=["bce", "contrastive", "triplet"],
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--emb-dim", type=int, default=512)
    parser.add_argument(
        "--tower-hidden",
        type=str,
        default="1024,512",
        help="Comma-separated hidden sizes for each tower, e.g. 256,128",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalize-emb", action="store_true")
    parser.add_argument(
        "--debug-sim",
        action="store_true",
        help="Print contrastive similarity matrix for each batch",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=project_root / "logs" / "train_progress.png",
    )
    parser.add_argument(
        "--eval-ks",
        type=str,
        default="10,50,100",
        help="Comma-separated K values for retrieval metrics (e.g. 10,50,100)",
    )
    parser.add_argument(
        "--max-pos-per-user",
        type=int,
        default=30,
        help="Max positives per user per batch (contrastive only)",
    )
    parser.add_argument(
        "--freshness-weighting",
        action="store_true",
        help="Enable freshness weighting on contrastive loss",
    )
    parser.add_argument(
        "--freshness-half-life-hours",
        type=float,
        default=24.0,
        help="Half-life in hours for freshness weighting",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping based on recall@K on validation set",
    )
    parser.add_argument(
        "--early-patience",
        type=int,
        default=3,
        help="Stop after N epochs without improvement",
    )
    parser.add_argument(
        "--early-k",
        type=int,
        default=100,
        help="Use recall@K for early stopping",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=project_root / "models" / "best_tower.pt",
        help="Path to save the best model checkpoint",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=5,
        help="Save periodic model checkpoints every N epochs (0 disables periodic saves).",
    )
    parser.add_argument(
        "--periodic-checkpoint-dir",
        type=Path,
        default=project_root / "models",
        help="Directory for periodic epoch checkpoints.",
    )
    parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Print retrieval debug stats (positives/users/pool size)",
    )
    parser.add_argument(
        "--multi-interest-k",
        type=int,
        default=2,
        choices=[1, 2],
        help="Use 2 interest vectors per user at retrieval time (k=2)",
    )
    parser.add_argument(
        "--test-topk-k",
        type=int,
        default=100,
        help="Top-K recommendations to export per user on test split",
    )
    parser.add_argument(
        "--test-topk-output-path",
        type=Path,
        default=project_root / "outputs" / "test_topk_recommendations.csv",
        help="Path to save per-user test top-K recommendations",
    )
    parser.add_argument(
        "--unseen-path",
        type=Path,
        default=None,
        help="Optional unseen CSV path for retrieval-only top-K export",
    )
    parser.add_argument(
        "--unseen-topk-k",
        type=int,
        default=100,
        help="Top-K recommendations to export per user on unseen data",
    )
    parser.add_argument(
        "--unseen-topk-output-path",
        type=Path,
        default=None,
        help="Optional output CSV path for unseen top-K recommendations",
    )
    parser.add_argument(
        "--bce-pos-weight",
        type=float,
        default=None,
        help="Optional positive class weight for BCE; if unset, auto uses neg_count/pos_count",
    )
    parser.add_argument(
        "--bce-weighted-sampler",
        action="store_true",
        help="Use WeightedRandomSampler to rebalance BCE mini-batches",
    )
    parser.add_argument(
        "--weak-positive-weight",
        type=float,
        default=0.3,
        help="For BCE: sample weight multiplier for positives with watch_live_time < 30s",
    )
    return parser.parse_args()
