from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List


def _parse_int_list(raw: str) -> List[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    out = []
    for v in vals:
        out.append(int(v))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _fmt_float(v: object, ndigits: int = 6) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return ""


def _to_float(v: object) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Grid-sweep wrapper for pipeline_eval.py and summary table generation."
    )
    parser.add_argument("--topks", type=str, default="100")
    parser.add_argument("--multi-interest-ks", type=str, default="2")
    parser.add_argument("--output-prefix-base", type=str, default="pipeline_eval_sweep")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--pipeline-script", type=Path, default=root / "src" / "pipeline_eval.py")

    parser.add_argument("--ttnn-val-path", type=Path, default=root / "data" / "TTNN_full_test.csv")
    parser.add_argument("--ttnn-checkpoint-path", type=Path, default=root / "models" / "best_tower.pt")
    parser.add_argument("--ttnn-label-col", type=str, default="is_click")
    parser.add_argument("--dcn-data-path", type=Path, default=root / "data" / "dcnv2_full_retry_test.csv")
    parser.add_argument("--dcn-train-path", type=Path, default=root / "data" / "dcnv2_full_retry_train.csv")
    parser.add_argument("--dcn-checkpoint-path", type=Path, default=root / "models" / "dcnv2_rerun_with_id_v1.pt")
    parser.add_argument("--dcn-batch-size", type=int, default=4096)
    parser.add_argument("--dcn-num-workers", type=int, default=0)
    parser.add_argument("--dcn-reg-loss", type=str, default="huber")
    parser.add_argument("--dcn-watch-log-train-clip", type=float, default=8.0)
    parser.add_argument("--dcn-precision-ks", type=str, default="1,3,5,10")
    parser.add_argument("--dcn-ndcg-ks", type=str, default="10,50,100")
    parser.add_argument("--dcn-w-ctr", type=float, default=1.0)
    parser.add_argument("--dcn-w-watch30", type=float, default=1.0)
    parser.add_argument("--dcn-w-like", type=float, default=1.0)
    parser.add_argument("--dcn-w-watch-time", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--outputs-dir", type=Path, default=root / "outputs")
    parser.add_argument("--logs-dir", type=Path, default=root / "logs")
    return parser.parse_args()


def _iter_grid(topks: Iterable[int], multi_interest_ks: Iterable[int]) -> Iterable[tuple[int, int]]:
    for topk in topks:
        for mi in multi_interest_ks:
            yield int(topk), int(mi)


def _build_cmd(args: argparse.Namespace, topk: int, mi: int, prefix: str) -> list[str]:
    return [
        args.python_exe,
        str(args.pipeline_script),
        "--ttnn-val-path",
        str(args.ttnn_val_path),
        "--ttnn-checkpoint-path",
        str(args.ttnn_checkpoint_path),
        "--ttnn-label-col",
        str(args.ttnn_label_col),
        "--ttnn-topk",
        str(topk),
        "--ttnn-multi-interest-k",
        str(mi),
        "--dcn-data-path",
        str(args.dcn_data_path),
        "--dcn-train-path",
        str(args.dcn_train_path),
        "--dcn-checkpoint-path",
        str(args.dcn_checkpoint_path),
        "--dcn-batch-size",
        str(args.dcn_batch_size),
        "--dcn-num-workers",
        str(args.dcn_num_workers),
        "--dcn-reg-loss",
        str(args.dcn_reg_loss),
        "--dcn-watch-log-train-clip",
        str(args.dcn_watch_log_train_clip),
        "--dcn-precision-ks",
        str(args.dcn_precision_ks),
        "--dcn-ndcg-ks",
        str(args.dcn_ndcg_ks),
        "--dcn-w-ctr",
        str(args.dcn_w_ctr),
        "--dcn-w-watch30",
        str(args.dcn_w_watch30),
        "--dcn-w-like",
        str(args.dcn_w_like),
        "--dcn-w-watch-time",
        str(args.dcn_w_watch_time),
        "--device",
        str(args.device),
        "--output-prefix",
        prefix,
    ]


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_markdown(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |\n")


def _select_ok_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        status = str(r.get("status", ""))
        if status.startswith("failed_"):
            continue
        out.append(r)
    return out


def _rank_by_metric(rows: list[dict], metric: str) -> list[dict]:
    scored: list[tuple[float, int, int, dict]] = []
    for r in rows:
        v = _to_float(r.get(metric))
        if v is None:
            continue
        topk = int(r.get("topk", 10**9))
        mi = int(r.get("multi_interest_k", 10**9))
        scored.append((v, -topk, -mi, r))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [x[3] for x in scored]


def _build_best_summary(rows: list[dict]) -> dict:
    ok_rows = _select_ok_rows(rows)
    metrics = ["ttnn_recall_mean", "pair_overlap_rate", "auc_mean"]
    out: dict = {
        "num_runs_total": len(rows),
        "num_runs_ok": len(ok_rows),
        "best_by_metric": {},
        "leaderboard_top5": {},
    }
    for m in metrics:
        ranked = _rank_by_metric(ok_rows, m)
        out["best_by_metric"][m] = ranked[0] if ranked else None
        out["leaderboard_top5"][m] = ranked[:5]
    return out


def _append_best_to_markdown(path: Path, best: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n\n## Best Configs\n")
        f.write(f"- num_runs_total: {best.get('num_runs_total', 0)}\n")
        f.write(f"- num_runs_ok: {best.get('num_runs_ok', 0)}\n")
        best_map = best.get("best_by_metric", {})
        for m in ["ttnn_recall_mean", "pair_overlap_rate", "auc_mean"]:
            item = best_map.get(m)
            if not item:
                f.write(f"- best_{m}: <none>\n")
                continue
            f.write(
                f"- best_{m}: run={item.get('run')} "
                f"topk={item.get('topk')} mi={item.get('multi_interest_k')} value={item.get(m)}\n"
            )


def main() -> None:
    args = parse_args()
    topks = _parse_int_list(args.topks)
    multi_interest_ks = _parse_int_list(args.multi_interest_ks)

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for topk, mi in _iter_grid(topks, multi_interest_ks):
        prefix = f"{args.output_prefix_base}_topk{topk}_mi{mi}"
        log_path = args.logs_dir / f"{prefix}.log"
        summary_path = args.outputs_dir / f"{prefix}_summary.json"

        start_ts = time.time()
        rc = 0
        status = "ok"

        if args.skip_existing and summary_path.exists():
            status = "skipped_existing"
        else:
            cmd = _build_cmd(args=args, topk=topk, mi=mi, prefix=prefix)
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
                rc = int(proc.returncode)
            if rc != 0:
                status = f"failed_rc_{rc}"

        elapsed = time.time() - start_ts
        summary = _read_summary(summary_path)

        overlap = summary.get("overlap_diagnostics", {})
        overlap_rates = overlap.get("overlap_rates_wrt_candidates", {})
        overlap_counts = overlap.get("overlap_counts", {})
        cand_counts = overlap.get("candidate_unique_counts", {})
        dcn_metrics = summary.get("dcn_metrics", {})

        row = {
            "run": prefix,
            "status": status,
            "rc": rc,
            "topk": topk,
            "multi_interest_k": mi,
            "duration_sec": f"{elapsed:.1f}",
            "ttnn_recall_mean": _fmt_float(summary.get("ttnn_recall_mean")),
            "pair_overlap_rate": _fmt_float(overlap_rates.get("pairs")),
            "matched_pairs": overlap_counts.get("pairs", ""),
            "candidate_pairs": cand_counts.get("pairs", ""),
            "auc_mean": _fmt_float(dcn_metrics.get("auc_mean")),
            "auc_is_click": _fmt_float(dcn_metrics.get("auc_is_click")),
            "precision@10_is_click": _fmt_float(dcn_metrics.get("precision@10_is_click")),
            "ndcg@10_is_click": _fmt_float(dcn_metrics.get("ndcg@10_is_click")),
            "summary_path": str(summary_path),
            "log_path": str(log_path),
        }
        rows.append(row)

        print(
            f"[{status}] topk={topk} mi={mi} "
            f"recall={row['ttnn_recall_mean']} overlap={row['pair_overlap_rate']} "
            f"auc={row['auc_mean']}"
        )

        if args.fail_fast and status.startswith("failed_"):
            break

    csv_fields = [
        "run",
        "status",
        "rc",
        "topk",
        "multi_interest_k",
        "duration_sec",
        "ttnn_recall_mean",
        "pair_overlap_rate",
        "matched_pairs",
        "candidate_pairs",
        "auc_mean",
        "auc_is_click",
        "precision@10_is_click",
        "ndcg@10_is_click",
        "summary_path",
        "log_path",
    ]
    csv_path = args.outputs_dir / f"{args.output_prefix_base}_results.csv"
    md_path = args.outputs_dir / f"{args.output_prefix_base}_results.md"
    best_path = args.outputs_dir / f"{args.output_prefix_base}_best_configs.json"
    _write_csv(csv_path, rows, csv_fields)
    _write_markdown(md_path, rows, csv_fields)
    best = _build_best_summary(rows)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    _append_best_to_markdown(md_path, best)
    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    print(f"saved: {best_path}")


if __name__ == "__main__":
    main()
