import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_filename(file_path: Path, suffix: str) -> Optional[Tuple[str, str]]:
    """Extract model and probe from names like '<model>_<probe>.report.jsonl'."""
    name = file_path.name
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    if "_" not in stem:
        return None
    model_id, probe = stem.rsplit("_", 1)
    return model_id, probe


def read_jsonl(file_path: Path) -> Iterable[dict]:
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_report_rows(
    report_dir: Path,
    models_filter: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempt_rows = []
    eval_rows = []
    digest_group_rows = []
    digest_probe_rows = []

    for file_path in sorted(report_dir.glob("*.report.jsonl")):
        parsed = parse_filename(file_path, ".report.jsonl")
        if not parsed:
            continue
        model_id, probe_from_name = parsed
        if models_filter and model_id not in models_filter:
            continue

        for item in read_jsonl(file_path):
            entry_type = item.get("entry_type")

            if entry_type == "attempt":
                attempt_rows.append(
                    {
                        "model": model_id,
                        "probe": item.get("probe_classname", probe_from_name),
                        "attempt_uuid": item.get("uuid"),
                        "seq": item.get("seq"),
                        "status": item.get("status"),
                        "goal": item.get("goal"),
                        "source_file": file_path.name,
                    }
                )

            if entry_type == "eval":
                passed = item.get("passed")
                total = item.get("total")
                if passed is None or total in (None, 0):
                    continue
                try:
                    passed = float(passed)
                    total = float(total)
                except (TypeError, ValueError):
                    continue
                eval_rows.append(
                    {
                        "model": model_id,
                        "probe": item.get("probe", probe_from_name),
                        "detector": item.get("detector"),
                        "passed": passed,
                        "total": total,
                        "pass_rate": passed / total if total else None,
                        "source_file": file_path.name,
                    }
                )

            if entry_type == "digest":
                eval_block = item.get("eval", {})
                for group_name, group_data in eval_block.items():
                    if not isinstance(group_data, dict):
                        continue
                    group_summary = group_data.get("_summary", {})
                    digest_group_rows.append(
                        {
                            "model": model_id,
                            "group": group_name,
                            "group_score": group_summary.get("score"),
                            "group_defcon": group_summary.get("group_defcon"),
                            "source_file": file_path.name,
                        }
                    )

                    for probe_name, probe_data in group_data.items():
                        if probe_name == "_summary" or not isinstance(probe_data, dict):
                            continue
                        probe_summary = probe_data.get("_summary", {})
                        digest_probe_rows.append(
                            {
                                "model": model_id,
                                "group": group_name,
                                "probe": probe_name,
                                "probe_score": probe_summary.get("probe_score"),
                                "probe_severity": probe_summary.get("probe_severity"),
                                "source_file": file_path.name,
                            }
                        )

    return (
        pd.DataFrame(attempt_rows),
        pd.DataFrame(eval_rows),
        pd.DataFrame(digest_group_rows),
        pd.DataFrame(digest_probe_rows),
    )


def summarize_attempt_statuses(attempts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if attempts.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    attempt_status = attempts.copy()
    attempt_status["status"] = pd.to_numeric(attempt_status["status"], errors="coerce")
    attempt_status = attempt_status.dropna(subset=["attempt_uuid", "status"]).copy()

    # Keep one row per (attempt, status) because report files can emit repeated states.
    attempt_status = attempt_status.drop_duplicates(
        subset=["model", "probe", "attempt_uuid", "status"]
    )

    attempt_max_status = (
        attempt_status.groupby(["model", "probe", "attempt_uuid"], as_index=False)
        .agg(max_status=("status", "max"))
    )
    attempt_max_status["is_hit"] = attempt_max_status["max_status"] >= 2

    by_probe = (
        attempt_max_status.groupby(["model", "probe"], as_index=False)
        .agg(
            total_attempts=("attempt_uuid", "count"),
            hit_attempts=("is_hit", "sum"),
        )
    )
    by_probe["hit_rate"] = by_probe["hit_attempts"] / by_probe["total_attempts"]
    by_probe["hit_percent"] = by_probe["hit_rate"] * 100.0

    by_model = (
        attempt_max_status.groupby("model", as_index=False)
        .agg(
            total_attempts=("attempt_uuid", "count"),
            hit_attempts=("is_hit", "sum"),
        )
    )
    by_model["hit_rate"] = by_model["hit_attempts"] / by_model["total_attempts"]
    by_model["hit_percent"] = by_model["hit_rate"] * 100.0

    status_distribution = (
        attempt_status.groupby(["model", "status"], as_index=False)
        .agg(count=("attempt_uuid", "count"))
    )
    total_by_model = status_distribution.groupby("model")["count"].transform("sum")
    status_distribution["fraction"] = status_distribution["count"] / total_by_model

    return by_model, by_probe, status_distribution


def summarize_report_eval(report_eval: pd.DataFrame) -> pd.DataFrame:
    if report_eval.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = (
        report_eval.groupby(["model", "probe", "detector"])
        .agg(total_passed=("passed", "sum"), total_samples=("total", "sum"))
        .reset_index()
    )
    grouped["pass_rate"] = grouped["total_passed"] / grouped["total_samples"]
    grouped["pass_percent"] = grouped["pass_rate"] * 100.0
    return grouped


def summarize_eval_by_probe(report_eval_summary: pd.DataFrame) -> pd.DataFrame:
    if report_eval_summary.empty:
        return pd.DataFrame()

    by_probe = (
        report_eval_summary.groupby(["model", "probe"], as_index=False)
        .agg(total_passed=("total_passed", "sum"), total_samples=("total_samples", "sum"))
    )
    by_probe["pass_rate"] = by_probe["total_passed"] / by_probe["total_samples"]
    by_probe["pass_percent"] = by_probe["pass_rate"] * 100.0
    return by_probe


def summarize_eval_by_model(report_eval_summary: pd.DataFrame) -> pd.DataFrame:
    if report_eval_summary.empty:
        return pd.DataFrame()

    by_model = (
        report_eval_summary.groupby("model", as_index=False)
        .agg(total_passed=("total_passed", "sum"), total_samples=("total_samples", "sum"))
    )
    by_model["pass_rate"] = by_model["total_passed"] / by_model["total_samples"]
    by_model["pass_percent"] = by_model["pass_rate"] * 100.0
    return by_model


def save_plots(
    attempt_hit_by_probe: pd.DataFrame,
    eval_by_model: pd.DataFrame,
    eval_by_probe: pd.DataFrame,
    report_eval_summary: pd.DataFrame,
    digest_group_rows: pd.DataFrame,
    digest_probe_rows: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    # This plot was intentionally disabled; remove stale artifact from previous runs.
    status_plot_path = output_dir / "plot_report_status_distribution.png"
    if status_plot_path.exists():
        status_plot_path.unlink()

    # Remove stale file name kept from older versions.
    old_probe_rate_path = output_dir / "plot_report_hit_rate_by_probe.png"
    if old_probe_rate_path.exists():
        old_probe_rate_path.unlink()

    # This plot was removed from the report; delete stale artifact from previous runs.
    old_hit_attempts_path = output_dir / "plot_report_hit_attempts_by_model.png"
    if old_hit_attempts_path.exists():
        old_hit_attempts_path.unlink()

    if not eval_by_model.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(data=eval_by_model, x="model", y="pass_percent")
        plt.title("Eval Pass Rate by Model (%)")
        plt.xlabel("Model")
        plt.ylabel("Pass Rate (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_report_hit_rate_by_model.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5))
        sns.barplot(data=eval_by_model, x="model", y="total_passed")
        plt.title("Total Passed by Model (absolute count)")
        plt.xlabel("Model")
        plt.ylabel("Total Passed")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_report_total_passed_by_model.png", dpi=180)
        plt.close()

    if not eval_by_probe.empty:
        all_probes = eval_by_probe.sort_values(["total_samples", "probe"], ascending=[False, True]).copy()
        probe_count = all_probes["probe"].nunique()
        figure_width = max(12, min(40, probe_count * 0.85))

        plt.figure(figsize=(figure_width, 7))
        sns.barplot(data=all_probes, x="probe", y="pass_percent", hue="model")
        plt.title("Eval Pass Rate by Probe and Model (%)")
        plt.xlabel("Probe")
        plt.ylabel("Pass Rate (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_report_pass_rate_by_probe.png", dpi=180)
        plt.close()

        plt.figure(figsize=(figure_width, 7))
        sns.barplot(data=all_probes, x="probe", y="total_passed", hue="model")
        plt.title("Total Passed by Probe and Model (absolute count)")
        plt.xlabel("Probe")
        plt.ylabel("Total Passed")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_report_total_passed_by_probe.png", dpi=180)
        plt.close()

    if not report_eval_summary.empty:
        eval_sorted = report_eval_summary.copy()
        eval_sorted["probe_detector"] = eval_sorted["probe"] + " | " + eval_sorted["detector"]

        top_probe_detectors = (
            eval_sorted.groupby("probe_detector", as_index=False)
            .agg(total_samples_all_models=("total_samples", "sum"))
            .sort_values("total_samples_all_models", ascending=False)
            .head(25)["probe_detector"]
            .tolist()
        )

        eval_plot = eval_sorted[eval_sorted["probe_detector"].isin(top_probe_detectors)].copy()
        pivot_eval = eval_plot.pivot_table(
            index="probe_detector", columns="model", values="pass_rate", aggfunc="mean"
        )
        if not pivot_eval.empty:
            pivot_eval = pivot_eval.reindex(top_probe_detectors)
        if not pivot_eval.empty:
            plt.figure(figsize=(10, max(4, len(pivot_eval) * 0.35)))
            sns.heatmap(pivot_eval * 100.0, annot=True, cmap="magma", fmt=".1f", vmin=0, vmax=100)
            plt.title("Detector Pass Rate Heatmap (%)")
            plt.xlabel("Model")
            plt.ylabel("Probe | Detector")
            plt.tight_layout()
            plt.savefig(output_dir / "plot_report_detector_passrate_heatmap.png", dpi=180)
            plt.close()

    if not digest_group_rows.empty:
        digest_copy = digest_group_rows.copy()
        digest_copy["group_score"] = pd.to_numeric(digest_copy["group_score"], errors="coerce")
        digest_copy = digest_copy.dropna(subset=["group_score"])
        if not digest_copy.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=digest_copy, x="group", y="group_score", hue="model")
            plt.title("Digest Group Score by Model")
            plt.xlabel("Group")
            plt.ylabel("Score")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "plot_digest_group_score.png", dpi=180)
            plt.close()

    if not digest_probe_rows.empty:
        pivot = digest_probe_rows.pivot_table(
            index="probe", columns="model", values="probe_score", aggfunc="mean"
        )
        if not pivot.empty:
            plt.figure(figsize=(10, max(4, len(pivot) * 0.35)))
            sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".3f")
            plt.title("Digest Probe Score Heatmap")
            plt.xlabel("Model")
            plt.ylabel("Probe")
            plt.tight_layout()
            plt.savefig(output_dir / "plot_digest_probe_heatmap.png", dpi=180)
            plt.close()

    if not attempt_hit_by_probe.empty or not eval_by_probe.empty:
        hit_cols = ["model", "probe", "hit_attempts"]
        pass_cols = ["model", "probe", "total_passed"]
        hit_by_probe = (
            attempt_hit_by_probe[hit_cols].copy()
            if not attempt_hit_by_probe.empty
            else pd.DataFrame(columns=hit_cols)
        )
        pass_by_probe = (
            eval_by_probe[pass_cols].copy() if not eval_by_probe.empty else pd.DataFrame(columns=pass_cols)
        )

        per_probe_summary = hit_by_probe.merge(pass_by_probe, on=["model", "probe"], how="outer")
        per_probe_summary[["hit_attempts", "total_passed"]] = per_probe_summary[
            ["hit_attempts", "total_passed"]
        ].fillna(0)

        for model_name, model_data in per_probe_summary.groupby("model"):
            model_data = model_data.copy()
            if model_data.empty:
                continue

            model_data["hit_attempts"] = pd.to_numeric(model_data["hit_attempts"], errors="coerce").fillna(0)
            model_data["total_passed"] = pd.to_numeric(model_data["total_passed"], errors="coerce").fillna(0)
            model_data = model_data.sort_values("probe")

            long_data = model_data.melt(
                id_vars=["probe"],
                value_vars=["hit_attempts", "total_passed"],
                var_name="metric",
                value_name="count",
            )
            long_data["metric"] = long_data["metric"].map(
                {
                    "hit_attempts": "Hits",
                    "total_passed": "Passes",
                }
            )

            plt.figure(figsize=(13, 6))
            sns.barplot(data=long_data, x="probe", y="count", hue="metric")
            plt.title(f"Hits and Passes by Probe - {model_name}")
            plt.xlabel("Probe")
            plt.ylabel("Count")
            plt.legend(title="Metric", loc="upper right")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name).strip())
            plt.savefig(output_dir / f"plot_hits_passes_by_probe_{safe_model}.png", dpi=180)
            plt.close()


def create_overview_table(
    attempt_hit_by_model: pd.DataFrame,
    eval_by_model: pd.DataFrame,
    digest_group_rows: pd.DataFrame,
    report_eval_summary: pd.DataFrame,
) -> pd.DataFrame:
    overview = eval_by_model.copy() if not eval_by_model.empty else pd.DataFrame()

    if not attempt_hit_by_model.empty:
        status_hits = attempt_hit_by_model[["model", "total_attempts", "hit_attempts", "hit_rate"]].copy()
        overview = status_hits if overview.empty else overview.merge(status_hits, on="model", how="outer")

    if not digest_group_rows.empty:
        digest_copy = digest_group_rows.copy()
        digest_copy["group_score"] = pd.to_numeric(digest_copy["group_score"], errors="coerce")
        digest_summary = (
            digest_copy.groupby("model")
            .agg(digest_group_score_mean=("group_score", "mean"), groups_seen=("group", "nunique"))
            .reset_index()
        )
        overview = digest_summary if overview.empty else overview.merge(digest_summary, on="model", how="outer")

    if not report_eval_summary.empty:
        detector_summary = (
            report_eval_summary.groupby("model")
            .agg(eval_detector_pass_rate_mean=("pass_rate", "mean"), eval_detector_count=("detector", "nunique"))
            .reset_index()
        )
        overview = detector_summary if overview.empty else overview.merge(detector_summary, on="model", how="outer")

    return overview.sort_values("model") if not overview.empty else overview


def write_markdown_summary(
    output_dir: Path,
    models: List[str],
    eval_by_model: pd.DataFrame,
    overview: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Garak Model Comparison Summary")
    lines.append("")
    lines.append(f"- Models found: {', '.join(models) if models else 'none'}")
    lines.append("- Primary source: report JSONL (attempt status + eval + digest)")
    lines.append("")

    if not eval_by_model.empty:
        lines.append("## Eval Pass Stats")
        lines.append(eval_by_model.to_markdown(index=False))
        lines.append("")

    if not overview.empty:
        lines.append("## Consolidated Overview")
        lines.append(overview.to_markdown(index=False))
        lines.append("")

    lines.append("## Generated Files")
    lines.append("- raw_report_attempts.csv")
    lines.append("- summary_report_attempt_hits_by_model.csv")
    lines.append("- summary_report_attempt_hits_by_probe.csv")
    lines.append("- summary_report_status_counts.csv")
    lines.append("- summary_report_eval_detector.csv")
    lines.append("- summary_report_eval_by_model.csv")
    lines.append("- summary_report_eval_by_probe.csv")
    lines.append("- summary_digest_group.csv")
    lines.append("- summary_digest_probe.csv")
    lines.append("- summary_model_overview.csv")
    lines.append("- plot_report_hit_rate_by_model.png")
    lines.append("- plot_report_pass_rate_by_probe.png")
    lines.append("- plot_report_total_passed_by_model.png")
    lines.append("- plot_report_total_passed_by_probe.png")
    lines.append("- plot_report_detector_passrate_heatmap.png")
    lines.append("- plot_digest_group_score.png")
    lines.append("- plot_digest_probe_heatmap.png")
    lines.append("- plot_hits_passes_by_probe_<model>.png")

    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    report_dir: Path,
    output_dir: Path,
    models_filter: Optional[List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts, report_eval, digest_group_rows, digest_probe_rows = collect_report_rows(
        report_dir, models_filter=models_filter
    )

    attempt_hit_by_model, attempt_hit_by_probe, status_distribution = summarize_attempt_statuses(attempts)
    report_eval_summary = summarize_report_eval(report_eval)
    eval_by_model = summarize_eval_by_model(report_eval_summary)
    eval_by_probe = summarize_eval_by_probe(report_eval_summary)
    overview = create_overview_table(attempt_hit_by_model, eval_by_model, digest_group_rows, report_eval_summary)

    if not attempts.empty:
        attempts.to_csv(output_dir / "raw_report_attempts.csv", index=False)
    if not report_eval.empty:
        report_eval.to_csv(output_dir / "raw_report_eval.csv", index=False)

    if not attempt_hit_by_model.empty:
        attempt_hit_by_model.to_csv(output_dir / "summary_report_attempt_hits_by_model.csv", index=False)
    if not attempt_hit_by_probe.empty:
        attempt_hit_by_probe.to_csv(output_dir / "summary_report_attempt_hits_by_probe.csv", index=False)
    if not status_distribution.empty:
        status_distribution.to_csv(output_dir / "summary_report_status_counts.csv", index=False)
    if not report_eval_summary.empty:
        report_eval_summary.to_csv(output_dir / "summary_report_eval_detector.csv", index=False)
    if not eval_by_model.empty:
        eval_by_model.to_csv(output_dir / "summary_report_eval_by_model.csv", index=False)
    if not eval_by_probe.empty:
        eval_by_probe.to_csv(output_dir / "summary_report_eval_by_probe.csv", index=False)
    if not digest_group_rows.empty:
        digest_group_rows.to_csv(output_dir / "summary_digest_group.csv", index=False)
    if not digest_probe_rows.empty:
        digest_probe_rows.to_csv(output_dir / "summary_digest_probe.csv", index=False)
    if not overview.empty:
        overview.to_csv(output_dir / "summary_model_overview.csv", index=False)

    save_plots(
        attempt_hit_by_probe,
        eval_by_model,
        eval_by_probe,
        report_eval_summary,
        digest_group_rows,
        digest_probe_rows,
        output_dir,
    )

    models_found = sorted(
        set(attempts.get("model", pd.Series(dtype=str)).dropna().tolist())
        | set(report_eval.get("model", pd.Series(dtype=str)).dropna().tolist())
        | set(digest_group_rows.get("model", pd.Series(dtype=str)).dropna().tolist())
    )
    write_markdown_summary(output_dir, models_found, eval_by_model, overview)

    print(f"Analysis completed. Output folder: {output_dir}")
    print(f"Models compared: {models_found}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Garak models using report JSONL files (attempt status, eval, digest)."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("garakProbesReports"),
        help="Directory containing *.report.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("garakComparisonOutput"),
        help="Directory where CSVs, charts and markdown summary will be written",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model IDs inferred from filename prefixes",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_analysis(
        report_dir=args.report_dir,
        output_dir=args.output_dir,
        models_filter=args.models,
    )


if __name__ == "__main__":
    main()
