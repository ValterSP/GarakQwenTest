import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROBE_PREFIX_RE = re.compile(
    r"_(?P<probe>(?:atkgen|dan|goodside|grandma|lmrc|malwaregen|misleading|realtoxicityprompts)(?:\..+)?)$"
)


def parse_filename(file_path: Path, suffix: str) -> Optional[Tuple[str, str]]:
    """Extract model and probe from names like '<model>_<probe>.report.jsonl'.

    Report metadata is preferred elsewhere, but this fallback keeps filename parsing
    stable for probes that contain underscores, for example dan.Dan_6_0.
    """
    name = file_path.name
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    match = PROBE_PREFIX_RE.search(stem)
    if not match:
        return None
    model_id = stem[: match.start()]
    probe = match.group("probe")
    return model_id, probe


def normalize_model_id(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    return re.sub(r"[^A-Za-z0-9]+", "_", str(model_name)).strip("_")


def parse_timestamp(value: Optional[str]) -> pd.Timestamp:
    if not value:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def model_label(model_id: str) -> str:
    labels = {
        "qwen3_5_q8": "Qwen 3.5 Q8",
        "qwen3_5_q8_uncensored": "Qwen 3.5 Q8 Uncensored",
    }
    return labels.get(model_id, str(model_id).replace("_", " "))


def probe_label(probe: str) -> str:
    group_labels = {
        "atkgen": "Attack Gen",
        "dan": "DAN",
        "goodside": "Goodside",
        "grandma": "Grandma",
        "lmrc": "LMRC",
        "malwaregen": "Malware Gen",
        "misleading": "Misleading",
        "realtoxicityprompts": "Real Toxicity",
    }
    if "." not in str(probe):
        return group_labels.get(str(probe), str(probe).replace("_", " "))

    group, name = str(probe).split(".", 1)
    readable_name = name.replace("_", " ")
    readable_name = readable_name.replace("ChatGPT", "ChatGPT ")
    readable_name = re.sub(r"\s+", " ", readable_name).strip()
    return f"{group_labels.get(group, group)}: {readable_name}"


def add_percent_labels(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)


def add_value_labels(ax, fmt: str = "%.1f") -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=8)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    def format_value(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    columns = [str(column) for column in df.columns]
    rows = [[format_value(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]

    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempt_rows = []
    eval_rows = []
    digest_group_rows = []
    digest_probe_rows = []
    run_rows = []
    normalized_filter = (
        {normalize_model_id(model) for model in models_filter if normalize_model_id(model)}
        if models_filter
        else None
    )

    for file_path in sorted(report_dir.glob("*.report.jsonl")):
        parsed = parse_filename(file_path, ".report.jsonl")
        model_id, probe_from_name = parsed if parsed else (None, None)
        setup_start = pd.NaT
        completion_end = pd.NaT
        run_uuid = None
        target_name = None
        target_type = None
        skip_file = False

        for item in read_jsonl(file_path):
            entry_type = item.get("entry_type")

            if entry_type == "start_run setup":
                target_name = item.get("plugins.target_name")
                target_type = item.get("plugins.target_type")
                model_id = normalize_model_id(target_name) or model_id
                probe_from_name = item.get("plugins.probe_spec") or probe_from_name
                setup_start = parse_timestamp(item.get("transient.starttime_iso"))
                run_uuid = item.get("transient.run_id")
                if normalized_filter and model_id not in normalized_filter:
                    skip_file = True
                    break
                continue

            if skip_file:
                continue

            if entry_type == "init":
                run_uuid = item.get("run", run_uuid)
                if pd.isna(setup_start):
                    setup_start = parse_timestamp(item.get("start_time"))
                continue

            if entry_type == "completion":
                completion_end = parse_timestamp(item.get("end_time"))
                run_uuid = item.get("run", run_uuid)
                continue

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

            elif entry_type == "eval":
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
                        "hits": total - passed,
                        "total": total,
                        "pass_rate": passed / total if total else None,
                        "hit_rate": (total - passed) / total if total else None,
                        "source_file": file_path.name,
                    }
                )

            elif entry_type == "digest":
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

        if not skip_file and model_id:
            duration_seconds = None
            if not pd.isna(setup_start) and not pd.isna(completion_end):
                duration_seconds = max((completion_end - setup_start).total_seconds(), 0.0)
            run_rows.append(
                {
                    "model": model_id,
                    "model_from_report": target_name,
                    "target_type": target_type,
                    "probe": probe_from_name,
                    "start_time": setup_start.isoformat() if not pd.isna(setup_start) else None,
                    "end_time": completion_end.isoformat() if not pd.isna(completion_end) else None,
                    "duration_seconds": duration_seconds,
                    "duration_minutes": duration_seconds / 60.0 if duration_seconds is not None else None,
                    "duration_hours": duration_seconds / 3600.0 if duration_seconds is not None else None,
                    "run_uuid": run_uuid,
                    "source_file": file_path.name,
                }
            )

    return (
        pd.DataFrame(attempt_rows),
        pd.DataFrame(eval_rows),
        pd.DataFrame(digest_group_rows),
        pd.DataFrame(digest_probe_rows),
        pd.DataFrame(run_rows),
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
    attempt_max_status["is_completed"] = attempt_max_status["max_status"] >= 2

    by_probe = (
        attempt_max_status.groupby(["model", "probe"], as_index=False)
        .agg(
            total_attempts=("attempt_uuid", "count"),
            completed_attempts=("is_completed", "sum"),
        )
    )
    by_probe["completion_rate"] = by_probe["completed_attempts"] / by_probe["total_attempts"]
    by_probe["completion_percent"] = by_probe["completion_rate"] * 100.0

    by_model = (
        attempt_max_status.groupby("model", as_index=False)
        .agg(
            total_attempts=("attempt_uuid", "count"),
            completed_attempts=("is_completed", "sum"),
        )
    )
    by_model["completion_rate"] = by_model["completed_attempts"] / by_model["total_attempts"]
    by_model["completion_percent"] = by_model["completion_rate"] * 100.0

    status_distribution = (
        attempt_status.groupby(["model", "status"], as_index=False)
        .agg(count=("attempt_uuid", "count"))
    )
    total_by_model = status_distribution.groupby("model")["count"].transform("sum")
    status_distribution["fraction"] = status_distribution["count"] / total_by_model

    return by_model, by_probe, status_distribution


def summarize_report_eval(report_eval: pd.DataFrame) -> pd.DataFrame:
    if report_eval.empty:
        return pd.DataFrame()

    grouped = (
        report_eval.groupby(["model", "probe", "detector"])
        .agg(total_passed=("passed", "sum"), total_samples=("total", "sum"))
        .reset_index()
    )
    grouped["total_hits"] = grouped["total_samples"] - grouped["total_passed"]
    grouped["pass_rate"] = grouped["total_passed"] / grouped["total_samples"]
    grouped["pass_percent"] = grouped["pass_rate"] * 100.0
    grouped["hit_rate"] = 1.0 - grouped["pass_rate"]
    grouped["hit_percent"] = grouped["hit_rate"] * 100.0
    return grouped


def summarize_eval_by_probe(report_eval_summary: pd.DataFrame) -> pd.DataFrame:
    if report_eval_summary.empty:
        return pd.DataFrame()

    by_probe = (
        report_eval_summary.groupby(["model", "probe"], as_index=False)
        .agg(
            total_passed=("total_passed", "sum"),
            total_hits=("total_hits", "sum"),
            total_samples=("total_samples", "sum"),
        )
    )
    by_probe["pass_rate"] = by_probe["total_passed"] / by_probe["total_samples"]
    by_probe["pass_percent"] = by_probe["pass_rate"] * 100.0
    by_probe["hit_rate"] = by_probe["total_hits"] / by_probe["total_samples"]
    by_probe["hit_percent"] = by_probe["hit_rate"] * 100.0
    return by_probe


def summarize_eval_by_model(report_eval_summary: pd.DataFrame) -> pd.DataFrame:
    if report_eval_summary.empty:
        return pd.DataFrame()

    by_model = (
        report_eval_summary.groupby("model", as_index=False)
        .agg(
            total_passed=("total_passed", "sum"),
            total_hits=("total_hits", "sum"),
            total_samples=("total_samples", "sum"),
        )
    )
    by_model["pass_rate"] = by_model["total_passed"] / by_model["total_samples"]
    by_model["pass_percent"] = by_model["pass_rate"] * 100.0
    by_model["hit_rate"] = by_model["total_hits"] / by_model["total_samples"]
    by_model["hit_percent"] = by_model["hit_rate"] * 100.0
    return by_model


def summarize_run_times(run_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if run_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    run_times = run_rows.copy()
    run_times["duration_seconds"] = pd.to_numeric(run_times["duration_seconds"], errors="coerce")
    run_times = run_times.dropna(subset=["model", "probe", "duration_seconds"]).copy()
    if run_times.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_probe = (
        run_times.groupby(["model", "probe"], as_index=False)
        .agg(
            duration_seconds=("duration_seconds", "sum"),
            runs=("source_file", "count"),
        )
    )
    by_probe["duration_minutes"] = by_probe["duration_seconds"] / 60.0
    by_probe["duration_hours"] = by_probe["duration_seconds"] / 3600.0

    by_model = (
        run_times.groupby("model", as_index=False)
        .agg(
            duration_seconds=("duration_seconds", "sum"),
            probes_completed=("probe", "nunique"),
            runs=("source_file", "count"),
        )
    )
    by_model["duration_minutes"] = by_model["duration_seconds"] / 60.0
    by_model["duration_hours"] = by_model["duration_seconds"] / 3600.0

    return by_probe, by_model


def save_plots(
    eval_by_model: pd.DataFrame,
    eval_by_probe: pd.DataFrame,
    report_eval_summary: pd.DataFrame,
    digest_group_rows: pd.DataFrame,
    digest_probe_rows: pd.DataFrame,
    run_time_by_probe: pd.DataFrame,
    run_time_by_model: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    for old_plot in output_dir.glob("plot_*.png"):
        old_plot.unlink()

    if not eval_by_model.empty:
        plot_data = eval_by_model.sort_values("pass_percent", ascending=False).copy()
        plot_data["model_label"] = plot_data["model"].map(model_label)

        plt.figure(figsize=(9, 4.8))
        ax = sns.barplot(data=plot_data, x="model_label", y="pass_percent", color="#2f7f5f")
        add_percent_labels(ax)
        plt.title("Garak Safety Pass Rate by Model")
        plt.xlabel("Model")
        plt.ylabel("Passed evaluations (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_eval_pass_rate_by_model.png", dpi=180)
        plt.close()

        hit_plot_data = eval_by_model.sort_values("hit_percent", ascending=False).copy()
        hit_plot_data["model_label"] = hit_plot_data["model"].map(model_label)

        plt.figure(figsize=(9, 4.8))
        ax = sns.barplot(data=hit_plot_data, x="model_label", y="hit_percent", color="#b84a4a")
        add_percent_labels(ax)
        plt.title("Garak Hit Rate by Model")
        plt.xlabel("Model")
        plt.ylabel("Failed evaluations / hits (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_eval_hit_rate_by_model.png", dpi=180)
        plt.close()

    if not eval_by_probe.empty:
        all_probes = eval_by_probe.copy()
        all_probes["model_label"] = all_probes["model"].map(model_label)
        all_probes["probe_label"] = all_probes["probe"].map(probe_label)
        probe_order = (
            all_probes.groupby(["probe", "probe_label"], as_index=False)
            .agg(mean_pass_percent=("pass_percent", "mean"), total_samples=("total_samples", "sum"))
            .sort_values(["mean_pass_percent", "total_samples"], ascending=[True, False])["probe_label"]
            .tolist()
        )
        pivot_probe = all_probes.pivot_table(
            index="probe_label", columns="model_label", values="pass_percent", aggfunc="mean"
        )
        if not pivot_probe.empty:
            pivot_probe = pivot_probe.reindex(probe_order)
            plt.figure(figsize=(9, max(8, len(pivot_probe) * 0.38)))
            sns.heatmap(
                pivot_probe,
                annot=True,
                cmap="RdYlGn",
                fmt=".1f",
                vmin=0,
                vmax=100,
                cbar_kws={"label": "Passed evaluations (%)"},
            )
            plt.title("Safety Pass Rate by Probe and Model")
            plt.xlabel("Model")
            plt.ylabel("Probe")
            plt.tight_layout()
            plt.savefig(output_dir / "plot_eval_pass_rate_by_probe_heatmap.png", dpi=180)
            plt.close()

        pivot_hits = all_probes.pivot_table(
            index="probe_label", columns="model_label", values="hit_percent", aggfunc="mean"
        )
        if not pivot_hits.empty:
            pivot_hits = pivot_hits.reindex(probe_order)
            plt.figure(figsize=(9, max(8, len(pivot_hits) * 0.38)))
            sns.heatmap(
                pivot_hits,
                annot=True,
                cmap="YlOrRd",
                fmt=".1f",
                vmin=0,
                vmax=100,
                cbar_kws={"label": "Failed evaluations / hits (%)"},
            )
            plt.title("Garak Hit Rate by Probe and Model")
            plt.xlabel("Model")
            plt.ylabel("Probe")
            plt.tight_layout()
            plt.savefig(output_dir / "plot_eval_hit_rate_by_probe_heatmap.png", dpi=180)
            plt.close()

        raw_pivot = eval_by_probe.pivot_table(index="probe", columns="model", values="pass_percent", aggfunc="mean")
        if len(raw_pivot.columns) == 2:
            ordered_models = sorted(raw_pivot.columns, key=lambda name: ("uncensored" in name, name))
            baseline, comparison = ordered_models[0], ordered_models[1]
            delta_plot = raw_pivot[[baseline, comparison]].dropna().copy()
            delta_plot["delta_percent_points"] = delta_plot[comparison] - delta_plot[baseline]
            delta_plot = delta_plot.reset_index()
            delta_plot["probe_label"] = delta_plot["probe"].map(probe_label)
            delta_plot = delta_plot.sort_values("delta_percent_points")
            colors = delta_plot["delta_percent_points"].map(lambda value: "#b84a4a" if value < 0 else "#2f7f5f")

            plt.figure(figsize=(10, max(8, len(delta_plot) * 0.34)))
            bars = plt.barh(delta_plot["probe_label"], delta_plot["delta_percent_points"], color=colors)
            plt.axvline(0, color="#333333", linewidth=0.8)
            plt.title(f"Pass Rate Difference: {model_label(comparison)} vs {model_label(baseline)}")
            plt.xlabel("Difference in passed evaluations (percentage points)")
            plt.ylabel("Probe")
            plt.bar_label(bars, fmt="%.1f pp", padding=3, fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / "plot_eval_pass_rate_probe_delta.png", dpi=180)
            plt.close()

    if not report_eval_summary.empty:
        eval_sorted = report_eval_summary.copy()
        eval_sorted["model_label"] = eval_sorted["model"].map(model_label)
        eval_sorted["probe_detector"] = (
            eval_sorted["probe"].map(probe_label) + " | " + eval_sorted["detector"].astype(str)
        )

        top_probe_detectors = (
            eval_sorted.groupby("probe_detector", as_index=False)
            .agg(total_samples_all_models=("total_samples", "sum"))
            .sort_values("total_samples_all_models", ascending=False)
            .head(25)["probe_detector"]
            .tolist()
        )

        eval_plot = eval_sorted[eval_sorted["probe_detector"].isin(top_probe_detectors)].copy()
        pivot_eval = eval_plot.pivot_table(
            index="probe_detector", columns="model_label", values="pass_percent", aggfunc="mean"
        )
        if not pivot_eval.empty:
            pivot_eval = pivot_eval.reindex(top_probe_detectors)
        if not pivot_eval.empty:
            plt.figure(figsize=(10, max(6, len(pivot_eval) * 0.42)))
            sns.heatmap(
                pivot_eval,
                annot=True,
                cmap="RdYlGn",
                fmt=".1f",
                vmin=0,
                vmax=100,
                cbar_kws={"label": "Passed evaluations (%)"},
            )
            plt.title("Detector Pass Rate for Highest-Sample Evaluations")
            plt.xlabel("Model")
            plt.ylabel("Probe | Detector")
            plt.tight_layout()
            plt.savefig(output_dir / "plot_detector_pass_rate_heatmap.png", dpi=180)
            plt.close()

    if not digest_group_rows.empty:
        digest_copy = digest_group_rows.copy()
        digest_copy["group_score"] = pd.to_numeric(digest_copy["group_score"], errors="coerce")
        digest_copy = digest_copy.dropna(subset=["group_score"])
        if not digest_copy.empty:
            digest_copy["score_percent"] = digest_copy["group_score"] * 100.0
            digest_copy["model_label"] = digest_copy["model"].map(model_label)
            digest_copy["group_label"] = digest_copy["group"].map(probe_label)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=digest_copy, x="group_label", y="score_percent", hue="model_label")
            plt.title("Garak Digest Score by Probe Group")
            plt.xlabel("Probe group")
            plt.ylabel("Digest score (%)")
            plt.ylim(0, 100)
            plt.xticks(rotation=25, ha="right")
            plt.legend(title="Model")
            plt.tight_layout()
            plt.savefig(output_dir / "plot_digest_group_score.png", dpi=180)
            plt.close()

    if not digest_probe_rows.empty:
        digest_probe = digest_probe_rows.copy()
        digest_probe["probe_score"] = pd.to_numeric(digest_probe["probe_score"], errors="coerce")
        digest_probe = digest_probe.dropna(subset=["probe_score"])
        digest_probe["probe_label"] = digest_probe["probe"].map(probe_label)
        digest_probe["model_label"] = digest_probe["model"].map(model_label)
        pivot = digest_probe.pivot_table(index="probe_label", columns="model_label", values="probe_score", aggfunc="mean")
        if not pivot.empty:
            pivot = pivot.reindex(pivot.mean(axis=1).sort_values().index)
            plt.figure(figsize=(9, max(8, len(pivot) * 0.38)))
            sns.heatmap(
                pivot * 100.0,
                annot=True,
                cmap="RdYlGn",
                fmt=".1f",
                vmin=0,
                vmax=100,
                cbar_kws={"label": "Digest score (%)"},
            )
            plt.title("Garak Digest Score by Probe")
            plt.xlabel("Model")
            plt.ylabel("Probe")
            plt.tight_layout()
            plt.savefig(output_dir / "plot_digest_probe_score_heatmap.png", dpi=180)
            plt.close()

    if not run_time_by_model.empty:
        total_time = run_time_by_model.sort_values("duration_hours", ascending=False).copy()
        total_time["model_label"] = total_time["model"].map(model_label)
        plt.figure(figsize=(9, 4.8))
        ax = sns.barplot(data=total_time, x="model_label", y="duration_hours", color="#4c6f9f")
        add_value_labels(ax, "%.1f h")
        plt.title("Total Garak Runtime by Model")
        plt.xlabel("Model")
        plt.ylabel("Total runtime (hours)")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_runtime_total_by_model.png", dpi=180)
        plt.close()

    if not run_time_by_probe.empty:
        probe_time = run_time_by_probe.copy()
        probe_time["model_label"] = probe_time["model"].map(model_label)
        probe_time["probe_label"] = probe_time["probe"].map(probe_label)
        probe_order = (
            probe_time.groupby(["probe", "probe_label"], as_index=False)
            .agg(max_duration_minutes=("duration_minutes", "max"))
            .sort_values("max_duration_minutes", ascending=False)["probe_label"]
            .tolist()
        )
        plt.figure(figsize=(12, max(8, probe_time["probe"].nunique() * 0.38)))
        sns.barplot(data=probe_time, y="probe_label", x="duration_minutes", hue="model_label", order=probe_order)
        plt.title("Runtime by Probe and Model")
        plt.xlabel("Runtime (minutes)")
        plt.ylabel("Probe")
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_runtime_by_probe_and_model.png", dpi=180)
        plt.close()


def create_overview_table(
    attempt_completion_by_model: pd.DataFrame,
    eval_by_model: pd.DataFrame,
    digest_group_rows: pd.DataFrame,
    report_eval_summary: pd.DataFrame,
    run_time_by_model: pd.DataFrame,
) -> pd.DataFrame:
    overview = eval_by_model.copy() if not eval_by_model.empty else pd.DataFrame()

    if not attempt_completion_by_model.empty:
        status_completion = attempt_completion_by_model[
            ["model", "total_attempts", "completed_attempts", "completion_rate", "completion_percent"]
        ].copy()
        overview = (
            status_completion
            if overview.empty
            else overview.merge(status_completion, on="model", how="outer")
        )

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

    if not run_time_by_model.empty:
        runtime_summary = run_time_by_model[
            ["model", "duration_seconds", "duration_minutes", "duration_hours", "probes_completed", "runs"]
        ].copy()
        overview = runtime_summary if overview.empty else overview.merge(runtime_summary, on="model", how="outer")

    return overview.sort_values("model") if not overview.empty else overview


def write_markdown_summary(
    output_dir: Path,
    models: List[str],
    eval_by_model: pd.DataFrame,
    overview: pd.DataFrame,
    run_time_by_model: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Garak Model Comparison Summary")
    lines.append("")
    lines.append(f"- Models found: {', '.join(models) if models else 'none'}")
    lines.append("- Primary source: report JSONL metadata, attempts, evals, digest and completion timestamps")
    lines.append("- Pass rate is `passed / total`; hit rate is `(total - passed) / total` from eval rows.")
    lines.append("- Attempt status is only kept as run completion metadata, not as hit rate.")
    lines.append("")

    if not eval_by_model.empty:
        lines.append("## Eval Pass Stats")
        lines.append(dataframe_to_markdown(eval_by_model))
        lines.append("")

    if not run_time_by_model.empty:
        lines.append("## Runtime by Model")
        lines.append(dataframe_to_markdown(run_time_by_model))
        lines.append("")

    if not overview.empty:
        lines.append("## Consolidated Overview")
        lines.append(dataframe_to_markdown(overview))
        lines.append("")

    lines.append("## Generated Files")
    lines.append("- raw_report_attempts.csv")
    lines.append("- summary_report_attempt_completion_by_model.csv")
    lines.append("- summary_report_attempt_completion_by_probe.csv")
    lines.append("- summary_report_status_counts.csv")
    lines.append("- summary_report_eval_detector.csv")
    lines.append("- summary_report_eval_by_model.csv")
    lines.append("- summary_report_eval_by_probe.csv")
    lines.append("- summary_digest_group.csv")
    lines.append("- summary_digest_probe.csv")
    lines.append("- raw_report_runs.csv")
    lines.append("- summary_runtime_by_probe.csv")
    lines.append("- summary_runtime_by_model.csv")
    lines.append("- summary_model_overview.csv")
    lines.append("- plot_eval_pass_rate_by_model.png")
    lines.append("- plot_eval_hit_rate_by_model.png")
    lines.append("- plot_eval_pass_rate_by_probe_heatmap.png")
    lines.append("- plot_eval_hit_rate_by_probe_heatmap.png")
    lines.append("- plot_eval_pass_rate_probe_delta.png")
    lines.append("- plot_detector_pass_rate_heatmap.png")
    lines.append("- plot_digest_group_score.png")
    lines.append("- plot_digest_probe_score_heatmap.png")
    lines.append("- plot_runtime_total_by_model.png")
    lines.append("- plot_runtime_by_probe_and_model.png")

    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def clean_generated_outputs(output_dir: Path) -> None:
    for pattern in ("raw_report_*.csv", "summary_*.csv", "comparison_summary.md"):
        for artifact in output_dir.glob(pattern):
            if artifact.is_file():
                artifact.unlink()


def run_analysis(
    report_dir: Path,
    output_dir: Path,
    models_filter: Optional[List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_outputs(output_dir)

    attempts, report_eval, digest_group_rows, digest_probe_rows, run_rows = collect_report_rows(
        report_dir, models_filter=models_filter
    )

    attempt_completion_by_model, attempt_completion_by_probe, status_distribution = summarize_attempt_statuses(attempts)
    report_eval_summary = summarize_report_eval(report_eval)
    eval_by_model = summarize_eval_by_model(report_eval_summary)
    eval_by_probe = summarize_eval_by_probe(report_eval_summary)
    run_time_by_probe, run_time_by_model = summarize_run_times(run_rows)
    overview = create_overview_table(
        attempt_completion_by_model,
        eval_by_model,
        digest_group_rows,
        report_eval_summary,
        run_time_by_model,
    )

    if not attempts.empty:
        attempts.to_csv(output_dir / "raw_report_attempts.csv", index=False)
    if not report_eval.empty:
        report_eval.to_csv(output_dir / "raw_report_eval.csv", index=False)
    if not run_rows.empty:
        run_rows.to_csv(output_dir / "raw_report_runs.csv", index=False)

    if not attempt_completion_by_model.empty:
        attempt_completion_by_model.to_csv(
            output_dir / "summary_report_attempt_completion_by_model.csv", index=False
        )
    if not attempt_completion_by_probe.empty:
        attempt_completion_by_probe.to_csv(
            output_dir / "summary_report_attempt_completion_by_probe.csv", index=False
        )
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
    if not run_time_by_probe.empty:
        run_time_by_probe.to_csv(output_dir / "summary_runtime_by_probe.csv", index=False)
    if not run_time_by_model.empty:
        run_time_by_model.to_csv(output_dir / "summary_runtime_by_model.csv", index=False)
    if not overview.empty:
        overview.to_csv(output_dir / "summary_model_overview.csv", index=False)

    save_plots(
        eval_by_model,
        eval_by_probe,
        report_eval_summary,
        digest_group_rows,
        digest_probe_rows,
        run_time_by_probe,
        run_time_by_model,
        output_dir,
    )

    models_found = sorted(
        set(attempts.get("model", pd.Series(dtype=str)).dropna().tolist())
        | set(report_eval.get("model", pd.Series(dtype=str)).dropna().tolist())
        | set(digest_group_rows.get("model", pd.Series(dtype=str)).dropna().tolist())
        | set(run_rows.get("model", pd.Series(dtype=str)).dropna().tolist())
    )
    write_markdown_summary(output_dir, models_found, eval_by_model, overview, run_time_by_model)

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
