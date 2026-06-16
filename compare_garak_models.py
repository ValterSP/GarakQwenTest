import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "garak-matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "garak-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROBE_PREFIX_RE = re.compile(
    r"_(?P<probe>(?:atkgen|dan|goodside|grandma|lmrc|malwaregen|misleading|realtoxicityprompts|tap|topic|exploitation|ansiescape)(?:\..+)?)$"
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


def is_uncensored_model(model_id: str) -> bool:
    return "uncensored" in str(model_id).lower()


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


def outcome_label(outcome: str) -> str:
    labels = {
        "hit": "positive",
        "pass": "negative",
    }
    return labels.get(str(outcome), str(outcome))


def outcome_display(outcome: str) -> str:
    labels = {
        "hit": "Positive",
        "pass": "Negative",
    }
    return labels.get(str(outcome), str(outcome).title())


RAW_ATTEMPTS_FILE = "dados_brutos_tentativas.csv"
RAW_EVAL_FILE = "dados_brutos_avaliacoes.csv"
RAW_RUNS_FILE = "dados_brutos_execucoes.csv"

SUMMARY_ATTEMPTS_BY_MODEL_FILE = "resumo_conclusao_tentativas_por_modelo.csv"
SUMMARY_ATTEMPTS_BY_PROBE_FILE = "resumo_conclusao_tentativas_por_probe.csv"
SUMMARY_STATUS_COUNTS_FILE = "resumo_contagem_estados.csv"
SUMMARY_EVAL_DETECTOR_FILE = "resumo_avaliacoes_por_detector.csv"
SUMMARY_EVAL_BY_MODEL_FILE = "resumo_avaliacoes_por_modelo.csv"
SUMMARY_EVAL_BY_PROBE_FILE = "resumo_avaliacoes_por_probe.csv"
SUMMARY_DIGEST_GROUP_FILE = "resumo_digest_por_grupo.csv"
SUMMARY_DIGEST_PROBE_FILE = "resumo_digest_por_probe.csv"
SUMMARY_RUNTIME_BY_PROBE_FILE = "resumo_tempo_execucao_por_probe.csv"
SUMMARY_RUNTIME_BY_MODEL_FILE = "resumo_tempo_execucao_por_modelo.csv"
SUMMARY_MODEL_OVERVIEW_FILE = "resumo_geral_modelos.csv"
SUMMARY_MANUAL_MISCLASSIFICATION_FILE = "resumo_misclassificacao_manual_por_probe.csv"

PLOT_POSITIVE_BY_MODEL_FILE = "grafico_taxa_positiva_por_modelo.png"
PLOT_NEGATIVE_BY_MODEL_FILE = "grafico_taxa_negativa_por_modelo.png"
PLOT_POSITIVE_BY_PROBE_FILE = "mapa_calor_taxa_positiva_por_probe.png"
PLOT_NEGATIVE_BY_PROBE_FILE = "mapa_calor_taxa_negativa_por_probe.png"
PLOT_POSITIVE_DELTA_FILE = "grafico_diferenca_taxa_positiva_por_probe.png"
PLOT_DETECTOR_POSITIVE_FILE = "mapa_calor_taxa_positiva_por_detector.png"
PLOT_DIGEST_GROUP_FILE = "grafico_digest_por_grupo.png"
PLOT_DIGEST_PROBE_FILE = "mapa_calor_digest_por_probe.png"
PLOT_RUNTIME_TOTAL_FILE = "grafico_tempo_total_execucao_por_modelo.png"
PLOT_RUNTIME_BY_PROBE_FILE = "grafico_tempo_execucao_por_probe_e_modelo.png"
PLOT_MANUAL_HITS_CENSORED_PROBE_FILE = "grafico_misclassificacao_manual_deteccoes_positivas_modelo_censurado_por_probe.png"
PLOT_MANUAL_HITS_CENSORED_GLOBAL_FILE = "grafico_misclassificacao_manual_deteccoes_positivas_modelo_censurado_global.png"
PLOT_MANUAL_PASSES_UNCENSORED_PROBE_FILE = "grafico_misclassificacao_manual_passes_modelo_descensurado_por_probe.png"
PLOT_MANUAL_PASSES_UNCENSORED_GLOBAL_FILE = "grafico_misclassificacao_manual_passes_modelo_descensurado_global.png"

SUMMARY_OUTPUT_FILE = "resumo_comparacao.md"


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


def collect_manual_review_rows(review_dir: Path) -> pd.DataFrame:
    rows = {}
    if not review_dir.exists():
        return pd.DataFrame()

    for file_path in sorted(review_dir.glob("*_misclassified_reviews.jsonl")):
        for row in read_jsonl(file_path):
            review_key = row.get("review_key")
            model = normalize_model_id(row.get("model")) or row.get("model")
            probe = row.get("probe_classname") or row.get("probe")
            if not review_key or not model or not probe:
                continue
            if row.get("is_misclassified") is False:
                continue

            original_result = str(
                row.get("original_result") or row.get("original_classification") or ""
            ).lower()
            was_hit = row.get("was_hit")
            if isinstance(was_hit, bool):
                original_kind = "hit" if was_hit else "pass"
            elif original_result == "hit":
                original_kind = "hit"
            elif original_result in {"miss", "pass"}:
                original_kind = "pass"
            else:
                original_kind = ""

            rows[str(review_key)] = {
                "review_key": str(review_key),
                "model": model,
                "probe": probe,
                "original_kind": original_kind,
                "manual_review": row.get("manual_review", "misclassified"),
                "reviewed_at": row.get("reviewed_at"),
                "source_file": file_path.name,
            }

    return pd.DataFrame(rows.values())


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


def summarize_manual_misclassification_by_probe(
    eval_by_probe: pd.DataFrame,
    manual_reviews: pd.DataFrame,
) -> pd.DataFrame:
    if eval_by_probe.empty:
        return pd.DataFrame()

    base = eval_by_probe[
        ["model", "probe", "total_passed", "total_hits", "total_samples"]
    ].copy()

    scenarios = [
        {
            "target_model_type": "censored",
            "original_kind": "hit",
            "denominator_column": "total_hits",
            "denominator_label": "hits",
            "include_model": lambda model: not is_uncensored_model(model),
        },
        {
            "target_model_type": "uncensored",
            "original_kind": "pass",
            "denominator_column": "total_passed",
            "denominator_label": "passes",
            "include_model": is_uncensored_model,
        },
    ]

    summary_rows = []
    for scenario in scenarios:
        model_base = base[base["model"].map(scenario["include_model"])].copy()
        if model_base.empty:
            continue

        if manual_reviews.empty:
            counts = pd.DataFrame(columns=["model", "probe", "misclassified_count"])
        else:
            review_rows = manual_reviews[
                (manual_reviews["original_kind"] == scenario["original_kind"])
                & (manual_reviews["model"].map(scenario["include_model"]))
            ].copy()
            counts = (
                review_rows.groupby(["model", "probe"], as_index=False)
                .agg(misclassified_count=("review_key", "nunique"))
                if not review_rows.empty
                else pd.DataFrame(columns=["model", "probe", "misclassified_count"])
            )

        merged = model_base.merge(counts, on=["model", "probe"], how="left")
        merged["misclassified_count"] = merged["misclassified_count"].fillna(0).astype(int)
        merged["denominator"] = pd.to_numeric(
            merged[scenario["denominator_column"]], errors="coerce"
        ).fillna(0)
        merged["misclassified_rate"] = merged.apply(
            lambda row: row["misclassified_count"] / row["denominator"]
            if row["denominator"]
            else 0.0,
            axis=1,
        )
        merged["misclassified_percent"] = merged["misclassified_rate"] * 100.0
        merged["scope"] = "probe"
        merged["target_model_type"] = scenario["target_model_type"]
        merged["original_kind"] = scenario["original_kind"]
        merged["denominator_label"] = scenario["denominator_label"]

        global_rows = (
            merged.groupby("model", as_index=False)
            .agg(
                misclassified_count=("misclassified_count", "sum"),
                denominator=("denominator", "sum"),
                total_samples=("total_samples", "sum"),
            )
        )
        global_rows["probe"] = "Global"
        global_rows["misclassified_rate"] = global_rows.apply(
            lambda row: row["misclassified_count"] / row["denominator"]
            if row["denominator"]
            else 0.0,
            axis=1,
        )
        global_rows["misclassified_percent"] = global_rows["misclassified_rate"] * 100.0
        global_rows["scope"] = "global"
        global_rows["target_model_type"] = scenario["target_model_type"]
        global_rows["original_kind"] = scenario["original_kind"]
        global_rows["denominator_label"] = scenario["denominator_label"]

        summary_rows.append(
            pd.concat(
                [
                    global_rows[
                        [
                            "model",
                            "probe",
                            "scope",
                            "target_model_type",
                            "original_kind",
                            "misclassified_count",
                            "denominator",
                            "denominator_label",
                            "misclassified_rate",
                            "misclassified_percent",
                            "total_samples",
                        ]
                    ],
                    merged[
                        [
                            "model",
                            "probe",
                            "scope",
                            "target_model_type",
                            "original_kind",
                            "misclassified_count",
                            "denominator",
                            "denominator_label",
                            "misclassified_rate",
                            "misclassified_percent",
                            "total_samples",
                        ]
                    ],
                ],
                ignore_index=True,
            )
        )

    if not summary_rows:
        return pd.DataFrame()

    summary = pd.concat(summary_rows, ignore_index=True)
    summary["model_label"] = summary["model"].map(model_label)
    summary["probe_label"] = summary["probe"].map(probe_label)
    return summary.sort_values(
        ["target_model_type", "model", "scope", "probe"],
        ascending=[True, True, True, True],
    )


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


def save_manual_misclassification_plot(
    manual_misclassification: pd.DataFrame,
    target_model_type: str,
    original_kind: str,
    output_path: Path,
    title: str,
) -> None:
    plot_data = manual_misclassification[
        (manual_misclassification["target_model_type"] == target_model_type)
        & (manual_misclassification["original_kind"] == original_kind)
        & (manual_misclassification["scope"] == "probe")
    ].copy()
    plot_data = plot_data[pd.to_numeric(plot_data["denominator"], errors="coerce").fillna(0) > 0]
    if plot_data.empty:
        return

    plot_data = plot_data.sort_values(
        ["misclassified_percent", "denominator", "probe_label"],
        ascending=[False, False, True],
    )
    order = plot_data["probe_label"].drop_duplicates().tolist()

    height = max(6, len(order) * 0.38)
    plt.figure(figsize=(12, height))
    ax = sns.barplot(
        data=plot_data,
        y="probe_label",
        x="misclassified_percent",
        hue="model_label",
        order=order,
    )
    add_percent_labels(ax)
    plt.title(title)
    plt.xlabel("Manually marked misclassified (%)")
    plt.ylabel("Probe")
    plt.xlim(0, 100)
    plt.legend(title=f"Model - {outcome_display(original_kind)}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_manual_misclassification_global_plot(
    manual_misclassification: pd.DataFrame,
    target_model_type: str,
    original_kind: str,
    output_path: Path,
    title: str,
) -> None:
    plot_data = manual_misclassification[
        (manual_misclassification["target_model_type"] == target_model_type)
        & (manual_misclassification["original_kind"] == original_kind)
        & (manual_misclassification["scope"] == "global")
    ].copy()
    if plot_data.empty:
        return

    plot_data = plot_data.sort_values("model_label")
    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(
        data=plot_data,
        x="model_label",
        y="misclassified_percent",
        color="#5f6f9f" if target_model_type == "uncensored" else "#9f4c4c",
    )
    add_percent_labels(ax)
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(f"Manually marked misclassified ({outcome_display(original_kind)})")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_plots(
    eval_by_model: pd.DataFrame,
    eval_by_probe: pd.DataFrame,
    report_eval_summary: pd.DataFrame,
    digest_group_rows: pd.DataFrame,
    digest_probe_rows: pd.DataFrame,
    run_time_by_probe: pd.DataFrame,
    run_time_by_model: pd.DataFrame,
    manual_misclassification: pd.DataFrame,
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
        plt.title("Positive Rate by Model")
        plt.xlabel("Model")
        plt.ylabel("Positive evaluations (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(output_dir / PLOT_POSITIVE_BY_MODEL_FILE, dpi=180)
        plt.close()

        hit_plot_data = eval_by_model.sort_values("hit_percent", ascending=False).copy()
        hit_plot_data["model_label"] = hit_plot_data["model"].map(model_label)

        plt.figure(figsize=(9, 4.8))
        ax = sns.barplot(data=hit_plot_data, x="model_label", y="hit_percent", color="#b84a4a")
        add_percent_labels(ax)
        plt.title("Negative Rate by Model")
        plt.xlabel("Model")
        plt.ylabel("Negative evaluations (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(output_dir / PLOT_NEGATIVE_BY_MODEL_FILE, dpi=180)
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
                cbar_kws={"label": "Positive evaluations (%)"},
            )
            plt.title("Positive Rate by Probe and Model")
            plt.xlabel("Model")
            plt.ylabel("Probe")
            plt.tight_layout()
            plt.savefig(output_dir / PLOT_POSITIVE_BY_PROBE_FILE, dpi=180)
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
                cbar_kws={"label": "Negative evaluations (%)"},
            )
            plt.title("Negative Rate by Probe and Model")
            plt.xlabel("Model")
            plt.ylabel("Probe")
            plt.tight_layout()
            plt.savefig(output_dir / PLOT_NEGATIVE_BY_PROBE_FILE, dpi=180)
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
            plt.title(f"Positive Rate Difference: {model_label(comparison)} vs {model_label(baseline)}")
            plt.xlabel("Difference in positive evaluations (percentage points)")
            plt.ylabel("Probe")
            plt.bar_label(bars, fmt="%.1f pp", padding=3, fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / PLOT_POSITIVE_DELTA_FILE, dpi=180)
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
                cbar_kws={"label": "Positive evaluations (%)"},
            )
            plt.title("Detector Positive Rate for Highest-Sample Evaluations")
            plt.xlabel("Model")
            plt.ylabel("Probe | Detector")
            plt.tight_layout()
            plt.savefig(output_dir / PLOT_DETECTOR_POSITIVE_FILE, dpi=180)
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
            plt.title("Digest Score by Probe Group")
            plt.xlabel("Probe group")
            plt.ylabel("Digest score (%)")
            plt.ylim(0, 100)
            plt.xticks(rotation=25, ha="right")
            plt.legend(title="Model")
            plt.tight_layout()
            plt.savefig(output_dir / PLOT_DIGEST_GROUP_FILE, dpi=180)
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
            plt.title("Digest Score by Probe")
            plt.xlabel("Model")
            plt.ylabel("Probe")
            plt.tight_layout()
            plt.savefig(output_dir / PLOT_DIGEST_PROBE_FILE, dpi=180)
            plt.close()

    if not run_time_by_model.empty:
        total_time = run_time_by_model.sort_values("duration_hours", ascending=False).copy()
        total_time["model_label"] = total_time["model"].map(model_label)
        plt.figure(figsize=(9, 4.8))
        ax = sns.barplot(data=total_time, x="model_label", y="duration_hours", color="#4c6f9f")
        add_value_labels(ax, "%.1f h")
        plt.title("Total Runtime by Model")
        plt.xlabel("Model")
        plt.ylabel("Total runtime (hours)")
        plt.tight_layout()
        plt.savefig(output_dir / PLOT_RUNTIME_TOTAL_FILE, dpi=180)
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
        plt.savefig(output_dir / PLOT_RUNTIME_BY_PROBE_FILE, dpi=180)
        plt.close()

    if not manual_misclassification.empty:
        save_manual_misclassification_plot(
            manual_misclassification,
            target_model_type="censored",
            original_kind="hit",
            output_path=output_dir / PLOT_MANUAL_HITS_CENSORED_PROBE_FILE,
            title="Manual Misclassified Positives by Probe - Censored Model",
        )
        save_manual_misclassification_global_plot(
            manual_misclassification,
            target_model_type="censored",
            original_kind="hit",
            output_path=output_dir / PLOT_MANUAL_HITS_CENSORED_GLOBAL_FILE,
            title="Global Manual Misclassified Positives - Censored Model",
        )
        save_manual_misclassification_plot(
            manual_misclassification,
            target_model_type="uncensored",
            original_kind="pass",
            output_path=output_dir / PLOT_MANUAL_PASSES_UNCENSORED_PROBE_FILE,
            title="Manual Misclassified Negatives by Probe - Uncensored Model",
        )
        save_manual_misclassification_global_plot(
            manual_misclassification,
            target_model_type="uncensored",
            original_kind="pass",
            output_path=output_dir / PLOT_MANUAL_PASSES_UNCENSORED_GLOBAL_FILE,
            title="Global Manual Misclassified Negatives - Uncensored Model",
        )


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
    manual_misclassification: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Resumo de Comparação de Modelos Garak")
    lines.append("")
    lines.append(f"- Modelos encontrados: {', '.join(models) if models else 'nenhum'}")
    lines.append("- Fonte principal: metadados JSONL dos relatórios, tentativas, avaliações, digest e tempos de conclusão")
    lines.append("- A taxa positiva corresponde a hit: `passed / total`; a taxa negativa corresponde a pass: `(total - passed) / total`.")
    lines.append("- O estado das tentativas é guardado apenas como metadados de conclusão da execução, não como taxa positiva/negativa.")
    lines.append("- Os gráficos de misclassificação manual usam as linhas guardadas em `garakManualReviews` como numerador.")
    lines.append("")

    if not eval_by_model.empty:
        lines.append("## Estatísticas de Taxa Positiva")
        lines.append(dataframe_to_markdown(eval_by_model))
        lines.append("")

    if not run_time_by_model.empty:
        lines.append("## Tempo de Execução por Modelo")
        lines.append(dataframe_to_markdown(run_time_by_model))
        lines.append("")

    if not overview.empty:
        lines.append("## Visão Consolidada")
        lines.append(dataframe_to_markdown(overview))
        lines.append("")

    if not manual_misclassification.empty:
        lines.append("## Resumo de Misclassificação Manual")
        manual_global = manual_misclassification[manual_misclassification["scope"] == "global"].copy()
        lines.append(dataframe_to_markdown(manual_global))
        lines.append("")

    lines.append("## Ficheiros Gerados")
    lines.append(f"- {RAW_ATTEMPTS_FILE}")
    lines.append(f"- {SUMMARY_ATTEMPTS_BY_MODEL_FILE}")
    lines.append(f"- {SUMMARY_ATTEMPTS_BY_PROBE_FILE}")
    lines.append(f"- {SUMMARY_STATUS_COUNTS_FILE}")
    lines.append(f"- {SUMMARY_EVAL_DETECTOR_FILE}")
    lines.append(f"- {SUMMARY_EVAL_BY_MODEL_FILE}")
    lines.append(f"- {SUMMARY_EVAL_BY_PROBE_FILE}")
    lines.append(f"- {SUMMARY_DIGEST_GROUP_FILE}")
    lines.append(f"- {SUMMARY_DIGEST_PROBE_FILE}")
    lines.append(f"- {RAW_RUNS_FILE}")
    lines.append(f"- {SUMMARY_RUNTIME_BY_PROBE_FILE}")
    lines.append(f"- {SUMMARY_RUNTIME_BY_MODEL_FILE}")
    lines.append(f"- {SUMMARY_MODEL_OVERVIEW_FILE}")
    lines.append(f"- {SUMMARY_MANUAL_MISCLASSIFICATION_FILE}")
    lines.append(f"- {PLOT_POSITIVE_BY_MODEL_FILE}")
    lines.append(f"- {PLOT_NEGATIVE_BY_MODEL_FILE}")
    lines.append(f"- {PLOT_POSITIVE_BY_PROBE_FILE}")
    lines.append(f"- {PLOT_NEGATIVE_BY_PROBE_FILE}")
    lines.append(f"- {PLOT_POSITIVE_DELTA_FILE}")
    lines.append(f"- {PLOT_DETECTOR_POSITIVE_FILE}")
    lines.append(f"- {PLOT_DIGEST_GROUP_FILE}")
    lines.append(f"- {PLOT_DIGEST_PROBE_FILE}")
    lines.append(f"- {PLOT_RUNTIME_TOTAL_FILE}")
    lines.append(f"- {PLOT_RUNTIME_BY_PROBE_FILE}")
    lines.append(f"- {PLOT_MANUAL_HITS_CENSORED_PROBE_FILE}")
    lines.append(f"- {PLOT_MANUAL_HITS_CENSORED_GLOBAL_FILE}")
    lines.append(f"- {PLOT_MANUAL_PASSES_UNCENSORED_PROBE_FILE}")
    lines.append(f"- {PLOT_MANUAL_PASSES_UNCENSORED_GLOBAL_FILE}")

    (output_dir / SUMMARY_OUTPUT_FILE).write_text("\n".join(lines), encoding="utf-8")


def clean_generated_outputs(output_dir: Path) -> None:
    for pattern in (
        "raw_report_*.csv",
        "summary_*.csv",
        "dados_brutos_*.csv",
        "resumo_*.csv",
        "comparison_summary.md",
        "resumo_comparacao.md",
    ):
        for artifact in output_dir.glob(pattern):
            if artifact.is_file():
                try:
                    artifact.unlink()
                except PermissionError:
                    continue


def run_analysis(
    report_dir: Path,
    review_dir: Path,
    output_dir: Path,
    models_filter: Optional[List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_outputs(output_dir)

    attempts, report_eval, digest_group_rows, digest_probe_rows, run_rows = collect_report_rows(
        report_dir, models_filter=models_filter
    )
    manual_reviews = collect_manual_review_rows(review_dir)

    attempt_completion_by_model, attempt_completion_by_probe, status_distribution = summarize_attempt_statuses(attempts)
    report_eval_summary = summarize_report_eval(report_eval)
    eval_by_model = summarize_eval_by_model(report_eval_summary)
    eval_by_probe = summarize_eval_by_probe(report_eval_summary)
    manual_misclassification = summarize_manual_misclassification_by_probe(
        eval_by_probe,
        manual_reviews,
    )
    run_time_by_probe, run_time_by_model = summarize_run_times(run_rows)
    overview = create_overview_table(
        attempt_completion_by_model,
        eval_by_model,
        digest_group_rows,
        report_eval_summary,
        run_time_by_model,
    )

    if not attempts.empty:
        attempts.to_csv(output_dir / RAW_ATTEMPTS_FILE, index=False)
    if not report_eval.empty:
        report_eval.to_csv(output_dir / RAW_EVAL_FILE, index=False)
    if not run_rows.empty:
        run_rows.to_csv(output_dir / RAW_RUNS_FILE, index=False)

    if not attempt_completion_by_model.empty:
        attempt_completion_by_model.to_csv(output_dir / SUMMARY_ATTEMPTS_BY_MODEL_FILE, index=False)
    if not attempt_completion_by_probe.empty:
        attempt_completion_by_probe.to_csv(output_dir / SUMMARY_ATTEMPTS_BY_PROBE_FILE, index=False)
    if not status_distribution.empty:
        status_distribution.to_csv(output_dir / SUMMARY_STATUS_COUNTS_FILE, index=False)
    if not report_eval_summary.empty:
        report_eval_summary.to_csv(output_dir / SUMMARY_EVAL_DETECTOR_FILE, index=False)
    if not eval_by_model.empty:
        eval_by_model.to_csv(output_dir / SUMMARY_EVAL_BY_MODEL_FILE, index=False)
    if not eval_by_probe.empty:
        eval_by_probe.to_csv(output_dir / SUMMARY_EVAL_BY_PROBE_FILE, index=False)
    if not digest_group_rows.empty:
        digest_group_rows.to_csv(output_dir / SUMMARY_DIGEST_GROUP_FILE, index=False)
    if not digest_probe_rows.empty:
        digest_probe_rows.to_csv(output_dir / SUMMARY_DIGEST_PROBE_FILE, index=False)
    if not run_time_by_probe.empty:
        run_time_by_probe.to_csv(output_dir / SUMMARY_RUNTIME_BY_PROBE_FILE, index=False)
    if not run_time_by_model.empty:
        run_time_by_model.to_csv(output_dir / SUMMARY_RUNTIME_BY_MODEL_FILE, index=False)
    if not overview.empty:
        overview.to_csv(output_dir / SUMMARY_MODEL_OVERVIEW_FILE, index=False)
    if not manual_misclassification.empty:
        manual_misclassification.to_csv(output_dir / SUMMARY_MANUAL_MISCLASSIFICATION_FILE, index=False)

    save_plots(
        eval_by_model,
        eval_by_probe,
        report_eval_summary,
        digest_group_rows,
        digest_probe_rows,
        run_time_by_probe,
        run_time_by_model,
        manual_misclassification,
        output_dir,
    )

    models_found = sorted(
        set(attempts.get("model", pd.Series(dtype=str)).dropna().tolist())
        | set(report_eval.get("model", pd.Series(dtype=str)).dropna().tolist())
        | set(digest_group_rows.get("model", pd.Series(dtype=str)).dropna().tolist())
        | set(run_rows.get("model", pd.Series(dtype=str)).dropna().tolist())
    )
    write_markdown_summary(
        output_dir,
        models_found,
        eval_by_model,
        overview,
        run_time_by_model,
        manual_misclassification,
    )

    print(f"Analysis completed. Output folder: {output_dir}")
    print(f"Models compared: {models_found}")
    print(f"Manual review rows loaded: {len(manual_reviews)}")


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
        "--review-dir",
        type=Path,
        default=Path("garakManualReviews"),
        help="Directory containing *_misclassified_reviews.jsonl manual review files",
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
        review_dir=args.review_dir,
        output_dir=args.output_dir,
        models_filter=args.models,
    )


if __name__ == "__main__":
    main()
