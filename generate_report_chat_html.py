import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROBE_PREFIX_RE = re.compile(
    r"_(?P<probe>(?:atkgen|continuation|dan|goodside|grandma|lmrc|malwaregen|misleading|realtoxicityprompts|tap|topic|exploitation|ansiescape)(?:\..+)?)$"
)


def parse_filename(file_path: Path, suffix: str) -> Optional[Tuple[str, str]]:
    name = file_path.name
    if not name.endswith(suffix):
        return None

    stem = name[: -len(suffix)]
    match = PROBE_PREFIX_RE.search(stem)
    if not match:
        return None

    model = stem[: match.start()]
    probe = match.group("probe")
    return model, probe


def normalize_model_id(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    return re.sub(r"[^A-Za-z0-9]+", "_", str(model_name)).strip("_")


def parse_timestamp(value: Any) -> Optional[str]:
    if not value:
        return None
    return str(value)


def seconds_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    if not start or not end:
        return None
    try:
        from datetime import datetime

        return max((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 0.0)
    except ValueError:
        return None


def read_jsonl(file_path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def has_detector_results(item: Dict[str, Any]) -> bool:
    detector_results = item.get("detector_results")
    return isinstance(detector_results, dict) and bool(detector_results)


def numeric_status(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("status"))
    except (TypeError, ValueError):
        return -1.0


def pick_final_attempt(entries: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    # Garak usually emits one pre-eval attempt with empty detector_results and a later
    # evaluated attempt with detector_results. Prefer the evaluated form.
    return max(
        entries,
        key=lambda pair: (has_detector_results(pair[1]), numeric_status(pair[1]), pair[0]),
    )[1]


def score_is_hit(score: Any) -> bool:
    try:
        return float(score) == 1.0
    except (TypeError, ValueError):
        return False


def detector_scores_for_index(detector_results: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
    scores = []
    if not isinstance(detector_results, dict):
        return scores

    for detector_name, values in detector_results.items():
        if not isinstance(values, list) or idx >= len(values):
            continue
        scores.append({"name": detector_name, "score": values[idx]})
    return scores


def normalize_idx(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def text_from_content(content: Any) -> str:
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if content is None:
        return ""
    return str(content)


def extract_turns_from_conversation(conversation: Any) -> List[Dict[str, str]]:
    if not isinstance(conversation, dict):
        return []

    turns = []
    for turn in conversation.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        turns.append(
            {
                "role": str(turn.get("role") or "unknown"),
                "text": text_from_content(turn.get("content")),
            }
        )
    return turns


def extract_turns_from_prompt_and_output(item: Dict[str, Any], idx: int) -> List[Dict[str, str]]:
    turns = []

    prompt_turns = (item.get("prompt") or {}).get("turns") or []
    for turn in prompt_turns:
        if not isinstance(turn, dict):
            continue
        turns.append(
            {
                "role": str(turn.get("role") or "user"),
                "text": text_from_content(turn.get("content")),
            }
        )

    outputs = item.get("outputs") or []
    if idx < len(outputs) and isinstance(outputs[idx], dict):
        turns.append({"role": "assistant", "text": str(outputs[idx].get("text") or "")})

    return turns


def extract_turns(item: Dict[str, Any], idx: int) -> List[Dict[str, str]]:
    conversations = item.get("conversations") or []
    if idx < len(conversations):
        turns = extract_turns_from_conversation(conversations[idx])
        if turns:
            return turns
    return extract_turns_from_prompt_and_output(item, idx)


def extract_turns_from_hitlog(item: Dict[str, Any]) -> List[Dict[str, str]]:
    turns = []

    prompt_turns = (item.get("prompt") or {}).get("turns") or []
    for turn in prompt_turns:
        if not isinstance(turn, dict):
            continue
        turns.append(
            {
                "role": str(turn.get("role") or "user"),
                "text": text_from_content(turn.get("content")),
            }
        )

    output = item.get("output")
    if isinstance(output, dict):
        turns.append({"role": "assistant", "text": str(output.get("text") or "")})

    return turns


def conversation_count(item: Dict[str, Any]) -> int:
    detector_results = item.get("detector_results") or {}
    detector_lengths = [
        len(values)
        for values in detector_results.values()
        if isinstance(values, list)
    ]
    return max(
        len(item.get("conversations") or []),
        len(item.get("outputs") or []),
        max(detector_lengths, default=0),
        1,
    )


def conversation_rows(
    item: Dict[str, Any],
    hitlog_by_key: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    total = conversation_count(item)
    detector_results = item.get("detector_results") or {}
    rows = []
    uuid = str(item.get("uuid") or "")

    for idx in range(total):
        detector_scores = detector_scores_for_index(detector_results, idx)
        hitlog_hit = hitlog_by_key.get((uuid, idx)) if hitlog_by_key is not None else None
        if hitlog_by_key is not None:
            hit_detectors = list(hitlog_hit.get("hit_detectors", [])) if hitlog_hit else []
            hitlog_scores = list(hitlog_hit.get("detector_scores", [])) if hitlog_hit else []
            is_hit = bool(hitlog_hit)
            hit_source = "hitlog" if is_hit else "hitlog-miss"
        else:
            hit_detectors = [
                score["name"] for score in detector_scores if score_is_hit(score.get("score"))
            ]
            hitlog_scores = []
            is_hit = bool(hit_detectors)
            hit_source = "detector_results"

        rows.append(
            {
                "uuid": uuid,
                "seq": item.get("seq"),
                "status": item.get("status"),
                "goal": item.get("goal", ""),
                "probe_classname": item.get("probe_classname", ""),
                "challenge": (item.get("notes") or {}).get("red_team_challenge", ""),
                "conversation_idx": idx,
                "conversation_total": total,
                "detector_scores": detector_scores,
                "hitlog_scores": hitlog_scores,
                "hit_detectors": hit_detectors,
                "is_hit": is_hit,
                "hit_source": hit_source,
                "turns": extract_turns(item, idx),
            }
        )

    return rows


def collect_hitlogs(hitlog_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], int, int]:
    hitlogs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    hitlog_file_count = 0
    hit_count = 0

    if not hitlog_dir.exists():
        return hitlogs, hitlog_file_count, hit_count

    for hitlog_path in sorted(hitlog_dir.glob("*.hitlog.jsonl")):
        parsed = parse_filename(hitlog_path, ".hitlog.jsonl")
        if not parsed:
            continue

        model, probe = parsed
        hitlog_file_count += 1
        bucket = hitlogs.setdefault((model, probe), {"hits": [], "by_key": {}})

        for _, item in read_jsonl(hitlog_path):
            attempt_id = str(item.get("attempt_id") or "")
            if not attempt_id:
                continue

            attempt_idx = normalize_idx(item.get("attempt_idx"))
            detector = str(item.get("detector") or "")
            score = item.get("score")
            score_row = {"name": detector, "score": score, "source": "hitlog"}
            key = (attempt_id, attempt_idx)
            existing = bucket["by_key"].get(key)

            if existing is None:
                existing = {
                    "uuid": attempt_id,
                    "seq": item.get("attempt_seq"),
                    "status": None,
                    "goal": item.get("goal", ""),
                    "probe_classname": item.get("probe", ""),
                    "challenge": "",
                    "conversation_idx": attempt_idx,
                    "conversation_total": item.get("generations_per_prompt") or 1,
                    "detector_scores": [score_row],
                    "hitlog_scores": [score_row],
                    "hit_detectors": [detector] if detector else [],
                    "is_hit": True,
                    "hit_source": "hitlog",
                    "source_file": hitlog_path.name,
                    "turns": extract_turns_from_hitlog(item),
                }
                bucket["by_key"][key] = existing
                bucket["hits"].append(existing)
            else:
                existing["detector_scores"].append(score_row)
                existing["hitlog_scores"].append(score_row)
                if detector and detector not in existing["hit_detectors"]:
                    existing["hit_detectors"].append(detector)

            hit_count += 1

    return hitlogs, hitlog_file_count, hit_count


def collect_reports(
    report_dir: Path,
    hitlogs: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], int, int, int]:
    dataset: Dict[str, Any] = {}
    report_count = 0
    attempt_count = 0
    conversation_count_total = 0

    for report_path in sorted(report_dir.glob("*.report.jsonl")):
        parsed = parse_filename(report_path, ".report.jsonl")
        if not parsed:
            continue

        model, probe = parsed
        report_count += 1
        attempts_by_uuid: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        eval_passed = 0.0
        eval_total = 0.0
        eval_rows = []
        start_time = None
        end_time = None
        target_type = ""
        run_uuid = ""

        for line_no, item in read_jsonl(report_path):
            entry_type = item.get("entry_type")

            if entry_type == "start_run setup":
                model = normalize_model_id(item.get("plugins.target_name")) or model
                probe = str(item.get("plugins.probe_spec") or probe)
                start_time = parse_timestamp(item.get("transient.starttime_iso"))
                target_type = str(item.get("plugins.target_type") or "")
                run_uuid = str(item.get("transient.run_id") or "")
                continue

            if entry_type == "completion":
                end_time = parse_timestamp(item.get("end_time"))
                run_uuid = str(item.get("run") or run_uuid)
                continue

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
                eval_passed += passed
                eval_total += total
                eval_rows.append(
                    {
                        "probe": item.get("probe", probe),
                        "detector": item.get("detector"),
                        "passed": passed,
                        "hits": total - passed,
                        "total": total,
                        "pass_rate": passed / total,
                        "hit_rate": (total - passed) / total,
                    }
                )
                continue

            if entry_type != "attempt":
                continue

            uuid = str(item.get("uuid") or f"{report_path.name}:{line_no}")
            attempts_by_uuid.setdefault(uuid, []).append((line_no, item))

        hitlog_data = (hitlogs or {}).get((model, probe))
        hitlog_by_key = hitlog_data["by_key"] if hitlog_data else None

        attempts = []
        for entries in attempts_by_uuid.values():
            final_attempt = pick_final_attempt(entries)
            attempt_count += 1
            rows = conversation_rows(final_attempt, hitlog_by_key=hitlog_by_key)
            conversation_count_total += len(rows)
            attempts.extend(rows)

        eval_hits = eval_total - eval_passed
        duration_seconds = seconds_between(start_time, end_time)
        dataset.setdefault(model, {})[probe] = {
            "report_file": report_path.name,
            "hitlog_file": f"{model}_{probe}.hitlog.jsonl" if hitlog_data else "",
            "run": {
                "run_uuid": run_uuid,
                "target_type": target_type,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_seconds / 60.0 if duration_seconds is not None else None,
            },
            "metrics": {
                "passed": eval_passed,
                "hits": eval_hits,
                "total": eval_total,
                "pass_rate": eval_passed / eval_total if eval_total else None,
                "hit_rate": eval_hits / eval_total if eval_total else None,
                "attempts": len(attempts_by_uuid),
                "conversations": len(attempts),
                "hitlog_hits": len(hitlog_data["hits"]) if hitlog_data else 0,
                "eval_rows": eval_rows,
            },
            "attempts": attempts,
            "hitlog_hits": list(hitlog_data["hits"]) if hitlog_data else [],
        }

    return dataset, report_count, attempt_count, conversation_count_total


def build_html(title: str, payload: str) -> str:
    escaped_title = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <style>
    :root {{
    --bg: #f6f4ef;
    --panel: #fffdf8;
    --ink: #1f1f1f;
    --muted: #665f54;
    --accent: #136f63;
    --accent-2: #bc5d2a;
    --user: #ddf3ee;
    --assistant: #ffeede;
    --line: #d7d0c5;
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(circle at 12% 12%, #fef9ef 0%, transparent 34%),
        radial-gradient(circle at 90% 25%, #eef9f5 0%, transparent 30%),
        linear-gradient(180deg, #f4f0e7 0%, #ece3d2 100%);
      min-height: 100vh;
    }}

    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 22px;
    }}

    header {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 18px;
      margin-bottom: 12px;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.1rem, 3vw, 1.7rem);
      letter-spacing: 0.2px;
    }}

    .sub {{ color: var(--muted); font-size: 0.95rem; }}

    .controls {{
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr 1fr 1fr;
      margin-top: 14px;
    }}

    .controls.secondary {{
      grid-template-columns: 1fr 140px 140px;
      margin-top: 10px;
    }}

    .controls.navigation {{
      grid-template-columns: minmax(180px, 1fr) 130px minmax(220px, 1.35fr) minmax(180px, 1fr);
      margin-top: 12px;
    }}

    input, select {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      background: #fff;
      color: var(--ink);
      font-size: 0.95rem;
    }}

    button {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      background: #fff;
      color: var(--ink);
      font-size: 0.95rem;
      cursor: pointer;
    }}

    button:hover {{
      background: #f8f8f8;
    }}

    button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}

    .count {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }}

    .tabs {{
      display: flex;
      gap: 8px;
      margin-top: 12px;
    }}

    .tabs button {{
      width: auto;
      min-width: 150px;
    }}

    .tabs button.active {{
      background: var(--ink);
      border-color: var(--ink);
      color: #fff;
    }}

    .review {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fffaf0;
      padding: 10px;
      margin-bottom: 12px;
    }}

    .review-status {{
      color: var(--muted);
      font-size: 0.88rem;
      margin: 0;
    }}

    .review-actions {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 170px;
      gap: 10px;
      align-items: center;
    }}

    .review-actions button {{
      padding: 8px 10px;
      font-size: 0.85rem;
    }}

    .button-danger {{
      background: #b42318;
      border-color: #8f1f16;
      color: #fff;
      font-weight: 700;
    }}

    .button-danger:hover {{
      background: #8f1f16;
    }}

    .attempt {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 14px;
      margin-top: 12px;
      animation: rise 220ms ease;
    }}

    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
      font-size: 0.88rem;
      color: var(--muted);
    }}

    .badge {{
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 3px 10px;
      background: #fff;
    }}

    .badge.status-hit {{ background: #d8f1e7; border-color: #90d3bc; color: #164e3f; }}
    .badge.status-mid {{ background: #fff3db; border-color: #e8c98d; color: #6b4a0f; }}
    .badge.status-low {{ background: #f0f0f0; border-color: #d4d4d4; color: #4f4f4f; }}

    .goal {{
      margin: 2px 0 12px;
      font-size: 0.92rem;
      color: var(--muted);
    }}

    .chat {{
      display: grid;
      gap: 10px;
      max-height: 62vh;
      overflow: auto;
      padding-right: 4px;
    }}

    .msg {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      white-space: pre-wrap;
      line-height: 1.4;
      font-size: 0.95rem;
    }}

    .msg .role {{
      display: block;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 5px;
      color: var(--muted);
    }}

    .msg.user {{ background: var(--user); border-left: 4px solid var(--accent); }}
    .msg.assistant {{ background: var(--assistant); border-left: 4px solid var(--accent-2); }}

    .empty {{
      margin-top: 16px;
      border: 1px dashed var(--line);
      border-radius: 12px;
      background: #fff;
      padding: 14px;
      color: var(--muted);
    }}

    .dashboard {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 14px;
      margin-top: 12px;
      overflow: hidden;
    }}

    .dashboard-controls {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }}

    .dashboard-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}

    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      padding: 12px;
    }}

    .metric-card .label {{
      display: block;
      color: var(--muted);
      font-size: 0.82rem;
      margin-bottom: 6px;
    }}

    .metric-card .value {{
      display: block;
      font-size: 1.35rem;
      font-weight: 800;
    }}

    .comparison-table {{
      width: 100%;
      min-width: 860px;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
      background: #fff;
      border: 1px solid var(--line);
    }}

    .comparison-wrap {{
      width: 100%;
      overflow-x: auto;
      border-radius: 12px;
    }}

    .comparison-table th,
    .comparison-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px;
      text-align: left;
      vertical-align: middle;
      font-size: 0.92rem;
    }}

    .comparison-table th {{
      background: #f5efe3;
      color: var(--muted);
      font-weight: 700;
    }}

    .comparison-table tr:last-child td {{
      border-bottom: 0;
    }}

    .rate-bar {{
      min-width: 120px;
    }}

    .track {{
      height: 8px;
      border-radius: 999px;
      background: #ece6dc;
      overflow: hidden;
      margin-top: 5px;
    }}

    .fill-pass,
    .fill-hit {{
      height: 100%;
      border-radius: 999px;
    }}

    .fill-pass {{ background: var(--accent); }}
    .fill-hit {{ background: #b42318; }}

    .dashboard-note {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 0.88rem;
    }}

    .dashboard-charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}

    .chart-panel {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      padding: 12px;
      min-width: 0;
    }}

    .chart-panel h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
      letter-spacing: 0;
    }}

    .chart-row {{
      display: grid;
      grid-template-columns: minmax(92px, 0.8fr) minmax(120px, 2fr) minmax(58px, auto);
      gap: 8px;
      align-items: center;
      margin: 9px 0;
      min-width: 0;
    }}

    .chart-label {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-size: 0.86rem;
    }}

    .chart-value {{
      text-align: right;
      font-size: 0.86rem;
      color: var(--muted);
      white-space: nowrap;
    }}

    .chart-track {{
      height: 12px;
      border-radius: 999px;
      background: #ece6dc;
      overflow: hidden;
      min-width: 0;
    }}

    .chart-fill {{
      height: 100%;
      min-width: 2px;
      border-radius: 999px;
    }}

    .chart-fill.pass {{ background: var(--accent); }}
    .chart-fill.hit {{ background: #b42318; }}
    .chart-fill.time {{ background: #456c99; }}
    .chart-fill.evals {{ background: var(--accent-2); }}

    .hidden {{ display: none; }}

    @media (max-width: 860px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .controls.secondary {{ grid-template-columns: 1fr; }}
      .controls.navigation {{ grid-template-columns: 1fr; }}
      .dashboard-controls {{ grid-template-columns: 1fr; }}
      .chart-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .chart-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>{escaped_title}</h1>
      <div class="sub">Choose model and probe, then browse each generated conversation with its detector classification.</div>
      <div class="controls">
        <select id="model"></select>
        <select id="probe"></select>
        <input id="q" type="text" placeholder="Search in chat text, goal, detector, challenge, UUID" />
      </div>
      <div class="controls secondary">
        <select id="result">
          <option value="">All classifications</option>
          <option value="hit">Hit</option>
          <option value="pass">Pass</option>
        </select>
        <select id="sort">
          <option value="new">Newest first</option>
          <option value="old">Oldest first</option>
        </select>
        <button id="reset" type="button">Reset Filters</button>
      </div>
      <div class="tabs">
        <button id="tabAll" class="active" type="button">All conversations</button>
        <button id="tabHitlog" type="button">Hitlog hits</button>
        <button id="tabDashboard" type="button">Dashboard</button>
      </div>
      <div class="count" id="count"></div>
    </header>

    <section class="dashboard hidden" id="dashboard">
      <div class="dashboard-controls">
        <select id="dashboardProbe"></select>
        <select id="dashboardSort">
          <option value="model">Sort by model</option>
          <option value="hit">Sort by highest hit rate</option>
          <option value="time">Sort by slowest runtime</option>
        </select>
      </div>
      <div class="dashboard-grid" id="dashboardTotals"></div>
      <div id="dashboardComparison"></div>
      <div class="dashboard-charts" id="dashboardCharts"></div>
      <p class="dashboard-note">Pass and hit rates are computed from report eval rows: pass = passed / total, hit = (total - passed) / total.</p>
    </section>

    <section class="attempt hidden" id="viewer">
      <div class="meta" id="meta"></div>
      <div class="review">
        <div class="review-actions">
          <div class="review-status" id="reviewStatus"></div>
          <button id="syncReviews" type="button">Load saved reviews</button>
        </div>
      </div>
      <div class="goal" id="goal"></div>
      <div class="chat" id="chat"></div>
      <div class="controls navigation">
        <button id="prev" type="button">Previous Conversation</button>
        <button id="markWrong" class="button-danger" type="button">Misclassified</button>
        <select id="attempt"></select>
        <button id="next" type="button">Next Conversation</button>
      </div>
    </section>

    <div class="empty hidden" id="empty">No conversations matched the current filters.</div>
  </div>

  <script>
    const dataset = {payload};

    const modelSel = document.getElementById("model");
    const probeSel = document.getElementById("probe");
    const count = document.getElementById("count");
    const q = document.getElementById("q");
    const result = document.getElementById("result");
    const sort = document.getElementById("sort");
    const reset = document.getElementById("reset");
    const tabAll = document.getElementById("tabAll");
    const tabHitlog = document.getElementById("tabHitlog");
    const tabDashboard = document.getElementById("tabDashboard");
    const reviewStatus = document.getElementById("reviewStatus");
    const markWrong = document.getElementById("markWrong");
    const syncReviews = document.getElementById("syncReviews");

    const dashboard = document.getElementById("dashboard");
    const dashboardProbe = document.getElementById("dashboardProbe");
    const dashboardSort = document.getElementById("dashboardSort");
    const dashboardTotals = document.getElementById("dashboardTotals");
    const dashboardComparison = document.getElementById("dashboardComparison");
    const dashboardCharts = document.getElementById("dashboardCharts");

    const viewer = document.getElementById("viewer");
    const empty = document.getElementById("empty");
    const meta = document.getElementById("meta");
    const goal = document.getElementById("goal");
    const chat = document.getElementById("chat");
    const attemptSel = document.getElementById("attempt");
    const prevBtn = document.getElementById("prev");
    const nextBtn = document.getElementById("next");

    let filtered = [];
    let selectedIndex = 0;
    let activeView = "all";
    const reviewStorageKey = "garakClassificationReviews.v1";
    let reviews = loadReviews();
    let reviewDirectoryHandle = null;
    const reviewFileHandles = new Map();
    const reviewFileHandleSources = new Map();
    const syncedReviewFiles = new Set();
    const syncingReviewFiles = new Set();
    let lastReviewSaveStatus = "";

    function safe(v) {{
      return (v ?? "").toString();
    }}

    function escapeHtml(s) {{
      return safe(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function resultClass(isHit) {{
      return isHit ? "status-hit" : "status-low";
    }}

    function resultLabel(isHit) {{
      return isHit ? "HIT" : "PASS";
    }}

    function selectedModel() {{
      return modelSel.value;
    }}

    function selectedProbe() {{
      return probeSel.value;
    }}

    function rowsForModelProbe() {{
      const model = selectedModel();
      const probe = selectedProbe();
      const probeData = dataset?.[model]?.[probe];
      return activeView === "hitlog" ? (probeData?.hitlog_hits || []) : (probeData?.attempts || []);
    }}

    function currentViewLabel() {{
      return activeView === "hitlog" ? "hitlog hits" : "conversations";
    }}

    function formatScore(score) {{
      if (typeof score === "number" && Number.isFinite(score)) {{
        if (score === 0 || score === 1) return String(score);
        return score.toPrecision(4);
      }}
      return safe(score);
    }}

    function formatPercent(rate) {{
      return typeof rate === "number" && Number.isFinite(rate)
        ? `${{(rate * 100).toFixed(1)}}%`
        : "n/a";
    }}

    function formatNumber(value) {{
      return typeof value === "number" && Number.isFinite(value)
        ? value.toLocaleString(undefined, {{ maximumFractionDigits: 0 }})
        : "0";
    }}

    function formatMinutes(seconds) {{
      if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "n/a";
      const totalSeconds = Math.max(0, Math.round(seconds));
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const remainingSeconds = totalSeconds % 60;
      if (hours > 0) return `${{hours}}h ${{minutes}}m ${{remainingSeconds}}s`;
      if (minutes > 0) return `${{minutes}}m ${{remainingSeconds}}s`;
      return `${{remainingSeconds}}s`;
    }}

    function allModels() {{
      return Object.keys(dataset).sort();
    }}

    function allProbeNames() {{
      const probes = new Set();
      allModels().forEach(model => {{
        Object.keys(dataset?.[model] || {{}}).forEach(probe => probes.add(probe));
      }});
      return Array.from(probes).sort();
    }}

    function metricFor(model, probe) {{
      return dataset?.[model]?.[probe]?.metrics || {{}};
    }}

    function runFor(model, probe) {{
      return dataset?.[model]?.[probe]?.run || {{}};
    }}

    function modelTotals(model) {{
      const probes = Object.values(dataset?.[model] || {{}});
      const total = probes.reduce((sum, row) => sum + Number(row.metrics?.total || 0), 0);
      const passed = probes.reduce((sum, row) => sum + Number(row.metrics?.passed || 0), 0);
      const hits = probes.reduce((sum, row) => sum + Number(row.metrics?.hits || 0), 0);
      const runtime = probes.reduce((sum, row) => sum + Number(row.run?.duration_seconds || 0), 0);
      return {{
        probes: probes.length,
        total,
        passed,
        hits,
        runtime,
        pass_rate: total ? passed / total : null,
        hit_rate: total ? hits / total : null,
      }};
    }}

    function detectorSummary(a) {{
      const scores = a.detector_scores || [];
      if (!scores.length) return "none";
      return scores.map(d => `${{d.name}}=${{formatScore(d.score)}}`).join(" | ");
    }}

    function loadReviews() {{
      try {{
        return JSON.parse(localStorage.getItem(reviewStorageKey) || "{{}}");
      }} catch (err) {{
        console.warn("Could not load saved reviews", err);
        return {{}};
      }}
    }}

    function persistReviewsLocal() {{
      try {{
        localStorage.setItem(reviewStorageKey, JSON.stringify(reviews));
      }} catch (err) {{
        console.warn("Could not persist reviews locally", err);
      }}
    }}

    function parseReviewJsonl(text) {{
      const rows = {{}};
      text.split(/\\r?\\n/).forEach(line => {{
        const trimmed = line.trim();
        if (!trimmed) return;
        try {{
          const row = JSON.parse(trimmed);
          if (row && row.review_key) {{
            rows[row.review_key] = row;
          }}
        }} catch (err) {{
          console.warn("Skipping invalid review row", err);
        }}
      }});
      return rows;
    }}

    async function readReviewsFromHandle(handle) {{
      try {{
        const file = await handle.getFile();
        const text = await file.text();
        return parseReviewJsonl(text);
      }} catch (err) {{
        if (err?.name === "NotFoundError") return {{}};
        throw err;
      }}
    }}

    async function ensureReadWritePermission(handle) {{
      const options = {{ mode: "readwrite" }};
      if (!handle.queryPermission || !handle.requestPermission) return true;
      if (await handle.queryPermission(options) === "granted") return true;
      return await handle.requestPermission(options) === "granted";
    }}

    async function chooseReviewDirectoryHandle(promptUser = true) {{
      if (reviewDirectoryHandle) return reviewDirectoryHandle;
      if (!promptUser || !window.showDirectoryPicker) return null;
      reviewStatus.textContent = `Choose the garakManualReviews folder to save ${{currentReviewFileName()}}.`;
      const handle = await window.showDirectoryPicker({{
        id: "garakManualReviews",
        mode: "readwrite",
      }});
      if (handle.name !== "garakManualReviews") {{
        throw new Error(`Selected "${{handle.name}}". Choose the garakManualReviews folder instead.`);
      }}
      if (!(await ensureReadWritePermission(handle))) {{
        throw new Error("Write permission was not granted for the review directory.");
      }}
      reviewDirectoryHandle = handle;
      return reviewDirectoryHandle;
    }}

    async function chooseReviewFileHandle({{ create = true, promptUser = true }} = {{}}) {{
      const fileName = currentReviewFileName();
      const existingHandle = reviewFileHandles.get(fileName);
      if (existingHandle) {{
        return {{
          handle: existingHandle,
          source: reviewFileHandleSources.get(fileName) || "selected file",
        }};
      }}
      const directoryHandle = await chooseReviewDirectoryHandle(promptUser);
      if (directoryHandle) {{
        let handle = null;
        try {{
          handle = await directoryHandle.getFileHandle(fileName, {{ create }});
        }} catch (err) {{
          if (err?.name === "NotFoundError" && !create) return null;
          throw err;
        }}
        reviewFileHandles.set(fileName, handle);
        reviewFileHandleSources.set(fileName, "garakManualReviews folder");
        return {{ handle, source: "garakManualReviews folder" }};
      }}
      return null;
    }}

    async function writeReviewsToHandle(handle, rows) {{
      const writable = await handle.createWritable();
      try {{
        await writable.write(reviewsAsJsonl(rows));
      }} finally {{
        await writable.close();
      }}
    }}

    function reviewKey(a) {{
      return [
        selectedModel(),
        selectedProbe(),
        safe(a.uuid),
        safe(a.conversation_idx),
      ].join("::");
    }}

    function currentClassification(a) {{
      return Boolean(a.is_hit) ? "hit" : "pass";
    }}

    function correctedClassification(a) {{
      return Boolean(a.is_hit) ? "pass" : "hit";
    }}

    function currentHitMissResult(a) {{
      return Boolean(a.is_hit) ? "hit" : "miss";
    }}

    function correctedHitMissResult(a) {{
      return Boolean(a.is_hit) ? "miss" : "hit";
    }}

    function buildMisclassifiedReviewRecord(a) {{
      const model = selectedModel();
      const probe = selectedProbe();
      const probeData = dataset?.[model]?.[probe] || {{}};
      const key = reviewKey(a);
      return {{
        schema_version: 1,
        review_key: key,
        reviewed_at: new Date().toISOString(),
        manual_review: "misclassified",
        is_misclassified: true,
        was_hit: Boolean(a.is_hit),
        original_result: currentHitMissResult(a),
        corrected_result: correctedHitMissResult(a),
        original_classification: currentClassification(a),
        corrected_classification: correctedClassification(a),
        model,
        probe,
        report_file: probeData.report_file || "",
        hitlog_file: probeData.hitlog_file || a.source_file || "",
        uuid: a.uuid,
        seq: a.seq,
        status: a.status,
        conversation_idx: a.conversation_idx,
        conversation_total: a.conversation_total,
        hit_source: a.hit_source || "",
        goal: a.goal || "",
        probe_classname: a.probe_classname || "",
        challenge: a.challenge || "",
        hit_detectors: a.hit_detectors || [],
        detector_scores: a.detector_scores || [],
        hitlog_scores: a.hitlog_scores || [],
        turns: a.turns || [],
      }};
    }}

    function safeFilePart(s) {{
      return safe(s).trim().replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^_+|_+$/g, "") || "unknown";
    }}

    function currentReviewFileName() {{
      return `${{safeFilePart(selectedModel())}}_${{safeFilePart(selectedProbe())}}_misclassified_reviews.jsonl`;
    }}

    function reviewsForCurrentProbe() {{
      const model = selectedModel();
      const probe = selectedProbe();
      return Object.values(reviews).filter(row => row.model === model && row.probe === probe);
    }}

    function reviewsAsJsonl(rows = reviewsForCurrentProbe()) {{
      const items = Array.isArray(rows) ? rows.slice() : Object.values(rows || {{}});
      items.sort((a, b) => safe(a.reviewed_at).localeCompare(safe(b.reviewed_at)));
      return items.map(row => JSON.stringify(row)).join("\\n") + (items.length ? "\\n" : "");
    }}

    function removeCachedReviewsForCurrentProbe() {{
      const model = selectedModel();
      const probe = selectedProbe();
      Object.keys(reviews).forEach(key => {{
        const row = reviews[key];
        if (row?.model === model && row?.probe === probe) {{
          delete reviews[key];
        }}
      }});
    }}

    function replaceReviewsForCurrentProbe(rows) {{
      removeCachedReviewsForCurrentProbe();
      Object.values(rows || {{}}).forEach(row => {{
        if (row && row.review_key) reviews[row.review_key] = row;
      }});
      persistReviewsLocal();
    }}

    function mergeReviewRows(rows) {{
      Object.values(rows || {{}}).forEach(row => {{
        if (row && row.review_key) reviews[row.review_key] = row;
      }});
      persistReviewsLocal();
    }}

    async function syncReviewsForCurrentProbe({{ promptUser = false }} = {{}}) {{
      const fileName = currentReviewFileName();
      if (!fileName || syncingReviewFiles.has(fileName)) return false;
      if (!promptUser && (!reviewDirectoryHandle || syncedReviewFiles.has(fileName))) return false;

      syncingReviewFiles.add(fileName);
      try {{
        const selection = await chooseReviewFileHandle({{
          create: false,
          promptUser,
        }});
        if (!selection) {{
          if (promptUser) {{
            lastReviewSaveStatus = `No existing review file found for ${{fileName}}. It will be created on first save.`;
            updateReviewStatus();
          }}
          return false;
        }}

        const rows = await readReviewsFromHandle(selection.handle);
        replaceReviewsForCurrentProbe(rows);
        syncedReviewFiles.add(fileName);
        lastReviewSaveStatus = `Loaded ${{Object.keys(rows).length}} review(s) from ${{selection.source}}/${{fileName}} and refreshed the browser cache.`;
        updateReviewStatus();
        return true;
      }} catch (err) {{
        if (err?.name === "AbortError") return false;
        lastReviewSaveStatus = `Could not load saved reviews: ${{err?.message || err}}`;
        console.warn("Could not load saved reviews", err);
        updateReviewStatus();
        return false;
      }} finally {{
        syncingReviewFiles.delete(fileName);
      }}
    }}

    function updateMisclassifiedButton(existing) {{
      markWrong.disabled = Boolean(existing);
      markWrong.textContent = existing ? "Misclassified" : "Mark misclassified";
      markWrong.title = existing
        ? "This conversation is already marked as misclassified."
        : "Mark this conversation as misclassified.";
    }}

    async function saveMisclassifiedReview() {{
      if (!filtered.length) return;
      const a = filtered[selectedIndex];
      const record = buildMisclassifiedReviewRecord(a);
      lastReviewSaveStatus = "";
      try {{
        const selection = await chooseReviewFileHandle({{ create: true, promptUser: true }});
        if (!selection) {{
          lastReviewSaveStatus = "Could not write directly. Open this page in Chrome or Edge and choose the garakManualReviews folder when prompted.";
          updateReviewStatus();
          return;
        }}

        const handle = selection.handle;
        const existingRows = await readReviewsFromHandle(handle);
        if (existingRows[record.review_key]) {{
          replaceReviewsForCurrentProbe(existingRows);
          syncedReviewFiles.add(currentReviewFileName());
          lastReviewSaveStatus = `Already marked in ${{selection.source}}/${{currentReviewFileName()}}.`;
          updateReviewStatus();
          return;
        }}

        existingRows[record.review_key] = record;
        await writeReviewsToHandle(handle, existingRows);
        replaceReviewsForCurrentProbe(existingRows);
        syncedReviewFiles.add(currentReviewFileName());
        lastReviewSaveStatus = `Saved JSONL to ${{selection.source}}/${{currentReviewFileName()}}.`;
        updateReviewStatus();
        goToNextConversation();
      }} catch (err) {{
        if (err?.name === "AbortError") return;
        lastReviewSaveStatus = `Could not save review: ${{err?.message || err}}`;
        console.warn("Could not save misclassified review", err);
        updateReviewStatus();
      }}
    }}

    function goToNextConversation() {{
      if (selectedIndex < filtered.length - 1) {{
        selectedIndex += 1;
        attemptSel.value = String(selectedIndex);
        renderAttempt();
      }}
    }}

    function updateReviewStatus() {{
      if (!filtered.length) {{
        reviewStatus.textContent = "";
        updateMisclassifiedButton(null);
        return;
      }}

      const a = filtered[selectedIndex];
      const existing = reviews[reviewKey(a)];
      updateMisclassifiedButton(existing);
      const count = reviewsForCurrentProbe().length;
      const existingText = existing
        ? `Current conversation marked as misclassified at ${{existing.reviewed_at}}.`
        : "Current conversation has not been marked as misclassified.";
      const saveText = lastReviewSaveStatus || `Target file: garakManualReviews/${{currentReviewFileName()}}.`;
      reviewStatus.textContent = `${{existingText}} ${{count}} misclassification(s) saved for this model/probe. ${{saveText}}`;
    }}

    function populateModels() {{
      const models = Object.keys(dataset).sort();
      modelSel.innerHTML = "";
      if (!models.length) {{
        modelSel.innerHTML = '<option value="">No models found</option>';
        return;
      }}
      modelSel.innerHTML = models.map(m => `<option value="${{escapeHtml(m)}}">${{escapeHtml(m)}}</option>`).join("");
    }}

    function populateProbes() {{
      const model = selectedModel();
      const probes = Object.keys(dataset?.[model] || {{}}).sort();
      probeSel.innerHTML = "";
      if (!probes.length) {{
        probeSel.innerHTML = '<option value="">No probes found</option>';
        return;
      }}
      probeSel.innerHTML = probes.map(p => `<option value="${{escapeHtml(p)}}">${{escapeHtml(p)}}</option>`).join("");
    }}

    function populateDashboardProbes() {{
      const probes = allProbeNames();
      dashboardProbe.innerHTML = "";
      if (!probes.length) {{
        dashboardProbe.innerHTML = '<option value="">No probes found</option>';
        return;
      }}
      dashboardProbe.innerHTML = probes.map(p => `<option value="${{escapeHtml(p)}}">${{escapeHtml(p)}}</option>`).join("");
      if (probeSel.value && probes.includes(probeSel.value)) {{
        dashboardProbe.value = probeSel.value;
      }}
    }}

    function rateCell(rate, kind) {{
      const value = typeof rate === "number" && Number.isFinite(rate) ? Math.max(0, Math.min(100, rate * 100)) : 0;
      const fillClass = kind === "hit" ? "fill-hit" : "fill-pass";
      return `
        <div class="rate-bar">
          <strong>${{formatPercent(rate)}}</strong>
          <div class="track"><div class="${{fillClass}}" style="width: ${{value.toFixed(1)}}%"></div></div>
        </div>
      `;
    }}

    function chartRows(rows, valueGetter, labelGetter, className, maxValue = null) {{
      const values = rows.map(row => Number(valueGetter(row) || 0));
      const max = maxValue ?? Math.max(...values, 1);
      return rows.map(row => {{
        const rawValue = Number(valueGetter(row) || 0);
        const width = max ? Math.max(0, Math.min(100, (rawValue / max) * 100)) : 0;
        return `
          <div class="chart-row">
            <div class="chart-label" title="${{escapeHtml(row.model)}}">${{escapeHtml(row.model)}}</div>
            <div class="chart-track"><div class="chart-fill ${{className}}" style="width: ${{width.toFixed(1)}}%"></div></div>
            <div class="chart-value">${{escapeHtml(labelGetter(row))}}</div>
          </div>
        `;
      }}).join("");
    }}

    function renderDashboardCharts(rows) {{
      const chartOrder = rows.slice().sort((a, b) => a.model.localeCompare(b.model));
      dashboardCharts.innerHTML = `
        <div class="chart-panel">
          <h2>Pass Rate</h2>
          ${{chartRows(chartOrder, row => Number(row.pass_rate || 0), row => formatPercent(row.pass_rate), "pass", 1)}}
        </div>
        <div class="chart-panel">
          <h2>Hit Rate</h2>
          ${{chartRows(chartOrder, row => Number(row.hit_rate || 0), row => formatPercent(row.hit_rate), "hit", 1)}}
        </div>
        <div class="chart-panel">
          <h2>Runtime</h2>
          ${{chartRows(chartOrder, row => row.runtime, row => formatMinutes(row.runtime), "time")}}
        </div>
        <div class="chart-panel">
          <h2>Total Evaluations</h2>
          ${{chartRows(chartOrder, row => row.total, row => formatNumber(row.total), "evals")}}
        </div>
      `;
    }}

    function renderDashboard() {{
      const probe = dashboardProbe.value || allProbeNames()[0] || "";
      if (probe && dashboardProbe.value !== probe) dashboardProbe.value = probe;

      const totals = allModels().map(model => ({{ model, ...modelTotals(model) }}));
      dashboardTotals.innerHTML = totals.map(row => `
        <div class="metric-card">
          <span class="label">${{escapeHtml(row.model)}}</span>
          <span class="value">${{formatPercent(row.hit_rate)}} hit</span>
          <span class="label">${{formatNumber(row.probes)}} probes · ${{formatNumber(row.total)}} evals · ${{formatMinutes(row.runtime)}}</span>
        </div>
      `).join("");

      let rows = allModels().map(model => {{
        const metrics = metricFor(model, probe);
        const run = runFor(model, probe);
        return {{
          model,
          report_file: dataset?.[model]?.[probe]?.report_file || "",
          passed: Number(metrics.passed || 0),
          hits: Number(metrics.hits || 0),
          total: Number(metrics.total || 0),
          pass_rate: metrics.pass_rate,
          hit_rate: metrics.hit_rate,
          attempts: Number(metrics.attempts || 0),
          conversations: Number(metrics.conversations || 0),
          hitlog_hits: Number(metrics.hitlog_hits || 0),
          runtime: Number(run.duration_seconds || 0),
        }};
      }});

      if (dashboardSort.value === "hit") {{
        rows.sort((a, b) => Number(b.hit_rate || 0) - Number(a.hit_rate || 0));
      }} else if (dashboardSort.value === "time") {{
        rows.sort((a, b) => b.runtime - a.runtime);
      }} else {{
        rows.sort((a, b) => a.model.localeCompare(b.model));
      }}

      const tableRows = rows.map(row => `
        <tr>
          <td><strong>${{escapeHtml(row.model)}}</strong><br><span class="sub">${{escapeHtml(row.report_file || "missing report")}}</span></td>
          <td>${{rateCell(row.pass_rate, "pass")}}</td>
          <td>${{rateCell(row.hit_rate, "hit")}}</td>
          <td>${{formatNumber(row.total)}}</td>
          <td>${{formatNumber(row.passed)}} / ${{formatNumber(row.hits)}}</td>
          <td>${{formatNumber(row.attempts)}} / ${{formatNumber(row.conversations)}}</td>
          <td>${{formatNumber(row.hitlog_hits)}}</td>
          <td>${{formatMinutes(row.runtime)}}</td>
        </tr>
      `).join("");

      dashboardComparison.innerHTML = `
        <div class="comparison-wrap">
          <table class="comparison-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Pass rate</th>
                <th>Hit rate</th>
                <th>Total evals</th>
                <th>Passed / Hits</th>
                <th>Attempts / Chats</th>
                <th>Hitlog hits</th>
                <th>Runtime</th>
              </tr>
            </thead>
            <tbody>${{tableRows || '<tr><td colspan="8">No dashboard data for this probe.</td></tr>'}}</tbody>
          </table>
        </div>
      `;

      renderDashboardCharts(rows);

      count.textContent = probe
        ? `Dashboard comparing ${{rows.length}} model(s) for probe "${{probe}}"; ${{allProbeNames().length}} total probe groups available`
        : "Dashboard has no probes to compare";
    }}

    function applyFilters() {{
      if (activeView === "dashboard") {{
        renderDashboard();
        return;
      }}

      const query = q.value.trim().toLowerCase();
      const selectedResult = result.value;
      const reverse = sort.value === "new";

      let rows = rowsForModelProbe().map((a, idx) => ({{ ...a, __idx: idx }}));

      if (selectedResult === "hit") {{
        rows = rows.filter(a => Boolean(a.is_hit));
      }}

      if (selectedResult === "pass") {{
        rows = rows.filter(a => !Boolean(a.is_hit));
      }}

      if (query) {{
        rows = rows.filter(a => {{
          const text = [
            safe(a.uuid),
            safe(a.seq),
            safe(a.goal),
            safe(a.probe_classname),
            safe(a.challenge),
            detectorSummary(a),
            ...(a.hit_detectors || []).map(safe),
            ...(a.turns || []).map(t => safe(t.text)),
          ].join("\\n").toLowerCase();
          return text.includes(query);
        }});
      }}

      rows.sort((a, b) => reverse ? b.__idx - a.__idx : a.__idx - b.__idx);

      filtered = rows;

      const totalRaw = rowsForModelProbe().length;
      count.textContent = `${{rows.length}} ${{currentViewLabel()}} shown of ${{totalRaw}} in this model/probe`;

      if (!filtered.length) {{
        viewer.classList.add("hidden");
        empty.classList.remove("hidden");
        attemptSel.innerHTML = "";
        return;
      }}

      empty.classList.add("hidden");
      viewer.classList.remove("hidden");

      attemptSel.innerHTML = filtered
        .map((a, i) => {{
          const seq = safe(a.seq || 0);
          const conv = Number(a.conversation_idx || 0) + 1;
          const total = Number(a.conversation_total || 1);
          return `<option value="${{i}}">Seq ${{escapeHtml(seq)}}.${{conv}}/${{total}} | ${{resultLabel(Boolean(a.is_hit))}} | ${{escapeHtml(safe(a.uuid).slice(0, 8))}}</option>`;
        }})
        .join("");

      if (selectedIndex >= filtered.length) selectedIndex = filtered.length - 1;
      if (selectedIndex < 0) selectedIndex = 0;
      attemptSel.value = String(selectedIndex);
      renderAttempt();
    }}

    function renderAttempt() {{
      if (!filtered.length) return;

      const a = filtered[selectedIndex];
      const model = selectedModel();
      const probe = selectedProbe();
      const probeData = dataset?.[model]?.[probe] || {{}};
      const reportName = probeData.report_file || "";
      const hitlogName = probeData.hitlog_file || a.source_file || "";
      const isHit = Boolean(a.is_hit);
      const convNumber = Number(a.conversation_idx || 0) + 1;
      const convTotal = Number(a.conversation_total || 1);

      const challengeHtml = a.challenge
        ? `<span class="badge">challenge: ${{escapeHtml(a.challenge)}}</span>`
        : "";

      const hitDetectorHtml = isHit
        ? `<span class="badge status-hit">Hit detector: ${{escapeHtml((a.hit_detectors || []).join(", "))}}</span>`
        : "";

      const hitSourceHtml = a.hit_source
        ? `<span class="badge">Hit source: ${{escapeHtml(a.hit_source)}}</span>`
        : "";

      meta.innerHTML = `
        <span class="badge">Model: ${{escapeHtml(model)}}</span>
        <span class="badge">Probe: ${{escapeHtml(probe)}}</span>
        <span class="badge">Report: ${{escapeHtml(reportName)}}</span>
        ${{hitlogName ? `<span class="badge">Hitlog: ${{escapeHtml(hitlogName)}}</span>` : ""}}
        <span class="badge">UUID: ${{escapeHtml(a.uuid)}}</span>
        <span class="badge">Seq: ${{escapeHtml(a.seq)}}</span>
        <span class="badge">Conversation: #${{convNumber}}/${{convTotal}}</span>
        <span class="badge ${{resultClass(isHit)}}">Classification: ${{resultLabel(isHit)}}</span>
        <span class="badge">Detector scores: ${{escapeHtml(detectorSummary(a))}}</span>
        ${{hitSourceHtml}}
        ${{hitDetectorHtml}}
        ${{challengeHtml}}
      `;

      goal.textContent = `Goal: ${{safe(a.goal || "(none)")}}`;

      const turns = a.turns || [];
      chat.innerHTML = turns.length
        ? turns
            .map((t) => `
              <div class="msg ${{escapeHtml(safe(t.role).toLowerCase())}}">
                <span class="role">${{escapeHtml(t.role)}}</span>
                ${{escapeHtml(t.text)}}
              </div>
            `)
            .join("")
        : '<div class="empty">No conversation turns found for this attempt.</div>';

      prevBtn.disabled = selectedIndex <= 0;
      nextBtn.disabled = selectedIndex >= filtered.length - 1;
      updateReviewStatus();
      syncReviewsForCurrentProbe({{ promptUser: false }});
    }}

    modelSel.addEventListener("change", () => {{
      populateProbes();
      selectedIndex = 0;
      applyFilters();
    }});

    probeSel.addEventListener("change", () => {{
      selectedIndex = 0;
      applyFilters();
    }});

    function setActiveView(view) {{
      activeView = view;
      tabAll.classList.toggle("active", activeView === "all");
      tabHitlog.classList.toggle("active", activeView === "hitlog");
      tabDashboard.classList.toggle("active", activeView === "dashboard");
      selectedIndex = 0;
      const isDashboard = activeView === "dashboard";
      dashboard.classList.toggle("hidden", !isDashboard);
      viewer.classList.toggle("hidden", isDashboard);
      empty.classList.toggle("hidden", isDashboard);
      applyFilters();
    }}

    tabAll.addEventListener("click", () => setActiveView("all"));
    tabHitlog.addEventListener("click", () => setActiveView("hitlog"));
    tabDashboard.addEventListener("click", () => setActiveView("dashboard"));
    dashboardProbe.addEventListener("change", () => renderDashboard());
    dashboardSort.addEventListener("change", () => renderDashboard());
    markWrong.addEventListener("click", () => saveMisclassifiedReview());
    syncReviews.addEventListener("click", () => syncReviewsForCurrentProbe({{ promptUser: true }}));

    [q, result, sort].forEach(el =>
      el.addEventListener("input", () => {{
        selectedIndex = 0;
        applyFilters();
      }})
    );

    attemptSel.addEventListener("change", () => {{
      selectedIndex = Number(attemptSel.value) || 0;
      renderAttempt();
    }});

    prevBtn.addEventListener("click", () => {{
      if (selectedIndex > 0) {{
        selectedIndex -= 1;
        attemptSel.value = String(selectedIndex);
        renderAttempt();
      }}
    }});

    nextBtn.addEventListener("click", () => {{
      goToNextConversation();
    }});

    reset.addEventListener("click", () => {{
      q.value = "";
      result.value = "";
      sort.value = "new";
      selectedIndex = 0;
      applyFilters();
    }});

    populateModels();
    populateProbes();
    populateDashboardProbes();
    applyFilters();
  </script>
</body>
</html>
"""


def json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML chat browser from Garak report JSONL files."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("garakProbesReports"),
        help="Directory with *.report.jsonl files",
    )
    parser.add_argument(
        "--hitlog-dir",
        type=Path,
        default=Path("garakProbesHitlog"),
        help="Directory with *.hitlog.jsonl files used as the hit source of truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report_chat_browser.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()

    if not args.report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {args.report_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    hitlogs, hitlog_file_count, hit_count = collect_hitlogs(args.hitlog_dir)
    dataset, report_count, attempt_count, row_count = collect_reports(args.report_dir, hitlogs=hitlogs)
    html = build_html("Garak Chat Browser", json_for_script(dataset))
    args.output.write_text(html, encoding="utf-8")

    print(f"HTML generated: {args.output}")
    print(f"Reports loaded: {report_count}")
    print(f"Hitlogs loaded: {hitlog_file_count}")
    print(f"Hitlog rows loaded: {hit_count}")
    print(f"Unique attempts loaded: {attempt_count}")
    print(f"Conversation rows loaded: {row_count}")


if __name__ == "__main__":
    main()
