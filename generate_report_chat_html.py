import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_filename(file_path: Path, suffix: str) -> Optional[Tuple[str, str]]:
    name = file_path.name
    if not name.endswith(suffix):
        return None

    stem = name[: -len(suffix)]
    if "_" not in stem:
        return None

    model, probe = stem.rsplit("_", 1)
    return model, probe


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
        hitlog_data = (hitlogs or {}).get((model, probe))
        hitlog_by_key = hitlog_data["by_key"] if hitlog_data else None
        report_count += 1
        attempts_by_uuid: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}

        for line_no, item in read_jsonl(report_path):
            if item.get("entry_type") != "attempt":
                continue
            uuid = str(item.get("uuid") or f"{report_path.name}:{line_no}")
            attempts_by_uuid.setdefault(uuid, []).append((line_no, item))

        attempts = []
        for entries in attempts_by_uuid.values():
            final_attempt = pick_final_attempt(entries)
            attempt_count += 1
            rows = conversation_rows(final_attempt, hitlog_by_key=hitlog_by_key)
            conversation_count_total += len(rows)
            attempts.extend(rows)

        dataset.setdefault(model, {})[probe] = {
            "report_file": report_path.name,
            "hitlog_file": f"{model}_{probe}.hitlog.jsonl" if hitlog_data else "",
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

    .hidden {{ display: none; }}

    @media (max-width: 860px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .controls.secondary {{ grid-template-columns: 1fr; }}
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
      </div>
      <div class="count" id="count"></div>
    </header>

    <section class="attempt hidden" id="viewer">
      <div class="meta" id="meta"></div>
      <div class="goal" id="goal"></div>
      <div class="chat" id="chat"></div>
      <div class="controls secondary" style="margin-top: 12px;">
        <button id="prev" type="button">Previous Conversation</button>
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

    function detectorSummary(a) {{
      const scores = a.detector_scores || [];
      if (!scores.length) return "none";
      return scores.map(d => `${{d.name}}=${{formatScore(d.score)}}`).join(" | ");
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

    function applyFilters() {{
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
      selectedIndex = 0;
      applyFilters();
    }}

    tabAll.addEventListener("click", () => setActiveView("all"));
    tabHitlog.addEventListener("click", () => setActiveView("hitlog"));

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
      if (selectedIndex < filtered.length - 1) {{
        selectedIndex += 1;
        attemptSel.value = String(selectedIndex);
        renderAttempt();
      }}
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
