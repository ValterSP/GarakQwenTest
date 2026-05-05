import argparse
import json
from html import escape
from pathlib import Path


def parse_filename(file_path: Path):
  name = file_path.name
  suffix = ".report.jsonl"
  if not name.endswith(suffix):
    return None
  stem = name[: -len(suffix)]
  if "_" not in stem:
    return None
  model, probe = stem.rsplit("_", 1)
  return model, probe


def get_hit_conversation_index(detector_results):
  """
  Find the index of the first conversation that had a hit (1 in detector results array).
  Returns the index (0-4) or None if no hit found.
  """
  if isinstance(detector_results, dict):
    for value in detector_results.values():
      if isinstance(value, list):
        for idx, v in enumerate(value):
          if v == 1:
            return idx
  return None


def extract_turns(item: dict, conversation_idx: int = 0):
  """
  Extract turns from a specific conversation.
  If conversation_idx is -1, try to find the conversation with a hit.
  """
  turns = []
  conversations = item.get("conversations") or []
  
  # If we should find the hit conversation
  if conversation_idx == -1:
    hit_idx = get_hit_conversation_index(item.get("detector_results") or {})
    conversation_idx = hit_idx if hit_idx is not None else 0
  
  # Extract from the specified conversation
  if conversations and 0 <= conversation_idx < len(conversations):
    conv = conversations[conversation_idx]
    if isinstance(conv, dict):
      conv_turns = conv.get("turns") or []
      for turn in conv_turns:
        role = turn.get("role", "unknown")
        content = turn.get("content") or {}
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        turns.append({"role": role, "text": text})

  if turns:
    return turns

  # Fallback to prompt + outputs if conversations not available
  prompt_turns = (item.get("prompt") or {}).get("turns") or []
  for p_turn in prompt_turns:
    content = p_turn.get("content") or {}
    turns.append(
      {
        "role": p_turn.get("role", "user"),
        "text": content.get("text", "") if isinstance(content, dict) else str(content),
      }
    )

  outputs = item.get("outputs") or []
  for out in outputs:
    if isinstance(out, dict):
      turns.append({"role": "assistant", "text": out.get("text", "")})

  return turns


def has_detector_hit(detector_results):
  if isinstance(detector_results, dict):
    return any(has_detector_hit(value) for value in detector_results.values())
  if isinstance(detector_results, list):
    return any(has_detector_hit(value) for value in detector_results)
  if isinstance(detector_results, (int, float)):
    return detector_results == 1
  return False


def collect_reports(report_dir: Path):
  dataset = {}
  report_count = 0
  attempt_count = 0

  for report_path in sorted(report_dir.glob("*.report.jsonl")):
    parsed = parse_filename(report_path)
    if not parsed:
      continue
    model, probe = parsed
    report_count += 1

    attempts = []
    with report_path.open("r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          item = json.loads(line)
        except json.JSONDecodeError:
          continue

        if item.get("entry_type") != "attempt":
          continue

        detector_results = item.get("detector_results") or {}
        is_hit = has_detector_hit(detector_results)
        hit_conv_idx = get_hit_conversation_index(detector_results) if is_hit else None

        attempts.append(
          {
            "uuid": item.get("uuid", ""),
            "status": item.get("status"),
            "goal": item.get("goal", ""),
            "probe_classname": item.get("probe_classname", ""),
            "challenge": (item.get("notes") or {}).get("red_team_challenge", ""),
            "detector_results": detector_results,
            "is_hit": is_hit,
            "hit_conversation_idx": hit_conv_idx,
            "turns": extract_turns(item, conversation_idx=-1 if is_hit else 0),
          }
        )
        attempt_count += 1

    dataset.setdefault(model, {})[probe] = {
      "report_file": report_path.name,
      "attempts": attempts,
    }

  return dataset, report_count, attempt_count


def build_html(title: str, payload: str):
  return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escape(title)}</title>
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
  <div class=\"wrap\">
    <header>
      <h1>{escape(title)}</h1>
      <div class=\"sub\">Choose model and probe, then browse each attempt as a chat conversation.</div>
      <div class=\"controls\">
        <select id=\"model\"></select>
        <select id=\"probe\"></select>
        <input id=\"q\" type=\"text\" placeholder=\"Search in chat text, goal, challenge, UUID\" />
      </div>
      <div class=\"controls secondary\">
        <select id="result">
          <option value="">All classifications</option>
          <option value="hit">Hit</option>
          <option value="pass">Pass</option>
        </select>
        <select id=\"sort\">
          <option value=\"new\">Newest first</option>
          <option value=\"old\">Oldest first</option>
        </select>
        <button id=\"reset\" type=\"button\">Reset Filters</button>
      </div>
      <div class=\"count\" id=\"count\"></div>
    </header>

    <section class=\"attempt hidden\" id=\"viewer\">
      <div class=\"meta\" id=\"meta\"></div>
      <div class=\"goal\" id=\"goal\"></div>
      <div class=\"chat\" id=\"chat\"></div>
      <div class=\"controls secondary\" style=\"margin-top: 12px;\">
        <button id=\"prev\" type=\"button\">Previous Attempt</button>
        <select id=\"attempt\"></select>
        <button id=\"next\" type=\"button\">Next Attempt</button>
      </div>
    </section>

    <div class=\"empty hidden\" id=\"empty\">No attempts matched the current filters.</div>
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
      return probeData?.attempts || [];
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
            safe(a.goal),
            safe(a.probe),
            safe(a.challenge),
            ...(a.turns || []).map(t => safe(t.text)),
          ].join("\n").toLowerCase();
          return text.includes(query);
        }});
      }}

      rows.sort((a, b) => reverse ? b.__idx - a.__idx : a.__idx - b.__idx);

      filtered = rows;

      const totalRaw = rowsForModelProbe().length;
      count.textContent = `${{rows.length}} attempts shown of ${{totalRaw}} in this model/probe`;

      if (!filtered.length) {{
        viewer.classList.add("hidden");
        empty.classList.remove("hidden");
        attemptSel.innerHTML = "";
        return;
      }}

      empty.classList.add("hidden");
      viewer.classList.remove("hidden");

      attemptSel.innerHTML = filtered
        .map((a, i) => `<option value="${{i}}">Attempt ${{i + 1}} | ${{resultLabel(Boolean(a.is_hit))}} | ${{escapeHtml(safe(a.uuid).slice(0, 8))}}</option>`)
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
      const reportName = dataset?.[model]?.[probe]?.report_file || "";
      const isHit = Boolean(a.is_hit);

      const challengeHtml = a.challenge
        ? `<span class="badge">challenge: ${{escapeHtml(a.challenge)}}</span>`
        : "";

      meta.innerHTML = `
        <span class="badge">Model: ${{escapeHtml(model)}}</span>
        <span class="badge">Probe: ${{escapeHtml(probe)}}</span>
        <span class="badge">Report: ${{escapeHtml(reportName)}}</span>
        <span class="badge">UUID: ${{escapeHtml(a.uuid)}}</span>
        <span class="badge ${{resultClass(isHit)}}">Classification: ${{resultLabel(isHit)}}</span>
        ${{a.hit_conversation_idx !== null && isHit ? `<span class="badge">Conversation with HIT: #${{a.hit_conversation_idx + 1}}</span>` : ""}}
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML chat browser from Garak report JSONL files"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("garakProbesReports"),
        help="Directory with *.report.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("garakComparisonOutput") / "chat_report_browser.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    report_dir = args.report_dir
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    output_file = args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset, report_count, attempt_count = collect_reports(report_dir)
    html = build_html("Garak Chat Browser", json.dumps(dataset, ensure_ascii=False))
    output_file.write_text(html, encoding="utf-8")

    print(f"HTML generated: {output_file}")
    print(f"Reports loaded: {report_count}")
    print(f"Attempts loaded: {attempt_count}")


if __name__ == "__main__":
    main()
