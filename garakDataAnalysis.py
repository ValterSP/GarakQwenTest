import subprocess
import json
import os
import shutil

#"atkgen", "continuation", "dan.Ablation_Dan_11_0", "dan.AntiDAN", "dan.AutoDAN", "dan.AutoDANCached", "dan.ChatGPT_Developer_Mode_RANTI", "dan.ChatGPT_Developer_Mode_v2", "dan.ChatGPT_Image_Markdown", "dan.DAN_Jailbreak", "dan.DUDE", "dan.Dan_10_0", "dan.Dan_11_0", "dan.Dan_6_0", "dan.Dan_6_2", "dan.Dan_7_0", "dan.Dan_8_0", "dan.Dan_9_0", "dan.STAN","malwaregen.Evasion", "malwaregen.Payload", "malwaregen.SubFunctions", "malwaregen.TopLevel", "goodside", "grandma", "lmrc", "misleading", "realtoxicityprompts"
probes = []


output_dir = "garakRunsResults"
os.makedirs(output_dir, exist_ok=True)

reports_output_dir = "garakProbesReports"
hitlog_output_dir = "garakProbesHitlog"
os.makedirs(reports_output_dir, exist_ok=True)
os.makedirs(hitlog_output_dir, exist_ok=True)

model = "qwen3.5_q8_uncensored"
modelToFileSafe = model.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")

all_reports_file = os.path.join(output_dir, f"{modelToFileSafe}_all_probes.report.jsonl")
all_hitlogs_file = os.path.join(output_dir, f"{modelToFileSafe}_all_probes.hitlog.jsonl")


def append_jsonl_file(source_path, target_path):
    if not os.path.exists(source_path):
        return
    with open(source_path, "r", encoding="utf-8") as src, open(target_path, "a", encoding="utf-8") as dst:
        for line in src:
            item = json.loads(line)
            dst.write(json.dumps(item) + "\n")


def copy_if_exists(source_path, target_path):
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)


open(all_reports_file, "w", encoding="utf-8").close()
open(all_hitlogs_file, "w", encoding="utf-8").close()

for probe in probes:
    report_prefix = f"llama_3_q4_K_M_RunAnalysisPython_{probe}"

    cmd = [
        "garak",
        "--verbose",
        "--target_name", model,
        "--target_type", "ollama",
        "--report_prefix", report_prefix,
        "--probes", probe
    ]
    subprocess.run(cmd, check=True) 

    report_file = f"C:\\Users\\valte\\.local\\share\\garak\\garak_runs\\{report_prefix}.report.jsonl"
    hitlog_file = f"C:\\Users\\valte\\.local\\share\\garak\\garak_runs\\{report_prefix}.hitlog.jsonl"

    report_target_file = os.path.join(reports_output_dir, f"{modelToFileSafe}_{probe}.report.jsonl")
    hitlog_target_file = os.path.join(hitlog_output_dir, f"{modelToFileSafe}_{probe}.hitlog.jsonl")

    append_jsonl_file(report_file, all_reports_file)
    append_jsonl_file(hitlog_file, all_hitlogs_file)
    copy_if_exists(report_file, report_target_file)
    copy_if_exists(hitlog_file, hitlog_target_file)

    print(f"Probe '{probe}' processada.")