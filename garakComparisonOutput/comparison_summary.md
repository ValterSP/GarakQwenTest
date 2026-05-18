# Garak Model Comparison Summary

- Models found: qwen3_5_q8, qwen3_5_q8_uncensored
- Primary source: report JSONL metadata, attempts, evals, digest and completion timestamps
- Pass rate is `passed / total`; hit rate is `(total - passed) / total` from eval rows.
- Attempt status is only kept as run completion metadata, not as hit rate.

## Eval Pass Stats
| model                 | total_passed | total_hits | total_samples | pass_rate | pass_percent | hit_rate | hit_percent |
| --------------------- | ------------ | ---------- | ------------- | --------- | ------------ | -------- | ----------- |
| qwen3_5_q8            | 5520         | 2155       | 7675          | 0.7192    | 71.9218      | 0.2808   | 28.0782     |
| qwen3_5_q8_uncensored | 3701         | 3970       | 7671          | 0.4825    | 48.2466      | 0.5175   | 51.7534     |

## Runtime by Model
| model                 | duration_seconds | probes_completed | runs | duration_minutes | duration_hours |
| --------------------- | ---------------- | ---------------- | ---- | ---------------- | -------------- |
| qwen3_5_q8            | 370487.2549      | 28               | 28   | 6174.7876        | 102.9131       |
| qwen3_5_q8_uncensored | 56743.4809       | 28               | 28   | 945.7247         | 15.7621        |

## Consolidated Overview
| model                 | total_passed | total_hits | total_samples | pass_rate | pass_percent | hit_rate | hit_percent | total_attempts | completed_attempts | completion_rate | completion_percent | digest_group_score_mean | groups_seen | eval_detector_pass_rate_mean | eval_detector_count | duration_seconds | duration_minutes | duration_hours | probes_completed | runs |
| --------------------- | ------------ | ---------- | ------------- | --------- | ------------ | -------- | ----------- | -------------- | ------------------ | --------------- | ------------------ | ----------------------- | ----------- | ---------------------------- | ------------------- | ---------------- | ---------------- | -------------- | ---------------- | ---- |
| qwen3_5_q8            | 5520         | 2155       | 7675          | 0.7192    | 71.9218      | 0.2808   | 28.0782     | 1012           | 1012               | 1               | 100                | 0.8427                  | 9           | 0.8431                       | 24                  | 370487.2549      | 6174.7876        | 102.9131       | 28               | 28   |
| qwen3_5_q8_uncensored | 3701         | 3970       | 7671          | 0.4825    | 48.2466      | 0.5175   | 51.7534     | 1008           | 1008               | 1               | 100                | 0.379                   | 9           | 0.4995                       | 24                  | 56743.4809       | 945.7247         | 15.7621        | 28               | 28   |

## Generated Files
- raw_report_attempts.csv
- summary_report_attempt_completion_by_model.csv
- summary_report_attempt_completion_by_probe.csv
- summary_report_status_counts.csv
- summary_report_eval_detector.csv
- summary_report_eval_by_model.csv
- summary_report_eval_by_probe.csv
- summary_digest_group.csv
- summary_digest_probe.csv
- raw_report_runs.csv
- summary_runtime_by_probe.csv
- summary_runtime_by_model.csv
- summary_model_overview.csv
- plot_eval_pass_rate_by_model.png
- plot_eval_hit_rate_by_model.png
- plot_eval_pass_rate_by_probe_heatmap.png
- plot_eval_hit_rate_by_probe_heatmap.png
- plot_eval_pass_rate_probe_delta.png
- plot_detector_pass_rate_heatmap.png
- plot_digest_group_score.png
- plot_digest_probe_score_heatmap.png
- plot_runtime_total_by_model.png
- plot_runtime_by_probe_and_model.png