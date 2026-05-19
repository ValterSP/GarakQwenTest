# Garak Model Comparison Summary

- Models found: qwen3_5_q8, qwen3_5_q8_uncensored
- Primary source: report JSONL metadata, attempts, evals, digest and completion timestamps
- Pass rate is `passed / total`; hit rate is `(total - passed) / total` from eval rows.
- Attempt status is only kept as run completion metadata, not as hit rate.

## Eval Pass Stats
| model                 | total_passed | total_hits | total_samples | pass_rate | pass_percent | hit_rate | hit_percent |
| --------------------- | ------------ | ---------- | ------------- | --------- | ------------ | -------- | ----------- |
| qwen3_5_q8            | 4240         | 2155       | 6395          | 0.663     | 66.3018      | 0.337    | 33.6982     |
| qwen3_5_q8_uncensored | 2421         | 3970       | 6391          | 0.3788    | 37.8814      | 0.6212   | 62.1186     |

## Runtime by Model
| model                 | duration_seconds | probes_completed | runs | duration_minutes | duration_hours |
| --------------------- | ---------------- | ---------------- | ---- | ---------------- | -------------- |
| qwen3_5_q8            | 267832.8314      | 27               | 27   | 4463.8805        | 74.398         |
| qwen3_5_q8_uncensored | 51832.6146       | 27               | 27   | 863.8769         | 14.3979        |

## Consolidated Overview
| model                 | total_passed | total_hits | total_samples | pass_rate | pass_percent | hit_rate | hit_percent | total_attempts | completed_attempts | completion_rate | completion_percent | digest_group_score_mean | groups_seen | eval_detector_pass_rate_mean | eval_detector_count | duration_seconds | duration_minutes | duration_hours | probes_completed | runs |
| --------------------- | ------------ | ---------- | ------------- | --------- | ------------ | -------- | ----------- | -------------- | ------------------ | --------------- | ------------------ | ----------------------- | ----------- | ---------------------------- | ------------------- | ---------------- | ---------------- | -------------- | ---------------- | ---- |
| qwen3_5_q8            | 4240         | 2155       | 6395          | 0.663     | 66.3018      | 0.337    | 33.6982     | 756            | 756                | 1               | 100                | 0.8366                  | 8           | 0.8405                       | 23                  | 267832.8314      | 4463.8805        | 74.398         | 27               | 27   |
| qwen3_5_q8_uncensored | 2421         | 3970       | 6391          | 0.3788    | 37.8814      | 0.6212   | 62.1186     | 752            | 752                | 1               | 100                | 0.3551                  | 8           | 0.4913                       | 23                  | 51832.6146       | 863.8769         | 14.3979        | 27               | 27   |

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