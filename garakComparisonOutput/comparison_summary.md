# Garak Model Comparison Summary

- Models found: qwen3_5_q8, qwen3_5_q8_uncensored
- Primary source: report JSONL metadata, attempts, evals, digest and completion timestamps
- Pass rate is `passed / total`; hit rate is `(total - passed) / total` from eval rows.
- Attempt status is only kept as run completion metadata, not as hit rate.
- Manual misclassification charts use saved `garakManualReviews` rows as the numerator.

## Eval Pass Stats
| model                 | total_passed | total_hits | total_samples | pass_rate | pass_percent | hit_rate | hit_percent |
| --------------------- | ------------ | ---------- | ------------- | --------- | ------------ | -------- | ----------- |
| qwen3_5_q8            | 4823         | 2582       | 7405          | 0.6513    | 65.1317      | 0.3487   | 34.8683     |
| qwen3_5_q8_uncensored | 2934         | 4467       | 7401          | 0.3964    | 39.6433      | 0.6036   | 60.3567     |

## Runtime by Model
| model                 | duration_seconds | probes_completed | runs | duration_minutes | duration_hours |
| --------------------- | ---------------- | ---------------- | ---- | ---------------- | -------------- |
| qwen3_5_q8            | 328227.8031      | 31               | 31   | 5470.4634        | 91.1744        |
| qwen3_5_q8_uncensored | 58933.5877       | 31               | 31   | 982.2265         | 16.3704        |

## Consolidated Overview
| model                 | total_passed | total_hits | total_samples | pass_rate | pass_percent | hit_rate | hit_percent | total_attempts | completed_attempts | completion_rate | completion_percent | digest_group_score_mean | groups_seen | eval_detector_pass_rate_mean | eval_detector_count | duration_seconds | duration_minutes | duration_hours | probes_completed | runs |
| --------------------- | ------------ | ---------- | ------------- | --------- | ------------ | -------- | ----------- | -------------- | ------------------ | --------------- | ------------------ | ----------------------- | ----------- | ---------------------------- | ------------------- | ---------------- | ---------------- | -------------- | ---------------- | ---- |
| qwen3_5_q8            | 4823         | 2582       | 7405          | 0.6513    | 65.1317      | 0.3487   | 34.8683     | 949            | 949                | 1               | 100                | 0.7948                  | 12          | 0.8066                       | 28                  | 328227.8031      | 5470.4634        | 91.1744        | 31               | 31   |
| qwen3_5_q8_uncensored | 2934         | 4467       | 7401          | 0.3964    | 39.6433      | 0.6036   | 60.3567     | 945            | 945                | 1               | 100                | 0.3669                  | 12          | 0.4983                       | 28                  | 58933.5877       | 982.2265         | 16.3704        | 31               | 31   |

## Manual Misclassification Summary
| model                 | probe  | scope  | target_model_type | original_kind | misclassified_count | denominator | denominator_label | misclassified_rate | misclassified_percent | total_samples | model_label            | probe_label |
| --------------------- | ------ | ------ | ----------------- | ------------- | ------------------- | ----------- | ----------------- | ------------------ | --------------------- | ------------- | ---------------------- | ----------- |
| qwen3_5_q8            | Global | global | censored          | hit           | 946                 | 2582        | hits              | 0.3664             | 36.6383               | 7405          | Qwen 3.5 Q8            | Global      |
| qwen3_5_q8_uncensored | Global | global | uncensored        | pass          | 0                   | 2934        | passes            | 0                  | 0                     | 7401          | Qwen 3.5 Q8 Uncensored | Global      |

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
- summary_manual_misclassification_by_probe.csv
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
- plot_manual_misclassified_hits_censored_by_probe.png
- plot_manual_misclassified_hits_censored_global.png
- plot_manual_misclassified_passes_uncensored_by_probe.png
- plot_manual_misclassified_passes_uncensored_global.png