# Garak Model Comparison Summary

- Models found: llama3_uncensored_q4, qwen3_5_q8, qwen3_5_q8_uncensored
- Primary source: report JSONL (attempt status + eval + digest)

## Eval Pass Stats
| model                 |   total_passed |   total_samples |   pass_rate |   pass_percent |
|:----------------------|---------------:|----------------:|------------:|---------------:|
| llama3_uncensored_q4  |           3579 |            8815 |    0.406012 |        40.6012 |
| qwen3_5_q8            |            123 |             125 |    0.984    |        98.4    |
| qwen3_5_q8_uncensored |           4006 |            8811 |    0.454659 |        45.4659 |

## Consolidated Overview
| model                 |   total_passed |   total_samples |   pass_rate |   pass_percent |   total_attempts |   hit_attempts |   hit_rate |   digest_group_score_mean |   groups_seen |   eval_detector_pass_rate_mean |   eval_detector_count |
|:----------------------|---------------:|----------------:|------------:|---------------:|-----------------:|---------------:|-----------:|--------------------------:|--------------:|-------------------------------:|----------------------:|
| llama3_uncensored_q4  |           3579 |            8815 |    0.406012 |        40.6012 |             1254 |           1254 |          1 |                  0.480869 |             9 |                       0.504892 |                    18 |
| qwen3_5_q8            |            123 |             125 |    0.984    |        98.4    |              125 |            125 |          1 |                  0.984    |             1 |                       0.984    |                     1 |
| qwen3_5_q8_uncensored |           4006 |            8811 |    0.454659 |        45.4659 |             1250 |           1250 |          1 |                  0.523297 |             9 |                       0.544501 |                    18 |

## Generated Files
- raw_report_attempts.csv
- summary_report_attempt_hits_by_model.csv
- summary_report_attempt_hits_by_probe.csv
- summary_report_status_counts.csv
- summary_report_eval_detector.csv
- summary_report_eval_by_model.csv
- summary_report_eval_by_probe.csv
- summary_digest_group.csv
- summary_digest_probe.csv
- summary_model_overview.csv
- plot_report_hit_rate_by_model.png
- plot_report_pass_rate_by_probe.png
- plot_report_total_passed_by_model.png
- plot_report_total_passed_by_probe.png
- plot_report_detector_passrate_heatmap.png
- plot_digest_group_score.png
- plot_digest_probe_heatmap.png
- plot_hits_passes_by_probe_<model>.png