# Resumo de Comparação de Modelos Garak

- Modelos encontrados: qwen3_5_q8, qwen3_5_q8_uncensored
- Fonte principal: metadados JSONL dos relatórios, tentativas, avaliações, digest e tempos de conclusão
- A taxa positiva corresponde a hit: `passed / total`; a taxa negativa corresponde a pass: `(total - passed) / total`.
- O estado das tentativas é guardado apenas como metadados de conclusão da execução, não como taxa positiva/negativa.
- Os gráficos de misclassificação manual usam as linhas guardadas em `garakManualReviews` como numerador.

## Estatísticas de Taxa Positiva
| model                 | total_passed | total_hits | total_samples | pass_rate | pass_percent | hit_rate | hit_percent |
| --------------------- | ------------ | ---------- | ------------- | --------- | ------------ | -------- | ----------- |
| qwen3_5_q8            | 4823         | 2582       | 7405          | 0.6513    | 65.1317      | 0.3487   | 34.8683     |
| qwen3_5_q8_uncensored | 2934         | 4467       | 7401          | 0.3964    | 39.6433      | 0.6036   | 60.3567     |

## Tempo de Execução por Modelo
| model                 | duration_seconds | probes_completed | runs | duration_minutes | duration_hours |
| --------------------- | ---------------- | ---------------- | ---- | ---------------- | -------------- |
| qwen3_5_q8            | 328227.8031      | 31               | 31   | 5470.4634        | 91.1744        |
| qwen3_5_q8_uncensored | 58933.5877       | 31               | 31   | 982.2265         | 16.3704        |

## Visão Consolidada
| model                 | total_passed | total_hits | total_samples | pass_rate | pass_percent | hit_rate | hit_percent | total_attempts | completed_attempts | completion_rate | completion_percent | digest_group_score_mean | groups_seen | eval_detector_pass_rate_mean | eval_detector_count | duration_seconds | duration_minutes | duration_hours | probes_completed | runs |
| --------------------- | ------------ | ---------- | ------------- | --------- | ------------ | -------- | ----------- | -------------- | ------------------ | --------------- | ------------------ | ----------------------- | ----------- | ---------------------------- | ------------------- | ---------------- | ---------------- | -------------- | ---------------- | ---- |
| qwen3_5_q8            | 4823         | 2582       | 7405          | 0.6513    | 65.1317      | 0.3487   | 34.8683     | 949            | 949                | 1               | 100                | 0.7948                  | 12          | 0.8066                       | 28                  | 328227.8031      | 5470.4634        | 91.1744        | 31               | 31   |
| qwen3_5_q8_uncensored | 2934         | 4467       | 7401          | 0.3964    | 39.6433      | 0.6036   | 60.3567     | 945            | 945                | 1               | 100                | 0.3669                  | 12          | 0.4983                       | 28                  | 58933.5877       | 982.2265         | 16.3704        | 31               | 31   |

## Resumo de Misclassificação Manual
| model                 | probe  | scope  | target_model_type | original_kind | misclassified_count | denominator | denominator_label | misclassified_rate | misclassified_percent | total_samples | model_label            | probe_label |
| --------------------- | ------ | ------ | ----------------- | ------------- | ------------------- | ----------- | ----------------- | ------------------ | --------------------- | ------------- | ---------------------- | ----------- |
| qwen3_5_q8            | Global | global | censored          | hit           | 946                 | 2582        | hits              | 0.3664             | 36.6383               | 7405          | Qwen 3.5 Q8            | Global      |
| qwen3_5_q8_uncensored | Global | global | uncensored        | pass          | 175                 | 2934        | passes            | 0.0596             | 5.9646                | 7401          | Qwen 3.5 Q8 Uncensored | Global      |

## Ficheiros Gerados
- dados_brutos_tentativas.csv
- resumo_conclusao_tentativas_por_modelo.csv
- resumo_conclusao_tentativas_por_probe.csv
- resumo_contagem_estados.csv
- resumo_avaliacoes_por_detector.csv
- resumo_avaliacoes_por_modelo.csv
- resumo_avaliacoes_por_probe.csv
- resumo_digest_por_grupo.csv
- resumo_digest_por_probe.csv
- dados_brutos_execucoes.csv
- resumo_tempo_execucao_por_probe.csv
- resumo_tempo_execucao_por_modelo.csv
- resumo_geral_modelos.csv
- resumo_misclassificacao_manual_por_probe.csv
- grafico_taxa_positiva_por_modelo.png
- grafico_taxa_negativa_por_modelo.png
- mapa_calor_taxa_positiva_por_probe.png
- mapa_calor_taxa_negativa_por_probe.png
- grafico_diferenca_taxa_positiva_por_probe.png
- mapa_calor_taxa_positiva_por_detector.png
- grafico_digest_por_grupo.png
- mapa_calor_digest_por_probe.png
- grafico_tempo_total_execucao_por_modelo.png
- grafico_tempo_execucao_por_probe_e_modelo.png
- grafico_misclassificacao_manual_deteccoes_positivas_modelo_censurado_por_probe.png
- grafico_misclassificacao_manual_deteccoes_positivas_modelo_censurado_global.png
- grafico_misclassificacao_manual_passes_modelo_descensurado_por_probe.png
- grafico_misclassificacao_manual_passes_modelo_descensurado_global.png