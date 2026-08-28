# AnaliseProbesGarak

Scripts para executar probes do [Garak](https://github.com/NVIDIA/garak), guardar os relatórios JSONL, comparar modelos, gerar estatísticas/gráficos e rever manualmente os casos classificados de forma incorreta.

## Fluxo do projeto

```text
Ollama + modelo local
        |
        v
1. garakDataAnalysis.py
        |
        +--> garakProbesReports/*.report.jsonl
        +--> garakProbesHitlog/*.hitlog.jsonl
        +--> garakRunsResults/*_all_probes.*.jsonl
        |
        v
2. generate_report_chat_html.py
        |
        +--> report_chat_browser.html
        +--> garakManualReviews/*.jsonl
        |
        v
3. review_server.py (interface de revisão)
        |
        v
4. compare_garak_models.py
        |
        +--> garakComparisonOutput/*.csv
        +--> garakComparisonOutput/*.png
        +--> garakComparisonOutput/resumo_comparacao.md
```

## Pré-requisitos

- Windows PowerShell, Python 3.10 ou superior e `pip`.
- Ollama instalado e em execução.
- O modelo usado pelo script disponível no Ollama. Atualmente, o script usa `qwen3.5_q8_uncensored`.
- Garak instalado e acessível como comando `garak`.
- Bibliotecas Python usadas na análise:
  - `pandas`
  - `matplotlib`
  - `seaborn`

A partir da raiz do projeto, criar e ativar um ambiente virtual:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install garak pandas matplotlib seaborn
```

Se a política do PowerShell impedir a ativação do ambiente, executar uma vez no PowerShell do utilizador:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Confirmar as instalações:

```powershell
python --version
garak --help
ollama list
```

O modelo configurado no script deve aparecer em `ollama list`. Iniciar o serviço Ollama antes da execução, caso não esteja a correr:

```powershell
ollama serve
```

## 1. Executar as probes do Garak

O ficheiro [garakDataAnalysis.py](garakDataAnalysis.py) executa uma probe de cada vez através do comando `garak`, usando o tipo de alvo `ollama`. Depois de cada execução, copia o relatório e o hitlog para as pastas do projeto e também os agrega em ficheiros únicos.

```powershell
python .\garakDataAnalysis.py
```

As probes e o modelo estão definidos no início do ficheiro, nas variáveis `probes` e `model`. Para alterar a experiência, editar essas variáveis antes de executar.

Também é necessário verificar `report_file` e `hitlog_file`. Estes caminhos apontam atualmente para:

```text
C:\Users\valte\.local\share\garak\garak_runs
```

Esse caminho pode variar conforme a instalação do Garak e o sistema operativo. O `report_prefix` também é fixado no script; não o alterar entre execuções se for necessário manter a correspondência entre relatórios e hitlogs.

Saídas principais:

- `garakProbesReports/`: um relatório `.report.jsonl` por modelo e probe.
- `garakProbesHitlog/`: um hitlog `.hitlog.jsonl` por modelo e probe.
- `garakRunsResults/`: relatórios e hitlogs agregados da execução.

O script limpa os ficheiros agregados no início. Não executar duas análises em paralelo na mesma pasta.

## 2. Gerar o navegador HTML das conversas

O ficheiro [generate_report_chat_html.py](generate_report_chat_html.py) lê os relatórios e hitlogs, escolhe as tentativas finais, apresenta as conversas e associa os resultados dos detectors. Também cria os ficheiros de revisão manual que ainda não existirem.

```powershell
python .\generate_report_chat_html.py
```

Opções disponíveis:

```powershell
python .\generate_report_chat_html.py `
  --report-dir garakProbesReports `
  --hitlog-dir garakProbesHitlog `
  --output report_chat_browser.html `
  --review-dir garakManualReviews
```

Para abrir uma cópia com outro nome ou noutra pasta, alterar `--output`. Os caminhos relativos são resolvidos a partir da pasta onde o comando é executado.

## 3. Rever classificações manualmente

O ficheiro [review_server.py](review_server.py) serve o HTML e disponibiliza uma API local para gravar e remover revisões em `garakManualReviews`.

Primeiro iniciar o servidor:

```powershell
python .\review_server.py
```

Abrir no navegador:

```text
http://127.0.0.1:8765/
```

Opções do servidor:

```powershell
python .\review_server.py `
  --host 127.0.0.1 `
  --port 8765 `
  --root . `
  --index report_chat_browser.html `
  --review-dir garakManualReviews
```

Para terminar o servidor, premir `Ctrl+C` no terminal. As revisões ficam em ficheiros JSONL com o padrão:

```text
<Model>_<Probe>_misclassified_reviews.jsonl
```

O servidor deve ser usado apenas localmente, salvo se forem configuradas medidas de segurança adicionais. O HTML também suporta seleção da pasta `garakManualReviews` no navegador, mas o servidor é a forma mais simples de gravar as revisões diretamente no projeto.

## 4. Comparar modelos e gerar resultados

O ficheiro [compare_garak_models.py](compare_garak_models.py) lê os relatórios Garak, as revisões manuais e o ficheiro `model_evaluation.json`. Calcula taxas de `pass`/`hit`, estados das tentativas, resultados por detector/probe/modelo, digest, tempos de execução e métricas de classificação manual. Também gera gráficos e um resumo Markdown.

```powershell
python .\compare_garak_models.py
```

Argumentos disponíveis:

```text
--report-dir          Pasta com os ficheiros *.report.jsonl
                      (predefinição: garakProbesReports)
--output-dir          Pasta dos CSVs, gráficos e resumo
                      (predefinição: garakComparisonOutput)
--review-dir          Pasta com as revisões manuais
                      (predefinição: garakManualReviews)
--models              Lista opcional de IDs de modelos a incluir
--classification-json JSON com a classificação manual TP/FN/FP/TN
                      (predefinição: model_evaluation.json)
```

Exemplo filtrando os modelos analisados:

```powershell
python .\compare_garak_models.py `
  --models qwen3_5_q8 qwen3_5_q8_uncensored
```

O programa remove os artefactos gerados conhecidos da pasta de saída antes de os criar novamente. Guardar nessa pasta apenas resultados que possam ser regenerados.

## Ficheiro `model_evaluation.json`

Este ficheiro contém a classificação manual agregada por modelo e probe. Para cada modelo, as entradas podem conter:

- `positivos` e `negativos`: total de exemplos classificados como positivos/negativos;
- `fp`: falsos positivos;
- `fn`: falsos negativos;
- `summary`: totais globais de `TP`, `FN`, `FP` e `TN`.

O script calcula `TP = positivos - FP` e `TN = negativos - FN` por probe e escreve o resultado em `resumo_classificacao_json_por_probe.csv`.

## Principais ficheiros e pastas

| Caminho | Função |
| --- | --- |
| `garakDataAnalysis.py` | Executa as probes e organiza relatórios/hitlogs. |
| `generate_report_chat_html.py` | Gera o navegador HTML das conversas. |
| `review_server.py` | Servidor HTTP local para consultar e gravar revisões. |
| `compare_garak_models.py` | Produz comparações, tabelas, gráficos e resumo. |
| `model_evaluation.json` | Classificações manuais agregadas usadas na análise. |
| `report_chat_browser.html` | Navegador HTML gerado para as conversas. |
| `garakProbesReports/` | Relatórios JSONL por probe. |
| `garakProbesHitlog/` | Hitlogs JSONL por probe. |
| `garakManualReviews/` | Revisões manuais em JSONL. |
| `garakComparisonOutput/` | CSVs, gráficos e `resumo_comparacao.md`. |
| `garakRunsResults/` | Ficheiros agregados das execuções. |
| `OldManualReview/` | Dados de revisão manual antigos. Não é usado automaticamente. |

## Ordem recomendada para uma nova experiência

1. Confirmar que o Ollama está ativo e que o modelo configurado existe.
2. Ajustar `model` e `probes` em `garakDataAnalysis.py`.
3. Executar `garakDataAnalysis.py`.
4. Executar `generate_report_chat_html.py`.
5. Iniciar `review_server.py` e rever os casos necessários.
6. Executar `compare_garak_models.py`.
7. Consultar `garakComparisonOutput/resumo_comparacao.md`, os CSVs e os gráficos.

## Notas

- Os scripts usam caminhos relativos à raiz do projeto; executar os comandos nessa pasta.
- Os ficheiros JSONL podem ser grandes e não devem ser editados manualmente sem necessidade.
- `hit` representa uma deteção positiva do Garak e `pass` representa uma resposta que passou o detector.
- Os resultados dependem da versão do Garak, do modelo, dos detectors, das probes e da configuração do Ollama.
