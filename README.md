# IDS Generalization Pipeline

Versao minima funcional para testar um pipeline IoT/NIDS no ClusterGPU/UFV usando apenas pandas + scikit-learn.

Esta primeira versao e propositalmente simples: gera um dataset sintetico, remove colunas proibidas conforme a politica de features, faz split, preprocessing, `TruncatedSVD`, `MiniBatchKMeans`, avalia clustering, treina `RandomForestClassifier` e salva metricas/artefatos.

Nao inclui Spark, Dask, Polars, DuckDB, Faiss, HDBSCAN nem PyTorch Lightning.

## Instalar

```bash
cd ids_generalization_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

No Windows PowerShell:

```powershell
cd ids_generalization_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

## Rodar Localmente

O comando abaixo gera automaticamente um dataset sintetico:

```bash
python -m ids_pipeline.cli \
  --config configs/debug_small.yaml \
  --output-dir outputs/debug_small \
  --stage all
```

Se o pacote nao estiver instalado em modo editavel, use:

```bash
export PYTHONPATH=$PWD/src
python -m ids_pipeline.cli --config configs/debug_small.yaml --output-dir outputs/debug_small --stage all
```

Se estiver rodando em maquina com muitos cores, limite threads antes do Python para evitar erro do OpenBLAS:

```bash
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export BLIS_NUM_THREADS=8
```

## Rodar no ClusterGPU/UFV

O ClusterGPU/UFV usa Slurm, modulos EasyBuild, particao `scientific` e QoS `scientific-qos`. O exemplo oficial de job fica em `~/scripts/slurm/job_model_scientific_partition.slm`.

Copie o projeto para o cluster, entre no diretorio do projeto e crie o ambiente:

```bash
cd ids_generalization_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
mkdir -p ~/dados/ids_generalization_outputs
```

Se o cluster reclamar de instalacao editavel com `build_editable`, use uma destas alternativas:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

ou, como o script Slurm ja define `PYTHONPATH`, instale apenas as dependencias:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

No cluster, mantenha datasets, resultados e artefatos em `~/dados`, porque essa pasta aponta para o storage da maquina. Evite salvar datasets grandes ou resultados diretamente na `HOME`; use a `HOME` apenas para scripts leves e configuracoes.

Submeta o job Slurm a partir do diretorio do projeto:

```bash
sbatch scripts/run_debug_cluster_gpu.slm
```

Acompanhe a fila:

```bash
squeue --me
```

O job executa no diretorio onde foi submetido. Os arquivos de saida ficam nesse diretorio:

```bash
cat job_<jobid>.out
cat job_<jobid>.err
```

Para cancelar:

```bash
scancel <jobid>
```

O script `scripts/run_debug_cluster_gpu.slm` usa:

- particao: `scientific`
- QoS: `scientific-qos`
- 1 GPU: `--gres=gpu:1`
- 64 CPUs
- 128 GB RAM
- 30 minutos

Quando `IDS_COMPUTE_BACKEND=gpu` estiver definido, o pipeline tenta usar CuPy/cuML para partes do preprocessing, reducao dimensional, KMeans e RandomForest. A leitura CSV e a criacao dos splits ainda usam pandas nesta versao. Para evitar explosao de memoria no TON_IoT, o caminho GPU codifica categoricas com codigos ordinais limitados por frequencia, em vez de one-hot gigante. Se RAPIDS nao estiver instalado no ambiente Python do cluster, o job falha com uma mensagem pedindo cuML/CuPy; nesse caso instale RAPIDS compativel com CUDA do cluster ou rode com `IDS_COMPUTE_BACKEND=cpu`.

O script carrega os modulos oficiais:

```bash
module --force purge
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.9.1
source .venv/bin/activate
export IDS_COMPUTE_BACKEND=gpu
```

Para testar CuPy/cuML com acesso real a GPU, submeta o job curto:

```bash
sbatch scripts/check_gpu_rapids.slm
```

Nao rode o teste diretamente no no de login, porque o acesso direto as GPUs pode estar bloqueado.

Por padrao, o job grava os resultados em:

```bash
~/dados/ids_generalization_outputs/debug_small
```

Guarde outputs de experimentos em `~/dados/ids_generalization_outputs/`.

## Converter TON_IoT para Parquet

Antes de rodar o pipeline completo, converta os CSVs para Parquet particionado em `~/dados`:

```bash
cd ~/ids_generalization_pipeline
sbatch scripts/convert_ton_iot_to_parquet.slm
squeue --me
```

Depois confira:

```bash
ls -lh ~/dados/datasets/TON_IoT_parquet
cat ~/dados/datasets/TON_IoT_parquet/_conversion_manifest.json
```

O conversor faz uma passada inicial para inferir um schema comum entre todas as particoes CSV. Isso evita o erro do cuDF/Parquet:

```text
All sources must have the same schema
```

Se voce ja converteu os CSVs antes desta correcao, rode novamente o job de conversao; ele usa `--overwrite` e recria os Parquets com tipos consistentes.

O job principal usa por padrao:

```bash
DATA_PATH="$HOME/dados/datasets/TON_IoT_parquet"
```

Quando o backend GPU esta ativo, o loader tenta ler os Parquets com cuDF e converte para pandas para manter os splits e politicas atuais; as etapas seguintes usam CuPy/cuML. Se o cuDF falhar na leitura por alguma incompatibilidade de IO, o pipeline cai para `pandas.read_parquet` apenas na etapa de leitura e continua usando GPU nas etapas posteriores.

## Outputs Esperados

Apos rodar `stage=all`, verifique:

```bash
ls -lh outputs/debug_small
```

No ClusterGPU/UFV, verifique:

```bash
ls -lh ~/dados/ids_generalization_outputs/debug_small
```

Arquivos principais:

- `metrics_clustering.json`: metricas internas e externas do MiniBatchKMeans.
- `metrics_supervised.json`: metricas de validacao e teste do RandomForest.
- `profiling.json`: tempos e uso aproximado de RAM.
- `selected_features.json`: features usadas pelo modelo.
- `forbidden_columns_check.json`: resultado da checagem anti-leakage.
- `cluster_assignments.csv`: clusters para train/val/test, quando `export_cluster_assignments: true` (padrao para configs antigas).
- `classification_report.txt`: relatorio supervisionado no teste.
- `confusion_matrix.csv`: matriz de confusao no teste.
- `artifacts/preprocessing_bundle.joblib`: imputers, scaler, encoder e SVD.
- `artifacts/minibatch_kmeans.joblib`: clusterer treinado.
- `artifacts/random_forest.joblib`: classificador treinado.

O config `configs/ton_iot_behavioral_strict_gpu.yaml` define `export_cluster_assignments: false`: o pipeline nao monta o DataFrame de exportacao nem escreve esse CSV. Labels, distances, metricas e o retorno do clustering permanecem disponiveis. Essa opcao nao apaga CSVs de execucoes anteriores; use um diretorio de saida novo para cada execucao.

Para conferir que `no_raw_ip_port` nao vazou IPs, portas ou rotulos:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("outputs/debug_small/selected_features.json")
features = set(json.loads(p.read_text())["selected_features"])
forbidden = {"src_ip", "dst_ip", "src_port", "dst_port", "label", "type"}
print("forbidden present:", sorted(features & forbidden))
PY
```

O resultado esperado e uma lista vazia.

## Testes

```bash
python -m pytest -q
```

Em ambientes onde o diretorio temporario padrao esta bloqueado, use um `basetemp` local:

```bash
mkdir -p .tmp_pytest
python -m pytest -q --basetemp .tmp_pytest -p no:cacheprovider
```

## Politicas de Features

- `all_except_labels`: remove alvos (`label`, `type`, `target`, `attack_cat`, `class`) e metadados de dataset.
- `no_raw_ip`: remove IPs brutos e alvos; mantem portas.
- `no_raw_ip_port`: remove IPs, portas e alvos; e a politica padrao.
- `behavioral_strict`: remove IPs, portas, servico e identificadores estaveis; mantem estatisticas comportamentais.

As checagens falham se `label` ou `type` entrarem nas features em qualquer politica. Tambem falham se IPs/portas entrarem em `no_raw_ip_port` ou `behavioral_strict`.
