# Fase 2 — proposta do baseline científico completo da variante GPU

Status atual: **IMPLEMENTADO — JOB NÃO SUBMETIDO; AGUARDA APROVAÇÃO FINAL**.

A proposta abaixo foi aprovada para implementação. O registro ao final documenta a entrega implementada; os trechos que descrevem a primeira entrega preservam seu contexto histórico.
Data: 09/09/2026. Código inspecionado: commit `f8bf2c585a971beaf1d21faff10d6db6f0adf1d6`.

## 1. Decisão aceita e escopo

A Fase 1G foi deliberadamente omitida por decisão do projeto. As diferenças da [Fase 1F](phase_1f_gpu_methodological_equivalence.md) foram aceitas. Esta proposta não condiciona a execução a comparações CPU/GPU ou controles da fase omitida.

O experimento responderá exclusivamente: **qual é o baseline completo da variante GPU da extensão no protocolo `group_stratified/2.0.0` congelado?** A identificação será:

```yaml
phase: "2"
scientific_result: true
integration_only: false
baseline_type: gpu_extension
exact_replication_of_original_article: false
```

Não se altera schema, quarentena, população, features, preprocessing, redução, algoritmo/parâmetros de clustering, RF, balanceamento, seed, targets ou critérios de split. Não haverá representantes, seleção por orçamento, outro protocolo, outra seed ou outro k. O treino usará **13.575.269 registros**; validação e teste completam a população elegível de **22.338.152**.

Esta primeira entrega cria apenas este documento. YAML e Slurm completos estão incluídos como conteúdo revisável; os arquivos executáveis indicados abaixo serão criados somente após aprovação.

## 2. Diagnóstico final do código e integração mínima

| Componente atual | Constatação | Encaminhamento proposto |
|---|---|---|
| `data_loading.load_dataset` | Loader oficial resolve fontes, lê as 23 partições em ordem lexicográfica, aplica qualidade por fonte e schema, preservando identidade. Concatena internamente e mantém a população em RAM. | Chamar uma única vez com o diretório completo e backend GPU, sem `sample_size`. Não substituir por concatenação externa. |
| I/O | cuDF pode recorrer a pandas se a leitura/conversão falhar. O retorno oficial é pandas. | Observar os logs como na 1E; registrar backend e motivos. Fallback de I/O isolado não invalida se todos os contratos de dados permanecerem válidos. |
| `config.load_config` | Carrega os campos conhecidos e guarda outros em `extra`; não valida sozinho todas as restrições nem faz herança de YAML. | Executor exclusivo verificará os campos operacionais e comparará integralmente os parâmetros de `reuse_split.yaml`. |
| `splitting.create_splits` | Reutiliza frozen quando configurado, mas contém também caminhos de geração. `save_splits` grava arrays novamente. | Chamar diretamente `split_protocols.load_frozen_splits`; não chamar `create_splits`, `group_split` ou `save_splits`. |
| `load_frozen_splits` | Carrega arrays e recalcula o manifesto contra a população/configuração atual; falha se divergir do salvo. | Acrescentar, no executor, confronto com o hash aprovado fixo, contagens e cobertura exigidas. Não mudar a função científica. |
| `cli.run_experiment` | `--stage all` passa por representantes e grava novamente os índices, mesmo quando o RF usa treino completo. | Não usar a CLI geral para este job. Orquestrar as mesmas funções científicas em script exclusivo. |
| Executor 1E | Tem guardas reutilizáveis como referência, mas impõe 60k/15k/25k linhas, RF de 20 árvores e marcação de integração. | Não executar nem modificar o smoke. Criar executor científico sem `derive_subset`, remapeamento de subconjunto ou marcadores de integração. |
| Preprocessing | Tipagem semântica, log explícito, mediana e estatísticas pandas de treino; matrizes CuPy, one-hot nominal, redutor cuML. | Chamar `fit_transform_preprocessing` intacta; comprovar objetos, dimensões e artefato serializado. Descrever o caminho como híbrido. |
| Redução | `effective=min(requested,max_components)`. Alternativa cuML PCA existente na importação/construção do redutor. | Manter requested=50 e comportamento existente; registrar o redutor real. PCA eventual será identificado como diferença, sem chamá-lo TruncatedSVD. |
| Clustering | cuML KMeans, k=30, max_iter=100, seed=42. `cluster_batch_size` e `cluster_n_init` não são encaminhados ao GPU. | Preservar chamada atual. Registrar parâmetros solicitados, aplicáveis, ignorados e `get_params()` efetivo. |
| Exportação de assignments | A construção de `assignment_rows` e a escrita estão dentro de `if config.export_cluster_assignments`. Labels/distances e métricas permanecem no retorno. | Exigir `false` antes da chamada e conferir ausência do arquivo sem ler seu conteúdo. |
| Métricas de clustering | Silhouette amostrado; Davies–Bouldin e Calinski–Harabasz no treino completo; AMI/V/purity em cada split. Cálculo sklearn/CPU e cópia da matriz de treino para host. | Preservar escopo e fórmulas; explicitar o custo CPU mesmo com treinamento GPU. Acrescentar apenas contagens compactas dos clusters ao relatório. |
| RF | cuML, 150 árvores, seed=42, streams derivados de `n_jobs` (16), sem pesos explícitos; targets label/type/cluster_id. | Chamar `train_random_forest` com treino completo e `representatives=None`, sem alterar parâmetros ou código científico. |
| Saídas supervisionadas | Métricas globais de val/test; relatórios por classe e matrizes atuais são gravados para **teste**. Algumas falhas de probabilidades resultam em métricas binárias `null`. | Preservar saídas; executor complementará relatórios exatos por classe e matriz de validação, usando predições do modelo salvo e as mesmas funções sklearn. Não prometer saídas que hoje não existem. |
| `Profiler` | Tempos por etapa incluem fit, métricas e I/O; RAM/VRAM observadas nas fronteiras. Não mede pico contínuo nem RF.fit isolado. | Reutilizar, sincronizando CUDA nos limites externos. Complementar com `/usr/bin/time -v`; não inventar tempos de kernels ou fit exclusivo. |

Referências locais: [loader](../src/ids_pipeline/data_loading.py), [config](../src/ids_pipeline/config.py), [CLI](../src/ids_pipeline/cli.py), [split](../src/ids_pipeline/split_protocols.py), [preprocessing](../src/ids_pipeline/preprocessing.py), [clustering](../src/ids_pipeline/clustering.py), [RF](../src/ids_pipeline/supervised.py), [profiler](../src/ids_pipeline/profiling.py), [executor 1E](../scripts/run_phase_1e_real_gpu_smoke.py).

Não será habilitado `class_weight='balanced'`, `sample_weight`, `balanced_subsample` ou bootstrap diferente. Não se encaminhará `cluster_n_init=5` ao cuML como uma suposta correção. A execução manterá as diferenças aceitas na Fase 1F.

## 3. Arquivos previstos após aprovação

| Arquivo | Finalidade |
|---|---|
| `scripts/run_phase_2_full_gpu_baseline.py` | Orquestração científica exclusiva, guardas, proveniência, manifesto e relatórios agregados. |
| `configs/phase_2_full_gpu_baseline.yaml` | Configuração completa abaixo. |
| `scripts/run_phase_2_full_gpu_baseline.slm` | Job exclusivo abaixo. |
| `tests/test_phase_2_full_gpu_baseline.py` | Fixtures pequenas de configuração, guardas, manifesto, exportação e integridade dos artefatos. |
| `provenance/phase_2_source_provenance.json` | Snapshot pequeno gerado antes da transferência/submissão, vinculando SHA Git, estado pendente e hashes dos arquivos executados. Não é código científico. |

Nenhum módulo de `src/ids_pipeline` precisa ser modificado para esta proposta. Os relatórios adicionais serão produzidos no executor; não mudam treinamento, predições ou definições das métricas.

## 4. Configuração YAML completa proposta

Destino após aprovação: `configs/phase_2_full_gpu_baseline.yaml`.

```yaml
dataset_name: ton_iot_network_full
data_path: /home/es112688/dados/datasets/TON_IoT_parquet
output_dir: /storage/dados/es112688/ids_generalization_outputs/phase_2_full_gpu_baseline_group_stratified
phase: "2"
scientific_result: true
integration_only: false
baseline_type: gpu_extension
exact_replication_of_original_article: false
phase_1g_omitted_by_project_decision: true
methodological_reference: docs/phase_1f_gpu_methodological_equivalence.md
source_provenance_path: provenance/phase_2_source_provenance.json

compute_backend: gpu
strict_gpu: true
feature_policy: behavioral_strict
expected_feature_policy_version: behavioral_strict/2.0.0
expected_selected_features:
  - duration
  - src_bytes
  - dst_bytes
  - conn_state
  - missed_bytes
  - src_pkts
  - src_ip_bytes
  - dst_pkts
  - dst_ip_bytes
label_col: label
type_col: type

split_strategy: group_stratified
frozen_splits_dir: /storage/dados/es112688/ids_generalization_splits/group_stratified_v2
reuse_split_config: /storage/dados/es112688/ids_generalization_splits/group_stratified_v2/reuse_split.yaml
expected_split_protocol_version: group_stratified/2.0.0
expected_split_hash: sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a
group_cols: [src_ip, dst_ip, service, proto]
test_size: 0.25
val_size: 0.20
timestamp_col: ts
timestamp_unit: s
temporal_bucket_freq: 1s
timestamp_policy: drop
service_policy: drop
group_split_tolerance: 0.02
group_split_candidates: 16
split_min_class_support: 1
split_small_support_threshold: 30
split_export_csv: false

expected_files: 23
expected_original_rows: 22339021
expected_quarantined_rows: 869
expected_eligible_rows: 22338152
expected_schema_version: ton_iot_network/1.0.0
expected_data_quality_policy_version: ton_iot_known_corruption/1.0.0
expected_full_split_rows: {train: 13575269, val: 3244049, test: 5518834}
expected_type_classes:
  - backdoor
  - ddos
  - dos
  - injection
  - mitm
  - normal
  - password
  - ransomware
  - scanning
  - xss

svd_components: 50
onehot_min_frequency: 10
gpu_max_categories_per_col: 64
log1p_numeric: true
log1p_columns: null  # lista explícita do schema congelado

selected_k: 30
cluster_batch_size: 16384  # compatibilidade; ignorado pelo KMeans GPU atual
cluster_n_init: 5         # compatibilidade; ignorado pelo KMeans GPU atual
cluster_max_iter: 100
silhouette_sample_size: 20000
export_cluster_assignments: false

representative_strategy: full
use_representatives_for_supervised: false
representatives_per_cluster: 20  # legado inativo; não será chamado
boundary_per_cluster: 5         # legado inativo; não será chamado
targets: [label, type, cluster_id]
random_forest_estimators: 150
random_state: 42
n_jobs: 32
log_level: INFO
```

O YAML não introduz sampling, selection budget, novos seletores ou override de pesos/defaults do RF. Os campos de expectativas são **guardas do executor futuro**, não parâmetros novos dos estimadores. O loader oficial aceita esses campos em `extra`; a validação semântica adicional terá testes próprios.

O executor lerá `reuse_split.yaml` como parte da resolução, aceitando somente as chaves de split reconhecidas na 1E, exigindo que todas existam no YAML completo e coincidam. `frozen_splits_dir` será comparado por caminho resolvido. Assim os parâmetros do arquivo de reutilização são incorporados e verificados, sem merge silencioso e sem regeneração.

## 5. Fluxo exato e validação do hash

1. Carregar o YAML pelo loader oficial. Validar os marcadores científicos, população completa, GPU estrito, RF=150, seed=42, k=30, targets, features, exportações desativadas e ausência de opções de sampling/seleção. Não admitir defaults sintéticos.
2. Verificar origem/hashes do código executado, manifesto e arquivos frozen, `reuse_split.yaml`, diretório de dataset e conjunto ordenado das 23 fontes retornado pelo resolvedor oficial. `_conversion_manifest.json` não é uma partição. A ordem é a oficial lexicográfica, não a ordenação numérica 1,2,3.
3. Confirmar CUDA real, uma GPU visível A100 de 80 GB e classes cuML necessárias **antes de carregar o dataset**. Verificar configuração e `IDS_COMPUTE_BACKEND=gpu`; não aceitar `auto`.
4. Exigir saída científica inexistente, inclusive ausência de symlink, e criá-la sem sobrescrita. Registrar manifesto `RUNNING`, timestamp UTC, hostname, ambiente, recursos e proveniência. Falha inicial não se torna resultado científico aprovado.
5. Chamar `load_dataset(data_path, compute_backend='gpu', schema_report_path=..., data_quality_report_path=...)`, **sem argumento de amostragem**. Conferir 22.339.021 originais, 869 quarentenados, 22.338.152 elegíveis, versões e `schema_report.ok=true`, identidade e ordenação preservadas. Registrar `io_backend`, `io_gpu_fallback` e motivo.
6. Conferir o ID de população com `data_quality_population.json` e `split_manifest.json` congelados. Chamar `load_frozen_splits(frozen_dir, df, config)` diretamente. Esta função recalcula seu manifesto e valida o hash salvo; depois o executor recalcula `manifest(df, splits, config)` para persistir a evidência observada e confrontá-la com o hash **fixo aprovado**, independentemente do arquivo salvo.
7. Exigir contagens exatas, união dos índices igual à população elegível, ausência de duplicatas/sobreposição, limites corretos e `validation_ok=true`. Conferir explicitamente os três pares de sobreposição de grupos igual a zero, `label={0,1}` e todas as dez classes de `type` em train/val/test. Classes com suporte <30 serão reportadas, sem mudar o split.
8. Conferir, em blocos, a correspondência de `eligible_original_positions.npy` (mmap somente leitura) com o índice original do DataFrame. Salvar somente hashes/relatórios compactos; não duplicar os arrays frozen nem criar associações por linha.
9. Só após essas guardas, selecionar as nove features pela política oficial, aplicar as políticas de proxy existentes (`drop` para timestamp/serviço) e verificar ordem, versão e colunas proibidas. Chamar preprocessing oficial sobre o DataFrame completo e **os mesmos arrays frozen**, sem recortar ou remapear a população experimental.
10. Verificar três matrizes CuPy finitas, tamanhos 13.575.269/3.244.049/5.518.834, mesma largura, oito quantitativas, one-hot nominal, unknown all-zero e vocabulário de treino. Confirmar requested/effective, objeto cuML do redutor e bundle serializado; não forcejar 50 componentes.
11. Executar `fit_predict_clustering` oficial. Conferir objeto real cuML KMeans, k, seed e tamanho/validade de labels e distances em todos os splits. Persistir parâmetros reais e distribuições dos 30 clusters, incluindo contagens zero. Não construir outro DataFrame de assignments.
12. Executar `train_random_forest` oficial com os três targets, treino completo, `representatives=None` e profiler. Não chamar `build_cluster_representatives`. Conferir tamanho de treino usado em cada target, classe cuML e parâmetros efetivos de cada RF salvo, sem reter simultaneamente cópias desnecessárias dos três modelos recarregados.
13. Completar os relatórios de validação descritos na seção 8. Auditar artefatos, probabilidades/métricas binárias e ausência de exportações proibidas. Fechar manifesto de forma atômica como `APPROVED` somente se todas as guardas obrigatórias passarem; falhas capturáveis resultam em `INVALID` e saída não zero.

O hash científico **não é o SHA256 do texto de `split_manifest.json`**. A função congelada calcula SHA256 da serialização canônica dos campos `protocol_version`, `population`, `parameters`, `seed` e `index_hashes`. Cada hash de índices usa inteiros little-endian int64; a identidade de população inclui a ordem/índice e o conteúdo das colunas do split, além da proveniência de qualidade. Não reimplementar esse algoritmo no executor.

Esperado = salvo = recalculado:

```text
sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a
```

Qualquer divergência interrompe o job **antes de preprocessing/modelos**. Não há tentativa alternativa de split. Chamadas às rotas de geração serão proibidas nos testes do executor.

## 6. Slurm completo proposto

Destino após aprovação: `scripts/run_phase_2_full_gpu_baseline.slm`. A solicitação de 256 GiB/12 horas é **provisória**, sujeita às verificações de recursos da seção 7. Não foi submetida.

```bash
#!/bin/bash
#SBATCH --job-name=ids_phase2_full
#SBATCH --partition=scientific
#SBATCH --qos=scientific-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=phase_2_full_gpu_%j.out
#SBATCH --error=phase_2_full_gpu_%j.err

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
module --force purge
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.9.1
source .venv/bin/activate

export IDS_COMPUTE_BACKEND=gpu
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"

config="configs/phase_2_full_gpu_baseline.yaml"
output="/storage/dados/es112688/ids_generalization_outputs/phase_2_full_gpu_baseline_group_stratified"
output_parent="${output%/*}"
evidence_prefix="$output_parent/phase_2_job_${SLURM_JOB_ID:?Missing job ID}"

test -f "$config"
test -f scripts/run_phase_2_full_gpu_baseline.py
test -f provenance/phase_2_source_provenance.json
test -d "$HOME/dados/datasets/TON_IoT_parquet"
test -r /storage/dados/es112688/ids_generalization_splits/group_stratified_v2/reuse_split.yaml
if [[ -e "$output" || -L "$output" ]]; then
    echo "Refusing existing scientific output: $output" >&2
    ls -ld -- "$output"
    du -sh -- "$output"
    find "$output" -maxdepth 1 -printf '%f\n' | sed -n '1,20p'
    exit 2
fi
mkdir -p "$output_parent"
export IDS_PHASE2_TIME_FILE="${evidence_prefix}.time.txt"
export IDS_PHASE2_ALLOCATION_FILE="${evidence_prefix}.allocation.txt"

scontrol show job "$SLURM_JOB_ID" > "$IDS_PHASE2_ALLOCATION_FILE"
df -h "$output_parent"
module list 2>&1
python --version
nvidia-smi -L
echo 'phase=2 scientific_result=true integration_only=false baseline_type=gpu_extension exact_replication_of_original_article=false'

set +e
if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v -o "$IDS_PHASE2_TIME_FILE" \
        python -B scripts/run_phase_2_full_gpu_baseline.py --config "$config"
else
    echo 'GNU time unavailable; stage profiling remains enabled.' > "$IDS_PHASE2_TIME_FILE"
    python -B scripts/run_phase_2_full_gpu_baseline.py --config "$config"
fi
run_status=$?
set -e
cat "$IDS_PHASE2_TIME_FILE"
exit "$run_status"
```

O executor também verificará o output definido no YAML, ambiente e alocação. O shell não cria o diretório científico antes dele, evitando conflito com `mkdir(exist_ok=False)`. Logs de accounting ficam como arquivos pequenos no diretório pai, referenciados no manifesto. O retorno Python é preservado; não há retomada, redução automática de população/árvores ou requeue automático.

`--gres=gpu:1` reserva uma GPU, mas **não garante o modelo A100** numa partição heterogênea. A guarda consulta CUDA e recusa outro modelo ou memória <75 GiB antes de leitura/treinamento. Antes da versão final do job, será conferido o nome GRES/constraint real de A100-80GB no cluster; não se inventa um nome como `a100_80gb`. Se houver heterogeneidade, o job final precisa selecionar a classe de nó correta além dessa guarda.

## 7. Recursos, estimativas e verificações remotas pendentes

### 7.1 Solicitação proposta

| Recurso | Proposta | Evidência / limite |
|---|---|---|
| GPU | 1 × A100-SXM4-80GB | Ambiente validado na 1E; disponibilidade/tipo de GRES ainda requer consulta. |
| CPU | 32 CPUs, 1 processo principal | Preserva `n_jobs=32` e permite trabalho CPU do pipeline híbrido/métricas. |
| RAM | 256 GiB por nó, provisórios | Mesma solicitação do job 1E local; MaxRSS real não foi fornecido nem encontrado no workspace. Não é estimativa medida de consumo. |
| Tempo | Limite de 12 horas, provisório | Envelope inicial solicitado pelo projeto. Não há medição válida do RF de 150 árvores/13,6 milhões para prever duração. |

O Slurm interpreta `--mem` como solicitação por nó e permite limites da partição; esses limites devem ser consultados antes da submissão. [Documentação oficial de sbatch](https://slurm.schedmd.com/sbatch.html).

### 7.2 Estimativa de memória, sem confundir matriz com processo

Os cálculos abaixo são analíticos, sem leitura dos dados. A hipótese de dimensão 20 vem do smoke e **não fixa a dimensão efetiva do treino completo**.

| Componente isolado | Estimativa |
|---|---:|
| Oito quantitativas, população completa, uma cópia float64 | 1,331 GiB |
| Matrizes reduzidas dos três splits, d=20, float32 | 1,664 GiB |
| Matriz reduzida de treino, d=20, float32 | 1,011 GiB |
| Matrizes dos três splits, cenário d=50, float32 | 4,161 GiB |
| Uma coleção de índices int64, população completa | 0,166 GiB |
| Labels int64 e distances float64 de todos os splits | 0,333 GiB |
| Probabilidades do teste para 30 classes, float32 | 0,617 GiB |
| 47 colunas × N × 8 bytes, apenas slots de valores/referências | 7,822 GiB, **não** footprint do DataFrame |

Strings, máscaras de nulidade, objetos pandas, normalização, DataFrames intermediários, cópias por split, arrays CuPy antes/depois da redução, pools de alocação, serialização e estruturas das árvores não estão contidos nessas estimativas. Várias cópias coexistem no código congelado. A construção de RF com profundidade padrão sem limite no cuML 26.08 também não pode ser dimensionada apenas pela matriz de entrada.

**Estimativa total de pico: ainda indeterminada com a evidência disponível.** Proponho reservar 256 GiB, sem apresentar esse valor como consumo esperado ou garantia de cabimento. O smoke 1E já carregou/validou a população inteira, mas descartou grande parte antes de preprocessing/RF; seu sucesso não demonstra a memória dos estágios completos. Se o MaxRSS ou a capacidade real do nó contraindicar essa reserva, a proposta deve voltar para ajuste de recursos, sem reduzir dataset/modelos.

### 7.3 Estimativa de tempo

Os logs fornecidos da 1E mostram início às 02:01:18 e término do carregamento/schema às 02:06:04, aproximadamente **4 min 46 s** nessa execução. Isso dá uma referência operacional somente para o início do fluxo. Não contém tempo confiável de fit completo, e o smoke usa 60.000 linhas de treino e 20 árvores.

**Não há estimativa pontual defensável para a duração total.** Para planejamento, reservar uma janela de 12 horas e considerar risco explícito de TIMEOUT. Não multiplicar linearmente o tempo do smoke por linhas × árvores e não prometer término dentro da janela. KMeans, RF e métricas CPU têm custos e alocações distintos. Nenhum ensaio adicional ou comparação CPU/GPU será exigido como Fase 1G disfarçada.

### 7.4 Consultas necessárias antes de submeter — não executadas localmente

Executar no login do cluster, somente leitura:

```bash
sacct -j 5428 --units=G --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,MaxVMSize
scontrol show partition scientific
sinfo -p scientific -N -o '%N|%c|%m|%G|%f|%l|%t'
sacctmgr show qos scientific-qos format=Name,MaxWall,MaxTRES,MaxTRESPU -P
df -h /storage/dados/es112688
df -i /storage/dados/es112688
quota -s
test ! -e /storage/dados/es112688/ids_generalization_outputs/phase_2_full_gpu_baseline_group_stratified
test ! -L /storage/dados/es112688/ids_generalization_outputs/phase_2_full_gpu_baseline_group_stratified
```

Consultar também `profiling.json` da 1E se disponível. Suas amostras de RAM não são MaxRSS contínuo. No `sacct`, não omitir os steps `.batch`/outros: o MaxRSS pode estar no step e não na linha principal. Se accounting, quota ou informações do QoS não forem expostos ao usuário, registrar a indisponibilidade e obter a confirmação necessária da administração; não declarar que os limites foram aprovados. [Documentação oficial de sacct](https://slurm.schedmd.com/sacct.html).

Não há sessão remota do cluster nesta execução local. Portanto espaço/quota, inexistência do output, limites de RAM/tempo, GRES e MaxRSS **continuam pendentes**, não aprovados por inferência. Espaço livre de filesystem não substitui quota pessoal. Tamanho final dos três RFs é desconhecido; a verificação de storage precisa contemplar os modelos exigidos, logs e margem para serialização, sem gravar previsões individuais ou cópias do dataset.

## 8. Métricas e artefatos produzidos

### 8.1 Clustering

Preservar exatamente as funções atuais:

| Métrica | População de cálculo |
|---|---|
| Inertia | Modelo ajustado no treino completo. |
| Silhouette | Amostra de treino de até 20.000 linhas, seed=42, seleção existente; não é amostra da população experimental usada nos modelos. |
| Davies–Bouldin | Treino completo, CPU/sklearn atual. |
| Calinski–Harabasz | Treino completo, CPU/sklearn atual. |
| AMI, V-measure e purity contra `label` e `type` | Separadamente em train/val/test completos. |
| Tamanho/fração dos clusters | Contagens compactas para IDs 0–29 em train/val/test, inclusive zeros. |

As métricas internas condicionais continuam obedecendo ao código quando faltar diversidade de clusters. Registrar indisponibilidade/degeneração, sem fabricar zeros ou forçar divisões. Não exigir 30 clusters ocupados se o resultado real não os ocupar; reportar esse achado sem reajustar k.

### 8.2 Supervisionado

Para `label`, `type` e `cluster_id`, em **validação e teste**: accuracy, macro/weighted precision, recall e F1; precision/recall/F1/support por classe e matriz de confusão. Para `label`, conservar ROC-AUC, Average Precision, FPR@TPR95, threshold correspondente e classe positiva. O campo legado `pr_auc` será identificado como **Average Precision**, sem mudar cálculo ou interpretá-lo como integral trapezoidal.

O código atual já grava as métricas globais e matrizes/relatórios de teste. Para evitar outro fit ou repetição desnecessária da predição do teste, os relatórios estruturados exatos de teste podem ser derivados de sua matriz de contagens já salva. Para validação, o executor carregará cada RF salvo por vez, fará uma predição adicional e gerará a matriz/relatório com as mesmas regras atuais (`zero_division=0`, classes ordenadas). Esse custo de auditoria será identificado separadamente, sem misturá-lo com RF.fit.

Conferir soma dos suportes e da matriz com o tamanho do split e a correspondência das métricas agregadas ao resultado oficial. Distribuições de `label/type` serão obtidas do relatório do split validado; para `cluster_id`, das labels internas. Manter todas as classes de referência no relatório de suporte, inclusive zeros, sem redefinir silenciosamente a lista usada pela fórmula macro-F1 atual.

O executor não aceitará silenciosamente ausência das métricas binárias esperadas para `label` com ambas as classes presentes: a disponibilidade de `predict_proba` e valores finitos precisa ser conferida. Erro de cálculo/reporting torna a entrega incompleta/INVALID; não se altera o RF para obter métricas melhores. O threshold de TPR95 é descritivo da curva avaliada, não um limiar operacional ajustado para deployment.

O relatório destacará `mitm`, `ransomware` e `backdoor`, seu suporte real em cada split e eventuais recall/F1 zero. Suporte pequeno não é motivo para excluir ou reamostrar dados, e ausência de predição de uma classe é resultado a reportar, não motivo para refazer o treino.

### 8.3 Organização de artefatos

```text
phase_2_full_gpu_baseline_group_stratified/
  experiment_manifest.json
  effective_config.json
  source_provenance.json
  dataset_info.json
  schema_report.json
  data_quality_report.json
  split_manifest_observed.json
  split_reference.json
  selected_features.json
  forbidden_columns_check.json
  clustering_metadata.json
  metrics_clustering.json
  metrics_supervised.json
  cluster_distribution.json
  target_distributions.json
  classification_report_label.txt
  classification_report_type.txt
  classification_report_cluster_id.txt
  confusion_matrix_label.csv
  confusion_matrix_type.csv
  confusion_matrix_cluster_id.csv
  metrics_by_class.json
  validation_confusion_matrices/
    label.csv
    type.csv
    cluster_id.csv
  profiling.json
  pipeline.log
  artifacts/
    preprocessing_metadata.json
    preprocessing_bundle.joblib
    cuml_kmeans.joblib
    random_forest_label.joblib
    random_forest_type.joblib
    random_forest_cluster_id.joblib
```

Manter também aliases pequenos produzidos pelo pipeline, como `classification_report.txt` e `confusion_matrix.csv`, quando existentes. O inventário final refletirá os arquivos realmente produzidos. `split_reference.json` aponta ao diretório persistente e registra hashes; não contém milhões de posições.

Nenhum CSV de assignments, dataset transformado, vetor completo de predições, tabela por observação, exportação de representantes ou cópia dos índices frozen. Labels/distances necessárias permanecem em memória pelo retorno oficial. Os modelos podem ser grandes, mas são artefatos explicitamente exigidos; não se fará uma segunda cópia desnecessária deles.

Logs Slurm `.out/.err`, arquivo de alocação e GNU time ficam referenciados por caminho no manifesto, ainda que localizados no diretório de submissão/pai. Não devem ser perdidos na coleta dos resultados.

## 9. Manifesto científico proposto

Estrutura abaixo: `null` identifica informação coletada em runtime, **não valor já observado**. Campos numéricos conhecidos são expectativas; só serão ratificados após as verificações.

```json
{
  "experiment_id": null,
  "phase": "2",
  "status": "RUNNING",
  "scientific_result": true,
  "integration_only": false,
  "baseline_type": "gpu_extension",
  "exact_replication_of_original_article": false,
  "phase_1g_omitted_by_project_decision": true,
  "started_at_utc": null,
  "completed_at_utc": null,
  "git_commit_sha": null,
  "git_dirty": null,
  "git_pending_changes": [],
  "code_sha256": {},
  "code_snapshot_hash": null,
  "methodological_reference": "docs/phase_1f_gpu_methodological_equivalence.md",
  "data": {
    "dataset": "ton_iot_network_full",
    "data_path": "/home/es112688/dados/datasets/TON_IoT_parquet",
    "files": 23,
    "ordered_source_ids": [],
    "original_rows": 22339021,
    "quarantined_rows": 869,
    "eligible_rows": 22338152,
    "schema_version": "ton_iot_network/1.0.0",
    "data_quality_policy_version": "ton_iot_known_corruption/1.0.0",
    "population_id": null,
    "io_backend": null,
    "io_gpu_fallback": null,
    "io_gpu_fallback_reason": null
  },
  "split": {
    "protocol": "group_stratified",
    "protocol_version": "group_stratified/2.0.0",
    "frozen_splits_dir": "/storage/dados/es112688/ids_generalization_splits/group_stratified_v2",
    "expected_hash": "sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a",
    "observed_hash": null,
    "hash_match": null,
    "train_rows": 13575269,
    "val_rows": 3244049,
    "test_rows": 5518834,
    "group_columns": ["src_ip", "dst_ip", "service", "proto"],
    "parameters": {},
    "index_hashes": {},
    "identity_hashes": {},
    "group_overlap": null,
    "class_coverage": null,
    "regenerated": false
  },
  "features": {
    "feature_policy": "behavioral_strict",
    "feature_policy_version": "behavioral_strict/2.0.0",
    "selected_features": ["duration", "src_bytes", "dst_bytes", "conn_state", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"],
    "numeric_features": ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"],
    "categorical_features": ["conn_state"],
    "conn_state_encoding": null
  },
  "preprocessing": {
    "log1p_columns": null,
    "imputation": "training_median_numeric_and_training_mode_conn_state",
    "scaling": "existing_training_statistics_numeric_only",
    "statistics_backend": "pandas_cpu",
    "numeric_preparation_backend": "pandas_cpu",
    "matrix_backend": "cupy_gpu",
    "encoded_dimension": null,
    "svd_components_requested": 50,
    "svd_components_effective": null,
    "reducer": null,
    "reducer_implementation": null,
    "reduction_backend": null,
    "matrix_shapes": {},
    "end_to_end_gpu": false
  },
  "clustering": {
    "implementation": null,
    "backend": "gpu",
    "k": 30,
    "random_state": 42,
    "parameters_effective": {},
    "configured_but_not_forwarded": {"cluster_batch_size": 16384, "cluster_n_init": 5},
    "algorithmic_difference_from_original": true,
    "original_algorithm": "sklearn MiniBatchKMeans",
    "current_algorithm": "cuML KMeans",
    "export_cluster_assignments": false,
    "metric_scopes": null
  },
  "supervised": {
    "implementation": null,
    "backend": "gpu",
    "estimators": 150,
    "targets": ["label", "type", "cluster_id"],
    "random_state": 42,
    "class_weight": null,
    "sample_weight_supplied": false,
    "difference_from_original_balanced_subsample": true,
    "per_target_parameters_effective": {},
    "per_target_train_rows": {},
    "training_index_hash": null
  },
  "software": {
    "python": null, "numpy": null, "pandas": null,
    "scipy": null, "scikit_learn": null, "joblib": null,
    "cupy": null, "cudf": null, "cuml": null,
    "cuda_toolkit": null, "cuda_runtime": null,
    "nvidia_driver": null, "gcc": null
  },
  "hardware": {
    "gpu": null, "gpu_uuid": null, "vram_bytes": null,
    "cpus_allocated": null, "ram_allocated": null,
    "SLURM_JOB_ID": null, "CUDA_VISIBLE_DEVICES": null,
    "hostname": null, "allocation_evidence_path": null
  },
  "timing": {
    "loading_seconds": null,
    "split_loading_validation_seconds": null,
    "preprocessing_and_reduction_seconds": null,
    "clustering_seconds": null,
    "supervised_label_seconds": null,
    "supervised_type_seconds": null,
    "supervised_cluster_id_seconds": null,
    "supplemental_reporting_seconds": null,
    "total_seconds": null,
    "scope": "stage_wall_time_including_metrics_and_io",
    "rf_fit_only_measured": false,
    "gnu_time_path": null,
    "continuous_vram_peak_measured": false
  },
  "validity": {
    "checks": {},
    "cluster_assignments_created": null,
    "artifact_inventory": [],
    "error": null
  }
}
```

`experiment_id` combinará fase, timestamp UTC e SLURM_JOB_ID. `class_weight:null` na versão final representa o **None efetivo do RF**, conferido pelo objeto, e não informação desconhecida. `bootstrap`, `n_bins`, `max_depth`, `n_streams` e demais defaults serão persistidos por target a partir de `get_params()`; não serão redefinidos pelo manifesto.

O SHA Git sozinho não basta quando há código transferido por rsync sem `.git` ou alterações pendentes. Antes de transferir, o snapshot de proveniência será gerado no workspace com o commit de origem, lista exata de alterações e hashes de módulos científicos, executor, config, Slurm, dependências declaradas e referência metodológica. No cluster esses hashes serão conferidos contra os arquivos recebidos. Sem Git nem proveniência verificável, o job falha; não inventa SHA e não atribui um arquivo alterado a um commit limpo. O manifesto de proveniência não inclui seu próprio hash de forma circular.

O resultado só é utilizável quando `status=APPROVED`, todos os checks obrigatórios passaram e os artefatos estão completos. `scientific_result=true` expressa a natureza planejada da execução; não aprova um estado RUNNING/INVALID. OOM, cancelamento ou TIMEOUT podem impedir o `finally`; um manifesto remanescente RUNNING deve ser tratado como incompleto, com estado final Slurm consultado externamente. Não aprovar parcial ou retomar silenciosamente.

## 10. Comprovação de GPU e de ausência de assignments

GPU será comprovada por evidências combinadas, sem afirmar que todas as operações são GPU:

- Módulos/versões reais, contexto CUDA funcional, propriedades do dispositivo A100 e memória, alocação Slurm e GPU/UUID quando disponível; uma GPU CUDA visível.
- Matrizes de todos os splits como `cupy.ndarray`, finitas, na GPU esperada e com tamanhos completos.
- Bundle com backend GPU, encoder nominal e objeto real cuML de redução quando effective >=1, dimensão conferida; solicitado/efetivo registrados.
- Objeto retornado de clustering e artefato salvo como instâncias cuML KMeans; RFs salvos como instâncias cuML RandomForestClassifier, dimensão e número de árvores conferidos.
- Guardas de backend antes dos estágios e sincronização CUDA nos limites; nenhuma opção `auto`, wrapper de aceleração sklearn ou fallback CPU para aprendizado.

O executor criará um output novo e exclusivo, validará `export_cluster_assignments=false` antes do clustering, conferirá a ausência imediatamente após e varrerá **somente os nomes** dos arquivos no output novo ao finalizar, inclusive em subdiretórios. Qualquer arquivo chamado `cluster_assignments.csv` causa INVALID. Não lerá seu conteúdo, não o apagará para ocultar uma violação e não consultará assignments antigos.

A evidência é composta pela guarda de configuração, ramo de exportação existente testado, verificação pós-clustering e inventário final. Não se propõe um monitor complexo de filesystem nem afirmar observação contínua que o código não fez.

## 11. Profiling, integridade e riscos conhecidos

O profiler atual registrará loading, frozen split validation, preprocessing+redução, clustering, cada target e total. Guardas/sincronizações serão externas às funções científicas; os tempos por target continuam contendo serialização, predição e métricas. Reporting suplementar terá seu tempo separado. Não haverá campo RF.fit exclusivo fabricado a partir desses números.

`/usr/bin/time -v` envolverá somente a execução principal Python e registrará wall-clock, user time, system time e Maximum resident set size em arquivo externo. Ele termina **depois** do processo Python; por isso o manifesto referencia seu caminho e não presume que o relatório final já exista enquanto está em execução. A coleta final lerá esse arquivo e os logs Slurm. A evidência de alocação também será preservada.

As leituras de VRAM já disponíveis no Profiler serão usadas como medições pontuais e máximo observado nesses pontos, **não pico contínuo de VRAM**. Não será criado outro sistema de profiling nesta fase. Usos momentâneos de memória, caches de alocação e atividades do driver limitam essa interpretação.

Riscos específicos que permanecem explícitos:

1. RAM real de DataFrames/cópias e estruturas dos RFs não foi medida em escala completa; 256 GiB é reserva proposta. Não alterar limpeza, features ou dtype para caber sem nova decisão.
2. Silhouette mantém amostra 20k, mas DB/CH e métricas externas são CPU no treino/splits completos. Clustering total inclui esses custos.
3. RF de 150 árvores com defaults atuais pode consumir memória/tempo significativo. Não limitar profundidade, reduzir árvores ou trocar balanceamento automaticamente.
4. Artefatos dos RFs e recarga sequencial para auditoria precisam de RAM/VRAM/storage; ausência do CSV de assignments não torna todos os artefatos pequenos.
5. A função de preprocessing apenas registra warning em uma falha de serialização; o executor científico precisa promover ausência/bundle inválido a falha, sem alterar seu cálculo.
6. Métricas binárias podem retornar null por exceção no código atual; não aceitar um baseline aparentemente completo sem investigar o motivo.
7. Vocabulário do treino completo pode diferir do smoke; guardar dimensão e redutor reais. PCA eventual não será rotulado como SVD nem ocultado.
8. Logs/documentos da 1E podem conter status anterior à execução real; a aprovação relatada pelo usuário não equivale à existência local de seus arquivos de accounting.
9. `group_stratified` mede generalização a combinações novas dos atributos de grupo, não garante dispositivos individualmente novos nem causalidade temporal. O protocolo não será trocado.
10. Os resultados constituem baseline da **variante GPU da extensão**, sem equivalência alegada a MiniBatchKMeans ou balanced_subsample do artigo. A omissão deliberada da 1G permanecerá registrada.

## 12. Validação local e pendências para submissão

### 12.1 Realizado nesta primeira entrega

| Verificação | Resultado |
|---|---|
| Suíte automatizada atual: `python -m pytest` | **206 passed, 1 warning, 33,95 s**. Warning de permissão do cache pytest no Windows; nenhum teste falhou. Fixtures pequenas, sem ler TON_IoT real. |
| Inspeção de configuração/contratos/métricas | Concluída nos arquivos atuais, preservados. |
| Loader oficial sobre YAML desta proposta | **PASS**. Bloco extraído para arquivo temporário local e carregado com `load_config`; parâmetros do split iguais aos da configuração 1D, hash/contagens/versões/flags/targets conferidos. Não acessou o caminho remoto. |
| `bash -n` sobre Slurm desta proposta | **PASS, exit code 0**, usando Git Bash no bloco extraído, sem executar seu conteúdo. |
| `git diff --check` e proveniência | **PASS**. HEAD `f8bf2c585a971beaf1d21faff10d6db6f0adf1d6`; nenhuma alteração em arquivos rastreados. Única adição pendente: este documento. |
| Manifesto JSON, links e formatação | **PASS**. JSON parseável, dez links locais existentes, cercas de código balanceadas e ausência de espaços finais no documento novo. |
| CUDA real e execução completa Fase 2 | Não executadas. Evidência GPU precedente: 1E aprovada pelo projeto. |
| MaxRSS 1E, storage/quota, output remoto, limites Slurm | Pendentes das consultas remotas da seção 7. |

O commit `f8bf2c5` consolidou os arquivos das fases anteriores durante esta sessão; a inspeção confirmou que ele adiciona as entregas 1D–1F, sem alterar os módulos científicos. Ao concluir a proposta, a única mudança desta entrega deve ser este documento. Não será feito commit automático.

### 12.2 Após aprovação da proposta, antes do job

Implementar os quatro arquivos de execução/config/teste, gerar proveniência e repetir a suíte completa. Testes novos deverão cobrir: flags científicos; RF=150 e três targets; proibição de subset/seleção/representantes; parâmetros frozen e `timestamp_unit=s`; falha por hash/contagem/população/schema/qualidade/cobertura/sobreposição divergentes; nenhuma chamada de geração de split; GPU obrigatório; matriz e redutor efetivos; I/O fallback permitido somente com identidade preservada; ausência do CSV; artefatos/manifesto completos e falha sem sobrescrita.

As guardas serão verificadas com fixtures pequenas e mocks explicitamente identificados; sucesso desses testes não será apresentado como teste CUDA real. Não executar MiniBatchKMeans ou comparações CPU/GPU como experimentos da Fase 2.

Validar novamente config pelo loader oficial e pela guarda específica, `bash -n`, `git diff --check`, proveniência e espaço/limites no cluster. Conferir o output inexistente imediatamente antes da execução; registrar a alocação efetiva. A aprovação desta proposta não será usada para submeter automaticamente o job sem a revisão final solicitada.

## 13. Comando de submissão proposto

Somente depois da implementação aprovada e das verificações remotas:

```bash
cd "$HOME/ids_generalization_pipeline"
sbatch scripts/run_phase_2_full_gpu_baseline.slm
```

O script foi implementado após aprovação da proposta. **Nenhum job foi submetido. A submissão aguarda aprovação final.**

## 14. Registro da implementação aprovada

Implementação entregue nos arquivos [executor](../scripts/run_phase_2_full_gpu_baseline.py),
[YAML final](../configs/phase_2_full_gpu_baseline.yaml), [Slurm final](../scripts/run_phase_2_full_gpu_baseline.slm)
e [testes específicos](../tests/test_phase_2_full_gpu_baseline.py). Os módulos `src/ids_pipeline` não foram alterados.

O executor reutiliza as funções oficiais de dados, schema, qualidade, split, features, preprocessing,
clustering e treinamento. Reutiliza também observação de I/O e verificações de objetos GPU da 1E;
não chama sua execução, seleção de subconjunto, marcação de integração ou configuração de 20 árvores.
Os testes da Fase 2 incluem 23 Parquets minúsculos, 230 linhas originais e 229 elegíveis, com
GPU explicitamente simulada. Esses testes exercitam o pipeline oficial; não constituem execução CUDA real.

As duas adaptações operacionais finais do Slurm são: mostrar metadados/tamanho e até vinte entradas
se o output já existir, sem removê-lo; e executar com o profiler existente caso `/usr/bin/time`
não esteja instalado, deixando sua indisponibilidade expressa no arquivo de accounting. CPU=32,
RAM=256 GiB, GPU=1 e limite=12 horas permanecem exatamente como aprovados. A guarda de A100/80GB
opera antes da leitura dos dados; o nome GRES/constraint local continua dependendo da consulta à partição.

### Proveniência e transferência

No workspace de origem, depois de qualquer alteração final e antes do rsync:

```bash
python scripts/run_phase_2_full_gpu_baseline.py --validate-config
python scripts/run_phase_2_full_gpu_baseline.py --write-provenance
```

Ambos os modos não carregam dataset nem iniciam CUDA/modelos. O segundo gera
`provenance/phase_2_source_provenance.json`, com SHA do commit de origem, estado pendente exato,
hashes dos arquivos e hash do snapshot. Esse arquivo e os arquivos que ele identifica precisam
acompanhar a transferência; o job o verifica antes da leitura dos dados e novamente ao finalizar.
Não editar config/código/documento de referência após gerar a proveniência sem regenerá-la.
Não substituir essa origem por um SHA inventado quando o cluster receber o projeto sem `.git`.

### Resultado das verificações desta implementação

- Testes específicos finais: **58 passed, 21,95 s**.
- Suíte completa final: **264 passed, 48,90 s**, com `python -m pytest -o addopts= -q -p no:cacheprovider`. Cache pytest desativado apenas para evitar a restrição de escrita local; nenhum teste omitido.
- YAML final: carregado pelo loader oficial; `--validate-config` aprovado, sem acesso remoto.
- Slurm final: `bash -n` aprovado; nenhum comando interno foi executado.
- `git diff --check`: aprovado. Como os arquivos da Fase 2 ainda não estão rastreados, foram conferidos também espaços finais/UTF-8 e o diff de adição de cada arquivo, sem staging automático.
- `export_cluster_assignments=false`; `representatives=None`; nenhuma chamada a gerador de split.
- Incrementos 1–3 preservados; Incrementos 4–5 não restaurados.
- Apenas arquivos novos da Fase 2 e este documento; nenhum commit ou staging automático.

### Checks remotos ainda necessários

Não houve sessão remota nem submissão durante a implementação. Partição, storage, quota e output
remoto permanecem **não verificados**; os comandos seguintes são somente de consulta e atendem à
verificação operacional solicitada, sem iniciar o baseline:

```bash
sinfo -p scientific -o "%P %a %l %c %m %G"
scontrol show partition scientific
sinfo -p scientific -N -o '%N|%c|%m|%G|%f|%l|%t'
df -h /storage/dados/es112688
if command -v quota >/dev/null 2>&1; then
    quota -s
else
    echo 'quota: comando indisponível; limite pessoal não verificado'
fi
output=/storage/dados/es112688/ids_generalization_outputs/phase_2_full_gpu_baseline_group_stratified
if [[ -e "$output" || -L "$output" ]]; then
    echo 'OUTPUT EXISTENTE: não submeter; não apagar'
    ls -ld -- "$output"
    du -sh -- "$output"
    find "$output" -maxdepth 1 -printf '%f\n' | sed -n '1,20p'
else
    echo 'OUTPUT AUSENTE no momento desta consulta'
fi
```

Se quota não estiver configurada, apenas registrar a resposta/indisponibilidade; isso não equivale a
afirmar quota ilimitada. Não reduzir RAM ou alterar os demais recursos automaticamente para agendar.
O job final ainda depende da aprovação do usuário e da avaliação desses resultados operacionais.
