# Fase 1E — implementação da proposta aprovada

Status: **IMPLEMENTED / REAL JOB NOT EXECUTED**. `integration_only: true`;
`scientific_result: false`. O executor foi implementado e testado localmente;
a submissão do job real continua aguardando a última revisão do usuário.
As saídas futuras não constituirão resultados do estudo.

## 1. Diagnóstico do código atual

| Componente | Evidência e consequência operacional |
|---|---|
| `data_loading.load_dataset` | Resolve o diretório em ordem lexicográfica, lê cada fonte, aplica a quarentena, concatena os frames e normaliza o frame completo. Não é out-of-core. `sample_size` não será usado. |
| I/O GPU | cuDF lê uma partição por vez e converte para pandas; existe fallback de I/O para pandas com warning. Registrar `io_backend`, `io_gpu_fallback` e `io_fallback_reason`. O fallback isolado não reprova: continuam obrigatórias todas as verificações de identidade/população/schema/quarentena/ordem/hash. |
| `split_protocols.load_frozen_splits` | Carrega arrays existentes e recalcula manifesto/contratos contra a população/configuração. Não chama a busca de grupos. |
| `cli.run_experiment` | Não oferece um ponto para derivar subconjuntos depois da validação congelada. Não será usada a CLI `--stage all`; o executor isolado chamará as mesmas funções de cada etapa. |
| Feature policy | As nove features e a ordem já são explícitas em `behavioral_strict/2.0.0`. |
| Preprocessing GPU | É híbrido: pandas calcula imputação, estatísticas, normalização numérica e vocabulário; CuPy materializa as matrizes e o one-hot. Registrar `backend=gpu` não significa que todas as operações ocorreram na GPU. Isso é o caminho congelado, não um fallback. |
| Redução | `_make_gpu_reducer` tenta cuML TruncatedSVD; se a importação/construção falhar, já existe alternativa cuML PCA. O algoritmo real será registrado. Falha no fit não implica uma nova alternativa. |
| Serialização | O preprocessing GPU pode apenas emitir warning se o joblib falhar. O smoke exigirá arquivo válido e recarregável para obter APPROVED. |
| Clustering | `fit_predict_clustering` usa cuML KMeans quando o backend é GPU; preserva labels/distances com exportação desativada. As métricas atuais usam sklearn/CPU. |
| RF | `train_random_forest` suporta label, type e cluster_id, usa cuML no ramo GPU e salva um bundle por alvo. Codificação dos alvos e métricas são CPU. |
| Métricas e profiling | Manter as definições atuais. O campo legado `pr_auc` é Average Precision. Os tempos são de etapas amplas, não RF.fit exclusivo, nem kernels CUDA isolados. Preprocessing e redução são medidos juntos. |

Não haverá alteração dos arquivos acima, dos critérios de split, de KMeans/RF,
das features, do schema ou da quarentena. Não serão chamados seletores de
representantes. Os Incrementos 4–5 permanecem fora do escopo.

## 2. Sequência e identidade dos 100 mil registros

1. Validar configuração, artefatos congelados e disponibilidade da GPU. Exigir
   exatamente um dispositivo CUDA visível, modelo A100 de 80 GB e backend gpu.
   Apenas consultar o ambiente; não repetir o smoke CUDA das fases anteriores.
2. Ler `reuse_split.yaml`, aceitar somente seus campos de configuração de split
   e conferir cada um contra o YAML abaixo. Divergência causa erro. O loader de
   YAML atual não implementa herança/merge; essa conferência será do executor.
3. Confirmar que o diretório completo contém exatamente as 23 fontes Parquet
   referenciadas na Fase 1D. `_conversion_manifest.json` não é uma partição.
4. Chamar **`load_dataset(data_path, compute_backend='gpu', ...)`**, sem
   `sample_size`, usando os caminhos de relatório de schema/qualidade da execução.
   Conferir 22.339.021 originais, 869 quarentenados, 22.338.152 elegíveis,
   versões congeladas e o ID de população do manifesto da Fase 1D.
5. Chamar diretamente **`load_frozen_splits`** com a população completa e a
   configuração conferida. Nenhuma chamada a `create_splits`, `group_split` ou
   `save_splits` nesta fase. Exigir protocolo, contagens e hash aprovados.
6. Recalcular o manifesto com a função existente `manifest(df, frozen, config)`
   para registrar o hash observado. Exigir igualdade com o manifesto salvo e com
   `sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a`.
   Essa segunda verificação tem custo, mas mantém a implementação congelada.
7. Somente depois selecionar dentro de cada array congelado, na ordem fixa
   train, val, test. Para o split de ordinal `i` (0, 1, 2), usar
   `Generator(PCG64(SeedSequence([42, i])))`; sortear offsets com
   `choice(len(frozen[name]), size=n, replace=False)` e ordenar as posições
   elegíveis resultantes. Tamanhos: 60.000 / 15.000 / 25.000. Nunca usar label/type
   na escolha. Registrar versões NumPy e o gerador, além dos índices efetivos.
8. Conferir tamanho, unicidade e pertencimento ao split de origem. Concatenar
   **apenas os vetores de índices** em ordem train/val/test e obter
   `smoke_df = df.iloc[positions].copy()`. Preservar o índice original do frame.
   As posições locais serão 0:60000, 60000:75000 e 75000:100000. Não concatenar
   manualmente partições do dataset, nem criar/rebatizar um protocolo de split.
9. Salvar para cada split um `.npz` com posição na população elegível, posição
   original global e posição local do smoke; registrar hashes vinculados ao
   hash do split pai. São 100 mil linhas de identidade, cerca de 2,3 MiB antes
   de compressão. Verificar as posições originais com o mapa da Fase 1D aberto
   via mmap. Não copiar nem regravar os milhões de índices congelados.
10. Registrar e imprimir, antes de ajustar qualquer modelo, contagens de label/type
    nos três subconjuntos, incluindo zeros e classes ausentes comparadas à população
    pai. Não completar classes raras, reamostrar ou buscar seed mais favorável.
11. Liberar a população completa e os arrays completos; manter apenas o frame de
    100 mil linhas, mapeamentos pequenos e resumos. Aplicar as funções oficiais de
    feature policy/proxies, preprocessing, clustering e RF aos índices locais.
    Não fornecer representantes; os três RF recebem o mesmo X_train e as mesmas
    60 mil identidades. cluster_id vem das labels produzidas pelo KMeans desta execução.

O hash aprovado da Fase 1D pertence ao split pai, não ao subconjunto. Ambos os
níveis de identidade serão separados no manifesto e nos nomes dos artefatos.

## 3. Arquivos implementados após aprovação

- `scripts/run_phase_1e_real_gpu_smoke.py`: orquestração, subconjunto, guardas e manifesto.
- `configs/phase_1e_real_gpu_smoke.yaml`: configuração exclusiva abaixo.
- `scripts/run_phase_1e_real_gpu_smoke.slm`: job separado abaixo.
- `tests/test_phase_1e_real_gpu_smoke.py`: testes pequenos das guardas e do mapeamento.
- Atualização deste documento com os resultados das verificações da implementação.

Nenhuma API científica será alterada. O baseline definitivo manterá 150 árvores.
Os quatro arquivos acima foram criados. Nenhum job real foi submetido.

## 4. YAML completo proposto

```yaml
dataset_name: ton_iot_network_full
data_path: /home/es112688/dados/datasets/TON_IoT_parquet
output_dir: /storage/dados/es112688/ids_generalization_outputs/phase_1e_real_gpu_smoke
integration_only: true
scientific_result: false

compute_backend: gpu
strict_gpu: true
feature_policy: behavioral_strict
expected_feature_policy_version: behavioral_strict/2.0.0
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

expected_original_rows: 22339021
expected_quarantined_rows: 869
expected_eligible_rows: 22338152
expected_full_split_rows: {train: 13575269, val: 3244049, test: 5518834}
subset_seed: 42
smoke_rows: {train: 60000, val: 15000, test: 25000}

svd_components: 50
onehot_min_frequency: 10
gpu_max_categories_per_col: 64
log1p_numeric: true
# Sem override de log1p: usar a lista explícita do schema congelado.
selected_k: 30
cluster_batch_size: 16384
cluster_n_init: 5
cluster_max_iter: 100
silhouette_sample_size: 20000
export_cluster_assignments: false

representative_strategy: full
use_representatives_for_supervised: false
targets: [label, type, cluster_id]
random_forest_estimators: 20
random_state: 42
n_jobs: 32
log_level: INFO
```

`data_path`, `output_dir`, `strict_gpu`, referências esperadas e campos do smoke
serão interpretados apenas pelo executor isolado, via `PipelineConfig.extra`.
Eles não acrescentam orçamento de seleção ou novos seletores à API científica.

## 5. Slurm completo proposto

```bash
#!/bin/bash
#SBATCH --job-name=ids_1e_smoke
#SBATCH --partition=scientific
#SBATCH --qos=scientific-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=phase_1e_integration_only_%j.out
#SBATCH --error=phase_1e_integration_only_%j.err

set -euo pipefail
echo 'integration_only=true scientific_result=false phase=1E'
module --force purge
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.9.1
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source .venv/bin/activate
export IDS_COMPUTE_BACKEND=gpu
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"

python -B scripts/run_phase_1e_real_gpu_smoke.py \
  --config configs/phase_1e_real_gpu_smoke.yaml
```

A GPU será atribuída pelo Slurm, sem sobrescrever `CUDA_VISIBLE_DEVICES`.
Como o pedido `gpu:1` não garante o modelo por si só, o executor rejeitará um
dispositivo diferente da A100 de 80 GB antes do carregamento. Não será inventado
um nome de recurso A100 específico do cluster.

Comando para a última revisão, ainda não executado:
`sbatch scripts/run_phase_1e_real_gpu_smoke.slm`.

## 6. Recursos e memória

Proposta: **32 CPUs, 256 GiB RAM, uma A100 de 80 GB e 4 horas**.

- O schema tem 47 colunas registradas: 32 strings, 12 float64 e três Int64.
  Cada vetor de oito bytes com 22.338.152 posições ocupa aproximadamente 0,166 GiB.
  Só os valores/referências básicos de 47 colunas já representam aproximadamente
  7,8 GiB, antes de textos, máscaras, índices e temporários.
- Como estimativa de capacidade, se cada célula textual custar 40–100 bytes
  incluindo representação, as 32 colunas textuais corresponderão a 27–67 GiB,
  além de aproximadamente 2,5 GiB numéricos. Frames por arquivo, concatenação,
  normalização e temporários podem elevar bastante o pico. Uma faixa operacional
  aproximada de 100–230 GiB motiva a reserva de 256 GiB.
- Essa faixa NÃO é medição ou limite demonstrado. Strings compartilhadas/Arrow,
  comprimentos reais e versão pandas alteram substancialmente o consumo. O JSON
  preservado informa apenas cerca de 0,217 GiB de Parquet comprimido; esse tamanho
  não estima RAM descomprimida. Não foi lido nem amostrado o dataset para esta estimativa.
- A GPU recebe uma partição por vez no loader e apenas 100 mil linhas nas etapas
  de modelos. O frame completo normalizado permanece na RAM do host até verificar
  o split e materializar o smoke. Não ficará todo na VRAM.
- Após o descarte do frame completo, a matriz é pequena: se houver C estados no
  vocabulário, são 8+C dimensões antes da redução, com 100.000 linhas float32.
  A redução efetiva será `min(50, 60000-1, 8+C-1)` pelo código atual.
- As 32 CPUs preservam `n_jobs: 32` e o limite atual de 16 streams do RF cuML;
  atendem I/O, normalização e métricas CPU. O split só é validado, não regenerado.
- Quatro horas são uma reserva inicial para leitura/normalização integral,
  validações, métricas CPU (incluindo Silhouette em até 20 mil linhas),
  serialização e RF de 20 árvores. Não são tempo previsto medido.

Antes da submissão, confirmar que a partição permite 256G junto de uma GPU.
OOM/timeout não autorizam reduzir população antes de validar o split ou mudar
metodologia; o resultado será incompleto e demandará revisão operacional.

## 7. Evidência dos backends e condição APPROVED

- Exigir config e resolução efetiva GPU; guardar versões de Python, NumPy,
  pandas, CuPy, cuDF, cuML, CUDA Toolkit (`nvcc --version`, se disponível),
  runtime/driver CUDA, versão NVIDIA do driver se disponível, GPU selecionada,
  `CUDA_VISIBLE_DEVICES` e `SLURM_JOB_ID`. Valores indisponíveis serão null, nunca
  preenchidos copiando o ambiente esperado. Não executar `cuml.accel`.
- Capturar o warning de fallback do loader e registrar backend cuDF, pandas ou
  misto, ocorrência e motivo. O fallback de I/O não impede APPROVED se as
  verificações de dados e split passarem; ele não autoriza fallback dos modelos.
- Preprocessing: exigir bundle GPU, oito quantitativas do schema, somente
  conn_state categórica, metadata one_hot/fit_split=train/unknown=all_zero,
  ausência de coordenadas ordinais e matrizes retornadas como arrays CuPy no
  dispositivo selecionado. Documentar explicitamente as operações pandas.
- Registrar vocabulário, ordem, desconhecidos e dimensões por split. A dimensão
  anterior à redução virá das oito quantitativas mais as colunas one-hot reais;
  a posterior virá de `X[split].shape[1]` e será estável nos três conjuntos.
- Redução: preservar `effective=min(requested,max_components)`, com o mesmo
  `max_components=max(0,min(n_train-1,encoded_width-1))`. Quando effective >= 1,
  exigir objeto efetivo `cuml.decomposition.TruncatedSVD` ou a alternativa cuML PCA
  já prevista, dimensão efetiva correspondente e saídas CuPy. Quando effective
  for zero, registrar redutor ausente e backend `not_applicable`. Não exigir
  exatamente 50 dimensões nem aceitar sklearn/omissão quando a redução se aplica.
- KMeans: exigir instância cuML KMeans com k=30, métricas indicando GPU, labels
  e distances com tamanho correto por split, valores finitos e modelo salvo.
- RF: exigir payload GPU, três alvos, contagem de treino 60 mil para cada alvo,
  sem representantes, métricas de val/test presentes e artefatos cuML de 20
  árvores. Recarregar os bundles gerados para confirmar classe/módulo reais e
  possibilidade de desserialização, incluindo o bundle de preprocessing.
- Verificar que a dimensão fornecida ao RF corresponde a X após a redução.
  Testes com spies verificarão que X e IDs não são trocados entre alvos.
- Sincronizar CUDA nas verificações finais para revelar falhas assíncronas.
  Isso não torna retroativamente o profiling existente uma medição de kernels.
- Não confundir métricas/codificação CPU previstas com fallback de modelos.
- Exigir inexistência de `cluster_assignments.csv` no diretório novo, sem ler
  arquivos antigos. Só publicar status APPROVED depois de concluir todas as
  etapas, conferir artefatos e salvar o manifesto.

Em exceções, salvar status INVALID e etapa/erro, quando o processo ainda puder
escrever. Inicialmente o status será RUNNING; término abrupto/OOM sem atualização
final nunca equivale a APPROVED. O executor retornará código não zero nas falhas.

## 8. Manifesto, métricas e artefatos

`integration_manifest.json` conterá os seguintes blocos:

| Bloco | Campos |
|---|---|
| Estado | status, phase=1E, integration_only=true, scientific_result=false, erro/etapa se houver |
| Dados | dataset_name, data_path, original_rows, quarantined_rows, eligible_rows, schema_version, policy_version, data_quality_population_id |
| Split pai | split_protocol_version, frozen_splits_dir, parâmetros efetivos, expected_split_hash, observed_split_hash, split_hash_match, full_train_rows, full_val_rows, full_test_rows |
| Smoke | nome integration smoke subset, subset_seed, gerador e ordem de sorteio, smoke_train_rows, smoke_val_rows, smoke_test_rows, smoke_total_rows, hashes/mapeamentos, distribuições label/type com ausências por split |
| Features | feature_policy, feature_policy_version, selected_features ordenadas, papéis semânticos, transformações, encoding/vocabulário conn_state, dimensões antes/depois da redução |
| Redução | svd_components_requested, svd_components_effective, algoritmo/classe/módulo reais, backend e shapes |
| Backends | backend_requested, preprocessing_backend, preprocessing_execution=hybrid_pandas_cupy, reduction_backend, clustering_backend, supervised_backend, evidências por objeto e resultado das guardas |
| Ambiente | versões observadas de Python/NumPy/pandas/CuPy/cuDF/cuML, toolkit/runtime/driver, GPU, CUDA_VISIBLE_DEVICES, SLURM_JOB_ID |
| KMeans | algoritmo, implementação, k=30, random_state=42 |
| RF | implementação por alvo, number_of_estimators=20, random_state=42, targets=[label,type,cluster_id], dimensões e linhas recebidas |
| Tempo | loading, frozen_split_validation, preprocessing_and_reduction, clustering, supervised_label, supervised_type, supervised_cluster_id, total; referência ao profiling |
| Proveniência | hashes dos arquivos científicos utilizados e config efetiva; inventário dos artefatos pequenos com marcadores integration-only |

`preprocessing_seconds` e `reduction_seconds` individuais ficarão null com motivo
`not_measured_separately`; o intervalo conjunto existente ficará disponível.
Os tempos supervisionados incluem fit, predição, métricas e I/O; não serão
apresentados como RF.fit exclusivo. A RAM do profiler é observada nas fronteiras
das etapas, não um pico contínuo. Não haverá interpretação de speedup.

Diretório exclusivo, novo, fora do Git:
`/storage/dados/es112688/ids_generalization_outputs/phase_1e_real_gpu_smoke`.
Se já existir, abortar; não misturar tentativas nem sobrescrever artefatos antigos.

Manter JSON de métricas normais, relatórios/confusões existentes, preprocessing
bundle, KMeans e três RF; índices apenas do smoke. Não duplicar os arrays do split
pai nem salvar matrizes de features desnecessariamente.

Todos os JSON produzidos pelo executor/etapas receberão os dois marcadores
integration_only/scientific_result, sem alterar valores de métricas. Relatórios
textuais e logs terão cabeçalho equivalente. CSVs e binários terão metadados
companheiros com esses marcadores e vínculo ao manifesto, preservando seus formatos.
Um aviso no diretório e o inventário explicitarão que TODOS os artefatos são de
integração. Não publicar tabelas de comparação científica.

Métricas opcionais null serão registradas como indisponíveis, não como zero;
o smoke não modificará seus cálculos. As matrizes/relatórios são os já gerados
pelo pipeline (relatórios/confusões em teste; métricas agregadas em val/test).

## 9. Testes e limitações antes do job real

Cobertura dos testes específicos do executor:

1. Loader oficial recebe o diretório completo e nenhum sample_size.
2. Validação congelada ocorre antes do sorteio; hash divergente impede modelos.
3. Nenhuma chamada de geração de splits; teste falha se a busca for invocada.
4. Subconjuntos determinísticos, sem reposição, pertencentes aos pais e com IDs
   originais/local/eligible consistentes, mesmo com lacunas de quarentena.
5. Mudar label/type não muda sorteio; classes ausentes são reportadas sem ajuste.
6. Policy e redução preservadas; RF recebe a mesma matriz pós-redução nos três alvos.
7. CPU disfarçada de GPU, redução aplicável omitida, fallback de modelos e
   serialização ausente impedem APPROVED. Fallback de I/O é aceito com identidade
   preservada. Usar doubles nos testes locais, sem alegar validação CUDA real.
8. Exportação desativada, marcadores em artefatos, manifesto de falha e recusa de
   diretório existente. A guarda de aprovação exige os três alvos completos.

A revisão atual não atesta memória real, compatibilidade da serialização com cuML
26.08 nem execução GPU da redução/RF: esses são objetivos do futuro job autorizado.
A GPU de 80 GB não substitui a necessidade de RAM do host. Nenhuma proporção será
alterada para garantir que uma classe rara apareça no smoke.

A implementação foi autorizada e concluída com as duas alterações aprovadas.
Não submeter Slurm antes da última revisão solicitada pelo usuário.

## 10. Verificações realizadas nesta preparação

```text
python -B -m pytest -p no:cacheprovider \
  tests/test_data_quality_policy.py tests/test_dataset_schema.py \
  tests/test_split_protocols_v2.py tests/test_behavioral_strict_v2.py \
  tests/test_cluster_assignment_export.py tests/test_supervised_multitarget.py \
  tests/test_metrics_binary.py tests/test_no_leakage.py

154 passed in 16.31s
```

São testes existentes com fixtures pequenas; não validam a futura orquestração
específica da Fase 1E nem constituem novo smoke CUDA real. Também passaram:

- YAML desta proposta carregado pelo `load_config` oficial e parâmetros conferidos.
- `bash -n` aplicado a uma cópia temporária do Slurm proposto; sem executar o script.
- `git diff --check` e conferência de espaços finais do documento novo.

A única adição desta preparação é este documento. Os arquivos já pendentes da
Fase 1D foram preservados. Nenhum arquivo de implementação científica foi alterado.

## 11. Validação da implementação (posterior à aprovação)

Executado o mesmo comando da seção 10 acrescido de
`tests/test_phase_1e_real_gpu_smoke.py`: **177 passed in 52.44s**.
São 23 testes novos e 154 testes de regressão. Foram usados 23 Parquet de dez
linhas cada, normalização/quarentena e carregamento congelado reais nas fixtures;
o caminho GPU utiliza doubles declarados, não CUDA local.

O teste de orquestração exercita os três alvos usando as mesmas matrizes e as
sequências corretas de alvos; confirma sucesso tanto sem fallback quanto com I/O
inteiramente em pandas. Há teste adicional de I/O misto. As fixtures confirmam
requested=50/effective=9, a alternativa GPU PCA e a regra de effective=0.

Também há rejeição de fontes/ordem/hash/schema/quarentena/mapeamento divergentes,
matrizes CPU, redutor CPU ou omitido, dimensão errada, bundle ausente, KMeans/RF
CPU, alvo ausente, dimensão de RF incompatível e exportação indevida de assignments.
