# Fase 2.1 — Otimização e estabilidade do clustering da extensão GPU

Status: metodologia aprovada; implementação preparada para revisão, sem execução científica. Ver [registro de implementação](phase_2_1_implementation.md).

## 1. Decisão proposta e limites

Executar 96 ajustes de cuML KMeans numa amostra fixa de 1.000.000 de posições do TRAIN congelado; promover no máximo três pares (representação, k) para três seeds cada no TRAIN completo. Separar a escolha não supervisionada da interpretação externa. Não existe vencedor determinado antecipadamente.

Recomendação: reutilizar o preprocessing ajustado na Fase 2, congelar cada representação uma única vez, e variar exclusivamente representação, k e seed do KMeans. Não reajustar imputação, scaling ou vocabulário na amostra de desenvolvimento. A amostra limita o ajuste exploratório do KMeans; não redefine a população que ajustou a representação de referência. R10 e R15 serão ajustados, uma única vez, sobre o TRAIN completo pré-processado; R20 reutiliza o redutor do baseline. Essa escolha de escopo de ajuste faz parte desta proposta e precisa ser aprovada antes da implementação.

Nenhuma mudança de schema, quarentena, features, one-hot, log1p, imputação, scaling, splits ou população. Sem RF, novos seletores de compressão, orçamento B, temporal, HDBSCAN, PCA, UMAP ou features adicionais. A amostra de desenvolvimento é infraestrutura desta busca, não restauração dos Incrementos 4–5. Sem novo split ou alteração dos índices congelados.

## 2. Evidência disponível e baseline pai

Foram inspecionados os arquivos locais [experiment_manifest.json](../resultados_fase_2/experiment_manifest.json) e [metrics_clustering.json](../resultados_fase_2/metrics_clustering.json), além do código indicado na seção 3. Não foram lidas matrizes, modelos binários ou `cluster_assignments.csv`; os modelos pai permanecem dependências a verificar no cluster. Não foram utilizadas métricas supervisionadas para projetar a escolha de candidatos.

| Referência | Valor verificado |
|---|---|
| Experimento pai | `phase2_20260909T232019.249671Z_5452` |
| Status do manifesto pai | `APPROVED` |
| Commit do pai | `bd46536ebdbbe4aeac96ffafd29cc0ba07d433b7` |
| Snapshot do pai | `sha256:996af854b5e64a5ec91e7ebdce571bf7121ddd34e6d811db053057ad127d8b1c` |
| Population ID | `sha256:81cf4c6e2c8b9a887a6542fbd7d10a0d0b1af4c068e869cd6810658b82cf6c99` |
| Original / quarentena / elegível | 22.339.021 / 869 / 22.338.152 |
| Protocolo | `group_stratified/2.0.0` |
| Split hash | `sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a` |
| Train / val / test | 13.575.269 / 3.244.049 / 5.518.834 |
| Grupos | `src_ip, dst_ip, service, proto` |
| Feature policy | `behavioral_strict/2.0.0` |
| Representação | 8 quantitativas + 13 indicadores de conn_state = 21 colunas; TruncatedSVD GPU para 20 |
| KMeans | cuML, k=30, seed=42 |

O manifesto confirma Python 3.13.5, CUDA 12.9, CuPy 14.2.0, cuDF 26.08.01, cuML 26.08.00, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0 e SciPy 1.18.1. Registrar todas as versões novamente; não inferir as versões do cluster a partir da instalação local Windows. A100-SXM4-80GB e módulos Python/3.13.5-GCCcore-14.3.0 e CUDA/12.9.1 permanecem a referência operacional.

| Métrica histórica | Valor |
|---|---:|
| Silhouette train, amostra de 20.000 | 0.7550276517868042 |
| Davies–Bouldin train completo | 0.8919131012936842 |
| Calinski–Harabasz train completo | 8407748.695393423 |
| Inertia train completo | 6320832.0 |
| Clusters observados no train | 30 |
| AMI / V-measure type no test | 0.5058840885453761 / 0.5058903256717766 |
| AMI / V-measure label no test | 0.05291828745241553 / 0.05292068183859816 |
| Purity type / label no test | 0.6937961895574318 / 0.9724247911787164 |

Os números externos acima são referências históricas já conhecidas, não entradas do Pareto ou da promoção. Purity binária alta pode refletir a classe majoritária. AMI binária baixa não implica independência estatística demonstrada. O conhecimento prévio do test limita qualquer alegação de um novo teste completamente intocado; nesta fase, o procedimento de escolha ficará explicitamente bloqueado às métricas externas.

## 3. Diagnóstico do código e reutilização segura

| Arquivo / função | Reutilização proposta e limitação |
|---|---|
| [data_loading.py](../src/ids_pipeline/data_loading.py), `load_dataset` | Carregamento, normalização e quarentena oficiais; mesma ordenação. Fallback cuDF → pandas de I/O permitido apenas com identidade/schema/hash intactos e registro do motivo. |
| [split_protocols.py](../src/ids_pipeline/split_protocols.py), `load_frozen_splits`, `manifest`, `array_hash`, `digest_json` | Reutilizar verificação integral e hashes; não chamar geradores de split. Os índices são posições na população elegível, distintas do índice original preservado em df.index. |
| [run_phase_2_full_gpu_baseline.py](../scripts/run_phase_2_full_gpu_baseline.py), `load_population`, `validate_split`, `stage`, observação de I/O | Reutilizar funções de validação com a configuração congelada do pai, separada da configuração de busca. Não chamar `run`, `validate_config` com uma config de busca, treinamento ou seu verificador de artefatos obrigatórios de RF. |
| [feature_policy.py](../src/ids_pipeline/feature_policy.py), `FeaturePolicy` | Exigir exatamente a lista e versão congeladas, sem alvos ou proxies. |
| [preprocessing.py](../src/ids_pipeline/preprocessing.py), `PreprocessingBundle`, `apply_log1p` | Reutilizar bundle do pai: medianas, médias, desvios, encoder nominal e redutor. O bundle GPU guarda um dicionário, não um transformer completo com método público de replay. Será necessário um adaptador de transformação sem fit, restrito à política atual. |
| `preprocessing._gpu_numeric_arrays` | Referência exata: estatísticas após log1p, mediana, média e desvio pandas (ddof=1), conversão final float32. Não substituir por novo scaler ou ddof=0. |
| [nominal_encoding.py](../src/ids_pipeline/nominal_encoding.py), `ConnStateOneHot.transform_gpu` | Reutilizar o encoder serializado; treino define vocabulário, missing usa moda do treino e desconhecido produz todos zeros. |
| `preprocessing._make_gpu_reducer` | Não usar diretamente: existe fallback para PCA em exceção. A Fase 2.1 deve construir TruncatedSVD explicitamente e abortar se indisponível. Nenhuma edição dessa função congelada é proposta. |
| [clustering.py](../src/ids_pipeline/clustering.py), `_fit_gpu_kmeans` | Referência dos parâmetros; o wrapper exige train/val/test, prediz todos e calcula distâncias que a etapa A não precisa. Não reutilizar o wrapper integral para a busca. |
| `fit_predict_clustering`, `compute_clustering_metrics` | Não chamar na busca: calculam automaticamente métricas externas; a amostra de silhouette depende da seed do modelo; copiam train para CPU e calculam DB/CH completos. A fase precisa de orquestração isolada de fit e métricas internas. |
| `_sample_for_silhouette` | Somente referência para recuperar exatamente a amostra histórica. Na busca, carregar índices fixos, sem sorteio por execução. |
| `cluster_purity` e métricas sklearn já usadas | Reutilizar somente no processo externo, após decisões congeladas. CPU nas métricas é explícito; não há KMeans CPU. |
| [profiling.py](../src/ids_pipeline/profiling.py), `Profiler` | Reutilizar medição e amostras de memória; nomes únicos por execução para não sobrescrever registros. Sincronizar CUDA nas fronteiras temporais. |

As verificações congeladas de cobertura de classes podem continuar na auditoria do split; elas não balanceiam a amostra e seus relatórios não entram no seletor de finalistas. Após essa auditoria, os processos de busca recebem somente matrizes sem rótulos, posições e hashes. Testes devem proibir acesso de seleção/Pareto a `label`, `type` ou scores externos.

## 4. Congelamento dos índices: mecanismo exato

Primeiro validar pai, população e carregar o frozen split. Definir `T = splits['train']` como int64 na ordem armazenada, `N=len(T)=13575269`. Não ordenar/regravar T; não usar df.loc com posições elegíveis.

O pseudocódigo abaixo é especificação, não foi executado:

```python
development_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026, 21])))
train_offsets = np.sort(development_rng.choice(N, size=1_000_000, replace=False))
development_indices = T[train_offsets].astype('<i8')

metric_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026, 21, 1])))
metric_offsets = np.sort(metric_rng.choice(1_000_000, size=20_000, replace=False))
metric_sample_indices = development_indices[metric_offsets]

# Etapa B: mesma seleção de linhas usada pela silhouette histórica da Fase 2.
# Preservar inclusive a ordem do sorteio do código original, sem sort.
baseline_rng = np.random.Generator(np.random.PCG64(42))
full_metric_offsets = baseline_rng.choice(N, size=20_000, replace=False)
full_metric_sample_indices = T[full_metric_offsets].astype('<i8')
```

`development_indices.npy` contém posições elegíveis globais, não offsets dentro do treino. `metric_sample_indices.npy` também; `metric_offsets.npy` é o vetor auxiliar que indexa a matriz de desenvolvimento. Salvar ainda `development_train_offsets.npy`, `full_metric_train_offsets.npy`, `full_metric_sample_indices.npy` e IDs originais das amostras, obtidos de `df.index` nas posições globais. Nomes e domínio dos índices constam no manifesto.

A ordenação dos offsets de desenvolvimento preserva a ordem relativa do TRAIN, sem alterar a uniformidade do conjunto sorteado. Linhas com features idênticas continuam distintas por identidade; não deduplicar. Congelar os arquivos .npy sem pickle e sem reposição. Hash SHA-256 dos bytes canônicos little-endian int64, além do SHA-256 dos arquivos. Identidade de amostra combina population ID, split hash, hash ordenado de posições e de IDs originais. Registrar PCG64, entropia das SeedSequences, versões NumPy, tamanho e algoritmo de sorteio.

Guardas: cardinalidade exata, unicidade, todos os índices pertencem a T, nenhuma interseção com val/test, mapeamento offset/global/original consistente e amostra de métricas contida no desenvolvimento. Arquivos já existentes só podem ser reutilizados após verificar todos os hashes e parâmetros, nunca substituídos por novo sorteio silencioso. Usar escrita temporária e publicação atômica.

Há duas amostras de silhouette: A, subconjunto do desenvolvimento; B, subconjunto do TRAIN completo correspondente à referência histórica. Elas têm hashes distintos e não se apresentam como a mesma amostra. Não reconstruir a amostra B a partir da seed do KMeans de cada finalista.

## 5. Quatro representações fixas e comparáveis

As nove features permanecem, nesta ordem: duration, src_bytes, dst_bytes, conn_state, missed_bytes, src_pkts, src_ip_bytes, dst_pkts, dst_ip_bytes. O layout matricial segue o bundle pai: oito quantitativas transformadas seguidas dos treze indicadores nominais; 21 colunas float32. A posição de conn_state na lista semântica não redefine o layout codificado.

1. Abrir, somente para leitura, `artifacts/preprocessing_bundle.joblib` e metadados do output aprovado 5452. Registrar SHA-256 dos arquivos e verificar vínculo com manifesto, versões, ordem das features, estatísticas finitas e 13 categorias. O bundle local não foi aberto nesta entrega; se estiver ausente/incompatível, parar, sem refit automático para substituir a referência.
2. O adaptador novo aplica o schema existente, log1p existente, estatísticas já ajustadas e `ConnStateOneHot.transform_gpu`. Ele não aprende estatística alguma. Testar equivalência com a transformação oficial em fixtures, incluindo ddof, missing, desconhecidos e ordem. Inputs das métricas/seleção não incluem alvos.
3. Gerar R0 para TRAIN completo uma vez, em chunks de transformação, mantendo a mesma ordem e sem criar matrizes val/test na busca. Persistir matriz float32 de 21 colunas e metadados em cache fora do Git. A transformação numérica inicial segue pandas/CuPy híbrido existente; não afirmar pipeline integralmente GPU.
4. Reutilizar o TruncatedSVD de 20 componentes do pai para R20. Construir R10 e R15 exclusivamente com `cuml.decomposition.TruncatedSVD`, mesmos parâmetros efetivos do redutor pai salvo n_components, random_state fixo 42. Ajustar cada um uma vez sobre R0 do TRAIN completo, na GPU, sem qualquer label/type. Não usar as três seeds de clustering para refazer SVD. Registrar get_params, fit_scope e hash de cada redutor. Não truncar simplesmente R20 para simular novos ajustes.
5. Materializar as quatro matrizes de desenvolvimento e as amostras de métricas a partir dos índices congelados. Em B transformar com esses mesmos artefatos, sem reajustar representação. R0 sem redução significa `svd_components=0`, redutor nulo e 21 colunas; R10/R15/R20 exigem 10/15/20 componentes efetivos. Uma dimensão diferente da esperada é incompatibilidade a investigar, não clipping silencioso da grade.

| Representação | Solicitado | Efetivo | Ajuste do redutor | Dimensão de entrada do KMeans |
|---|---:|---:|---|---:|
| R0 | 0 | 0 | Nenhum | 21 |
| R10 | 10 | 10 | TRAIN completo, uma vez, seed 42 | 10 |
| R15 | 15 | 15 | TRAIN completo, uma vez, seed 42 | 15 |
| R20 | 20 na grade; 50 no pai | 20 | Artefato do pai, seed 42 | 20 |

O emprego do TRAIN completo no ajuste dos redutores não vaza val/test. Deve ser reportado: esta não é uma comparação de preprocessors ajustados em 1M contra preprocessors ajustados em 13,5M. Isola-se o efeito da representação com a população de ajuste do preprocessing fixa.

Cada matriz terá row-index hash, feature-order hash, dtype, shape e hash dos bytes de valores em chunks. Todos os k/seeds de uma representação leem o mesmo arquivo e o mesmo objeto GPU de entrada; testar que fit não altera a matriz, verificar hash antes e depois do grupo de execuções e persistir a referência por run. Matrizes/artefatos de entrada são imutáveis. Manter uma representação por vez na GPU.

## 6. Matriz completa da Etapa A

Ordem fixa: representação R0, R10, R15, R20; k crescente; seeds 42, 123, 2026. As 32 células abaixo contêm três runs cada, com IDs `A_R{d}_k{k}_s{seed}`, total 96. Não há novas amostras por célula.

| k | R0, seeds | R10, seeds | R15, seeds | R20, seeds |
|---:|---|---|---|---|
| 10 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 |
| 15 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 |
| 20 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 |
| 25 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 |
| 30 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 |
| 40 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 |
| 50 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 |
| 60 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 | 42 / 123 / 2026 |

`A_R20_k30_s42` é controle obrigatório da configuração no desenvolvimento. Não é a execução histórica: o KMeans recebe 1M em vez de 13.575.269 linhas. Não subtrair sua inertia/CH do histórico como se a diferença representasse melhoria em condições iguais.

## 7. Parâmetros efetivos de cuML KMeans

Fonte primária operacional: `clustering.parameters_effective` no manifesto do job 5452. Esses são parâmetros registrados, não apenas defaults presumidos da documentação.

| Parâmetro | Valor a manter |
|---|---|
| implementation | `cuml.cluster.kmeans.KMeans` |
| init | `scalable-k-means++` |
| max_iter | 100 |
| tol | 0.0001 |
| n_init | `auto` |
| oversampling_factor | 2.0 |
| max_samples_per_batch | 32768 |
| device_buffer_samples | 0 |
| init_size | 0 |
| output_type | null, com tipo efetivo de saída registrado |
| verbose | false |
| n_clusters | Único parâmetro de estrutura variável: grade de k |
| random_state | 42, 123 ou 2026, exclusivamente KMeans |

Construir diretamente o mesmo estimador GPU com esses parâmetros; comparar get_params antes e depois. Se o ambiente não aceitar a assinatura registrada, abortar, sem eliminar parâmetros ou recorrer a sklearn. Não encaminhar `cluster_n_init=5` ou `cluster_batch_size=16384`: eram campos legados ignorados pelo caminho GPU pai.

A documentação oficial explica que `n_init='auto'` com inicialização escalável usa uma inicialização interna; as três seeds continuam sendo três execuções externas. `max_samples_per_batch` controla lotes de distâncias, não converte o estimador em MiniBatchKMeans. A documentação é móvel; em caso de divergência, investigar o binário 26.08 e o manifesto, sem atualizar o método silenciosamente. [API oficial KMeans](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.KMeans/).

Não passar sample_weight, não criar centros com informação externa, não escolher melhor seed pela inertia. Coletar n_iter_ quando exposto; max_iter atingido é advertência de convergência, não motivo para aumentar iterações ou remover um run sem registro.

## 8. Métricas internas e concentração

### Escopos

| Quantidade | Etapa A | Etapa B |
|---|---|---|
| Silhouette, Euclidiana | Mesmos 20.000 de desenvolvimento | Mesmos 20.000 do TRAIN, amostra histórica recuperada |
| Davies–Bouldin e Calinski–Harabasz | Mesmos 20.000 de desenvolvimento | TRAIN completo |
| Inertia | Desenvolvimento completo, 1M | TRAIN completo |
| Distribuição/tamanhos | Desenvolvimento completo, 1M | TRAIN completo |
| AMI de estabilidade | Atribuições dos mesmos 1M | Predições dos modelos completos nos mesmos 1M |

Manter as fórmulas e bibliotecas sklearn de métricas do baseline onde aplicável, com versão registrada; só o escopo de A é diferente e será explícito nos nomes/metadados. Silhouette deve usar as linhas congeladas com sample_size=None na função, para não reamostrar. Usar cálculo de distâncias em chunks com working_memory fixo de 512 MiB, evitando a matriz densa 20k×20k; não aproximar nem alterar a distância. DB/CH completos em B podem usar a implementação atual CPU sobre uma representação por vez, dentro da reserva de RAM. Não criar novo algoritmo streaming de métricas nesta primeira implementação.

Antes das métricas geométricas, verificar 2 ≤ número observado de clusters < número de linhas no escopo. Métrica indefinida recebe null e motivo, nunca zero artificial. Cluster sem presença na amostra de 20k deve ser registrado separadamente de cluster vazio no conjunto de ajuste. Não estratificar a amostra para obter cobertura. Emissões numéricas inesperadas ou valores não finitos levam a diagnóstico, não a um ranking válido.

Para n linhas e k clusters solicitados, calcular `counts = bincount(ids, minlength=k)`, incluindo zeros, com soma n. Definir n_j e p_j=n_j/n para j=0,...,k−1; nenhum ID fora de [0,k).

| Medida | Definição |
|---|---|
| Clusters observados | q = soma de 1[n_j > 0] |
| Mínimo / máximo | min(n_j) / max(n_j), incluindo clusters vazios |
| Média / mediana | n/k / mediana dos k tamanhos |
| Desvio-padrão | sqrt(soma((n_j−n/k)^2)/k), ddof=0 |
| Fração máxima | max(p_j) |
| Top-5 / top-10 | Soma das maiores min(5,k) / min(10,k) frações |
| Clusters pequenos | Soma de 1[n_j < t], t=10,100,1000, incluindo zeros |
| Clusters vazios | Soma de 1[n_j=0], também informada à parte |
| Entropia | H = −soma(p_j ln p_j), termo zero definido como 0 |
| Entropia normalizada | H/ln(k), denominador k solicitado, não q |
| Número efetivo entrópico | exp(H); diferente do número observado q |
| Fração de uso observado / entrópico | q/k e exp(H)/k, para comparar diferentes k |

Sob k≥10 e n>0, H_norm está em [0,1] e exp(H) em [1,k]. Nenhuma penalização implica que a distribuição deva ser uniforme. Top-10 para k=10 é sempre 1: informativa, mas inadequada como objetivo para comparar k. Limiares absolutos de tamanho têm significados diferentes em 1M e 13,5M; reportar também frações, sem confundir crescimento de população com desaparecimento de degeneração.

Inertia = soma_i ||x_i−centro_atribuído||², conferida com `inertia_` no escopo de ajuste; registrar também inertia/n apenas como descritivo. Não comparar inertia entre dimensões ou tamanhos de treino como escala comum, nem usá-la para promoção.

Silhouette, DB e CH calculados em R0/R10/R15/R20 descrevem geometrias distintas. Uma redução pode eliminar variação e elevar scores sem melhorar utilidade. Pareto entre representações é uma análise operacional multiobjetivo, não prova de superioridade absoluta. CH muda também com n; separar tabelas A e B. A interpretação deverá exigir estabilidade e distribuição, além da separação.

## 9. Estabilidade entre seeds

Para cada um dos 32 pares (R,k), guardar labels de desenvolvimento em uint8 (k≤60), na ordem fixa dos 1M, com hash e referência aos mesmos IDs. Calcular AMI aritmética ajustada ao acaso para os pares (42,123), (42,2026), (123,2026), sem acessar alvos. São 96 comparações A; B terá até nove comparações, na mesma amostra de 1M, com modelos ajustados no TRAIN completo. AMI é invariante à permutação de IDs de cluster; concordância de IDs brutos não serve. [Definição oficial de AMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html).

Registrar valores pareados, mean/min/max, n_evaluated e sample hash. Só formar a agregação se todas as três seeds terminaram; não promover uma configuração com duas seeds bem-sucedidas omitindo a terceira. AMI=1 não elimina flags de degeneração nem comprova boa qualidade: conferir a validade técnica e reportar a concentração separadamente. Três seeds não sustentam intervalos de confiança robustos ou testes de significância fortes; os três pares não são independentes.

Persistir esses pequenos vetores de atribuição em arquivos binários próprios: aproximadamente 96 MB para todos os runs A, até 9 MB para B. Eles correspondem à amostra de 1M, não a associações completas de 22M. Labels completos de B podem existir em memória para métricas, mas não serão exportados em CSV. No relatório, a estabilidade B se chama explicitamente `stability_on_fixed_1M_train_sample`, não AMI integral de 13,5M.

## 10. Pareto e promoção de até três finalistas

Unidade de escolha = (R,k), nunca seed. Agregar silhouette, DB, CH e concentração por média, desvio amostral entre seeds (ddof=1), min/max. Guardar o resultado de cada seed e as advertências.

### Guardas de invalidade técnica

As guardas determinam se o resultado pode participar da comparação, independentemente de parecer cientificamente interessante:

| Guarda | Tratamento |
|---|---|
| Erro de execução | Registrar exceção, estágio e tentativa; resultado INVALID, sem aproveitar saída parcial como run completo. |
| NaN/Inf | Invalidar valores não finitos na matriz transformada, centros, saídas ou métricas calculadas; não substituir por zero ou ocultar no ranking. Missing reconhecido no dado bruto continua seguindo o schema/imputação congelados. |
| Número de linhas divergente | Invalidar diferenças de cardinalidade ou mapeamento entre entrada, índices, atribuições e soma dos tamanhos, além das guardas de população/split. |
| Backend incorreto | Invalidar execução matricial/redutor/KMeans fora do backend exigido; métricas CPU e I/O pandas explicitamente autorizados não são fallback de clustering. |
| Número de clusters observado incompatível | No conjunto de ajuste, manter o contrato proposto q=k, com centros de shape (k,d) e IDs inteiros em [0,k). Descumprimento torna o run incompatível com o k solicitado. Não exigir q=k nas amostras de métricas, estabilidade ou val/test: ausência nesses subconjuntos não prova incompatibilidade do ajuste. |
| Artefato incompleto/corrompido | Invalidar arquivo obrigatório ausente/truncado, checksum divergente, falha de desserialização ou conteúdo incompatível com o manifesto. Arquivos RUNNING ainda em escrita não são resultados publicados. |

Configurações entram no Pareto apenas com três runs tecnicamente válidos e todas as métricas internas exigidas disponíveis, mesmos hashes e parâmetros congelados. Métrica matematicamente indefinida recebe null com motivo: não é NaN disfarçado nem zero, mas impede completar a comparação exigida. Não mudar a amostra para corrigir isso silenciosamente. Resultados inválidos/incompletos permanecem no relatório, fora do ranking válido.

### Flags de degeneração

As flags são diagnósticos científicos, independentes de `validity`; não transformam automaticamente um run válido em INVALID nem o excluem do Pareto:

| Flag | Evidência a registrar |
|---|---|
| Maior cluster muito dominante | max_cluster_fraction por seed e pior caso entre seeds. |
| Muitos microclusters | Número e fração de clusters com <10, <100 e <1000 elementos; considerar n e k. |
| Baixa entropia normalizada | H/ln(k), conforme definição fixa da seção 8. |
| Número efetivo muito abaixo de k | exp(H), k e razão exp(H)/k. |
| Concentração alta nos top-5/top-10 | Frações acumuladas, com contexto de k; top-10=1 quando k=10 não ativa por si só um diagnóstico de degeneração. |

As medidas contínuas serão sempre reportadas. Os limiares foram definidos antes da execução científica e estão congelados em `clustering_degeneracy/1.0.0`: maior fração ≥0,90; pelo menos 50% dos clusters abaixo de cada limite de microclusters; H/ln(k)<0,50; exp(H)/k<0,25; top-5≥0,95 quando k>5; top-10≥0,99 quando k>10. Não são critérios de invalidade ou exclusão automática. Definições e justificativas estão no [registro de implementação](phase_2_1_implementation.md). Não há exigência de clusters artificialmente balanceados.

Para todos os tecnicamente elegíveis, inclusive aqueles com flags, usar o vetor de objetivos: maximizar média de silhouette, maximizar média de CH, minimizar média de DB, maximizar AMI pareada mínima e minimizar a pior fração máxima entre seeds. O vetor tem cinco eixos explícitos; não somar métricas ou normalizá-las com pesos. Publicar também Pareto apenas de geometria/estabilidade para tornar visível o efeito da concentração. Ser dominado segundo esses objetivos não equivale a ser inválido.

A domina B quando não é pior em nenhum eixo e é estritamente melhor em pelo menos um. Usar valores não arredondados, comparações diretas de números finitos; empates exatos não dominam. Não inventar epsilon para produzir um único vencedor. Exportar todos os dominados e não dominados, indicadores de elegibilidade e razões. Uma frente grande é resultado possível.

Promoção: se há até três elegíveis na frente principal, apresentar todos para revisão, com suas flags explícitas. Se há mais, apresentar a frente com distribuições, incerteza entre seeds e custo, e registrar escolha humana de até três alternativas com trade-offs distintos. Flags podem fundamentar a discussão e uma decisão documentada de não promover, sem alterar o status técnico ou apagar a configuração da frente. Não automatizar escolha arbitrária por soma de ranks ou escolher três seeds. Tempo só auxilia uma escolha de viabilidade entre soluções defendíveis, sem dominar a decisão científica. Se nenhum candidato é elegível, parar com diagnóstico, sem relaxar regras automaticamente.

Salvar `finalists.json` contendo pares (R,k), as três seeds obrigatórias, argumentos da escolha, hash das tabelas internas e `external_metrics_used=false`. Congelar SHA-256 da decisão e aprovação antes de B. O programa de promoção receberá esquema fechado de colunas internas; recusar colunas AMI(type), AMI(label), purity, V-measure externa e métricas RF.

R20/k=30 participa obrigatoriamente de A e da tabela histórica final. Não forçar sua promoção fora do Pareto, nem acrescentar uma quarta configuração completa. Se não for finalista, reaproveitar o modelo histórico seed42 como referência sem refit; estabilidade entre três seeds desse baseline no TRAIN completo ficará não medida. A questão sobre estabilidade de k30 poderá ser respondida em A, com essa limitação explícita.

## 11. Etapa B e bloqueio da validação externa

Executar apenas os pares aprovados, cada um com seeds 42,123,2026, até nove novos fits. Entrada = exatamente os 13.575.269 registros do TRAIN congelado, mesma ordem e mesmos artefatos de representação de A. O carregador de B recusa finalistas desconhecidos, hashes diferentes e qualquer comando que expanda a grade.

Medir DB/CH/inertia/distribuições no TRAIN completo; silhouette nos mesmos 20k fixados para B. Calcular estabilidade nos mesmos 1M do desenvolvimento. Agregar novamente por configuração e inspecionar Pareto e flags internas sem rótulos, mantendo a validade técnica separada. Não esperar que o ranking de A sobreviva automaticamente ao aumento de população.

Congelar uma segunda decisão, `final_selection.json`, com os modelos/configurações retidos, trade-offs, ausência eventual de vencedor absoluto, hashes de métricas B e `external_metrics_used=false`. A decisão é anterior ao processo de métricas externas. Não escolher a seed mais favorável: apresentar sempre as três; se for necessário apontar uma instância para uso posterior, usar seed42 pré-declarada, condicionada à configuração retida, sem gerar cluster_id definitivo nesta fase.

Só então o processo externo poderá carregar `label/type` e transformar val/test com os artefatos já fixados. Calcular AMI, V-measure e purity para label e type em train/val/test, em todas as execuções finalistas, inclusive alternativas não retidas para implantação. Não reabrir seleção em resposta ao alinhamento externo. Fazer predições GPU em chunks contíguos para limitar memória, sem refit; compará-las a predições integrais em fixtures e registrar tamanho de lote. Não chamar RF.

Reavaliar o modelo histórico R20/k30/seed42 na amostra B recuperada e obter sua distribuição, se os artefatos necessários estiverem disponíveis. Isso é predição/avaliação do modelo salvo, sem novo fit completo. Manter métricas históricas e recalculadas em colunas separadas, investigando diferenças e sem sobrescrever a Fase 2. Identificar custo dessas operações fora dos até nove novos ajustes.

## 12. Artefatos e proveniência

Raiz externa: `/storage/dados/es112688/ids_generalization_outputs/phase_2_1_clustering_optimization`.

```text
phase_2_1_clustering_optimization/
    phase_2_1_manifest.json
    source_provenance.json
    parent_reference.json
    development_indices.npy
    development_train_offsets.npy
    development_original_ids.npy
    metric_sample_indices.npy
    metric_offsets.npy
    metric_original_ids.npy
    full_metric_sample_indices.npy
    full_metric_train_offsets.npy
    full_metric_original_ids.npy
    sample_manifest.json
    representations/
        manifest.json
        reducers/                 # R10/R15 e referência imutável ao R20 pai
        cache/                    # R0 train e quatro matrizes A; fora do Git
    stage_a/
        runs.csv
        stability.csv
        cluster_distribution.csv
        configuration_summary.csv
        pareto_candidates.csv
        finalists.json
        models/
        sample_assignments/
        runs/                     # um JSON por run, publicação atômica
    stage_b/
        runs.csv
        stability.csv
        cluster_distribution.csv
        configuration_summary.csv
        final_selection.json
        external_metrics.csv
        final_comparison.csv
        models/
        sample_assignments/
        runs/
    configs/
    logs/
    report.md
```

Manifesto único canônico: `phase_2_1_manifest.json`, sem segunda cópia divergente chamada manifest.json na raiz. Campos: phase=2.1, scientific_result=true, experiment_type=clustering_optimization, parent_baseline=phase2, parent_experiment_id, commit/snapshot atuais e do pai, hashes dos artefatos pai, population/split IDs, seeds/tamanhos/hashes das três amostras, fit_scope de preprocessors/redutores, grade, parâmetros efetivos, `expected_runs_A=96`, `expected_runs_B=3*len(finalists)`, successful/failed/pending por etapa, finalists, selection_methodology, decisões e hashes, ambiente/hardware, dtype/ordem/hash das matrizes e tempos.

Estados separados: PLANNED, RUNNING_A, A_COMPLETE_AWAITING_SELECTION, RUNNING_B, B_COMPLETE_AWAITING_SELECTION, EXTERNAL_EVALUATION, COMPLETE; falhas documentadas como INVALID ou INCOMPLETE com motivo. scientific_result=true indica intenção científica, não aprovação de runs incompletos. Não contar tentativas como execuções científicas novas. Retomada explícita só reutiliza runs íntegros com mesmos hashes, preserva falhas em diretórios de tentativa e não sorteia novas seeds. Nunca escrever sobre output do pai.

Cada run terá campos separados `validity.status`, `validity.reasons`, `degeneracy_flags` (valor, limiar, estado, motivo) e `promotion_decision`. Uma configuração tecnicamente válida pode ter flags e não ser promovida; essas três situações não podem ser codificadas como a mesma falha.

Tabelas de resultados derivadas dos JSON por run, sem append concorrente. Matrizes/cache binário estimados em poucos GiB; atribuições da amostra em aproximadamente 0,1 GiB. Modelos KMeans pequenos; nada de RF serializado. Não criar, ler ou exportar `cluster_assignments.csv`.

## 13. YAML proposto e separação de configurações

Criar futuramente `configs/phase_2_1_clustering_optimization.yaml` e snapshots resolvidos por run. O loader próprio lerá esse contrato de fase e a configuração científica pai para validar dados/split. Não adicionar os campos abaixo a PipelineConfig global nem reativar `selection_budget` ou `selection_methods`.

```yaml
phase: '2.1'
scientific_result: true
experiment_type: clustering_optimization
compute_backend: gpu
strict_gpu: true
parent_baseline_dir: /storage/dados/es112688/ids_generalization_outputs/phase_2_full_gpu_baseline_group_stratified
parent_experiment_id: phase2_20260909T232019.249671Z_5452
data_path: /home/es112688/dados/datasets/TON_IoT_parquet
frozen_splits_dir: /storage/dados/es112688/ids_generalization_splits/group_stratified_v2
output_dir: /storage/dados/es112688/ids_generalization_outputs/phase_2_1_clustering_optimization
expected_population_id: sha256:81cf4c6e2c8b9a887a6542fbd7d10a0d0b1af4c068e869cd6810658b82cf6c99
expected_split_hash: sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a
expected_original_rows: 22339021
expected_quarantined_rows: 869
expected_eligible_rows: 22338152
expected_split_rows:
  train: 13575269
  val: 3244049
  test: 5518834
feature_policy_version: behavioral_strict/2.0.0
export_cluster_assignments: false
development_sample:
  size: 1000000
  seed_sequence:
  - 2026
  - 21
  bit_generator: PCG64
metric_sample_a:
  size: 20000
  seed_sequence:
  - 2026
  - 21
  - 1
  bit_generator: PCG64
metric_sample_b:
  size: 20000
  seed: 42
  source: parent_silhouette_algorithm
stability_scope: development_1M
preprocessing_source: frozen_parent_bundle
reducer_fit_scope: full_train
reducer_seed: 42
representations:
  R0: 0
  R10: 10
  R15: 15
  R20: 20
k_values:
- 10
- 15
- 20
- 25
- 30
- 40
- 50
- 60
clustering_seeds:
- 42
- 123
- 2026
kmeans:
  init: scalable-k-means++
  max_iter: 100
  tol: 0.0001
  n_init: auto
  oversampling_factor: 2.0
  max_samples_per_batch: 32768
  device_buffer_samples: 0
  init_size: 0
  output_type: null
  verbose: false
metrics_backend: sklearn_cpu
metrics_working_memory_mib: 512
external_prediction_batch_rows: 65536
degeneracy_flags:
  version: clustering_degeneracy/1.0.0
  invalidate_run: false
  automatically_exclude_from_pareto: false
  thresholds:
    max_cluster_fraction: 0.9
    microcluster_fractions:
      lt10: 0.5
      lt100: 0.5
      lt1000: 0.5
    min_normalized_entropy: 0.5
    min_effective_clusters_ratio: 0.25
    top5_cluster_fraction: 0.95
    top10_cluster_fraction: 0.99
promotion:
  max_finalists: 3
  require_all_seeds: true
  require_all_requested_clusters: true
  method: pareto_then_documented_approval
  external_metrics_allowed: false
source_provenance_path: provenance/phase_2_1_source_provenance.json
```

Schema, qualidade, feature order e demais parâmetros congelados serão verificados contra as versões oficiais e o pai, mesmo quando não repetidos no YAML. A seed do split continua vindo do split congelado; jamais é substituída pela seed do KMeans.

## 14. Scripts e testes a implementar após aprovação

| Arquivo proposto | Responsabilidade |
|---|---|
| `scripts/phase_2_1_common.py` | Contrato da fase, transformação sem fit, hashes, amostras, métricas internas/concentração, Pareto, timers e serialização; não expõe seleção por rótulo. |
| `scripts/prepare_phase_2_1_clustering.py` | Guardas, parent bundle, índices e caches/redutores imutáveis; nenhuma execução KMeans. |
| `scripts/run_phase_2_1_clustering.py` | CLI restrita A/B, por representação/finalista, somente cuML KMeans, sem métricas externas. |
| `scripts/report_phase_2_1_clustering.py` | Agregação, estabilidade/Pareto, validação e congelamento das decisões, relatório final. |
| `scripts/evaluate_phase_2_1_external.py` | Leitura dos alvos somente após decisões verificadas; predições e métricas externas dos modelos salvos. |
| `tests/test_phase_2_1_clustering.py` | Fixtures pequenas; guardas, amostras, métricas, Pareto e integração simulada. |

Não editar `src/ids_pipeline` nesta implementação proposta. Se reutilização segura exigir uma alteração global não prevista, apresentar separadamente o diff e a justificativa antes de fazê-la. A verificação de proveniência 2.1 deve ser própria; não regenerar o snapshot histórico do baseline para fazê-lo corresponder ao novo código.

Testes necessários: amostra uniforme determinística e sem reposição; independência de alvos e seeds de modelo; índices globais versus offsets; hashes/replay; R0 sem reducer; R10/15/20 efetivos e PCA proibido; replay de estatísticas sem aprendizado; invariância de matrices entre k/seeds; saída GPU; reprodução de métricas nas fixtures; soma e zeros de distribuição; limites de H/H_norm/exp(H); top-m com k menor/igual a m; AMI sob permutação de IDs; solução colapsada não elegível; Pareto com empate/trade-off/dominância e NaN; nenhuma escolha de seed; máximo três finalistas e nove fits B; ausência de alvos e de RF em A/B; bloqueio externo antes do hash de decisão; no-clobber, retomada e matriz incompleta.

Testes específicos da separação: um run com q=k e concentração extrema continua válido, recebe flags e permanece elegível ao Pareto; muitos microclusters ou baixa entropia não alteram validade; NaN/Inf, backend errado, cardinalidade divergente e artefatos corrompidos invalidam; q diferente de k no conjunto de ajuste descumpre o contrato, mas q<k somente na amostra de métricas não ativa essa guarda. Os limiares agora são obrigatórios e protegidos pelo hash científico da configuração. A solução colapsada não elegível mencionada acima refere-se à incompatibilidade técnica q=1 com k solicitado maior que um, não a uma exclusão pela fração máxima.

A API de métricas sklearn permanece CPU, com entradas apenas de representação/cluster IDs. A guarda GPU se aplica a transformação matricial, redutor e KMeans; pandas e CPU nas métricas não constituem fallback de clustering. Nenhum teste grande é necessário para implementar o contrato; smoke CUDA pequeno ficará sujeito à aprovação da implementação.

## 15. Jobs e recursos propostos

Usar partition scientific, QoS scientific-qos, Python/3.13.5-GCCcore-14.3.0, CUDA/12.9.1, `.venv`, IDS_COMPUTE_BACKEND=gpu. O cluster informou GRES `gpu:a100:8`; propor `--gres=gpu:a100:1`, conferindo disponibilidade na submissão. Guardar scontrol da alocação e verificar A100 de 80 GB no processo. Usar `/usr/bin/time -v`, sem novo profiler complexo.

| Job futuro | Organização | CPUs | RAM | GPU | Walltime solicitado |
|---|---|---:|---:|---|---|
| `run_phase_2_1_prepare.slm` | Uma preparação comum | 32 | 256 GiB | 1 A100 80 GB | 2 h |
| `run_phase_2_1_stage_a.slm` | Array 0–3, limite simultâneo 1; 24 fits por representação | 32 | 256 GiB | 1 A100 80 GB | 12 h por tarefa |
| `run_phase_2_1_stage_b.slm` | Array de F finalistas, limite simultâneo 1; 3 fits por finalista | 32 | 256 GiB | 1 A100 80 GB | 12 h por tarefa |
| `run_phase_2_1_external.slm` | Uma avaliação pós-decisão dos modelos, sem fit | 32 | 256 GiB | 1 A100 80 GB | 12 h |

Somente agregação/Pareto usa CSVs e vetores amostrais: executar sem GPU, em job CPU curto se necessário, sem ocupar nó de login para cargas grandes. Não executar quatro representações simultaneamente na mesma GPU. Os limites de walltime são reservas conservadoras, não duração esperada. A redução de 256 GiB exige análise posterior; o MaxRSS de um baseline não limita toda a grade.

Preparação publica READY após conferir hashes; A só aceita READY. Depois de A, parar para escolha dos finalistas, sem encadear B automaticamente. B também para antes da decisão interna final e validação externa. Um job com timeout preserva runs completos; retomada explícita não sobrescreve resultados. Dependências Slurm não substituem essas aprovações.

Não se fornece comando de submissão pronto para executar nesta primeira entrega: os scripts ainda são propostos. Antes da implementação/submissão, verificar novamente limites, quota, espaço e outputs; aproveitar os módulos e guardas da Fase 2 sem executar seu baseline ou RF.

## 16. Custo computacional: evidência e estimativa

O manifesto 5452 fornece evidência melhor que usar o tempo total como tempo de KMeans:

| Etapa histórica | Segundos | Escopo |
|---|---:|---|
| Carregamento | 198.17 | Dataset elegível completo |
| Validação do split | 63.94 | Frozen split e invariantes |
| Preprocessing + SVD | 46.05 | Ajuste/transformação; não separados |
| Clustering | 314.34 | Fit, predições, distâncias, métricas internas/externas e I/O |
| Supervisionado total | 1203.13 | Excluído da Fase 2.1 |
| Total interno | 1962.86 | Quase todo o processo |
| Tempo GNU time | 1970.36 | Processo completo; log fornecido pelo usuário |

Não existe medida de fit KMeans isolado no pai. Assim, 96×314,34s = 8,38 GPU-horas de parede alocada como referência de escala, não previsão exata nem limite superior. A elimina métricas externas e usa menos dados, mas repete 96 silhouettes de 20k e inclui k até 60; não dividir linearmente por 13,575. Similarmente, 9×314,34s = 47,15 minutos é apenas referência de custo para nove estágios históricos, não promessa para B.

Planejamento provisório: preparação em dezenas de minutos, A em algumas horas (janela operacional de 2–12 h agregadas, baixa confiança), B em aproximadamente 1–3 h agregadas, avaliação externa podendo acrescentar dezenas de minutos a horas. Reservas Slurm maiores absorvem incerteza; não são estimativa de consumo. Registrar previsão revisada após os primeiros três runs já pertencentes à matriz A, sem criar piloto extra, alterar amostras ou escolher parâmetros com base em desempenho externo.

Silhouette pode dominar custo CPU em A. Medir fit_seconds, predict_seconds (0 quando nenhuma chamada separada), metrics_seconds, io_seconds e total_seconds com perf_counter; sincronizar CUDA antes/depois de fit/predict. Usar `model.fit` para fit e `labels_` para atribuições de ajuste; chamadas separadas a predict ficam em outro timer. Registrar setup/representações e estabilidade por grupo fora dos tempos de um run individual. Instanciação, serialização e hashing não são RF.fit nem KMeans.fit. Não contar timers aninhados duas vezes.

Registrar GPU UUID/VRAM antes/depois, CPU/RAM alocadas e disponíveis, versions, shapes e n_iter_. MemGetInfo é uma amostra de uso do dispositivo, não pico contínuo por processo; inclusive pode incluir uso alheio se não houver isolamento. GNU time mede RAM de host. Não apresentar speedup final ou pipeline 100% GPU.

Cache R0 full train ocupa cerca de 1,06 GiB (13.575.269×21×4 bytes); as quatro matrizes A juntas aproximadamente 0,25 GiB (1M×66×4). Uma matriz B por vez evita multiplicar memória. IDs, redutores e labels amostrais são modestos frente à quota pessoal informada de 200 GiB, mas conferir consumo real antes da submissão e não apagar o baseline.

## 17. Comparação final e perguntas científicas

`final_comparison.csv` terá uma linha por execução B e a referência histórica seed42, com colunas: configuração, representação, componentes solicitados/efetivos, k, seed, fit_n, metric_scope, sample hashes, silhouette, DB, CH, inertia, mean/min stability AMI, max/top-5/top-10 fractions, normalized entropy, observed_clusters, entropy_effective_clusters, AMI/V-measure/purity type e label no test e tempos. Métricas externas completas train/val/test ficam em `external_metrics.csv`.

Para a linha histórica, estabilidade mean/min é null se não houver três fits completos dessa configuração; não preencher com estabilidade de A. Concentração histórica ausente dos JSON locais será obtida do modelo/artefato no cluster após o bloqueio de escolha; se indisponível, registrar ausência. A tabela A separada conserva todos os 96 runs, inclusive falhas e degenerações.

O relatório responderá, distinguindo evidência A e B: competitividade de k30; efeito de SVD versus R0; separação, estabilidade e concentração; alternativas de k; variabilidade entre seeds; estabilidade da silhouette histórica; clusters pequenos/dominantes; número observado e efetivo; alinhamento externo com type; e persistência de AMI binária baixa. Comparar configurações por agregados de três seeds, sem tratar diferentes IDs de cluster como categorias equivalentes.

Se o Pareto produzir trade-offs, a conclusão será um conjunto de alternativas, não um campeão pela silhouette. Não chamar o alinhamento externo de critério de escolha. O objetivo atual é clustering estável e não degenerado; não demonstrar aumento de F1 do IDS, compressão, fidelidade ao artigo ou necessidade de retreinar RF. Essas decisões pertencem a etapas futuras.

## 18. Riscos e critério de conclusão

Riscos conhecidos: estrutura rara pode faltar na amostra uniforme; a amostra interna não é holdout e não prova generalização; três seeds dão estabilidade descritiva limitada; classes/ataques externos já são conhecidos historicamente; índices posicionais confundidos com originais causariam erro silencioso; métricas de espaços distintos e CH de populações distintas não têm comparação causal simples; SVD21→20 não é garantidamente inútil; PCA fallback existente não pode ser acionado; replay GPU não possui interface pública completa; artefato pai/versionamento precisam ser verificáveis; não determinismo numérico GPU pode permanecer apesar de seed fixa; excesso de objetivos pode ampliar o Pareto; concentração natural não equivale a erro; custo de silhouette e memória deve ser observado; cache/proveniência e retomadas precisam impedir mistura de versões.

Concluir a Fase 2.1 somente quando:

1. População, schema/quarentena, features e split forem verificados e hashes congelados preservados.
2. Amostras e representações estiverem congeladas/reutilizáveis; 96 runs A tiverem registro íntegro, sem resultados omitidos. Falhas operacionais devem ser resolvidas com repetição da mesma condição ou impedir status COMPLETE; degeneração válida deve ser reportada, não escondida.
3. Pareto, estabilidade e seleção de até três pares estiverem documentados e aprovados sem métricas externas.
4. As três seeds de cada finalista tiverem resultados B completos, com escopos de métricas corretos e no máximo nove novos fits completos.
5. A decisão B estiver congelada antes da avaliação externa; relatório apresentar todos os alinhamentos sem reotimização retroativa.
6. Referência histórica e referência A estiverem identificadas corretamente, custos medidos, limitações e perguntas da seção 17 respondidas, mesmo que a resposta seja ausência de vencedor absoluto.
7. Nenhum RF, split novo, selector de compressão ou cluster_assignments.csv tiver sido produzido; nenhum baseline pai sobrescrito.

## 19. Registro desta primeira entrega

Criado somente este documento. Os arquivos modificados de Fase 2 e resultados locais existentes antes desta tarefa foram preservados. Foram lidos código e JSONs pequenos, consultadas referências oficiais e conferida a consistência documental. Não houve carregamento TON_IoT, geração real de amostras, fit, benchmark, teste CUDA, criação de YAML/scripts/jobs ou submissão Slurm. A implementação e os limiares/escopos propostos aguardam aprovação do usuário.

## 20. Atualização após aprovação para implementação

A seção 19 preserva o registro da primeira entrega. Scripts, configuração, testes e Slurm foram implementados após aprovação, sem alterar o código científico congelado. As flags foram definidas antes da execução, conforme o [registro completo](phase_2_1_implementation.md). As instruções da proposta em tempo futuro são detalhadas nesse registro. Nenhum job ou execução real foi iniciado.
