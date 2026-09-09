# Fase 1F — equivalência metodológica do backend GPU

Data: 09/09/2026. Status: **ANÁLISE CONCLUÍDA; DECISÃO E IMPLEMENTAÇÃO AGUARDAM APROVAÇÃO**.

Esta entrega modifica somente este documento. Não executa treinamento, KMeans, leitura da população TON_IoT ou jobs Slurm. Não lê `cluster_assignments.csv`. Os Incrementos 1–3 e os artefatos congelados permanecem intactos; os Incrementos 4–5 não são restaurados.

## 1. Parecer e recomendação principal

**O backend GPU atual é uma variante metodológica do trabalho, não uma reprodução do baseline original acelerada por GPU.** A aprovação da Fase 1E demonstra integração operacional, mas não resolve essa diferença.

Há três constatações centrais:

1. O cuML 26.08.00 disponibiliza `KMeans`, mas não uma implementação pública de `MiniBatchKMeans`. Blocos de cálculo de distâncias e streaming de dados do KMeans não reproduzem as atualizações estocásticas do MiniBatchKMeans.
2. O RF cuML 26.08.00 **aceita `class_weight` e `fit(sample_weight=...)`**, porém rejeita `balanced_subsample`. O projeto não passa nenhum desses pesos ao RF GPU.
3. Há uma diferença adicional decisiva: no código C++ dessa versão, com `bootstrap=True`, pesos de amostra determinam as **probabilidades de sorteio do bootstrap**. Portanto, ativar `class_weight='balanced'` não restaura o balanceamento por árvore do sklearn. É uma mudança na distribuição dos exemplos vistos por cada árvore. A evidência está em `RowSampler.sample()` e `tree_sample_weight()`, e não somente na assinatura Python. [Implementação RF cuML 26.08](https://github.com/NVIDIA/cuml/blob/v26.08.00/python/cuml/cuml/ensemble/randomforestclassifier.py), [amostragem C++](https://github.com/NVIDIA/cuml/blob/v26.08.00/cpp/src/randomforest/randomforest.cuh).

**Recomendação:** para manter a direção de execução GPU, tratar o caminho atual explicitamente como variante GPU com KMeans completo e RF sem ponderação, preservando-o como controle. Antes de autorizar o baseline completo, realizar uma comparação pequena e controlada com os estimadores CPU originais, usando o protocolo congelado atual. Não acrescentar `balanced` ou pesos globais como suposta correção equivalente.

Se preservar `balanced_subsample` for requisito obrigatório do baseline principal, a alternativa pronta e defensável é **RF sklearn em CPU**, possivelmente em um fluxo híbrido. Se também for obrigatório preservar MiniBatchKMeans, esse estágio deve permanecer sklearn/CPU. A exigência simultânea de GPU em ambos os estimadores e equivalência estrita não é satisfeita pelas APIs investigadas. Um eventual RF GPU com bootstrap ponderado exige uma decisão metodológica própria; não é a recomendação de implementação desta etapa.

Esta recomendação não autoriza rodar o baseline completo nem altera automaticamente o RF. A escolha entre fidelidade dos estimadores e variante GPU precisa ser aprovada após a leitura das alternativas abaixo.

## 2. Evidências, versões e limites da verificação

Foram separados três referenciais:

| Referencial | Evidência examinada | Limite |
|---|---|---|
| Artigo e implementação original | [PDF local, especialmente pp. 4–6](../Fontes_projeto_IDS/artigo/Artigo_SBCUP_2026.pdf) e notebook `COLAB_02_PIPELINE_ARTIGO.ipynb` | O artigo descreve o método; o notebook explicita parâmetros e a distinção entre matrizes de clustering e RF. |
| Projeto atual | HEAD `02996e412e07f1205335b4e573bcf854eb1e2a8e`, arquivos científicos locais e arquivos das Fases 1D/1E presentes no workspace | O HEAD sozinho não identifica os arquivos ainda não versionados das Fases 1D/1E. Eles não foram alterados nesta análise. |
| RAPIDS/cuML | Documentação **26.08**, tag **v26.08.00**, commit `265b9da6a0e75dbef071a3168398b993a5ff6f0e` | Inspeção de código-fonte da release, sem executar o wheel instalado no ClusterGPU nesta fase. |

O notebook original foi consultado no commit `c283dc4e7f0d0d2b471ee40ec14764adca93208d`: [fonte fixada](https://github.com/Math-BUG/iot-fog-ids-two-phase/blob/c283dc4e7f0d0d2b471ee40ec14764adca93208d/codigo/COLAB_02_PIPELINE_ARTIGO.ipynb). Suas células de featurização, construção de `embedder`, `preprocess_sup` e `avaliar_modelo` sustentam a comparação. As dependências originais não fixam versões, limitando a reprodução binária exata: [requirements original](https://github.com/Math-BUG/iot-fog-ids-two-phase/blob/c283dc4e7f0d0d2b471ee40ec14764adca93208d/requirements.txt).

O ambiente informado e aprovado na Fase 1E é Python 3.13.5, GCC 14.3.0, CUDA 12.9.1, A100-SXM4-80GB, CuPy 14.2.0, cuDF 26.08.01 e cuML 26.08.00. Os logs fornecidos comprovam a conclusão do smoke e TruncatedSVD GPU efetivo com 20 componentes. **Não comprovam um fit ponderado**, que o projeto não solicitou. A correspondência das capacidades ponderadas com o binário instalado ainda requer uma verificação pequena nesse ambiente; não foi inventada uma execução remota.

Para a semântica CPU, também foram inspecionadas as implementações versionadas do sklearn 1.7.2 e 1.8.0. Ambas mantêm bootstrap uniforme e ponderação por árvore para `balanced_subsample`. Não se pressupõe que uma página `stable/latest`, uma versão futura do sklearn ou o docstring de um proxy sejam evidência suficiente sobre a release instalada.

## 3. Estado atual do projeto

### 3.1 População e split imutáveis

| Item | Valor aprovado |
|---|---:|
| Registros originais | 22.339.021 |
| Quarentena | 869 |
| População elegível | 22.338.152 |
| Treino | 13.575.269 |
| Validação | 3.244.049 |
| Teste | 5.518.834 |

Protocolo: `group_stratified/2.0.0`. Grupos: `src_ip, dst_ip, service, proto`. Todas as dez classes presentes em cada split; nenhuma sobreposição de grupos. Hash congelado:

```text
sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a
```

O fit completo futuro utilizará **13.575.269 linhas de treino**, não as 22.338.152 linhas elegíveis. Validação e teste permanecem fora do ajuste. O diretório persistente aprovado é `group_stratified_v2`, sob `$HOME/dados/ids_generalization_splits`.

As execuções futuras precisam aplicar os parâmetros de `reuse_split.yaml` e verificar o hash pelos mecanismos existentes. O YAML GPU genérico, isoladamente, não substitui o manifesto congelado. Por exemplo, o checkpoint fixa `timestamp_unit: s`; não se deve substituí-lo por `auto`. Referências: [configuração do checkpoint](../configs/ton_iot_group_stratified_checkpoint.yaml), [contratos e carregamento congelado](../src/ids_pipeline/split_protocols.py), [Fase 1D](phase_1d_group_split_checkpoint.md).

### 3.2 Features e representação

A lista ordenada de `behavioral_strict/2.0.0` permanece:

```text
duration, src_bytes, dst_bytes, conn_state, missed_bytes,
src_pkts, src_ip_bytes, dst_pkts, dst_ip_bytes
```

São oito quantitativas definidas pelo schema e uma categórica nominal (`conn_state`). As quantitativas recebem o `log1p` explícito existente, imputação por mediana de treino e padronização; `conn_state` recebe imputação pela moda de treino e one-hot com vocabulário de treino. Categorias desconhecidas geram vetor de indicadores zerado, sem aumentar a dimensionalidade. Não há códigos ordinais de `conn_state` entregues aos estimadores. [Preprocessing atual](../src/ids_pipeline/preprocessing.py), [encoder nominal compartilhado](../src/ids_pipeline/nominal_encoding.py).

CPU e GPU têm a mesma classificação semântica, mas não operações numericamente idênticas: o scaler CPU usa `StandardScaler`; `_gpu_numeric_arrays` calcula estatísticas em pandas, com `std()` amostral (`ddof=1`), e materializa matrizes CuPy `float32`. O scaler CPU usa variância populacional (`ddof=0`). Para colunas completas e não constantes após imputação, o fator relativo é `sqrt((n-1)/n)`; em treino grande a diferença é pequena, mas não nula. Essa diferença existente é registrada, sem modificar o Incremento 2.

O preprocessing denominado GPU é híbrido: estatísticas e parte da transformação numérica são pandas/CPU; construção matricial nominal e redução usam GPU. Isso já integrava o caminho aprovado e não constitui fallback novo. Fallback cuDF → pandas de I/O continua aceitável somente preservando identidade, população, schema, quarentena, ordenação e hash, conforme decisão da Fase 1E.

A redução mantém `effective=min(requested,max_components)`, com `max_components=max(0,min(n_train-1,n_encoded_features-1))`. `requested=50` não obriga dimensão efetiva 50. O smoke realizou TruncatedSVD com 20; o treino completo pode ter outro vocabulário e deve registrar sua dimensão efetiva. O solver CPU padrão é randomized; o cuML 26.08 usa `algorithm='full'` por padrão. Trata-se da mesma família de projeção SVD sem centralização, com diferenças numéricas e de solver. [Implementação cuML fixada](https://github.com/NVIDIA/cuml/blob/v26.08.00/python/cuml/cuml/decomposition/tsvd.pyx).

Existe alternativa PCA em `_make_gpu_reducer` caso a importação/construção de TruncatedSVD falhe. **PCA centraliza a matriz e não é intercambiável com TruncatedSVD**. O smoke observado não acionou essa alternativa. Não se modifica o mecanismo congelado; resultados com redutores reais diferentes não devem ser agrupados como o mesmo baseline.

### 3.3 Estimadores e dependências entre estágios

Na [configuração GPU completa](../configs/ton_iot_behavioral_strict_gpu.yaml): `k=30`, `cluster_batch_size=16384`, `cluster_n_init=5`, `cluster_max_iter=100`, RF com 150 árvores, `random_state=42`, `n_jobs=32`, sem representantes no treinamento. Os alvos configurados são `label`, `type` e `cluster_id`; nenhum alvo é alterado aqui.

O [clustering](../src/ids_pipeline/clustering.py) usa MiniBatchKMeans no ramo CPU e KMeans no GPU. O [supervisionado](../src/ids_pipeline/supervised.py) usa `balanced_subsample` somente no ramo CPU. No GPU, passa apenas quantidade de árvores, seed e `n_streams`, limitado a 16 a partir de `n_jobs`.

Os dois estágios recebem a mesma matriz transformada/reduzida. **`cluster_id` não é feature do RF de `label` ou `type`.** Com todas as linhas de treino e representantes desativados, mudar exclusivamente o clustering não muda os dados desses dois RFs. Afeta as análises de clustering e a definição do alvo `cluster_id`. Não se deve atribuir uma diferença de F1 de `label/type` à troca de KMeans se entradas, pesos e RF permanecerem iguais.

## 4. MiniBatchKMeans versus KMeans

### 4.1 Capacidade real da release

Não existe `MiniBatchKMeans` exportado por `cuml.cluster` na tag examinada; tampouco há override desse estimador no `cuml.accel`. Usar `cuml.accel` sobre sklearn MiniBatchKMeans não o transforma em uma implementação GPU. O KMeans distribuído continua sendo KMeans; distribuição entre dispositivos não cria a semântica de mini-batches. [Exports cuML](https://github.com/NVIDIA/cuml/blob/v26.08.00/python/cuml/cuml/cluster/__init__.py), [overrides de clustering](https://github.com/NVIDIA/cuml/blob/v26.08.00/python/cuml/cuml/accel/_overrides/sklearn/cluster.py).

O KMeans 26.08 dispõe de `max_samples_per_batch` para particionar distâncias e `device_buffer_samples` para streaming de entradas residentes no host. Ambos tratam custo/memória, não substituem atualizações MiniBatchKMeans. O caminho atual já passa arrays CuPy; o mecanismo de streaming de entrada host não é acionado. [API KMeans 26.08](https://docs.nvidia.com/cuml/26.08/api/generated/cuml.cluster.KMeans/).

Não foi demonstrada uma alternativa GPU pronta semanticamente equivalente ao sklearn MiniBatchKMeans, incluindo inicialização, atualização acumulada, reatribuição de centros e parada. Um port próprio em CuPy exigiria implementação e testes de conformidade; não é uma simples configuração e não é recomendado para liberar este baseline.

### 4.2 Parâmetros e equivalência

| Parâmetro | CPU atual | GPU atual | Avaliação |
|---|---|---|---|
| `n_clusters` | `selected_k=30` | `selected_k=30` | Mesmo número de centros. Não garante a mesma partição. |
| `batch_size` | `cluster_batch_size=16384` | Não passado; não existe equivalente de atualização mini-batch | Parâmetro do YAML ignorado no GPU. `max_samples_per_batch` não o substitui. |
| `n_init` | `cluster_n_init=5` | Não passado; `auto`, efetivamente 1 com inicializador padrão | Parâmetro do YAML ignorado no GPU. |
| `max_iter` | `cluster_max_iter=100` | `cluster_max_iter=100` | Mesmo valor, unidades operacionais distintas: teto de passagens equivalentes em mini-batches versus iterações completas. |
| `random_state` | `42` | `42` | Controle de RNG em cada biblioteca, sem equivalência de sequências, centros ou resultados. |
| `init` | sklearn `k-means++` | cuML `scalable-k-means++` | Inicializadores diferentes. |
| Parada | `tol=0`, `max_no_improvement=10`, defaults sklearn | `tol=1e-4`, convergência do KMeans | Critérios diferentes. |
| Reatribuição | `reassignment_ratio` do MiniBatchKMeans | Sem correspondência direta no construtor usado | Tratamento de centros pouco ocupados diferente. |
| Distância retornada | Mínimo das distâncias aos centros | Norma ao centro atribuído | Mesmo significado euclidiano para atribuição ao centro mais próximo, com centros/resultados diferentes. |

No MiniBatchKMeans, `n_init` avalia inicializações e executa a otimização mini-batch a partir da melhor; no KMeans cuML, conta execuções completas, escolhendo a de menor inércia. Portanto, passar futuramente `n_init=5` ao GPU não seria equivalência exata e elevaria seu custo. Não foi feito aqui. [Código sklearn MiniBatchKMeans](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/cluster/_kmeans.py), [código cuML KMeans](https://github.com/NVIDIA/cuml/blob/v26.08.00/python/cuml/cuml/cluster/kmeans.pyx).

O original publicado no notebook utilizava `batch_size=4096`, `n_init=10`, seed 42 e `k=30`. Assim, mesmo o CPU atual já não reproduz todos os hiperparâmetros do notebook. Essa divergência deve ser declarada, sem revertê-la silenciosamente.

### 4.3 Impacto esperado e escala

Ambos procuram reduzir a soma das distâncias quadráticas aos centros. MiniBatchKMeans aproxima a otimização por atualizações estocásticas; KMeans completo atualiza centros com a população de treino de cada iteração. Isso pode mudar centros, tamanhos de clusters, atribuições de fluxos raros, inércia e estabilidade. Não há garantia de melhor alinhamento com ataques nem de maior F1 ao reduzir a inércia. Inicialização e parada diferentes impedem atribuir toda diferença observada exclusivamente ao tamanho do lote.

Preservar MiniBatchKMeans CPU **não foi demonstrado inviável**: ele foi projetado para reduzir o custo de atualização. Com dimensão efetiva 20 apenas como cenário de cálculo, a matriz de treino densa `float32` ocupa aproximadamente 1,01 GiB; as três matrizes juntas, 1,66 GiB. São estimativas de `n*d*4`, não medições de pico de RAM/VRAM. DataFrames, cópias, SVD, índices, distâncias e árvores acrescentam memória; o loader atual não é out-of-core.

MiniBatchKMeans oferece `partial_fit`, mas substituir `fit` por um loop externo mudaria ordem, inicialização e parada, salvo reprodução cuidadosa. Não é um ajuste operacional automaticamente equivalente. A opção de menor risco para preservar o estimador é seu `fit` atual, medido primeiro em escala pequena. [API sklearn versionada](https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.MiniBatchKMeans.html).

### 4.4 Alternativas de clustering

| Alternativa | Vantagem | Desvantagem / classificação |
|---|---|---|
| C1. Manter sklearn MiniBatchKMeans em CPU, após representação congelada | Preserva o estimador CPU atual; implementação já existente | Estágio não executa em GPU; transferência e custo precisam ser medidos. Não restaura features/parâmetros do artigo. |
| C2. Manter cuML KMeans completo, declarado explicitamente | Caminho já aprovado operacionalmente em GPU; sem novo algoritmo implementado | Modificação metodológica em relação ao MiniBatchKMeans. Exige identificar defaults reais e parâmetros ignorados. |
| C3. KMeans cuML com streaming host / execução distribuída | Opções da mesma família para restrições de memória | Continua KMeans completo, não MiniBatchKMeans. Integração nova e desnecessária sem evidência de falta de memória. |
| C4. Port GPU de MiniBatchKMeans | Poderia buscar a mesma regra matemática de atualização | Não há equivalência pronta demonstrada; elevado esforço de validação de RNG, inicialização, parada e precisão. Não recomendado nesta fase. |

Para o objetivo GPU, C2 é a recomendação condicional, identificada como alteração de método. Para fidelidade do estimador, C1. Nenhuma delas foi implementada ou executada nesta entrega.

## 5. Random Forest e balanceamento

### 5.1 APIs 26.08 confirmadas no código da release

| Capacidade | cuML 26.08.00 | Uso no projeto |
|---|---|---|
| `class_weight=None` | Suportado | Default atual, sem ponderação. |
| `class_weight='balanced'` | Suportado; frequências calculadas no treino | Não utilizado. |
| `class_weight={classe: peso}` | Suportado | Não utilizado. |
| `class_weight='balanced_subsample'` | Rejeitado por `process_class_weight`; proxy sklearn sinaliza `UnsupportedOnGPU` | Não há reprodução GPU atual. |
| `fit(X, y, sample_weight=None, ...)` | Suportado; pesos encaminhados ao fit nativo | Projeto chama `fit(X, y)` sem pesos. |
| `sample_weight` em `score()` | Também existe | Não confundir peso de avaliação com peso de treinamento. |

Fontes: [API RF 26.08](https://docs.nvidia.com/cuml/26.08/api/generated/cuml.ensemble.RandomForestClassifier/), [classificador e rejeição no proxy](https://github.com/NVIDIA/cuml/blob/v26.08.00/python/cuml/cuml/ensemble/randomforestclassifier.py), [processamento dos pesos](https://github.com/NVIDIA/cuml/blob/v26.08.00/python/cuml/cuml/common/classification.py), [encaminhamento ao código nativo](https://github.com/NVIDIA/cuml/blob/v26.08.00/python/cuml/cuml/ensemble/randomforest_common.pyx). A solicitação pública de suporte por árvore também identifica a lacuna: [issue 8146](https://github.com/NVIDIA/cuml/issues/8146); o fundamento desta conclusão é a release fixada, independentemente do estado futuro da issue.

### 5.2 Por que pesos globais não reproduzem `balanced_subsample`

Sem pesos externos, sejam `n_c` as contagens de classe no treino, `N` seu tamanho e `C` o número de classes observadas. O balanceamento global usa:

```text
w(c) = N / (C * n_c)
```

No `balanced_subsample` sklearn, cada árvore `t` recebe um bootstrap uniforme com reposição. Depois, os pesos são calculados a partir das contagens desse bootstrap. Para classe presente:

```text
w_t(c) = N_t / (C_t * n_t,c)
peso efetivo da observação i = multiplicidade_i,t * w_t(y_i)
```

`C_t` representa as classes presentes no bootstrap; classes ausentes não ganham observações. Os pesos afetam a construção da árvore. Como `n_t,c` varia entre árvores, um vetor fixo `sample_weight[i]` não pode, em geral, produzir todos os vetores específicos de árvore. Frequências próximas às globais podem aproximar fatores, mas não estabelecem equivalência, sobretudo com poucas observações. [Implementação sklearn 1.8.0](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/ensemble/_forest.py), [cálculo por índices](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/utils/class_weight.py).

No cuML 26.08, `class_weight='balanced'` é convertido uma vez em pesos por observação. Com `bootstrap=True`, `RowSampler` sorteia linhas proporcionalmente a esses pesos e entrega `nullptr` como pesos de impureza da árvore, pois os pesos foram materializados pela amostragem. Com `bootstrap=False`, encaminha os pesos para a construção da árvore, mas elimina o bootstrap. [Implementação nativa, `RowSampler`](https://github.com/NVIDIA/cuml/blob/v26.08.00/cpp/src/randomforest/randomforest.cuh).

Consequência matemática: no caso global balanceado e sem outros pesos, a probabilidade total de sortear uma classe passa a `1/C`. Isso difere do bootstrap uniforme por observação do original, cuja probabilidade de classe é `n_c/N`, seguido de ponderação. Há mudança no conjunto de exemplos e nas multiplicidades dentro de cada árvore, não apenas nos custos de erro.

Passar manualmente `sample_weight[i]=N/(C*n_yi)` com `class_weight=None` reproduz a ponderação global de entrada do cuML; **não** corrige essa diferença nem implementa `balanced_subsample`. Passar simultaneamente pesos e `class_weight` exige cuidado adicional: o processamento pode considerar frequências ponderadas e multiplicar fatores. Não é necessário nem recomendado para este projeto.

Desativar bootstrap também não é uma solução equivalente: muda a diversidade do ensemble. Implementar lógica específica por árvore exigiria controle explícito dos bootstraps, ponderações, estimadores e agregação, além de continuar enfrentando as diferenças de construção das árvores GPU. Isso seria desenvolvimento científico novo e está fora desta entrega.

### 5.3 Outras diferenças do RF

| Item | sklearn CPU atual | cuML 26.08 atual | Consequência |
|---|---|---|---|
| Árvores | 150 | 150 | Contagem idêntica. |
| Seed | 42 no projeto; 10000 no notebook original | 42 | Mesma seed entre backends não gera as mesmas árvores. |
| Critério | Gini padrão | Gini padrão | Mesmo conceito de impureza. |
| Limiares candidatos | Busca sklearn nos valores das features | Discretização por quantis, `n_bins=128` padrão | Aproximação da busca de limiares; pode afetar fronteiras raras. |
| `max_features` | `sqrt` | `sqrt` | Mesma política nominal; arredondamento/RNG/implementação precisam ser distinguidos. |
| Profundidade máxima | `None` | **`None` na release 26.08** | Não afirmar que este ambiente ainda usa o antigo default 16. |
| Bootstrap sem pesos | Habilitado; tamanho total do treino | Habilitado; `max_samples=1.0` | Mesmo princípio, sorteios e implementações diferentes. |
| Ponderação | `balanced_subsample` | `None` | Modificação metodológica substantiva. |
| Paralelismo | `n_jobs=32` | `n_streams=16` derivado | Parâmetros operacionais distintos; nenhuma promessa de identidade bit a bit. |

O aumento de `n_bins` não garante reproduzir as árvores sklearn e não é proposto como alteração nesta fase. O default de profundidade sem limite também impede inferir custo de 150 árvores no treino completo a partir do smoke com um subconjunto. [API RF da release](https://docs.nvidia.com/cuml/26.08/api/generated/cuml.ensemble.RandomForestClassifier/).

### 5.4 Classes raras

`mitm`, `ransomware` e `backdoor` precisam ser avaliadas com suporte, precision, recall e F1 por classe; weighted-F1 e accuracy podem esconder perdas. A distribuição no recorte do artigo não é a distribuição da população completa: o artigo informa 50.000 normais, oito ataques com 20.000 registros cada e 1.043 MITM. Não se transfere automaticamente a conclusão sobre balanceamento desse recorte para a população atual.

Pesos não criam diversidade de ataques ou grupos. Sob bootstrap uniforme de tamanho `N`, uma classe com `m` observações pode estar ausente de uma árvore com probabilidade `(1-m/N)^N`, aproximadamente `exp(-m)`. Com uma única observação isso é cerca de 36,8%. É uma ilustração matemática, não a contagem da classe no treino completo. O único MITM do treino do smoke 1E não deve ser confundido com o suporte do treino congelado completo.

Bootstrap ponderado global pode repetir intensamente os mesmos exemplos raros; maior presença por árvore não prova melhor generalização a grupos novos. Pesos recalculados por árvore também não resolvem ausência de classe no treino nem diferenças entre dispositivos/tráfego. A inferência sobre impacto exige validação controlada, sem escolher pesos pelo teste final.

### 5.5 Alternativas de RF

| Alternativa | Vantagens | Desvantagens e comparabilidade |
|---|---|---|
| R1. sklearn RF com `balanced_subsample` | Preserva a ponderação por árvore e o estimador CPU; opção pronta | Fit CPU, possível custo elevado. Não é RF treinado em GPU; ainda usa features atuais, diferentes do artigo. |
| R2. cuML RF sem pesos, estado atual | GPU já validada; nenhuma intervenção de amostragem adicional | Remove balanceamento original. Aceitável somente como variante explicitamente não ponderada, com limitações por classe reportadas. |
| R3. cuML RF com `balanced`, bootstrap mantido | Suporte público a pesos; direciona massa de treino às classes raras | Produz bootstrap ponderado; não equivale a `balanced_subsample` nem ao bootstrap uniforme ponderado na impureza. Exige aprovação metodológica específica. |
| R4. cuML RF com pesos e `bootstrap=False` | Aplica pesos à construção das árvores sem sorteio ponderado | Elimina bootstrap e altera o ensemble; não preserva RF original. Não recomendado como correção de compatibilidade. |
| R5. Implementação específica por árvore | Permite investigar reprodução da regra de ponderação | Engenharia nova, sem equivalência pronta ou validação; limiares GPU continuam diferentes. Fora da fase. |

`cuml.accel` não resolve R1 em GPU: a conversão do classificador com `balanced_subsample` é explicitamente não suportada na release. Fallback CPU deve ser identificado como tal. Não implementar oversampling, undersampling ou SMOTE; R3 é apresentada para tornar visível inclusive o efeito de reamostragem interna dos pesos, não para executá-lo sob outro nome.

## 6. Comparabilidade com o artigo

Nesta tabela, **idêntico** significa mesma escolha especificada; **equivalente** significa mesmo significado matemático sob as condições indicadas; **aproximação** não implica resultados intercambiáveis; **modificação** altera representação, algoritmo ou população avaliada.

| Componente | Artigo / notebook original | GPU atual | Equivalente? | Impacto potencial |
|---|---|---|---|---|
| População | Recorte de 211.043 fluxos | 22.338.152 elegíveis, schema e quarentena auditados | Modificação | Frequências, suporte, grupos e dificuldade distintos. |
| Feature policy | Clustering com engenharia de portas e conectividade; RF com colunas originais disponíveis após exclusão de alvos/artefatos | Nove features explícitas de `behavioral_strict/2.0.0` para ambos | Modificação congelada | Remove IPs, portas, serviço/protocolo, agregados e demais campos; altera informação disponível. |
| Tipagem | Seleção numérica/categórica baseada em dtype no notebook | Esquema semântico explícito `ton_iot_network/1.0.0` | Modificação/correção congelada | Corrige quantitativas como `src_bytes`; não é mera aceleração. |
| Preprocessing de clustering | Log das numéricas aplicáveis; `StandardScaler(with_mean=False)` antes do SVD; `StandardScaler()` depois | Log explícito, mediana e padronização centrada antes do SVD; sem scaler após SVD | Modificação | Muda geometria e peso relativo dos eixos/componentes. |
| Encoding | One-hot de diversas categorias com `min_frequency=10` | One-hot exclusivo de `conn_state`, sem agrupamento por frequência | Equivalência nominal parcial; representação modificada | Mesma noção de categoria, mas informação e dimensionalidade diferentes. |
| Encoding CPU/GPU atuais | `ConnStateOneHot` compartilhado | Mesma moda, vocabulário e regra de desconhecidos; matriz CuPy | Equivalente semanticamente | Sem distância ordinal artificial; não exige igualdade numérica após redução. |
| Dimensionality reduction | sklearn TruncatedSVD, 50 componentes no ramo clustering, seguido de scaler | cuML TruncatedSVD; 50 solicitados, dimensão limitada; também usado pelo RF | Equivalência da família SVD, modificação do pipeline | Bases e dimensões distintas; projeção antes do RF muda eixos de decisão. PCA eventual seria outra mudança. |
| Clustering | sklearn MiniBatchKMeans | cuML KMeans completo | Modificação | Atualização, inicialização, parada e partições distintas. |
| `k` | 30 | 30 | Idêntico como hiperparâmetro | Não identifica os mesmos clusters nem valida novamente a escolha de k. |
| Entrada do RF | `preprocess_sup` próprio: mediana numérica e one-hot categórico; sem SVD; IPs/portas não excluídos por sua lista de features | Mesma matriz comportamental reduzida do clustering | Modificação já existente | Pode alterar a classificação mais do que a troca de implementação do RF. |
| Random Forest | sklearn, 150 árvores, seed 10000 | cuML, 150 árvores, seed 42, limiares por quantis | Mesma família; aproximação de construção, parâmetros parcialmente distintos | Mesma quantidade de árvores não implica mesmo modelo. |
| Balanceamento | `balanced_subsample` | Sem pesos | Modificação | Mudança do tratamento de classes e potencial perda de recall raro. |
| Split | SGKF externo de 5 folds, seed 42; interno de 5 folds, seed 123, estratificação por `type` | `group_stratified/2.0.0`, busca e invariantes próprios, estratificação considerando `label/type`, índices congelados | Mesmo princípio de separação de grupos; protocolo diferente | População e holdout diferentes impedem reprodução direta dos números do artigo. |
| Chave dos grupos | `src_ip, dst_ip, service, proto` | Mesmas quatro colunas | Idêntica como definição | Generalização a combinações novas, sem garantir dispositivos individualmente novos. |
| Alvos | `label`, `type`, `cluster_kmeans30` | `label`, `type`, `cluster_id` | Mesmo papel; terceiro alvo depende do clusterer | Valores de IDs de clusters não são classes semanticamente estáveis entre ajustes. |
| Ajuste no treino | Featurização/encoders/modelos ajustados no treino | Estatísticas e modelos ajustados no treino congelado | Equivalente como princípio | Preserva separação de avaliação, sem garantir equivalência do protocolo. |
| Métricas | Accuracy, macro/weighted-F1, relatório por classe, matriz; também balanced accuracy/MCC no notebook | Núcleo de classificação compartilhado via sklearn; métricas binárias adicionais | Equivalente nas definições comuns, cobertura diferente | Não confundir mesmas fórmulas com populações ou resultados comparáveis. `pr_auc` legado é AP, não área trapezoidal. |

O PDF reporta 183.158/16.901/10.984 linhas em treino/validação/teste no recorte original. Essas contagens não correspondem às proporções efetivas atuais e não devem servir como meta para alterar o split congelado.

As diferenças de features, schema, população, split e matriz de entrada do RF já existiam antes desta análise. Não se propõe revertê-las. A denominação adequada é **extensão do estudo sob protocolo de generalização e representação comportamental explícita**, com variante GPU identificada. Não afirmar reprodução exata do artigo, nem atribuir eventual diferença de desempenho ou tempo exclusivamente ao hardware.

## 7. Verificações pequenas propostas antes do baseline completo

**São necessárias; não foram executadas nesta fase.** A integração 1E não isolou algoritmo, balanceamento e backend.

### 7.1 Conferência do binário instalado, sem dados reais

Após aprovação, registrar no ambiente real a versão, origem do pacote e caminhos das classes; inspecionar assinaturas e `get_params()` de KMeans/RF, presença ou ausência de MiniBatchKMeans, defaults de `init`, `n_init`, `n_bins`, `max_depth`, `bootstrap` e `class_weight`. A assinatura de `fit` deve incluir `sample_weight`.

Uma fixture sintética mínima pode confirmar a aceitação de pesos e a rejeição de `balanced_subsample`, sem tratá-la como teste de desempenho ou de equivalência. O fit ponderado é somente proposta de teste de capacidade; depende de aprovação e não habilita pesos nos experimentos. Para bootstrap ponderado, a evidência de semântica deve continuar vinculada ao código da release, não ser inferida apenas de uma variação de accuracy.

### 7.2 Comparação controlada de clustering

Reutilizar uma lista pequena e persistida de IDs pertencentes aos splits congelados, preferencialmente os subconjuntos 1E já registrados. Não gerar novo split nem incluir teste no ajuste. Manter as nove features, schema, ordem e `k=30`.

Primeiro, comparar MiniBatchKMeans CPU atual com KMeans GPU atual sobre **a mesma matriz já transformada**, apenas transferida entre dispositivos. Isso mede a diferença agregada dos estimadores/configurações. Registrar inércia na mesma matriz, tamanhos, clusters vazios, AMI/ARI entre partições (invariantes à permutação de IDs) e métricas existentes; não comparar números brutos de cluster como se fossem classes fixas.

Se for necessário atribuir causas, usar controles separados e pré-declarados: KMeans completo sklearn versus MiniBatchKMeans para efeito da regra de atualização, e KMeans completo CPU versus GPU para efeito da implementação. Centros iniciais comuns e uma inicialização podem ajudar nessa investigação, mas seriam condições de um diagnóstico isolado, **não mudanças nos parâmetros do baseline congelado**. Essa ampliação é opcional e requer aprovação própria; o primeiro par já detecta a falta de intercambialidade.

### 7.3 Comparação controlada de RF

Usar exatamente os mesmos IDs de treino/validação e a mesma matriz transformada; preservar 150 árvores, alvos configurados e seed atual 42. Para o diagnóstico do RF de `cluster_id`, fixar uma única referência de rótulos de clustering entre braços para não trocar simultaneamente modelo e alvo.

| Controle proposto | Comparação | Pergunta respondida |
|---|---|---|
| RF CPU `balanced_subsample` versus CPU sem pesos | Mesmo backend e entradas | Qual é o efeito de remover o balanceamento original nesse subconjunto? |
| RF CPU sem pesos versus GPU sem pesos | Mesmos dados e política de pesos | Qual é a diferença entre implementações sem o confundimento adicional da ponderação? |
| CPU `balanced_subsample` versus CPU `balanced`, opcional | Mesmo bootstrap uniforme e backend versionado | Quanto muda a ponderação por árvore versus global? |
| GPU ponderado, somente mediante decisão posterior | Identificar como bootstrap ponderado | Avaliar uma modificação metodológica, não demonstrar equivalência. Fora da recomendação mínima. |

Reportar macro/weighted-F1, precision/recall/F1 e suporte por classe; para `label`, manter ROC-AUC, AP e FPR@TPR95 existentes. Usar validação para a análise de alternativas; preservar teste final sem escolha de método baseada nele. Classes ausentes ou com suporte muito pequeno no subconjunto devem aparecer como limitação, sem rebalancear, completar ou regenerar os splits. Em particular, o suporte de MITM do smoke 1E é insuficiente para uma conclusão científica sobre essa classe.

Medir custo apenas como diagnóstico de viabilidade de cada estágio e uso de memória; não extrapolar speedup final a partir de poucos registros. Se a decisão envolver fidelidade CPU em escala completa, é necessário um ensaio de capacidade progressivo previamente aprovado; o documento não presume que CPU seja inviável nem que a A100 elimine todo risco de custo do RF.

### 7.4 Critério de liberação metodológica

Antes do baseline completo, deve existir decisão escrita identificando:

- C1 ou C2 para clustering e R1 ou R2 para o RF principal; outras opções exigem justificativa adicional.
- Algoritmo real, parâmetros efetivos e parâmetros que não se aplicam ao backend, sem renomear MiniBatchKMeans como KMeans equivalente.
- Ponderação ausente, por árvore ou bootstrap ponderado, sem apresentar essas três possibilidades como intercambiáveis.
- Preservação de população, hash do split, política de features e redutor real; nenhuma alteração dos Incrementos 1–3.
- Resultado dos controles pequenos e limitações das classes raras. Igualdade aproximada de métricas em um subconjunto não prova identidade metodológica.

Não é necessário nem autorizado executar agora os 22.338.152 registros, restaurar seletores ou mudar `k`, targets, features ou split. **A próxima ação depende da aprovação da alternativa e dos controles propostos neste documento.**

## 8. Registro desta entrega

Arquivo criado: `docs/phase_1f_gpu_methodological_equivalence.md`.

Nenhum arquivo científico, configuração, job ou teste foi alterado. Não houve fit de qualquer estimador nem submissão Slurm. A verificação desta entrega é documental: confronto com código local, notebook original, fontes da release cuML e consistência dos links locais. Testes de treinamento e CUDA ficam pendentes de aprovação, conforme o escopo solicitado.
