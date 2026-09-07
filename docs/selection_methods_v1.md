# Seletores sob orçamento fixo — Incremento 5

Os seis métodos em `selection_methods/1.0.0` entregam seus IDs ao contrato congelado `selection_budget/1.0.0`. Cada seleção contém exatamente B linhas reais e distintas do mesmo treino congelado. Não há alteração dos incrementos anteriores nem uso de RF nesta validação.

| Método | Informação permitida | Pesos dos estratos | Amostragem dentro do estrato |
|---|---|---|---|
| `random` | IDs de treino, selection_seed | não se aplica | B IDs uniformes sem reposição |
| `stratified_label` | IDs e label do treino, selection_seed | tamanho da classe | uniforme sem reposição |
| `stratified_type` | IDs e type do treino, selection_seed | tamanho da classe | uniforme sem reposição |
| `cluster_uniform` | IDs e cluster do treino, selection_seed | 1 | uniforme sem reposição |
| `cluster_proportional` | IDs e cluster do treino, selection_seed | tamanho do cluster | uniforme sem reposição |
| `cluster_sqrt` | IDs e cluster do treino, selection_seed | raiz quadrada do tamanho | uniforme sem reposição |

Os métodos estratificados são proporcionais com cobertura mínima. Não impõem 50/50. Se B for próximo do número de classes, a cobertura mínima necessariamente pode afastar bastante as proporções das originais. O relatório mostra quotas e capacidades para tornar esse efeito visível.

## Alocador central

`allocate_budget(capacities, weights, B, minimum_coverage=True)` implementa `bounded_largest_remainder/1.0.0`:

1. Valida capacidades inteiras não negativas, pesos finitos positivos nos estratos não vazios e B viável.
2. Se B=N, retorna todas as capacidades.
3. Se B for pelo menos o número de estratos não vazios, reserva uma unidade para cada um.
4. Distribui o orçamento restante proporcionalmente aos pesos originais dos estratos com capacidade residual.
5. Estratos cuja parcela excede a capacidade são saturados; o excedente é redistribuído entre os demais.
6. Aplica pisos e distribui o restante pelas maiores partes fracionárias. Empates usam a ordem canônica dos estratos.
7. Confere soma exata B e quotas entre zero e a capacidade.

Classes usam ordem lexical de sua representação textual normalizada; clusters usam ordem numérica dos IDs. Estratos vazios recebem zero. Quando B<C, não há garantia de cobertura de todos; os estratos omitidos são registrados. Nenhum ajuste é feito truncando a seleção final.

**Limite de `cluster_sqrt`:** a raiz reduz a razão entre os pesos, mas não é um teto rígido para o maior cluster. Com capacidades [100000, 1, 1] e B=1000, qualquer solução com cobertura e sem reposição precisa usar [998, 1, 1]. Isso é uma restrição da população, não uma falha a corrigir com duplicação ou relaxamento de B.

## Isolamento e identidade

`select_random` não possui parâmetros para labels, types ou clusters. Os demais seletores recebem um único `TrainStrata`, que declara o atributo permitido, os IDs ordenados e o hash da população de treino. Trocar o tipo de atributo ou seu alinhamento causa erro.

A fronteira `select_from_training_frame` recorta apenas a coluna necessária e apenas os IDs do treino antes de chamar o seletor. Random não consulta o dataframe; os seletores de cluster não consultam label/type nem distâncias. Classes existentes apenas em validação/teste não entram no vocabulário dos estratos.

Os kernels não aprendem nem alteram preprocessing ou clustering. Usam `numpy.random.default_rng(selection_seed)`. O mesmo conjunto de entradas, método, B e seed gera a mesma seleção. O hash de conteúdo continua sendo o do contrato congelado: métodos ou seeds diferentes que produzam o mesmo conjunto terão o mesmo hash, com método/seed registrados separadamente. Em `full` ou estratos saturados, seeds diferentes naturalmente podem produzir o mesmo conjunto.

`SelectionResult` registra versão do método, informação utilizada, versão do alocador, capacidades, pesos, quotas, cobertura, estratos omitidos e hash das atribuições aos estratos. A seleção é validada e ordenada pelo contrato existente. Os resumos não gravam associações de toda a população.

## Integração

```yaml
selection_budget: 1000
selection_method: cluster_sqrt
selection_seed: 42
frozen_splits_dir: reports/increment_3/real/group_stratified
feature_policy: behavioral_strict
selected_k: 30
```

Mantenha os parâmetros de split do arquivo `reuse_split.yaml` e os demais parâmetros já definidos. A CLI proíbe método sem orçamento, método desconhecido ou combinação de método com seleção explícita já fornecida. Métodos de cluster exigem behavioral_strict/2.0.0 e k=30.

Random e estratificados são selecionados após carregar o split. Métodos de cluster recebem apenas `clustering['labels']['train']`, produzido pela execução atual. Não leem CSV de associações antigo. A seleção comum é passada à integração de RF congelada; representantes legados não entram nesse caminho.

## Validação real deste incremento

`scripts/validate_selection_methods.py` usa as 13.575.269 linhas do treino principal e produz somente B=1000 e B=10000 para os seis métodos. Os SHA256 das 23 fontes são comparados com o checkpoint da população elegível. Quarentena e normalização usam as funções congeladas.

O ambiente não dispõe de CuPy/cuML; a execução usa o ramo CPU existente, MiniBatchKMeans, com k=30, batch_size=16384, n_init=5, max_iter=100 e clustering_seed=42. Nenhum algoritmo novo substitui o KMeans. O preprocessing utiliza os componentes congelados e mantém SVD solicitado de 50, limitado pela dimensão de entrada conforme a regra existente.

Para limitar cópias em memória, normalização/log1p são aplicados por bloco; medianas, scaler, vocabulário e SVD são ajustados sobre todo o treino. Matrizes intermediárias usam arquivos NPY mapeados. O teste de equivalência compara esse caminho com `fit_transform_preprocessing` e `_fit_cpu_minibatch_kmeans` nas mesmas linhas e verifica representação, centros e clusters. Holdouts não precisam ser transformados/preditos nesta auditoria, pois não participam das seleções.

O ajuste de clustering sobre o treino inteiro é executado uma única vez. Os novos IDs de cluster são armazenados em `train_cluster_ids.npy`, alinhados ao array congelado de treino, com proveniência e hash. São um artefato comum de entrada; cada seleção grava apenas 1000 ou 10000 IDs em `selected_ids.npy`, junto a resumos pequenos. A matriz transformada e os modelos auxiliares documentam essa execução, sem RF e sem F1.

Os testes cobrem restrições de capacidade, arredondamento, redistribuição, grupos dominantes/singletons, cobertura mínima, reprodução, isolamento de informação e integração. Não foram adicionados mixed, medoid, boundary, centroid, HDBSCAN ou outra política de clustering. Incremento 6 permanece fora do escopo.
