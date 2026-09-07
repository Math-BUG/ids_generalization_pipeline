# Contrato de orçamento total — Incremento 4

`selection_budget/1.0.0` valida a saída de futuros seletores. Não implementa `random`, estratificação, alocação por cluster ou novas políticas de representantes.

```yaml
selection_budget: 100000  # inteiro positivo ou full
split_seed: 42
clustering_seed: 42
selection_seed: 42
model_seed: 42
frozen_splits_dir: reports/increment_3/real/group_stratified
```

Os demais parâmetros de split devem coincidir com `reuse_split.yaml` do Incremento 3. O modo orçamento exige um split congelado validado. Não regenera splits nem exporta cópias dos arrays por seleção; salva uma referência pequena ao split utilizado.

## Invariantes

Para N linhas de treino e B inteiro, exige-se `1 <= B <= N`, exatamente B identificadores distintos, todos pertencentes ao treino congelado. Validação e teste não podem completar o orçamento. B impossível causa erro. Não há redução de B, truncamento de representantes, reposição, deduplicação corretiva ou pontos sintéticos.

`full` resolve para N. B=N e `full` produzem a mesma seleção e o mesmo hash de conteúdo. Outros valores inteiros positivos, além da grade planejada, são permitidos. Booleanos, floats, strings numéricas e abreviações como `1k` são rejeitados na configuração: escreva `1000`.

`Selection.accept(population, requested, seed, indices)` recebe identificadores fornecidos por um futuro seletor e verifica os invariantes. Para B<N, omitir os índices causa erro: nenhum seletor foi implementado neste incremento. Para B=N ou `full`, omitir índices significa utilizar todos os IDs do treino. Os IDs são ordenados canonicamente e protegidos contra mutação; duplicatas são rejeitadas, nunca removidas para corrigir a contagem.

Os identificadores são posições `iloc` na **população elegível congelada**, vinculadas ao seu manifesto e ao hash do split. Não são números locais de linha da matriz comprimida nem rótulos de índice pandas. O mapeamento para as posições originais dos arquivos continua sendo o artefato do Incremento 3.

```python
with FrozenTrainingPopulation.load(split_directory) as population:
    selection = Selection.accept(population, B, selection_seed, proposed_row_ids)
    selection.save_summary(population, output / "selection_summary.json")
```

## Identidade e metadados

O resumo registra orçamento solicitado e realizado, N, fração retida, seed da seleção, hash da população de treino, hash da seleção e hash do split congelado. Não grava lista de IDs, matriz ou associação individual em JSON/CSV.

Uma seleção diferente não pode sobrescrever o resumo do mesmo experimento. Isso também impede trocar o conjunto de amostras entre chamadas separadas de treinamento dos alvos no mesmo diretório. Use outro diretório para outra seleção/método.

O hash do treino vincula a população elegível e os hashes de índices e identidades do treino congelado. O hash da seleção vincula a versão do contrato, esse treino, o split e os IDs selecionados em ordem canônica, serializados em inteiros little-endian de 64 bits. São usados SHA256 e JSON canônico.

O hash da seleção identifica **conteúdo**: seed diferente que produza o mesmo conjunto terá o mesmo hash, mas a seed permanecerá registrada separadamente. A grafia `full` versus B=N também não muda esse hash. População ou conjunto diferente muda sua identidade.

`budget_report` registra apenas viabilidade, com `realized_budget: null` e `selection_hash: null`, quando nenhuma seleção foi realizada. Não confunde orçamento resolvido com observações efetivamente selecionadas. O checkpoint real deste incremento usa somente esse caminho.

## Seeds e compatibilidade das APIs congeladas

As quatro seeds são independentes. Campos omitidos herdam individualmente o antigo `random_state`, preservando a configuração legada. Alterar `selection_seed` não alimenta nenhuma outra etapa.

`config.seed_for(stage)` resolve a seed de uma etapa. `config.for_stage(stage)` adapta APIs congeladas que ainda recebem `random_state`, preservando as quatro seeds resolvidas no objeto adaptado. A CLI encaminha o adaptador de split às funções congeladas de criação/carregamento e manifesto, e o adaptador de clustering à entrada de clustering. As chamadas CPU/GPU de RF usam `model_seed`. Representantes legados recebem o adaptador de seleção. Preprocessing e geração/carregamento de dados mantêm o comportamento anterior de `random_state`.

Chamadas diretas às APIs antigas de split/clustering devem receber `config.for_stage('split')` ou `config.for_stage('clustering')`, respectivamente. Nenhum arquivo dos protocolos ou do preprocessing foi alterado.

Sem `selection_budget` (ou com `null`), o modo é legado. Com orçamento definido, `use_representatives_for_supervised`, estratégias de representantes diferentes de `full`, e overrides de `representatives_per_cluster`/`boundary_per_cluster` são incompatíveis e causam erro. Defaults inativos desses dois contadores não controlam B. O builder legado também rejeita chamadas diretas em modo orçamento. Não convertemos centroides em amostras reais nem truncamos resultados legados.

## Guarda anterior ao RF

`train_random_forest(..., selection=...)` carrega o split congelado e valida sua identidade contra o dataframe e os splits realmente recebidos. Resolve **uma única seleção**, compartilhada por todos os alvos solicitados. `label` e `type` usam exatamente os mesmos IDs e a mesma ordem; `cluster_id` permanece opcional e compatível.

O mapeamento dos IDs selecionados para as posições locais de `X['train']` é único e aplicado igualmente a X e y. Antes de cada `RandomForest.fit`, no caminho CPU e GPU, o contrato é revalidado: B, pertencimento, unicidade, hashes, ordem do treino e contagens de X/y. As entradas privadas de fit também rejeitam o modo orçamento sem contexto de seleção. Nenhuma implementação de RF ou hiperparâmetro foi substituído; apenas a seed e a guarda de entrada foram conectadas.

A identidade de cada linha de X continua dependendo do contrato já existente do preprocessing: a linha j de X de treino corresponde à posição j do array de treino congelado. Esta etapa preserva esse alinhamento ao recortar X e y; não tenta inferir identidades a partir dos valores transformados.

## Validação sem experimentos

```powershell
python -B scripts/check_selection_budgets.py
```

Lê o manifesto e os arrays congelados; verifica hashes, contagens, ausência de sobreposição e cobertura da população elegível. Arquivos grandes usam mapeamento de memória e hashing em blocos. Não lê o dataset ou `cluster_assignments.csv`, não seleciona linhas e não treina modelos.

Os testes de integração substituem RF por um objeto de teste que registra argumentos de `fit`. Não há ajuste real de RF nos testes deste incremento. Os novos métodos de seleção e a matriz experimental permanecem fora do escopo.
