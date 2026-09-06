# Incremento 2: behavioral_strict/2.0.0

O Incremento 1 permanece congelado: 22.339.021 registros originais, 869 quarentenados, 22.338.152 elegíveis e zero violações no checkpoint anterior. Nenhum dado real é relido nesta etapa. Schema, normalização semântica e quarentena não foram alterados.

## Contrato de seleção

`BEHAVIORAL_STRICT_FEATURES`, em feature_policy.py, é a única lista autorizada para esta política. A seleção retorna exatamente esta ordem, independentemente da ordem das colunas da entrada:

| Ordem | Feature | Semântica do schema | Tratamento |
|---:|---|---|---|
| 1 | duration | continuous | quantitativo |
| 2 | src_bytes | count | quantitativo |
| 3 | dst_bytes | count | quantitativo |
| 4 | conn_state | nominal | categórico one-hot |
| 5 | missed_bytes | count | quantitativo |
| 6 | src_pkts | count | quantitativo |
| 7 | src_ip_bytes | count | quantitativo |
| 8 | dst_pkts | count | quantitativo |
| 9 | dst_ip_bytes | count | quantitativo |

Uma feature ausente, nomes duplicados ou conflito com um alvo configurado causam erro explícito. Chamadas diretas ao preprocessing/clustering também devem satisfazer o contrato exato, inclusive a ordem. IPs, portas, service, proto, ts, label, type, ssl_resumed e qualquer feature futura não podem ser incluídos por substring. As constantes de hints antigas no schema congelado não são usadas para selecionar behavioral_strict.

## Representação e estatísticas

As oito quantitativas continuam no ramo determinado pelo schema, inclusive quando o armazenamento contém strings numéricas. O preprocessing preserva as transformações existentes: log1p somente nas colunas explicitamente autorizadas pelo schema e habilitadas pela configuração; imputação pela mediana e padronização ajustadas no treino. Com a configuração padrão, as oito recebem log1p. Não houve nova inferência por dtype ou por nome. Não foram modificados os métodos numéricos CPU/GPU já existentes, incluindo suas diferenças de precisão e cálculo do desvio padrão.

`ConnStateOneHot` aprende somente no treino um vocabulário ordenado lexicograficamente. Cada estado conhecido ocupa exatamente uma coluna binária, sem drop, escala ordinal ou agrupamento por frequência. `onehot_min_frequency` e `gpu_max_categories_per_col` continuam válidos nos ramos legados, mas não agrupam nem cortam o vocabulário de conn_state.

- Ausência reconhecida pelo schema: imputação pela moda do treino, com desempate lexicográfico.
- Treino inteiramente ausente: fallback local e explícito `__MISSING__`, com uma coluna preservada. Isso não acrescenta sentinelas ao schema.
- Estado desconhecido em validação/teste: vetor inteiro de zeros no bloco conn_state. Não cria coluna nem atualiza o vocabulário.
- CPU: construção CSR esparsa; ColumnTransformer pode devolver uma matriz densa conforme a densidade total.
- GPU: os mesmos parâmetros de vocabulário/moda; construção de indicadores float32 diretamente em CuPy por posicionamento dos valores 1. Os índices inteiros intermediários são usados somente para localizar colunas, nunca como coordenadas do modelo. Não é criada uma matriz densa N×C no host.

Para C estados no treino, o bloco nominal tem C colunas, com custo denso GPU de 4×N×C bytes por split. No caso normal das oito quantitativas utilizáveis, a matriz anterior à redução tem 8+C colunas: quantitativas na ordem relativa da lista e, depois, indicadores na ordem do vocabulário. A redução dimensional configurada anteriormente permanece posterior a essa representação e continua ajustada somente no treino. Se ativada, o clustering recebe a projeção do espaço quantitativo/one-hot, não códigos ordinais de conn_state.

A mudança nominal vale para conn_state também nas demais políticas. Os outros campos categóricos GPU permanecem no caminho legado; a garantia de ausência de coordenadas ordinais em toda a matriz aplica-se a behavioral_strict, cujo único campo categórico é conn_state. Não foi ampliado o escopo para redesenhar os outros encoders.

## Artefatos

`selected_features.json` registra versão, requested_features, found_features, lista selecionada, tipos semânticos e transformações por feature. No estágio prepare, `transformation_status=configured_not_fitted` evita afirmar que houve ajuste. Depois do preprocessing, recebe `fitted_on_train`, vocabulário, moda, regra de desconhecidos, ordem das saídas quantitativas, dimensões e redução efetivamente aplicada. O mesmo estado nominal fica no preprocessing_metadata.json e no bundle serializado.

## Validação e compatibilidade

Os testes usam apenas fixtures pequenas, incluindo ambas as rotas de preprocessing, geometria nominal, categorias exclusivas de holdout, serialização e redução. O caminho GPU dos testes unitários usa NumPy como substituto de CuPy e, quando necessário, um redutor CPU; isso não equivale a executar CUDA real. Um teste CPU de clustering usa 120 registros e k=30, sem Random Forest ou F1.

O smoke opcional `python -B scripts/smoke_behavioral_strict_cuda.py` verifica CuPy/cuML e um dispositivo utilizável. Quando disponíveis, usa 20 mil registros sintéticos e cuML KMeans com k=30. Se indisponíveis, registra skipped e o motivo em cuda_smoke.json. Não instala dependências, lê dataset, grava cluster_assignments.csv nem realiza avaliação supervisionada.

Resultados antigos não são comparáveis diretamente: o conjunto de features agora é fixo; conn_state mudou a geometria GPU e pode aumentar a dimensão anterior à redução. Em CPU, a ordem das saídas categóricas e o tratamento de frequência/treino inteiramente ausente de conn_state também podem diferir. Bundles antigos não devem ser reutilizados como se fossem behavioral_strict/2.0.0. Fixtures/configurações anteriores sem uma das nove colunas obrigatórias agora falham, inclusive o gerador sintético genérico atual quando lhe faltam os contadores requeridos.

Splits, orçamento, representantes, seletores, profiling, Random Forest e configuração de k não foram modificados. O Incremento 3 não foi iniciado.
