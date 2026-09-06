# Incremento 2 concluído

`behavioral_strict/2.0.0` exige nove colunas, nesta ordem: duration, src_bytes, dst_bytes, conn_state, missed_bytes, src_pkts, src_ip_bytes, dst_pkts, dst_ip_bytes. Não usa substrings. Ausência, duplicação de nomes, conflito com alvos, inclusão extra ou mudança da ordem contratada causam erro.

As oito quantitativas mantêm a classificação e o log1p explícitos do schema congelado. Imputação pela mediana, padronização e redução continuam usando somente treino. conn_state usa one-hot nominal em CPU e GPU, vocabulário ordenado aprendido no treino, ausência imputada pela moda de treino e categorias desconhecidas representadas por zeros. O fallback para treino de conn_state inteiramente ausente é explícito e local ao encoder. Nenhum código ordinal de conn_state entra no clustering.

`selected_features.json` registra versão, listas solicitada/encontrada/selecionada, semântica e transformações. Distingue configuração sem ajuste no prepare de transformações efetivas após preprocessing, incluindo vocabulário, ordem dos indicadores e redução aplicada.

**124 testes passaram**, incluindo **39 casos novos** em test_behavioral_strict_v2.py. Os testes verificam todos os requisitos, o contrato na entrada direta do preprocessing, desconhecidos e ausência, serialização, estabilidade com SVD, ambas as rotas com GPU simulada e clustering CPU com 120 registros e k=30. Consulte [tests.xml](tests.xml) e [verification.json](verification.json).

Comando executado:

```powershell
python -B -m pytest tests/test_behavioral_strict_v2.py tests/test_feature_policy.py tests/test_no_leakage.py tests/test_proxy_policy.py tests/test_dataset_schema.py tests/test_data_quality_policy.py tests/test_schema_audit.py -o addopts='' -p no:cacheprovider -q --junitxml=reports/increment_2/tests.xml
python -B scripts/smoke_behavioral_strict_cuda.py
```

O [smoke CUDA](cuda_smoke/cuda_smoke.json) foi **skipped** antes de gerar os 20 mil registros: CuPy e cuML não estão instalados no Python disponível. nvidia-smi detectou uma GTX 1650 de 4 GiB; a consulta WSL retornou acesso negado. A GPU dos testes unitários foi simulada com NumPy; não se afirma validação de execução CUDA/cuML real. O script está pronto para um ambiente com backend utilizável, usando k=30 e nenhuma avaliação supervisionada.

Hashes de **10 arquivos congelados** coincidem com o início da etapa: schema, normalizador, quarentena, loader, splits, clustering, representantes, profiling, supervisão e configuração. O Incremento 1 permanece em 22.339.021 originais, 869 quarentenados e 22.338.152 elegíveis, com o checkpoint anterior de zero violações. Não foi reexecutada uma auditoria grande nem lido cluster_assignments.csv.

Arquivos desta etapa:

- src/ids_pipeline/feature_policy.py — lista fixa e versão;
- src/ids_pipeline/leakage_checks.py — contrato obrigatório em chamadas diretas;
- src/ids_pipeline/nominal_encoding.py — novo encoder nominal compartilhado;
- src/ids_pipeline/preprocessing.py — conexão do encoder CPU/GPU e metadados;
- src/ids_pipeline/cli.py — selected_features.json antes/depois do ajuste;
- tests/test_feature_policy.py — fixture com todas as colunas obrigatórias;
- tests/test_behavioral_strict_v2.py — novos testes;
- scripts/smoke_behavioral_strict_cuda.py — smoke opcional real;
- docs/behavioral_strict_v2.md — contrato, representação e compatibilidade.

Resultados antigos e bundles não são intercambiáveis com a nova política: seleção de features, geometria GPU de conn_state, dimensões e eventualmente projeções mudam. Configurações/fixtures sem as nove colunas agora falham; a frequência mínima e o limite de categorias deixam de agrupar conn_state. Os demais encoders GPU legados não foram redesenhados. A redução existente continua posterior à representação nominal. Não houve alteração de k=30, novos seletores, orçamento, Random Forest ou avanço ao Incremento 3.
