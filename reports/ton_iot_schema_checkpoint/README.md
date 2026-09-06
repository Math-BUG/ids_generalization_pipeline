# Checkpoint de validação do esquema TON_IoT

**Parecer final: o dataset não está compatível com `ton_iot_network/1.0.0`.**
As duas varreduras terminaram: 23 partições, 22.339.021 registros em cada
representação e 869 falhas de conversão em `src_bytes` (175 na partição 1,
690 na 22 e 4 na 23), todas com o token `0.0.0.0` nos CSVs originais.
As demais verificações de valores passaram. Não houve treinamento de IDS,
clustering ou alterações nos dados. Resultados em `report.html` e nos agregados
`parquet/` e `csv_complete/`.

Os 104 controles de consistência dos agregados passaram; estão em
`aggregate_verification.json`. Os hashes dos códigos de produção permaneceram
iguais e os metadados tamanho/mtime das fontes foram preservados. O registro de
ambiente está em `environment.json`.

## Escopo e execução

- CSVs originais: `D:/IC/Dataset/Network_dataset_1.csv` até `Network_dataset_23.csv`.
- Parquets correspondentes ao manifesto do projeto:
  `D:/IC/Dataset/dados/dataset_parquet_total/Network_dataset_1.parquet` até
  `Network_dataset_23.parquet`.
- São 23 partições lógicas, auditadas em duas representações físicas. Não somar
  as linhas das duas representações como se fossem instâncias diferentes.
- A cópia `D:/IC/Dataset/dataset_parquet`, arquivos de amostra e outros arquivos
  da pasta foram excluídos. A seleção exige nomes exatos e partições 1 a 23.
- Blocos de no máximo 100.000 linhas por leitor; duas varreduras independentes
  em paralelo. CSV com strings e sem inferência automática de ausência;
  Parquet com `pyarrow.ParquetFile.iter_batches(use_threads=False)`.
- Sem reconversão, escrita ou remoção nos diretórios de origem. Tamanho e mtime
  são registrados antes/depois de cada arquivo; isto não substitui hash integral
  de conteúdo, que exigiria uma leitura adicional.

```powershell
python -B -u scripts/audit_ton_iot_schema.py --input-dir D:/IC/Dataset/dados/dataset_parquet_total --output-dir reports/ton_iot_schema_checkpoint/parquet --format parquet --chunk-size 100000
python -B -u scripts/audit_ton_iot_schema.py --input-dir D:/IC/Dataset --output-dir reports/ton_iot_schema_checkpoint/csv_complete --format csv --chunk-size 100000
```

O script exige uma pasta de saída nova para evitar sobrescrever uma auditoria.
Escolha outros nomes de saída ao reproduzir. O primeiro ensaio CSV identificou
um BOM UTF-8 no cabeçalho; o leitor de cabeçalho da auditoria foi corrigido para
`utf-8-sig`, sem alterar a normalização ou os arquivos de entrada. O diretório
`csv/` vazio pertence a esse ensaio interrompido; não é uma auditoria concluída.
A varredura Parquet iniciou antes dessa correção exclusiva do leitor CSV. Sua
versão exata do script está preservada em `audit_script_parquet_snapshot.py`;
cada `audit.json` registra os hashes SHA-256 dos códigos usados.

## Como interpretar as contagens

- Grão: uma célula de uma coluna em um registro de conexão. `invalid_cells`
  soma células inválidas; não é uma contagem de linhas distintas entre colunas.
- `observed_rows`: linhas em arquivos onde a coluna está presente.
  `absent_rows`: linhas de arquivos sem essa coluna. Ausência estrutural não é
  nulo nativo. Colunas adicionais e alvos obrigatórios ausentes bloqueiam o
  parecer de compatibilidade; ausência de coluna opcional é registrada.
- `native_nulls`: nulos presentes no dataframe recebido. CSV não tem dtype nem
  valor nulo nativo; campos vazios são tokens `""`. Os dtypes físicos Parquet
  são os tipos Arrow declarados nos arquivos; dtypes pandas também são listados.
- `sentinel_tokens`: contagens exatas de `""` e `"-"`, por coluna e arquivo.
  `recognized_missing` soma somente nulos nativos e essas sentinelas, nunca
  inclui valores inválidos mascarados internamente pelo normalizador.
- `conversion_failure`, `infinite`, `domain_violation`,
  `unsafe_integer_precision` e `missing_required` vêm do normalizador de produção.
  A exceção já contém o relatório de todas as colunas do bloco; é capturada para
  continuar, sem relaxar o comportamento de produção.
- Os campos negativos, integralidade e portas são detalhamentos da violação de
  domínio. Eles podem se sobrepor e não devem ser somados para contar células.
  Integralidade aplica-se a contagens, códigos integrais, portas e `label`.
  `invalid_label` inclui ausências obrigatórias e demais erros da coluna.
- Exemplos: no máximo cinco tokens exemplificados por regra/coluna/arquivo e no
  agregado, com posição zero-based do registro lógico no arquivo, excluído o
  cabeçalho. Não é número de linha física, pois CSV pode ter campos multilinha.
  A contagem é completa; os exemplos são limitados e não são um inventário de
  todos os tokens distintos. Valores exibidos são limitados a 80 caracteres.

## Evidências e limites

Cada diretório final contém `audit.json`, `by_column.csv`, `by_file.csv`,
`by_file_column.csv` e um JSON por partição com colunas presentes/ausentes,
tipos físicos, contagens, exemplos e verificação de metadados de origem.
`by_file_column.csv` fornece o cruzamento completo solicitado, sem amostragem.

A auditoria responde à compatibilidade estrutural e de valores com esta versão
do esquema. Não valida a veracidade de cada registro, ontologias completas de
texto, consistência entre `label` e `type`, vazamento, splits ou validade de
uma comparação experimental. Compatibilidade de contagens CSV/Parquet não é
prova de igualdade célula a célula. Nenhum desses limites altera a decisão sobre
violações explicitamente detectadas.

O relatório usa tabelas para consulta exata por coluna e arquivo e um gráfico de
barras das três partições afetadas (partição no eixo X, falhas no eixo Y; fonte:
`comparison.json`). Ele mostra concentração de contagens, sem confundi-las com
taxas; as 20 partições sem erro são explicitamente informadas no texto. A organização segue
resumo técnico, evidências, escopo/definições, método, limites e próximos passos;
perguntas de proveniência ficam junto às regras que ainda exigem decisão.

Validação pequena do código de auditoria: dois testes em
`tests/test_schema_audit.py`, incluindo continuidade após erro, BOM CSV, tipos
Arrow, preservação dos arquivos e detalhamento de domínio. Os 53 testes de
esquema também passaram. As fixtures verificam pré-processamento e estatísticas
em dados pequenos; nenhum classificador IDS foi treinado neste checkpoint.
`review_aggregates.ipynb` permite revisar os agregados sem reler os datasets.

Uma verificação complementar releu somente `src_bytes` do CSV da partição 1,
em blocos, para contar exatamente os tokens que falham em `pd.to_numeric`,
excluindo as duas sentinelas cadastradas. O resultado está em
`src_bytes_invalid_tokens_partition_1.json`: 175 ocorrências de `0.0.0.0`,
com cinco posições exemplificadas. Essa releitura não soma instâncias ao censo.
Após a conclusão dos Parquets, a verificação foi estendida às três partições
afetadas (1, 22 e 23), somando 869 ocorrências do mesmo token. O código está em
`verify_invalid_src_bytes.py` e o resultado completo em
`invalid_token_verification.json`.

O HTML foi validado e empacotado pelo renderer da skill: validação e verificação
estrutural passaram. Não havia Chromium headless instalado, portanto não houve
QA visual ou de interações em navegador. O arquivo inclui representação
semântica de leitura/impressão. Nenhum navegador foi instalado.

Para satisfazer a proveniência exigida pelo renderer, somente os agregados
prontos foram colocados em `report_tables.sqlite`; as consultas de apresentação
estão em `report_queries.sql`. Isso não relê registros brutos nem altera a
auditoria Python. `build_checkpoint_report.py` reproduz essa preparação.
