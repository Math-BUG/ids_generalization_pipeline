# Quarentena TON_IoT: Incremento 1

A política `ton_iot_known_corruption/1.0.0` define uma única regra, `quarantine_src_bytes_ipv4_literal`: excluir da população elegível toda observação cujo campo **src_bytes seja exatamente o texto `0.0.0.0`**. A justificativa é sua incompatibilidade com a variável quantitativa e a investigação forense D, sem recuperação comprovada. O número 869 é uma expectativa do checkpoint real, nunca uma constante usada para decidir a seleção.

A decisão não consulta `src_ip`, `label`, `type`, partição, tempo, modelo ou métricas. O token em outra coluna não ativa a regra. `src_ip="0"` e `src_ip="0.0.0.0"` isoladamente não ativam a regra. Não há reparo, imputação, normalização desse token, deduplicação, clipping ou ampliação de sentinelas.

`ton_iot_network/1.0.0` permanece inalterado e estrito: `src_bytes` é quantitativa, finita, integral e não negativa quando presente. Qualquer outro token inválido, infinito ou violação de domínio continua chegando ao normalizador e interrompe a carga. Quarentena de corrupção conhecida não é autorização para descartar todo erro de schema.

## Integração e identidade

`load_dataset` executa a política por arquivo antes da normalização e antes de devolver dados ao orquestrador. Consequentemente, nenhum registro quarentenado chega aos splits, ao preprocessing, ao clustering ou ao treinamento. O conversor CSV/Parquet não foi alterado: esta política define elegibilidade experimental, não uma nova versão dos arquivos de origem.

CPU e GPU usam a mesma implementação da política; o caminho cuDF lê cada arquivo separadamente para manter sua proveniência. Não são adicionadas colunas de proveniência que possam virar features. A ordem relativa e o índice pandas das linhas sobreviventes são preservados na aplicação da política. Em múltiplos arquivos, o loader mantém o ordinal original global, incluindo as lacunas dos registros quarentenados. Índices duplicados não levam à remoção de outras linhas.

A identidade reproduzível é `(source_id, row_position_0based)`: caminho relativo do arquivo dentro da raiz comum e posição lógica original após o cabeçalho, sem depender de UID, índice pandas ou do conteúdo de label/type. O manifesto informa a quantidade original e todas as posições quarentenadas de cada arquivo; as posições elegíveis são o complemento, na ordem original. Todas as cópias duplicadas continuam tendo suas próprias identidades.

O `population_id` é um SHA256 do JSON canônico do manifesto `source-content-and-membership/1`, que contém:

- versão da política, versão do schema e identificador da regra;
- arquivos na mesma ordem lexicográfica já usada pelo loader;
- SHA256 do conteúdo integral de cada fonte, calculado em blocos de 1 MiB;
- quantidade original e posições excluídas de cada arquivo.

O ID independe do tamanho dos chunks, backend e diretório absoluto. É sensível ao conteúdo e à ordem das fontes. CSV e Parquet têm IDs distintos por serem representações físicas diferentes; recompressão ou regravação também muda a proveniência. Trata-se de um identificador da população elegível vinculado às fontes, não de um hash canônico dos valores normalizados nem de uma prova de veracidade do tráfego. Alterar label/type no arquivo muda seu SHA256, mas **não muda a decisão de quarentena**.

A opção de amostragem pequena `sample_size` já existente continua posterior à qualidade. O relatório e `data_quality_population_id` identificam a população elegível completa **antes dessa opção**, não uma eventual subamostra de depuração. Nenhuma nova política de amostragem foi introduzida.

## Relatórios

O orquestrador grava `data_quality_report.json` e inclui o ID em `dataset_info.json`. A API aceita `data_quality_report_path`; se omitido, usa o diretório de `schema_report_path`, ou `artifacts/data_quality/<hash>.json` no diretório corrente quando nenhum caminho de relatório foi informado. Também disponibiliza o relatório em `df.attrs["data_quality_report"]`.

O relatório contém original/quarentenado/elegível, regra, contagens e posições por arquivo, fingerprints das fontes, ID da população e distribuições posteriores por label/type. Os alvos são contados somente depois da decisão de exclusão e são apresentados como texto observado, com nulos separados. Esses totais não são critérios de elegibilidade. O relatório de qualidade registra a aplicação da regra, não substitui o diagnóstico de schema: outros erros ainda podem impedir a população de ser usada.

## Auditoria integral, sem experimentos

`scripts/audit_ton_iot_quality.py` percorre exclusivamente os arquivos nomeados `Network_dataset_1..23` no diretório indicado, em blocos de até 100 mil registros. Reutiliza a política de produção e o normalizador estrito. A auditoria captura diagnósticos para continuar a contagem; nunca fornece frames inválidos ao pipeline nem escreve nas fontes.

```powershell
python -B scripts/audit_ton_iot_quality.py --input-dir D:/IC/Dataset/dados/dataset_parquet_total --output-dir reports/ton_iot_quarantine_checkpoint/parquet --format parquet
python -B scripts/audit_ton_iot_quality.py --input-dir D:/IC/Dataset --output-dir reports/ton_iot_quarantine_checkpoint/csv --format csv
```

Escolha uma pasta de saída nova para repetir a auditoria; o script recusa sobrescrever auditorias existentes. O relatório de qualidade é atualizado durante a varredura; só o `audit.json` final com `complete=true` e `compatible=true` certifica a conclusão. O código de saída é diferente de zero se houver violações residuais. Os arquivos de origem não são modificados, reconvertidos ou deduplicados.

O critério de fechamento é zero violações do schema após a quarentena. Isso não acrescenta validação de sintaxe IP nem certifica rótulos, eventos ou políticas posteriores. O Incremento 2 permanece separado.
