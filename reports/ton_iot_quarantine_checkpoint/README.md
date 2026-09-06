# Incremento 1 encerrado: quarentena de corrupção conhecida

**Zero violações de ton_iot_network/1.0.0 na população elegível após a quarentena.**

| Representação | Partições | Originais | Quarentenadas | Elegíveis | Violações restantes |
|---|---:|---:|---:|---:|---:|
| Parquet do projeto | 23 | 22.339.021 | 869 | 22.338.152 | 0 |
| CSV original | 23 | 22.339.021 | 869 | 22.338.152 | 0 |

As exclusões coincidem exatamente com as 869 identidades da investigação forense: 175 na partição 1, 690 na 22 e 4 na 23. Nenhuma exclusão adicional. Nenhuma deduplicação. Nenhuma escrita ou reconversão dos arquivos fonte. O esquema e o normalizador continuam com os mesmos hashes do Incremento 1 anterior à quarentena.

A regra em ton_iot_known_corruption/1.0.0 consulta exclusivamente src_bytes == "0.0.0.0". IPs, alvos e desempenho não participam da decisão. O loader aplica a regra antes de normalizar e antes de disponibilizar qualquer população aos estágios posteriores. O pipeline continua abortando se o restante contiver qualquer outro valor inválido.

Os relatórios de [qualidade Parquet](parquet/data_quality_report.json) e [qualidade CSV](csv/data_quality_report.json) registram regra, contagens, posições originais por arquivo, distribuições posteriores e identificadores. Os relatórios de [schema Parquet](parquet/audit.json) e [schema CSV](csv/audit.json) registram zero violações em todas as colunas presentes, integridade dos metadados e conclusão da varredura. by_file.csv de cada pasta resume os 23 arquivos.

As auditorias processaram até 100 mil linhas por bloco. O tamanho/mtime das fontes permaneceu igual durante a leitura. Os SHA256 integrais das fontes foram calculados incrementalmente. Os IDs abaixo diferem de forma esperada porque vinculam a população à representação física, aos nomes e à ordem dos arquivos; não são hashes de valores normalizados. As identidades excluídas e distribuições posteriores CSV/Parquet coincidem.

- **parquet:** `sha256:81cf4c6e2c8b9a887a6542fbd7d10a0d0b1af4c068e869cd6810658b82cf6c99`
- **csv:** `sha256:9ce48007cbda2e836076af2abbfd02ebc765b7d04c7002de625ab6d205682dc2`

Distribuição posterior por label: {"0": 795511, "1": 21542641}.

| type | Elegíveis |
|---|---:|
| backdoor | 508,116 |
| ddos | 6,165,008 |
| dos | 3,375,328 |
| injection | 452,659 |
| mitm | 1,052 |
| normal | 795,511 |
| password | 1,718,568 |
| ransomware | 72,805 |
| scanning | 7,140,161 |
| xss | 2,108,944 |

**74 testes relevantes passaram** (19 casos novos de quarentena e 55 testes existentes de schema/auditoria), com registro em [tests.xml](tests.xml). A revisão cruzada das evidências passou em **121 verificações**, registradas em [verification.json](verification.json). Nenhuma divergência inesperada foi encontrada. O caminho GPU foi testado com leitura simulada, sem exigir ou executar hardware GPU.

Arquivos de implementação desta etapa: src/ids_pipeline/data_quality_policy.py (novo); src/ids_pipeline/data_loading.py; src/ids_pipeline/cli.py; scripts/audit_ton_iot_quality.py (novo); tests/test_data_quality_policy.py (novo); tests/test_dataset_schema.py (caminho explícito do relatório no teste); docs/ton_iot_data_quality_policy.md (novo).

O identificador da qualidade se refere à população elegível completa, anterior à opção sample_size de depuração já existente. Não foram alterados behavioral_strict, splits, orçamento, seleção, clustering ou profiling. Não houve treinamento. cluster_assignments.csv não foi lido.

Compatibilidade científica: resultados anteriores à quarentena podem se referir a uma população diferente; futuros resultados devem registrar o ID desta população. O sucesso deste checkpoint demonstra conformidade com o schema atual, não veracidade independente de IPs, rótulos ou eventos. O Incremento 2 permanece fora do escopo.
