# Forense das 869 linhas TON_IoT

Resultado: **D. sem evidência suficiente**, para todas as 869 linhas. Consulte [report.html](report.html) para o parecer, exemplos completos, limitações e tratamento proposto. Nenhuma correção foi aplicada ao dataset ou ao pipeline.

- [records_evidence.jsonl](records_evidence.jsonl): 869 registros CSV originais completos, campos, partição, índice lógico a partir de zero, linhas físicas a partir de um, delimitadores/aspas e SHA256 do texto de cada registro.
- [target_rows.csv](target_rows.csv): as mesmas identidades e campos, em formato tabular de evidência; não é dataset corrigido.
- [findings.json](findings.json), [extra_findings.json](extra_findings.json): agregados, diagnósticos do esquema e três subpadrões.
- [csv_reinspection.json](csv_reinspection.json): segunda leitura dos CSVs originais, igualdade integral das evidências e controles do token nas colunas IP.
- [sources_and_comparisons.json](sources_and_comparisons.json): caminhos físicos, metadados e sete comparações CSV/Parquet. Apenas a forma textual de duration difere; os valores normalizados coincidem.
- [domains_by_column.csv](domains_by_column.csv): todas as 46 colunas presentes nas três partições. `uid` está ausente no CSV. Diagnóstico de sintaxe IP é adicional ao esquema atual, sem alterá-lo.
- `summary_*.csv`: distribuições completas de partição, label, type, dia UTC, origem/destino, protocolo, serviço, estado e campos de porta.
- [patterns.csv](patterns.csv): cruzamento de partição, alvos, endpoints, protocolo, serviço e estado, com intervalo temporal e posição representativa.
- [recovery_sources.json](recovery_sources.json): inventário por nomes de fontes anteriores e resultados da busca na amostra CSV local de 211.043 linhas. Dois destinos coincidentes não constituem correspondência recuperável.
- [verification.json](verification.json): 33 verificações de consistência aprovadas. Os hashes usam o checkpoint CSV mais recente, que já incorpora a versão final do auditor anterior.
- [review_evidence.ipynb](review_evidence.ipynb): revisão compacta sem ler novamente o dataset.

Os scripts [investigate.py](investigate.py) e [complete_evidence.py](complete_evidence.py) leem somente as fontes específicas autorizadas; escrevem apenas nesta pasta. Reexecução completa, a partir da raiz do repositório:

```powershell
python -B reports/ton_iot_src_bytes_forensics/investigate.py
python -B reports/ton_iot_src_bytes_forensics/complete_evidence.py
python -B reports/ton_iot_src_bytes_forensics/build_report.py
```

O primeiro script reutiliza evidências somente se houver metadados de reinspeção compatíveis; o segundo sempre reinspeciona os CSVs. Os metadados tamanho/mtime não substituem hashes integrais dos arquivos. A geração do relatório usa apenas os artefatos já revisados. As consultas em report_queries.sql foram efetivamente executadas sobre tabelas de apresentação e salvas em report_tables.sqlite; a investigação dos dados originais foi feita em Python.

A validação e o empacotamento HTML passaram. A verificação automática foi `structural_only`: o ambiente não disponibilizou Chromium headless, portanto a interação e a aparência em navegador não foram verificadas. O HTML inclui a representação semântica de tabelas para leitura alternativa.

Fontes primárias consultadas em 05/09/2026:

- [TON_IoT / UNSW](https://research.unsw.edu.au/projects/toniot-datasets): descreve coleta em PCAP e logs/CSV Zeek e processamento posterior; não explica o token anômalo. O acesso ao link SharePoint de distribuição falhou nesta sessão; os arquivos remotos originais não foram lidos.
- [Data_Analysis.R do autor](https://github.com/Nour-Moustafa/TON_IoT-Network-dataset/blob/master/Data_Analysis.R): código de análise da amostra, não proveniência completa dos CSVs de rede.
- [Artigo Network TON_IoT](https://www.sciencedirect.com/science/article/abs/pii/S2210670721002808), também consultado na página dos autores no ResearchGate: descrição do testbed e features; nenhuma explicação encontrada para a associação src_ip="0" / src_bytes="0.0.0.0".
- [Zeek ICMP.cc](https://github.com/zeek/zeek/blob/master/src/packet_analysis/protocol/icmp/ICMP.cc), funções InitConnKey e ICMP6_counterpart: pares de pseudportas ICMP. Código atual serve como referência contextual; não prova a versão usada em 2019.
- [Zeek conn fields](https://docs.zeek.org/en/current/scripts/base/protocols/conn/main.zeek.html): distinção entre contadores de bytes e bytes IP.
- [RFC 4861](https://www.rfc-editor.org/rfc/rfc4861.html): Router/Neighbor Solicitation e endereçamento.
- [RFC 4862, 5.4.2](https://datatracker.ietf.org/doc/html/rfc4862#section-5.4.2): origem não especificada em DAD.
- [RFC 3810](https://www.rfc-editor.org/rfc/rfc3810.html): ICMPv6 tipo 143, destino dos relatórios MLDv2 e condições para origem não especificada.
- [RFC 2131](https://www.rfc-editor.org/rfc/rfc2131.html): contexto DHCP para origem IPv4 não configurada.

Não foi encontrada explicação explícita para o token nas fontes consultadas; isso não prova que inexista documentação adicional. Os nomes de arquivos sob D:/IC/Dataset foram inventariados, excluindo ambientes virtuais. Nenhum PCAP/log/arquivo compactado candidato foi encontrado nesse escopo. Outros discos e o conteúdo de pickles não foram pesquisados.
