"""Publish reviewed forensic evidence only; does not reread the source dataset."""
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))
f, extra = read('findings.json'), read('extra_findings.json')
controls, recovery = read('csv_reinspection.json'), read('recovery_sources.json')
records = [json.loads(x) for x in (ROOT/'records_evidence.jsonl').read_text(encoding='utf-8').splitlines()]
checks = []
def check(name, result):
    checks.append({'check': name, 'passed': bool(result)})
    assert result, name
check('869 target rows', len(records) == f['target_rows'] == 869)
check('unique partition and logical row identities', len({(r['partition'],r['row_0based']) for r in records}) == 869)
check('all 46 raw fields match header; 45 delimiters; no quotes', all((r['field_count'],r['header_count'],r['comma_count'],r['quote_count']) == (46,46,45,0) for r in records))
check('all complete raw records roundtrip strictly', all(next(csv.reader(io.StringIO(r['raw_csv']),strict=True)) == list(r['fields'].values()) for r in records))
check('all raw record hashes match', all(hashlib.sha256(r['raw_csv'].encode()).hexdigest() == r['raw_sha256'] for r in records))
check('all records remain identical on second original CSV read', all(c['all_cached_records_identical_to_reinspection'] and c['source_unchanged'] for c in controls))
check('869 invalid source IPs; 869 valid destinations', f['valid_ip_counts'] == {'src_ip':0,'dst_ip':869})
check('only src_bytes fails current production rules', {k:v['invalid_count'] for k,v in f['schema_report']['columns'].items() if v['invalid_count']} == {'src_bytes':869})
check('all label/type pairs consistent', f['label_type_consistent'] == 869)
for dimension, rows in f['summaries'].items():
    check('summary reconciles: '+dimension, sum(r['count'] for r in rows) == 869)
for i,c in enumerate(f['parquet_comparisons']):
    check('Parquet comparison '+str(i+1), not c['semantic_mismatches'] and not c['missing_parquet_columns'] and len(c['common_columns']) == 46)
check('all three pattern counts reconcile', sorted(p['count'] for p in extra['patterns']) == [6,39,824])
check('partition pattern counts reconcile', all(sum(p['partitions'].values()) == p['count'] for p in extra['patterns']))
check('869 records and 562 distinct full rows', extra['exact_duplicate_rows_beyond_first'] == 307 and f['unique_ts_5tuples'] == 562)
prior = json.loads((REPO/'reports/ton_iot_schema_checkpoint/csv_complete/audit.json').read_text())
for name, expected in prior['code_sha256'].items():
    check('production/checkpoint SHA256 unchanged: '+name, hashlib.sha256((REPO/name).read_bytes()).hexdigest() == expected)
(ROOT/'verification.json').write_text(json.dumps({'passed':len(checks),'failed':0,'checks':checks},indent=2),encoding='utf-8')

now = datetime.now(timezone.utc).isoformat()
sources = [{'id':'forensics','label':'869 registros completos, comparação CSV/Parquet e domínios','path':'findings.json'},
           {'id':'extra','label':'Padrões, períodos e duplicatas nas 869 linhas','path':'extra_findings.json'},
           {'id':'controls','label':'Reinspeção dos CSVs e controles de IP','path':'csv_reinspection.json'},
           {'id':'recovery','label':'Inventário local de fontes e amostra Train_Test','path':'recovery_sources.json'},
           {'id':'raw','label':'Texto CSV original de todas as 869 linhas, posições e hashes','path':'records_evidence.jsonl'},
           {'id':'checks','label':'Verificações de consistência e hashes do esquema','path':'verification.json'}]
blocks, tables, charts, datasets, queries = [], [], [], {}, []
database = sqlite3.connect(':memory:')
database.row_factory = sqlite3.Row
def paragraph(name, body, source=None):
    blocks.append({'id':name,'type':'markdown','body':body,**({'sourceId':source} if source else {})})
def dataset(name, rows, sort, direction='ASC'):
    fields = list(rows[0])
    table_name='forensic_'+name
    declarations=', '.join('"'+k+'" '+('TEXT' if isinstance(rows[0][k],str) else 'REAL') for k in fields)
    database.execute(f'CREATE TABLE "{table_name}" ({declarations})')
    database.executemany(f'INSERT INTO "{table_name}" VALUES ({",".join("?" for _ in fields)})',[tuple(r[k] for k in fields) for r in rows])
    sql=f'SELECT * FROM "{table_name}" ORDER BY "{sort}" {direction};'
    datasets[name]=[dict(r) for r in database.execute(sql)]
    queries.append(sql)
    sid=name+'_sql'
    sources.append({'id':sid,'label':'Evidência forense: '+name,'path':'report_queries.sql',
                    'query':{'engine':'sqlite','language':'sql','sql':sql,'executed_at':now,'tables_used':[table_name],
                             'description':'Consulta executada sobre agregados revisados em report_tables.sqlite. Build_report.py carrega findings.json, extra_findings.json, domains_by_column.csv e csv_reinspection.json; investigação original em Python nos scripts locais. Não consulta o dataset original nesta etapa.'}})
    return sid
def table(name,title,rows,labels,sort,direction='asc'):
    sid=dataset(name,rows,sort,direction.upper())
    tables.append({'id':name,'title':title,'dataset':name,'sourceId':sid,'defaultSort':{'field':sort,'direction':direction},
                   'columns':[{'field':k,'label':v,**({'type':'text'} if isinstance(rows[0][k],str) else {'format':'number'})} for k,v in labels]})
    blocks.append({'id':name+'_block','type':'table','tableId':name})

title='Forense das 869 linhas TON_IoT'
paragraph('title','# '+title)
paragraph('summary','## Parecer: D. sem evidência suficiente\n\n'
          '**A classificação D aplica-se às 869 linhas, nos três padrões encontrados.** A estrutura CSV está íntegra, mas existem dois campos suspeitos: src_bytes="0.0.0.0" e src_ip="0". '
          'As cópias Parquet preservam a anomalia. Há evidência suficiente para rejeitar o valor quantitativo e a representação textual do IP; falta a fonte anterior ao CSV para estabelecer a causa, recuperar valores ou atestar o restante da linha. '
          'O parecer não autoriza transformar o token em zero ou numa sentinela global. Nenhuma regra de ton_iot_network/1.0.0 foi alterada.', 'forensics')
paragraph('structure','## O CSV não apresenta deslocamento sequencial detectável\n\n'
          'Todas as **869 linhas** têm **46 campos para 46 nomes no cabeçalho**, 45 vírgulas, nenhuma aspa e exatamente uma linha física. '
          'O parser CSV estrito aceita cada registro; a reconstrução dos campos coincide com o texto original. '
          'A ordem local é service → duration → src_bytes → dst_bytes → conn_state → missed_bytes. Os campos quantitativos e categóricos vizinhos ocupam posições coerentes. '
          'Um campo extra ou ausente por delimitador não explica a anomalia observada. Contagem correta de campos não exclui troca não contígua, substituição ou erro anterior à gravação.', 'forensics')
paragraph('domain','## src_ip também é suspeito; as outras colunas apenas passam nos domínios verificados\n\n'
          'src_bytes falha como número em 869/869 registros. src_ip contém o texto "0" em 869/869: ele falha em um parser estrito de IPv4/IPv6. '
          'O esquema atual trata IP como identificador textual e ainda não verifica sua sintaxe, razão pela qual o checkpoint anterior encontrou apenas src_bytes. '
          'Todos os destinos são endereços IPv6 multicast válidos. Nas outras 44 colunas presentes, não foi detectada violação dos domínios testados. '
          'Isso não equivale a confirmar os valores contra o tráfego capturado; categorias abertas e sentinelas não comprovam conteúdo correto.', 'forensics')

pattern_names={'135/136':'Neighbor Solicitation (hipótese)','133/134':'Router Solicitation (hipótese)','143/0':'MLDv2 Report (hipótese)'}
pattern_rows=[{'pattern':p['pattern'],'interpretation':pattern_names[p['pattern']],'rows':p['count'],'destinations':p['destinations'],
               'partition1':p['partitions'].get('1',0),'partition22':p['partitions'].get('22',0),'partition23':p['partitions'].get('23',0)} for p in extra['patterns']]
paragraph('patterns','## A anomalia acompanha três padrões de controle IPv6\n\n'
          'Todos os registros têm label=0, type=normal, proto=icmp, service="-" e conn_state=OTH. '
          'Os três grupos abaixo compartilham a mesma assinatura de campos suspeitos; são subpadrões de tráfego, não três causas demonstradas.', 'forensics')
table('patterns','Subpadrões das 869 linhas',pattern_rows,[('pattern','src_port / dst_port'),('interpretation','Interpretação candidata'),('rows','Linhas'),('destinations','Destinos distintos'),('partition1','Partição 1'),('partition22','Partição 22'),('partition23','Partição 23')],'rows','desc')
charts.append({'id':'patterns_chart','title':'Neighbor Solicitation concentra 824 das 869 linhas','type':'bar','dataset':'patterns','sourceId':'patterns_sql','valueFormat':'number',
               'encodings':{'x':{'field':'pattern','type':'nominal','label':'Campos src_port / dst_port'},'y':{'field':'rows','type':'quantitative','label':'Linhas','format':'number'}}})
blocks.append({'id':'pattern_chart_block','type':'chart','chartId':'patterns_chart'})
paragraph('protocol','## Esses números não identificam serviços TCP/UDP\n\n'
          'O [código atual do Zeek, ICMP6_counterpart](https://github.com/zeek/zeek/blob/master/src/packet_analysis/protocol/icmp/ICMP.cc) usa o tipo complementar para pares de solicitações/respostas ICMPv6: isso explica a coerência dos pares 135/136 e 133/134; outros tipos podem usar o código. '
          'A versão exata empregada na geração de 2019 não foi comprovada. Não interpretar dst_port=136 como prova de código ICMP inválido.\n\n'
          'Pelos [formatos de Neighbor Discovery](https://www.rfc-editor.org/rfc/rfc4861.html), 135 corresponde a Neighbor Solicitation e 133 a Router Solicitation. '
          'Os 824 destinos do primeiro grupo estão no espaço solicited-node; os 39 do segundo são ff02::2. '
          'Os seis registros 143/0 usam ff02::16, consistente com [MLDv2](https://www.rfc-editor.org/rfc/rfc3810.html). '
          'Os padrões são compatíveis com inicialização/autoconfiguração IPv6. [DAD](https://datatracker.ietf.org/doc/html/rfc4862#section-5.4.2), Router Solicitation e MLDv2 podem envolver origem não especificada sob condições definidas. '
          '**Isso torna plausível uma falha no tratamento dessa origem, mas não prova que src_ip era :: nem que src_bytes era zero.**')

paragraph('time','## 690 ocorrências estão concentradas em menos de três horas\n\n'
          'A partição 22 concentra 690/869 (79,40%) entre 03:49:58 e 06:31:56 UTC de 28/04/2019. '
          'As demais aparecem em 02–04/04 e 29/04. Datas calculadas interpretando ts como segundos Unix, apresentadas em UTC; não foi presumido o fuso local do laboratório.', 'extra')
table('partition_times','Períodos exatos por partição',extra['partition_times'],[('partition','Partição'),('count','Linhas'),('first_ts','ts mínimo'),('last_ts','ts máximo'),('first_utc','Primeiro UTC'),('last_utc','Último UTC')],'partition')
table('days','Distribuição diária UTC',f['summaries']['date_utc'],[('value','Data UTC'),('count','Linhas')],'value')
paragraph('endpoints','## Não é possível atribuir essas linhas a um único dispositivo\n\n'
          'Há 233 destinos distintos e 233 combinações distintas de src_ip, src_port, dst_ip, dst_port e proto. A origem foi perdida ou mal representada; destinos multicast identificam grupos, não dispositivos individuais. '
          'Há 562 combinações distintas de ts mais essa tupla e 307 repetições adicionais de linhas completas. Sem uid nos CSVs afetados e com ts em segundos, essas repetições não permitem decidir entre duplicação do dataset e eventos indistinguíveis na representação. '
          'Todas as 869 identidades originais foram preservadas; não houve deduplicação. label=0/type=normal é o rótulo registrado, cuja veracidade independente não foi reconstituída.', 'forensics')
table('destinations','Dez destinos mais frequentes; lista completa em summary_dst_ip.csv',f['summaries']['dst_ip'][:10],[('value','dst_ip'),('count','Linhas')],'count','desc')

paragraph('examples','## Três exemplos mantêm a identidade e os vizinhos originais\n\n'
          'Um exemplo por subpadrão. A linha física é contada a partir de 1, incluindo o cabeçalho; row_0based é o índice lógico do registro após o cabeçalho. '
          'Os registros completos e seus hashes estão em records_evidence.jsonl.', 'raw')
example_rows=[]
for p in extra['patterns']:
    r=p['example']
    example_rows.append({'pattern':p['pattern'],'partition':r['partition'],'line':r['physical_line'],'row_0based':r['row_0based'],'ts':r['ts'],'src_ip':r['src_ip'],'dst_ip':r['dst_ip'],'duration':r['duration'],'src_bytes':r['src_bytes'],'dst_bytes':r['dst_bytes'],'src_pkts':r['src_pkts'],'src_ip_bytes':r['src_ip_bytes']})
table('examples','Exemplos representativos',example_rows,[(k,k) for k in example_rows[0]],'pattern')
paragraph('raw_examples','### Linhas CSV originais completas dos três exemplos\n\n'+ '\n\n'.join('Padrão '+p['pattern']+'; partição '+str(p['example']['partition'])+'; linha física '+str(p['example']['physical_line'])+':\n\n```csv\n'+next(r['raw_csv'].strip() for r in records if r['partition']==p['example']['partition'] and r['row_0based']==p['example']['row_0based'])+'\n```' for p in extra['patterns']), 'raw')
paragraph('neighbors','## Contadores vizinhos não fornecem recuperação determinística\n\n'
          'duration varia de 0 a 29,075908 segundos; 773 registros têm duração zero. src_pkts é 1 em 773, 2 em 94 e 3 em 2 registros. '
          'dst_bytes, dst_pkts, dst_ip_bytes e missed_bytes são zero em todos. Os contadores passam em finitude, não negatividade e integralidade quando exigida. '
          'Os booleanos DNS/SSL/weird estão ausentes por "-"; os campos de aplicação apresentam sentinelas e zeros compatíveis com ausência de sessões dessas aplicações. '
          'Nenhum desses resultados estabelece o conteúdo original de src_bytes. '
          'A [documentação do Zeek](https://docs.zeek.org/en/current/scripts/base/protocols/conn/main.zeek.html) distingue orig_bytes de orig_ip_bytes; não é defensável substituir src_bytes por src_ip_bytes ou deduzir payload subtraindo um cabeçalho fixo sem os pacotes e a lógica de extração.')

paragraph('controls_heading','## 0.0.0.0 aparece em um contexto de IP legítimo\n\n'
          'Nos três CSVs afetados, a varredura de controle encontrou 77 ocorrências do token em src_ip, todas na partição 1; nenhuma em dst_ip. '
          'Os cinco exemplos preservados apresentam origem 0.0.0.0, destino 255.255.255.255, UDP 68→67, service=dhcp e src_bytes=0. '
          'Esse uso de origem antes da configuração do cliente é previsto pelo [DHCP](https://www.rfc-editor.org/rfc/rfc2131.html). '
          'Os 77 são a contagem de IPs com esse token nos arquivos 1/22/23; não são um censo novo das 23 partições nem uma validação dos pacotes DHCP. '
          'O token não deve tornar-se sentinela global de ausência.')
table('zero_ip_examples','Três controles de IP fora da coorte anômala',controls[0]['zero_ip_examples'][:3],[(k,k) for k in controls[0]['zero_ip_examples'][0]],'row_0based')

paragraph('recovery_heading','## Nenhuma representação local examinada recuperou os valores\n\n'
          'Foram confrontadas as 869 posições originais com os três Parquets principais e com os três arquivos correspondentes da cópia dataset_parquet; o Parquet adicional na raiz existe para a partição 1. '
          'São sete comparações por arquivo, com 46 campos comuns examinados em cada registro selecionado. Não houve divergência semântica; a representação textual de duration difere após armazenamento numérico. '
          'A coluna uid adicional nos arquivos principais não recupera identidade de origem. As cópias derivadas não constituem fontes independentes de verdade.', 'forensics')
compare_rows=[{'partition':c['partition'],'representation':Path(c['source']).parent.name,'rows':c['rows_compared'],'common_fields':len(c['common_columns']),'mismatches':len(c['semantic_mismatches'])} for c in f['parquet_comparisons']]
table('comparisons','Comparações CSV versus Parquet',compare_rows,[('partition','Partição'),('representation','Diretório da representação'),('rows','Linhas comparadas'),('common_fields','Campos'),('mismatches','Linhas divergentes')],'partition')
paragraph('recovery_limits','### A amostra não oferece correspondência confiável\n\n'
          'O train_test_network.csv local foi lido em blocos de 50 mil: tem 211.043 linhas e 44 colunas, sem ts ou uid. '
          'Não contém src_ip="0" nem src_bytes="0.0.0.0". Apenas duas linhas compartilham um destino da coorte, ff02::2, mas apresentam src_ip=fe80::ffff:ffff:ffff. '
          'Sem chave e com endereço divergente, não são evidência de correção das linhas investigadas. '
          'O inventário de nomes sob D:/IC/Dataset, excluindo ambientes virtuais, não encontrou candidatos PCAP/conn.log/Zeek ou arquivos compactados de origem. Nenhum pickle grande foi carregado.', 'recovery')
paragraph('generation','## As fontes públicas explicam o contexto, mas não o token corrompido\n\n'
          'A [página oficial TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) descreve rede coletada em PCAP e logs/CSV do Zeek e posterior filtragem para CSV. '
          'O arquivo [Data_Analysis.R do autor](https://github.com/Nour-Moustafa/TON_IoT-Network-dataset/blob/master/Data_Analysis.R) analisa a amostra e transforma variáveis para análises; não documenta a geração das 23 partições nem uma regra que explique este token em src_bytes. '
          'O [artigo da arquitetura Network TON_IoT](https://www.sciencedirect.com/science/article/abs/pii/S2210670721002808) contextualiza o testbed e as features; não foi localizada uma explicação para o par anômalo nas fontes consultadas. '
          'O link oficial de distribuição via SharePoint falhou no acesso disponível nesta sessão, de modo que os logs/PCAP remotos não foram inspecionados. '
          'O conversor local D:/IC/Dataset/a.py apenas lê CSV, remove BOM do cabeçalho e grava Parquet; não contém reparo ou troca desses campos. '
          'A correspondência com o CSV demonstra que a anomalia antecede as conversões examinadas, mas não identifica qual etapa anterior a introduziu.')

paragraph('scope','## Método: censo da coorte, com leituras incrementais\n\n'
          'A população investigada é exatamente src_bytes igual ao token literal "0.0.0.0" nas partições 1/22/23, identificadas pelo checkpoint anterior. '
          'As três fontes CSV foram percorridas sequencialmente para localizar os registros; somente a coorte, controles limitados e agregados foram retidos. '
          'Cada linha original da coorte foi reinspecionada, com igualdade integral dos registros salvos. Parquets foram acessados somente nos grupos que contêm as posições-alvo, em batches de até 50 mil linhas; cada grupo pode cobrir uma partição inteira. '
          'Os 46 campos comuns foram comparados semanticamente pelo normalizador de produção, separando a comparação literal do token inválido de src_bytes. '
          'As 23 partições completas não foram carregadas simultaneamente nem submetidas novamente à auditoria global. O arquivo proibido pelo usuário não foi lido. Não houve treinamento ou execução de experimento.')
domain_rows=list(csv.DictReader((ROOT/'domains_by_column.csv').open(encoding='utf-8',newline='')))
domain_table=[{'column':r['column'],'semantic_type':r['semantic_type'],'schema_invalid':int(r['schema_invalid']),'ip_invalid':int(r['extra_ip_syntax_invalid']),'sentinels':int(r['missing_sentinel']),'distinct':int(r['distinct_values']),'examples':r['values_top5']} for r in domain_rows]
table('domains','Domínios das 46 colunas presentes; uid está ausente nos três CSVs',domain_table,[('column','Coluna'),('semantic_type','Semântica atual'),('schema_invalid','Violações no esquema'),('ip_invalid','Falhas de sintaxe IP adicionais'),('sentinels','Sentinelas'),('distinct','Valores distintos'),('examples','Até cinco valores e contagens')],'column')
paragraph('validation','## Verificações e limites de confiança\n\n'
          f'**{len(checks)} verificações de consistência passaram**, incluindo somas dos recortes, identidade das 869 posições, hashes de cada linha exportada, sete comparações Parquet e hashes de schema.py, dataset_schema.py e do auditor anterior. '
          'Tamanho e mtime dos arquivos lidos permaneceram iguais; isso não equivale a calcular hashes integrais dos arquivos fonte. '
          'A estrutura e os tokens têm alta confiança por observação direta. A relação com autoconfiguração IPv6 é uma hipótese contextual; a atribuição a dispositivo, a correção dos rótulos e a causa de geração permanecem sem comprovação. '
          'Trocar src_ip e src_bytes produziria "0.0.0.0" como origem IPv4 com destino IPv6, sem resolver a inconsistência de família. Tratar "0" como codificação numérica de IP não está documentado no esquema textual. '
          'Essas alternativas não justificam reparo automático.', 'checks')
paragraph('decision','## Tratamento proposto para D, ainda não implementado\n\n'
          'Preservar as 869 linhas e seus valores originais em uma trilha de evidência e mantê-las em quarentena metodológica até confrontar PCAP/conn.log ou obter a regra de transformação original. '
          'Procurar primeiro as janelas UTC e destinos informados; recuperar os campos apenas mediante correspondência inequívoca e proveniência registrada. '
          'Manter src_bytes quantitativa, finita, integral e não negativa. Não cadastrar este IP como sentinela global; não converter para zero, trocar colunas ou preencher src_ip com :: por inferência. '
          'Se a fonte original permanecer inacessível, a opção conservadora para um futuro conjunto principal é excluir explicitamente os registros não verificáveis, com manifesto de identidades e contagens, nunca descartá-los silenciosamente. '
          'A decisão alternativa de mantê-los exige uma política futura de valores corrompidos ao menos para src_bytes e src_ip; ela não equivale a declarar o restante confiável. '
          'Uma análise de sensibilidade posterior deverá documentar o efeito dessa decisão, pois a exclusão atinge somente exemplos rotulados normais e um segmento IPv6 específico. Nada disso foi aplicado nesta investigação.')
paragraph('questions','## Evidência que falta para encerrar o caso\n\n'
          'São necessários os conn.log/PCAP anteriores aos CSVs para as janelas identificadas e o código/versionamento de extração, agregação e rotulagem que produziu as 23 partições. '
          'As perguntas pendentes são: como a origem IPv6 não especificada foi serializada; como orig_bytes foi mapeado para src_bytes em ICMP; e em que etapa surgiram os registros indistinguíveis. '
          'Sem essas evidências, a classificação permanece **D para todas as 869 linhas**. O Incremento 2 permanece fora do escopo.')

artifact={'surface':'report','manifest':{'version':1,'surface':'report','title':title,'generatedAt':now,'sources':sources,'blocks':blocks,'tables':tables,'charts':charts},'snapshot':{'version':1,'generatedAt':now,'status':'ready','datasets':datasets}}
(ROOT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'report_queries.sql').write_text('\n'.join(queries),encoding='utf-8')
database.commit()
with sqlite3.connect(ROOT/'report_tables.sqlite') as saved:
    database.backup(saved)
database.close()

# Inspectable companion uses saved evidence only. Running it does not reread or modify sources.
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
    {'cell_type':'markdown','metadata':{},'source':['# Forense TON_IoT: revisão das evidências\n','Resultado: D para as 869 linhas. Este notebook lê somente os artefatos locais; os scripts investigate.py e complete_evidence.py documentam as leituras originais.\n']},
    {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['from pathlib import Path\n','import json, pandas as pd\n','root = Path.cwd()\n',"if not (root / 'findings.json').exists():\n","    root = root / 'reports/ton_iot_src_bytes_forensics'\n","f = json.loads((root/'findings.json').read_text(encoding='utf-8'))\n","extra = json.loads((root/'extra_findings.json').read_text(encoding='utf-8'))\n","assert f['target_rows'] == 869\n","assert all(not c['semantic_mismatches'] for c in f['parquet_comparisons'])\n","assert sum(p['count'] for p in extra['patterns']) == 869\n","display(pd.DataFrame(extra['partition_times']))\n","display(pd.read_csv(root/'domains_by_column.csv', keep_default_na=False))\n"]} ]}
(ROOT/'review_evidence.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'checks_passed':len(checks),'tables':len(tables),'charts':len(charts),'records':len(records)}))
