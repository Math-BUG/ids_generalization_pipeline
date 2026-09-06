"""Reconcile completed audits with forensic identities; no original data reads."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

checks=[]
def check(name, condition):
    checks.append({'check':name,'passed':bool(condition)})
    assert condition,name

records=[json.loads(line) for line in (ROOT.parent/'ton_iot_src_bytes_forensics/records_evidence.jsonl').read_text(encoding='utf-8').splitlines()]
expected={(r['partition'],r['row_0based']) for r in records}
check('869 distinct original forensic identities',len(expected)==869)
reports={}
for fmt in ['parquet','csv']:
    audit=read(ROOT/fmt/'audit.json')
    quality=read(ROOT/fmt/'data_quality_report.json')
    check(fmt+' complete and compatible',audit['complete'] and audit['compatible'] and audit['file_count']==23)
    check(fmt+' count reconciliation',(audit['original_rows'],audit['quarantined_rows'],audit['eligible_rows'])==(22339021,869,22338152))
    check(fmt+' zero residual schema violations',audit['invalid_cells_after_quarantine']==0 and all(v['invalid_cells']==0 for v in audit['columns'].values()))
    observed={(int(Path(f['source_id']).stem.split('_')[-1]),p) for f in quality['files'] for p in f['quarantined_row_positions_0based']}
    check(fmt+' exact forensic identities, no others',observed==expected)
    check(fmt+' report/audit population IDs agree',quality['population_id']==audit['population_id'])
    manifest=json.dumps(quality['population_identity_manifest'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    check(fmt+' population ID independently recomputed',quality['population_id']=='sha256:'+hashlib.sha256(manifest).hexdigest())
    for target,stats in quality['eligible_target_distributions'].items():
        check(fmt+' target total '+target,sum(stats['values'].values())+stats['native_nulls']==22338152)
    for f in quality['files']:
        part=int(Path(f['source_id']).stem.split('_')[-1])
        check(fmt+' partition '+str(part)+' conservation',f['original_rows']==f['eligible_rows']+f['quarantined_rows'])
        check(fmt+' partition '+str(part)+' exact exclusions',f['quarantined_rows']=={1:175,22:690,23:4}.get(part,0))
    for path,digest in audit['code_sha256'].items():
        check(fmt+' audited code unchanged '+path,hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest)
    reports[fmt]={'audit':audit,'quality':quality}
check('CSV and Parquet posterior target distributions agree',reports['csv']['quality']['eligible_target_distributions']==reports['parquet']['quality']['eligible_target_distributions'])
old=read(ROOT.parent/'ton_iot_schema_checkpoint/csv_complete/audit.json')
for path,digest in old['code_sha256'].items():
    if Path(path).name in {'schema.py','dataset_schema.py'}:
        check('strict Increment 1 schema unchanged '+path,hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest)
suite=ET.parse(ROOT/'tests.xml').getroot().find('testsuite')
check('74 relevant tests passed',int(suite.attrib['tests'])==74 and int(suite.attrib['failures'])==0 and int(suite.attrib['errors'])==0)
summary={'passed':len(checks),'failed':0,'checks':checks,'original_rows':22339021,'quarantined_rows':869,'eligible_rows':22338152,
         'population_ids':{fmt:r['quality']['population_id'] for fmt,r in reports.items()},
         'eligible_target_distributions':reports['parquet']['quality']['eligible_target_distributions'],
         'unexpected_divergences':[]}
(ROOT/'verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
text='''# Incremento 1 encerrado: quarentena de corrupção conhecida

**Zero violações de ton_iot_network/1.0.0 na população elegível após a quarentena.**

| Representação | Partições | Originais | Quarentenadas | Elegíveis | Violações restantes |
|---|---:|---:|---:|---:|---:|
| Parquet do projeto | 23 | 22.339.021 | 869 | 22.338.152 | 0 |
| CSV original | 23 | 22.339.021 | 869 | 22.338.152 | 0 |

As exclusões coincidem exatamente com as 869 identidades da investigação forense: 175 na partição 1, 690 na 22 e 4 na 23. Nenhuma exclusão adicional. Nenhuma deduplicação. Nenhuma escrita ou reconversão dos arquivos fonte. O esquema e o normalizador continuam com os mesmos hashes do Incremento 1 anterior à quarentena.

A regra em ton_iot_known_corruption/1.0.0 consulta exclusivamente src_bytes == "0.0.0.0". IPs, alvos e desempenho não participam da decisão. O loader aplica a regra antes de normalizar e antes de disponibilizar qualquer população aos estágios posteriores. O pipeline continua abortando se o restante contiver qualquer outro valor inválido.

Os relatórios de [qualidade Parquet](parquet/data_quality_report.json) e [qualidade CSV](csv/data_quality_report.json) registram regra, contagens, posições originais por arquivo, distribuições posteriores e identificadores. Os relatórios de [schema Parquet](parquet/audit.json) e [schema CSV](csv/audit.json) registram zero violações em todas as colunas presentes, integridade dos metadados e conclusão da varredura. by_file.csv de cada pasta resume os 23 arquivos.

As auditorias processaram até 100 mil linhas por bloco. O tamanho/mtime das fontes permaneceu igual durante a leitura. Os SHA256 integrais das fontes foram calculados incrementalmente. Os IDs abaixo diferem de forma esperada porque vinculam a população à representação física, aos nomes e à ordem dos arquivos; não são hashes de valores normalizados. As identidades excluídas e distribuições posteriores CSV/Parquet coincidem.

'''
for fmt,pid in summary['population_ids'].items():
    text+=f'- **{fmt}:** `{pid}`\n'
text+='\nDistribuição posterior por label: '+json.dumps(summary['eligible_target_distributions']['label']['values'])+'.\n\n'
text+='| type | Elegíveis |\n|---|---:|\n'
for value,n in summary['eligible_target_distributions']['type']['values'].items():
    text+=f'| {value} | {n:,} |\n'
text+=f'''\n**74 testes relevantes passaram** (19 casos novos de quarentena e 55 testes existentes de schema/auditoria), com registro em [tests.xml](tests.xml). A revisão cruzada das evidências passou em **{len(checks)} verificações**, registradas em [verification.json](verification.json). Nenhuma divergência inesperada foi encontrada. O caminho GPU foi testado com leitura simulada, sem exigir ou executar hardware GPU.

Arquivos de implementação desta etapa: src/ids_pipeline/data_quality_policy.py (novo); src/ids_pipeline/data_loading.py; src/ids_pipeline/cli.py; scripts/audit_ton_iot_quality.py (novo); tests/test_data_quality_policy.py (novo); tests/test_dataset_schema.py (caminho explícito do relatório no teste); docs/ton_iot_data_quality_policy.md (novo).

O identificador da qualidade se refere à população elegível completa, anterior à opção sample_size de depuração já existente. Não foram alterados behavioral_strict, splits, orçamento, seleção, clustering ou profiling. Não houve treinamento. cluster_assignments.csv não foi lido.

Compatibilidade científica: resultados anteriores à quarentena podem se referir a uma população diferente; futuros resultados devem registrar o ID desta população. O sucesso deste checkpoint demonstra conformidade com o schema atual, não veracidade independente de IPs, rótulos ou eventos. O Incremento 2 permanece fora do escopo.
'''
(ROOT/'README.md').write_text(text,encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
