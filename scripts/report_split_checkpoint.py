"""Render and verify only split artifacts from the Increment 3 checkpoint."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import scipy
import pyarrow
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ids_pipeline.split_protocols import NAMES, array_hash, digest_json, row_validation
from ids_pipeline.utils import write_json


def main():
    root=Path('reports/increment_3'); real=root/'real'
    quality=json.loads((real/'data_quality_population.json').read_text(encoding='utf-8'))
    before=json.loads((root/'frozen_before.json').read_text(encoding='utf-8'))
    unchanged={f:hashlib.sha256(Path(f'src/ids_pipeline/{f}.py').read_bytes()).hexdigest()==h for f,h in before.items()}
    checks={'frozen_files_unchanged':unchanged,'protocols':{}}
    assert all(unchanged.values())
    xml=ET.parse(root/'tests.xml').getroot()
    checks['tests']={k:sum(int(s.attrib.get(k,0)) for s in xml.findall('testsuite')) for k in ('tests','failures','errors','skipped')}
    assert checks['tests']['failures']==0 and checks['tests']['errors']==0
    checks['environment']=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,pyarrow=pyarrow.__version__)
    original_positions=np.load(real/'eligible_original_positions.npy',allow_pickle=False,mmap_mode='r')
    assert len(original_positions)==quality['eligible_rows']
    assert np.all(np.diff(original_positions)>0)
    checks['eligible_original_positions_hash']=array_hash(original_positions)
    lines=['# Incremento 3 — checkpoint dos protocolos de split','',
           f"População: **{quality['original_rows']:,} originais; {quality['quarantined_rows']:,} quarentenados; {quality['eligible_rows']:,} elegíveis**.",
           '',f"Identificador: `{quality['population_id']}`.",'',
           'Fontes: 23 partições Parquet verificadas por SHA256 contra o Incremento 1; somente colunas de split e o campo de quarentena foram lidos. Sem treinamento ou experimentos de modelos.',
           '',f"Testes relevantes: **{checks['tests']['tests']} passaram**, {checks['tests']['failures']} falhas e {checks['tests']['errors']} erros. Os 13 arquivos congelados conferem com os hashes anteriores.",'',
           'Implementação: `src/ids_pipeline/splitting.py`, `src/ids_pipeline/split_protocols.py` (novo), `src/ids_pipeline/cli.py` (apenas persistência de erro de split); scripts de auditoria/relatório; `tests/test_split_protocols_v2.py`; `docs/split_protocols_v2.md`.',
           '', 'O alvo é 60/15/25; tolerância de grupos ±2 pontos percentuais. Suporte mínimo por classe em grupos: 1; alerta descritivo de suporte pequeno: abaixo de 30. Nenhuma tolerância foi relaxada.', '']
    for strategy in ('group_stratified','temporal','temporal_per_class'):
        directory=real/strategy
        path=directory/'split_manifest.json'
        approved=path.exists()
        if not approved: path=directory/'best_candidate_not_approved.json'
        lines += [f'## {strategy}', '']
        if not path.exists():
            lines += ['Protocolo inviável; nenhuma população parcial foi liberada.', '', '```json', (directory/'failure_diagnostics.json').read_text(encoding='utf-8'), '```', '']
            continue
        report=json.loads(path.read_text(encoding='utf-8'))
        assert report['population']['source_population_id']==quality['population_id']
        assert sum(report['counts'].values())==quality['eligible_rows']
        calculated=digest_json({k:report[k] for k in ['protocol_version','population','parameters','seed','index_hashes']})
        assert calculated==report['split_hash']
        if approved:
            arrays={s:np.load(directory/'splits'/f'{s}_indices.npy',allow_pickle=False) for s in NAMES}
            validation=row_validation(arrays,quality['eligible_rows'])
            assert validation['ok']
            for s in NAMES:
                assert array_hash(arrays[s])==report['index_hashes'][s]
                assert digest_json([quality['population_id'],array_hash(original_positions[arrays[s]])])==report['identity_hashes'][s]
            del arrays
            p=report['parameters']
            reuse={k:p[k] for k in ['test_size','val_size','group_cols','timestamp_col','timestamp_unit',
                                     'temporal_bucket_freq','label_col','type_col']}
            reuse.update(split_strategy=strategy,random_state=report['seed'] if report['seed'] is not None else 42,
                         group_split_tolerance=p['group_tolerance'],group_split_candidates=p['group_candidates'],
                         split_min_class_support=p['minimum_class_support'],
                         split_small_support_threshold=p['small_support_threshold'],
                         frozen_splits_dir=directory.as_posix(),split_export_csv=False)
            (directory/'reuse_split.yaml').write_text(yaml.safe_dump(reuse,sort_keys=False),encoding='utf-8')
        checks['protocols'][strategy]={'approved':approved,'manifest_hash_verified':True,'split_hash':report['split_hash'],
                                       'arrays_verified':approved,'validation_ok':report['validation_ok']}
        lines += [f"Versão: `{report['protocol_version']}`. Validação: **{report['validation_ok']}**. {'Congelado para reutilização.' if approved else 'Candidato diagnóstico; NÃO aprovado para treinamento.'}",
                  '',f"Hash completo: `{report['split_hash']}`.",'',
                  '| Split | Registros | % real | Início UTC | Fim UTC | Buckets | Grupos |', '|---|---:|---:|---|---|---:|---:|']
        for s in NAMES:
            interval=report['global_chronology']['intervals'][s]
            lines.append(f"| {s} | {report['counts'][s]:,} | {report['proportions'][s]*100:.6f} | {interval['start_utc']} | {interval['end_utc']} | {interval['buckets']:,} | {report['group_overlap']['n_'+s+'_groups']:,} |")
        lines += ['',f"Cronologia global preservada: **{report['global_chronology']['global_chronology_preserved']}**.",
                  '',f"Sobreposição de grupos (train/val, train/test, val/test): {report['group_overlap']['train_val_overlap']:,}; {report['group_overlap']['train_test_overlap']:,}; {report['group_overlap']['val_test_overlap']:,}. Em protocolos temporais isso é descritivo, não viola o contrato temporal.",
                  '',f"Verificação de linhas: `{json.dumps(report['row_validation'],sort_keys=True)}`.",'']
        for target,d in report['distributions'].items():
            lines += [f'### Distribuição de {target}', '', '| Classe | train | val | test |', '|---|---:|---:|---:|']
            for c in d['splits']['train']['counts']:
                lines.append('| '+c+' | '+' | '.join(f"{d['splits'][s]['counts'][c]:,}" for s in NAMES)+' |')
            lines += ['',f"Conhecidas no treino: {', '.join(d['known_in_train']) or 'nenhuma'}.",
                      f"Primeira aparição em validação: {', '.join(d['first_seen_validation']) or 'nenhuma'}.",
                      f"Primeira aparição em teste: {', '.join(d['first_seen_test']) or 'nenhuma'}.",
                      f"No teste, ausentes do treino (inclui as já vistas em validação): {', '.join(d['unseen_from_train_in_test']) or 'nenhuma'}.", '']
            for s in NAMES:
                entry=d['splits'][s]
                lines.append(f"- {s}: ausentes = {entry['missing']}; suporte pequeno = {entry['small_support']}.")
            lines.append('')
        if strategy=='group_stratified':
            search=report['group_search']
            lines += ['### Viabilidade e concentração dos grupos', '',
                      f"Grupos únicos: **{search['unique_groups']:,}**. Maior grupo: **{search['largest_group']:,}** ({search['largest_group_fraction']*100:.6f}%).",
                      '',f"Quantis de tamanho: `{json.dumps(search['group_size_quantiles'])}`.",
                      '', 'Percentuais dos 20 maiores grupos: '+', '.join(f'{v*100:.6f}%' for v in search['top_group_fractions'])+'.',
                      '',f"Busca: {len(search['candidates'])} candidatos; escolhido = {search['selected_candidate']}. Estado conjunto: **{search['feasibility_status']}**.",
                      '', f"Razões demonstradas de inviabilidade: {search['proven_infeasibility_reasons']}.", '',
                      '| type | Registros | Grupos com a classe | Fração da classe no maior grupo |','|---|---:|---:|---:|']
            for c,d in search['class_group_support']['type'].items():
                lines.append(f"| {c} | {d['rows']:,} | {d['groups']:,} | {d['largest_group_class_fraction']*100:.4f}% |")
            lines += ['', 'Todos os candidatos e escores estão no manifesto/diagnóstico JSON. Nenhum desempenho de modelo foi consultado.', '']
        if strategy=='temporal_per_class':
            lines += [f"Validação interna por classe: **{report['per_class_validation']['ok']}**. Classes inviáveis: {report['per_class_validation']['invalid_classes']}.",
                      '', 'Relatório detalhado: `real/temporal_per_class/splits/temporal_per_class_report.csv`. Este protocolo permanece auxiliar.', '']
    lines += ['## Reutilização e limites', '',
              'Use os parâmetros de `reuse_split.yaml` do protocolo aprovado ao compor a configuração futura. O carregamento verifica população, configuração, índices, hashes e contratos; não faz nova busca. Os arquivos `.npy` contêm posições elegíveis; `real/eligible_original_positions.npy` mapeia para as posições originais globais.', '',
              'Resultados antigos não são diretamente intercambiáveis: mudaram a alocação de grupos e os cortes temporais. A presença de classes inéditas exige uma futura decisão sobre avaliação closed-set versus unseen/open-set; esta etapa só as identifica. Suportes pequenos merecem análise de incerteza futura. Nenhum algoritmo de classificação open-set foi adicionado. Incremento 4 não iniciado.', '']
    write_json(root/'verification.json',checks)
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
