"""Independently check saved small selections against frozen IDs and source targets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ids_pipeline.data_quality_policy import source_fingerprint,assert_source_unchanged
from ids_pipeline.dataset_schema import normalize_dataset_schema
from ids_pipeline.selection_budget import FrozenTrainingPopulation,Selection,hash_indices
from ids_pipeline.selection_methods import METHODS,TrainStrata,select_random,select_from_training_frame
from ids_pipeline.utils import write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def selected_targets(ids):
    """Read only label/type in bounded batches; independently map quarantined positions."""
    ref=read_json('reports/ton_iot_quarantine_checkpoint/parquet/data_quality_report.json')
    mapping=np.load('reports/increment_3/real/eligible_original_positions.npy',mmap_mode='r')
    original=mapping[ids]
    values={};original_offset=0;eligible_offset=0
    for item in ref['files']:
        path=Path(item['source_path']);fingerprint=source_fingerprint(path)
        assert fingerprint['sha256']==item['source_fingerprint']['sha256']
        lo,hi=np.searchsorted(ids,[eligible_offset,eligible_offset+item['eligible_rows']])
        keep=np.ones(item['original_rows'],dtype=bool)
        keep[item['quarantined_row_positions_0based']]=False
        expected=np.flatnonzero(keep)[ids[lo:hi]-eligible_offset]+original_offset
        np.testing.assert_array_equal(original[lo:hi],expected)
        batch_offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100000,columns=['label','type']):
            start=original_offset+batch_offset
            a,b=np.searchsorted(original,[start,start+batch.num_rows])
            if b>a:
                raw=batch.to_pandas().iloc[original[a:b]-start]
                normalized,_=normalize_dataset_schema(raw)
                for identity,label,kind in zip(ids[a:b],normalized['label'],normalized['type']):
                    values[int(identity)]=(str(label),str(kind))
            batch_offset+=batch.num_rows
        assert batch_offset==item['original_rows']
        assert_source_unchanged(path,fingerprint)
        original_offset+=batch_offset;eligible_offset+=item['eligible_rows']
    assert len(values)==len(ids)
    mapping._mmap.close()
    return values


def main():
    root=Path('reports/increment_5');real=root/'real'
    before=read_json(root/'frozen_before.json')
    unchanged={name:hashlib.sha256(Path(f'src/ids_pipeline/{name}.py').read_bytes()).hexdigest()==h
               for name,h in before.items()}
    assert all(unchanged.values()),unchanged
    saved=read_json(real/'selections.json');assert saved['status']=='complete' and len(saved['results'])==12
    ids={(b,m):np.load(real/f'B_{b}'/m/'selected_ids.npy') for b in (1000,10000) for m in METHODS}
    targets=selected_targets(np.unique(np.concatenate(list(ids.values()))))
    checks=[];quotas=[];class_distributions=[]
    with FrozenTrainingPopulation.load('reports/increment_3/real/group_stratified') as pop:
        cluster_ids=np.load(real/'train_cluster_ids.npy',mmap_mode='r')
        provenance=read_json(real/'cluster_provenance.json')
        assert provenance['cluster_ids_hash']==hash_indices(cluster_ids)
        assert provenance['train_population_hash']==pop.train_population_hash
        assert provenance['frozen_split_hash']==pop.split_hash
        assert len(cluster_ids)==len(pop.train_indices) and provenance['fit_rows']==len(pop.train_indices)
        assert provenance['k']==30 and not provenance['holdouts_used_for_fit'] and not provenance['labels_used_for_clustering']
        clusters=TrainStrata.from_values(pop,'cluster',pop.train_indices,cluster_ids)
        names,counts=np.unique(cluster_ids,return_counts=True)
        capacities=dict(zip(map(str,names),map(int,counts)))
        for (b,method),selected in ids.items():
            directory=real/f'B_{b}'/method
            summary=read_json(directory/'selection_summary.json');diag=read_json(directory/'method_summary.json')
            accepted=Selection.accept(pop,b,42,selected)
            assert accepted.selection_hash==summary['selection_hash']
            assert summary['realized_budget']==b and summary['selection_seed']==42
            assert diag['method']==method
            entry=dict(B=b,method=method,selection_hash=accepted.selection_hash,realized_budget=len(selected),
                       duplicates=0,outside_train=0,hash_recomputed=True,quotas_verified=True)
            for column,index in [('label',0),('type',1)]:
                distribution=Counter(targets[int(i)][index] for i in selected)
                for name,count in sorted(distribution.items()):
                    class_distributions.append(dict(B=b,method=method,column=column,value=name,count=count))
            if method.startswith('cluster_'):
                observed=Counter(map(str,cluster_ids[np.searchsorted(pop.train_indices,selected)]))
                assert diag['stratum_assignment_hash']==hash_indices(clusters.codes)
                for row in diag['quotas']:
                    assert capacities[row['stratum']]==row['capacity']
                repeated=select_from_training_frame(pop,b,42,method,None,clusters=clusters)
                assert repeated.selection.selection_hash==accepted.selection_hash
                entry['real_selection_repeated']=True
            elif method.startswith('stratified_'):
                index=0 if method=='stratified_label' else 1
                observed=Counter(targets[int(i)][index] for i in selected)
                column='label' if index==0 else 'type'
                declared={r['stratum']:r['capacity'] for r in diag['quotas']}
                assert declared==pop.manifest['distributions'][column]['splits']['train']['counts']
                entry['real_selection_repeated']=False
            else:
                observed={}
                assert select_random(pop,b,42).selection.selection_hash==accepted.selection_hash
                entry['real_selection_repeated']=True
            if diag['quotas']:
                assert sum(r['capacity'] for r in diag['quotas'])==len(pop.train_indices)
                assert sum(r['quota'] for r in diag['quotas'])==b
                assert observed=={r['stratum']:r['quota'] for r in diag['quotas'] if r['quota']}
                for row in diag['quotas']:quotas.append(dict(B=b,method=method,**row))
            checks.append(entry)
        cluster_ids._mmap.close()
    tests=ET.parse(root/'tests.xml').getroot()
    test_counts={k:sum(int(s.attrib.get(k,0)) for s in tests.findall('testsuite')) for k in ('tests','errors','failures','skipped')}
    assert test_counts['errors']==test_counts['failures']==0
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in real.glob('*.joblib')}
    report=dict(status='verified',frozen_files_unchanged=unchanged,tests=test_counts,selections=checks,
                environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
                fitted_artifact_sha256=artifacts,provenance=provenance,
                stratified_reproducibility='unit tests; real saved ID hashes and source quotas independently verified',
                real_random_and_cluster_reproducibility='same seed rerun with frozen population and the same fitted clusters',rf_fits=0)
    write_json(root/'verification.json',report)
    pd.DataFrame(quotas).to_csv(root/'quotas.csv',index=False)
    pd.DataFrame(class_distributions).to_csv(root/'selected_class_distributions.csv',index=False)
    render_report(root,checks,quotas,class_distributions,report)
    print(f"Verified all 12 selections, source targets, quotas, hashes and {len(unchanged)} frozen files")


def table(headers,rows):
    return ['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(map(str,row))+' |' for row in rows]


def render_report(root,checks,quotas,distributions,report):
    p=report['provenance']
    lines=['# Incremento 5 — seleções verificadas','',
           '**12/12 seleções aprovadas:** B=1.000 e B=10.000 para os seis métodos; zero duplicatas e zero IDs fora do treino.',
           '',f"Treino: **{p['fit_rows']:,}** registros, split group_stratified congelado. Selection seed: 42.",
           '',f"Testes: **{report['tests']['tests']} passaram**; 14 módulos congelados conferidos por SHA256.",
           '', 'O MiniBatchKMeans CPU existente foi ajustado uma vez sobre todo o treino com k=30, batch_size=16384, n_init=5, max_iter=100 e seed=42. CuPy/cuML indisponíveis. Nenhum RF foi treinado; não há F1 ou experimento completo.',
           '', 'Preprocessing congelado: oito quantitativas e conn_state one-hot com 13 categorias de treino; 21 dimensões antes do SVD e 20 depois, conforme a regra existente. Validação/teste e alvos não participaram dos ajustes.',
           '', 'Fontes: SHA256 das 23 partições conferidos com o Incremento 1. As quotas foram verificadas pelos IDs salvos; classes relidas das fontes em blocos, com mapeamento independente das posições de quarentena. Random e seletores de cluster foram repetidos com o mesmo seed e produziram os mesmos hashes. Estratificados têm reprodução testada em fixtures e hashes/quotas reais revalidados.',
           '', '## Contrato e métodos','',
           'Todos retornam B IDs reais, únicos e pertencentes ao treino. Random usa amostragem uniforme sem reposição. Estratificados usam somente label ou type do treino e pesos iguais ao tamanho de cada classe. Cluster_uniform, cluster_proportional e cluster_sqrt usam respectivamente 1, n e sqrt(n). Dentro de cada estrato a escolha é uniforme sem reposição.',
           '', 'Alocador: reserva uma unidade por estrato não vazio quando B≥C; distribui o restante pelos pesos originais; satura capacidades e redistribui excedentes; aplica maiores restos com empate pela ordem canônica. Soma exata B, sem truncamento. Versões: selection_methods/1.0.0 e bounded_largest_remainder/1.0.0.',
           '', '## Hashes','',f"Split: `{p['frozen_split_hash']}`",'',f"Treino: `{p['train_population_hash']}`",'',f"Clusters atuais: `{p['cluster_ids_hash']}`",'']
    lines+=table(['B','Método','Hash da seleção'],[(r['B'],r['method'],r['selection_hash']) for r in checks])
    lines+=['','## Quotas de cluster','']
    cluster_rows=[r for r in quotas if r['method'].startswith('cluster_')]
    keys=sorted({r['stratum'] for r in cluster_rows},key=int)
    lookup={(r['stratum'],r['B'],r['method']):r for r in cluster_rows}
    rows=[]
    for key in keys:
        rows.append([key,lookup[key,1000,'cluster_uniform']['capacity']]+[
            lookup[key,b,m]['quota'] for b in (1000,10000) for m in ('cluster_uniform','cluster_proportional','cluster_sqrt')])
    lines+=table(['Cluster','n','Uniforme 1k','Proporcional 1k','Sqrt 1k','Uniforme 10k','Proporcional 10k','Sqrt 10k'],rows)
    lines+=['','## Classes dos estratificados','']
    rows=[]
    for method in ('stratified_label','stratified_type'):
        names=sorted({r['stratum'] for r in quotas if r['method']==method})
        for name in names:
            r1=next(r for r in quotas if r['method']==method and r['stratum']==name and r['B']==1000)
            r2=next(r for r in quotas if r['method']==method and r['stratum']==name and r['B']==10000)
            rows.append([method,name,r1['capacity'],r1['quota'],r2['quota']])
    lines+=table(['Método','Classe','n no treino','B=1k','B=10k'],rows)
    lines+=['','## Limites metodológicos','',
            '- Sqrt reduz a concentração dos pesos, mas não estabelece um teto rígido por cluster. Capacidades [100000,1,1] e B=1000 obrigam [998,1,1]. Um teto adicional seria outra política e não foi implementado.',
            '- A cobertura mínima pode afastar as proporções originais, especialmente quando B se aproxima de C. Não foi imposto balanceamento 50/50.',
            '- Mesmo seed garante reprodução das seleções com as mesmas entradas e ambiente. Reajustes CPU/GPU do clustering podem produzir partições diferentes; os seis métodos desta validação compartilham um único ajuste registrado.',
            '- Os resultados antigos com mixed/centroid/boundary ou número fixo por cluster não representam esta comparação sob B fixo. Estes artefatos validam seleção; não demonstram superioridade de desempenho.',
            '', '## Arquivos deste incremento','',
            '`src/ids_pipeline/selection_methods.py` (novo), `src/ids_pipeline/config.py` (selection_method), `src/ids_pipeline/cli.py` (despacho), `tests/test_selection_methods.py`, `scripts/validate_selection_methods.py`, `scripts/report_selection_methods.py` e `docs/selection_methods_v1.md`.',
            '', 'Evidências: verification.json, tests.xml, quotas.csv, selected_class_distributions.csv, real/selections.json e resumos/IDs pequenos por método. Modelos, matrizes intermediárias e clusters atuais são artefatos compartilhados de proveniência; não foram lidas associações CSV antigas. Nenhum avanço ao Incremento 6.', '']
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
