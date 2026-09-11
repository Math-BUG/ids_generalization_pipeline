"""External interpretation: only after both internal-only decisions are frozen."""
from pathlib import Path
import time

import joblib
import numpy as np

import phase_2_1_common as C
from report_phase_2_1_clustering import write_csv


def final_report(root,stage_a,stage_b,comparison):
    """Descriptive readout; never feeds external scores back into selection."""
    a=C.read_json(root/'stage_a/internal_evidence.json')
    b=C.read_json(root/'stage_b/internal_evidence.json')
    lines=['# Fase 2.1 — resultados', '', 'As decisões abaixo precederam a abertura das métricas externas.',
        '', 'Finalistas A: '+str(stage_a['configurations']), 'Configurações retidas B: '+str(stage_b['configurations']),
        '', '## Geometria, estabilidade e concentração', '']
    for name,evidence in [('A (KMeans no desenvolvimento)',a),('B (KMeans no TRAIN completo)',b)]:
        lines += ['### '+name,'','Frente de Pareto: '+', '.join(evidence['pareto']), '',
            '| Configuração | Silhouette média ± sd | DB médio | CH médio | AMI estabilidade mínima | Maior fração (pior seed) | Seeds com flags |',
            '|---|---:|---:|---:|---:|---:|---|']
        for row in evidence['configurations']:
            if not row['eligible']:
                lines.append(f"| {row['configuration']} | Não elegível: {row['reason']} | — | — | — | — | — |")
                continue
            lines.append(f"| {row['configuration']} | {row['mean_silhouette']:.6f} ± {row['std_silhouette']:.6f} | "
                f"{row['mean_davies_bouldin']:.6f} | {row['mean_calinski_harabasz']:.3f} | "
                f"{row['min_pairwise_stability_ami']:.6f} | {row['worst_max_cluster_fraction']:.6f} | {row['flagged_seeds']} |")
        thirty=[r['configuration'] for r in evidence['configurations'] if r['k']==30 and r['pareto']]
        lines += ['', 'k=30 na frente desta etapa: '+(', '.join(thirty) or 'nenhum / não avaliado entre os finalistas')+'.',
            'Os demais k da frente oferecem trade-offs; isto não estabelece um k absolutamente melhor.', '']
    lines += ['## Efeito do SVD na Etapa A', '',
        'Diferenças pareadas em relação a R0 no mesmo k. Valores positivos de DB e concentração significam aumento dessas medidas.', '',
        '| Configuração − R0 | Δ silhouette | Δ DB | Δ AMI mínima | Δ maior fração |', '|---|---:|---:|---:|---:|']
    index={(r['representation'],r['k']):r for r in a['configurations'] if r['eligible']}
    for (rep,k),row in index.items():
        raw=index.get(('R0',k))
        if rep=='R0' or raw is None:
            continue
        values=[row[key]-raw[key] for key in ('mean_silhouette','mean_davies_bouldin','min_pairwise_stability_ami','worst_max_cluster_fraction')]
        lines.append('| '+f'{rep}:k{k}'+' | '+' | '.join(f'{v:+.6f}' for v in values)+' |')
    lines += ['', 'SVD muda a geometria: estas diferenças não provam melhoria universal. '
        'A confirmação completa só existe para os finalistas de B; não extrapolar os demais.', '',
        '## Interpretação externa posterior', '',
        '| Run | AMI type test | AMI label test | Clusters observados | Clusters efetivos |', '|---|---:|---:|---:|---:|']
    for row in comparison:
        lines.append(f"| {row['configuration']} | {row['ami_type_test']:.6f} | {row['ami_label_test']:.6f} | "
                     f"{row['observed_clusters']} | {row['effective_number_of_clusters']:.3f} |")
    historical=next(row for row in comparison if row['historical_reference'])
    lines += ['', f"Referência histórica: AMI type={historical['ami_type_test']:.6f}, AMI label={historical['ami_label_test']:.6f}. "
        'Comparar os valores sem reabrir seleção; AMI baixa não é prova de independência.',
        '', 'Distribuições por cluster, microclusters, entropia e flags por seed estão nas tabelas internas. '
        'Custos de fit/predição/métricas constam nos runs e no profiling externo.',
        'Estabilidade B foi medida nos mesmos 1M; a referência histórica isolada não possui três seeds completas. '
        'A estabilidade de R20/k30 na população completa só foi medida se esse par foi promovido.',
        'Três seeds produzem evidência descritiva; não há seleção pela maior silhouette ou por alinhamento externo.',
        'Esta fase não mediu F1 do IDS nem speedup final.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def external_gate(c):
    prepared=C.load_prepared(c)
    first=C.verify_decision(c,'A',prepared)
    second=C.verify_decision(c,'B',prepared)
    return prepared,first,second


def external_scores(y,clusters):
    # Imports are deliberately confined to post-decision external execution.
    from sklearn.metrics import adjusted_mutual_info_score, v_measure_score
    from ids_pipeline.clustering import cluster_purity
    result=dict(ami=float(adjusted_mutual_info_score(y,clusters)),
        v_measure=float(v_measure_score(y,clusters)),purity=float(cluster_purity(y,clusters)))
    C.require(np.isfinite(list(result.values())).all(),'Nonfinite external metrics')
    return result


def predict_split(df,indices,bundle,reducer,model,k,batch_rows):
    import cupy as cp
    output=np.empty(len(indices),dtype=np.uint8)
    for start in range(0,len(indices),batch_rows):
        idx=indices[start:start+batch_rows]
        X=C.replay_preprocessing(df.iloc[idx][bundle.feature_cols],bundle)
        if reducer is not None:
            X=cp.asarray(reducer.transform(X),dtype=cp.float32)
        C.gpu_matrix(X,len(idx))
        labels=np.asarray(C.to_numpy_array(model.predict(X)))
        C.cluster_counts(labels,len(idx),k)
        output[start:start+len(idx)]=labels
    return output


def evaluate(c):
    root=Path(c['output_dir'])
    with C.lock(root):
        # These checks precede any dataset/label or historical metric file read.
        prepared,stage_a,stage_b=external_gate(c)
        destination=root/'stage_b/external_evaluation'
        destination.mkdir(parents=True,exist_ok=False)
        report=dict(status='RUNNING',opened_at=C.utcnow(),stage_a_decision_hash=stage_a['digest'],
            stage_b_decision_hash=stage_b['digest'],external_metrics_used_for_selection=False)
        C.atomic_json(destination/'manifest.json',report)
        meta=C.read_json(root/C.MANIFEST)
        meta['status']='EXTERNAL_EVALUATION'
        C.atomic_json(root/C.MANIFEST,meta)
        profiler=C.Profiler()
        try:
            env=C.gpu_environment(c)
            C.require(env['cuml_version']==prepared['environment']['cuml_version'] and env['software']==prepared['environment']['software'], 'External environment changed')
            from run_phase_2_full_gpu_baseline import load_population
            audit=destination/'audit'
            audit.mkdir()
            df,splits=load_population(C.parent_config(c),audit,{},profiler)
            C.require(np.array_equal(splits['train'],C.load_array(root,prepared['samples']['arrays']['train_indices'])), 'External train identity mismatch')
            bundle=joblib.load(C.checked_artifact(root,prepared['bundle']))
            C.validate_bundle(bundle)
            evidence=C.read_json(root/'stage_b/internal_evidence.json')
            summary={row['configuration']:row for row in evidence['configurations']}
            specs=[]
            for r,k,seed in C.expected_runs(c,'B',stage_a):
                rec=C.validate_run(root,C.read_json(root/f'stage_b/runs/{C.run_id(r,k,seed)}.json'),prepared)
                specs.append((r,k,seed,rec))
            specs.append(('R20',30,42,None))
            external,comparison=[],[]
            for r,k,seed,rec in specs:
                historical=rec is None
                name='historical_phase2_R20_k30_s42' if historical else C.run_id(r,k,seed)
                if historical:
                    path=Path(c['parent_baseline_dir'])/'artifacts/cuml_kmeans.joblib'
                    C.require(C.file_hash(path)==prepared['parent']['kmeans_sha256'],'Historical model changed')
                else:
                    path=C.checked_artifact(root,rec['artifacts']['model'])
                model=joblib.load(path)
                C.gpu_class(model,'KMeans')
                C.require(model.get_params(deep=False)=={**c['kmeans'],'n_clusters':k,'random_state':seed},'External model parameters changed')
                reducer=None if r=='R0' else joblib.load(C.checked_artifact(root,prepared['reducers'][r]['artifact']))
                if reducer is not None:
                    C.gpu_class(reducer,'TruncatedSVD')
                row=dict(configuration=name,representation=r,components=c['representations'][r],k=k,seed=seed,
                    fit_n=c['expected_split_rows']['train'],historical_reference=historical,
                    components_requested=50 if historical else c['representations'][r],components_effective=c['representations'][r],
                    metric_scope='full_train_DB_CH_inertia; fixed_train_sample_silhouette',
                    metric_sample_hash=prepared['samples']['full_metric_sample_indices_hash'],
                    stage_a_decision_hash=stage_a['digest'],stage_b_decision_hash=stage_b['digest'])
                if rec is not None:
                    row.update({x:rec['metrics'][x] for x in ('silhouette','davies_bouldin','calinski_harabasz')},
                        **rec['concentration'],inertia=rec['inertia'],fit_seconds=rec['fit_seconds'],
                        mean_pairwise_stability_ami=summary[f'{r}:k{k}']['mean_pairwise_stability_ami'])
                for split,idx in splits.items():
                    with C.timed(profiler,name+'_'+split+'_predict',gpu=True):
                        labels=predict_split(df,idx,bundle,reducer,model,k,c['external_prediction_batch_rows'])
                    C.cluster_counts(labels,len(idx),k,require_all=split=='train')
                    with C.timed(profiler,name+'_'+split+'_metrics'):
                        for target in ('label','type'):
                            y=df.iloc[idx][target].astype(str).to_numpy()
                            scores=external_scores(y,labels)
                            external.append(dict(configuration=name,split=split,target=target,n=len(idx),**scores))
                            if split=='test':
                                row.update({f'{metric}_{target}_test':v for metric,v in scores.items()})
                        if historical and split=='train':
                            from run_phase_2_1_clustering import stage_matrix
                            _,X,_=stage_matrix(c,prepared,'B','R20')
                            offsets=C.load_array(root,prepared['samples']['arrays']['full_metric_train_offsets'])
                            metrics=C.internal_metrics(X,labels,offsets,full_geometry=True)
                            row.update({key:metrics[key] for key in ('silhouette','davies_bouldin','calinski_harabasz')},
                                **C.concentration(C.cluster_counts(labels,len(idx),k)),inertia=float(model.inertia_),
                                mean_pairwise_stability_ami=None,fit_seconds=None)
                            # History is opened only here, after final selection.
                            old=C.read_json(Path(c['parent_baseline_dir'])/'metrics_clustering.json')
                            row.update(historical_silhouette=old['silhouette_train'],historical_db=old['davies_bouldin_train'],
                                historical_ch=old['calinski_harabasz_train'],historical_inertia=old['inertia'])
                comparison.append(row)
            # Ensure no source/decision changed while labels were being evaluated.
            external_gate(c)
            write_csv(root/'stage_b/external_metrics.csv',external)
            write_csv(root/'stage_b/final_comparison.csv',comparison)
            report.update(status='COMPLETE',environment=env,completed_at=C.utcnow())
            meta['status']='COMPLETE' if meta.get('excluded_or_pending_runs_A',0)==0 else 'INCOMPLETE_WITH_EXCLUDED_STAGE_A_RUNS'
            final_report(root,stage_a,stage_b,comparison)
        except BaseException as exc:
            report.update(status='INVALID',error=f'{type(exc).__name__}: {exc}')
            meta.update(status='INCOMPLETE',error=report['error'])
            raise
        finally:
            report['profiling']=profiler.to_dict()
            C.atomic_json(destination/'manifest.json',report)
            C.atomic_json(root/C.MANIFEST,meta)


def main():
    p=C.parser(__doc__)
    args=p.parse_args()
    evaluate(C.load_config(args.config))


if __name__=='__main__':
    main()
