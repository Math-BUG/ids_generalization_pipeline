"""Internal aggregation/Pareto and explicit, immutable human decisions."""
from pathlib import Path
import time

import numpy as np
import pandas as pd

import phase_2_1_common as C


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.pending')
    pd.DataFrame(rows).to_csv(tmp,index=False)
    tmp.replace(path)


def summarize(c,stage,prepared):
    root=Path(c['output_dir'])
    if C.decision_path(root,stage).exists():
        C.verify_decision(c,stage,prepared)
        return C.check_seal(C.read_json(root/f'stage_{stage.lower()}/internal_evidence.json'))
    previous=C.verify_decision(c,'A',prepared) if stage=='B' else None
    expected=C.expected_runs(c,stage,previous)
    summaries,runs,distributions,pair_rows,records=[],[],[],[],{}
    for r,k,seed in expected:
        name=C.run_id(r,k,seed)
        path=root/f'stage_{stage.lower()}/runs/{name}.json'
        if not path.exists():
            runs.append(dict(run_id=name,representation=r,k=k,seed=seed,status='MISSING'))
            continue
        record=C.check_seal(C.read_json(path))
        C.require(record['preparation_digest']==prepared['digest'],'Run has incompatible preparation')
        C.require((record['representation'],record['k'],record['seed'],record['stage'])==(r,k,seed,stage),'Run identity mismatch')
        records[name]=record
        row=dict(run_id=name,representation=r,k=k,seed=seed,status=record['validity']['status'],run_digest=record['digest'])
        if row['status']=='VALID':
            C.validate_run(root,record,prepared)
            row.update(record['metrics'],**record['concentration'],inertia=record['inertia'],
                fit_seconds=record['fit_seconds'],predict_seconds=record['predict_seconds'],
                metrics_seconds=record['metrics_seconds'],total_seconds=record['total_seconds'],
                degeneracy_flags=C.json.dumps(record['degeneracy_flags'],sort_keys=True))
            row.pop('undefined_reasons',None)
            for cluster_id,n in enumerate(record['counts']):
                distributions.append(dict(run_id=name,cluster_id=cluster_id,count=n,fraction=n/record['fit_n']))
        runs.append(row)
    for r,k in dict.fromkeys((r,k) for r,k,_ in expected):
        group=[records.get(C.run_id(r,k,s)) for s in c['clustering_seeds']]
        summary=dict(configuration=f'{r}:k{k}',representation=r,k=k,eligible=False)
        if not all(rec and rec['validity']['status']=='VALID' for rec in group):
            summary['reason']='Missing or invalid seed execution'
            summaries.append(summary)
            continue
        C.require(len({rec['input_hash'] for rec in group})==1,'Seeds used different representation matrices')
        assignments={rec['seed']:C.load_array(root,rec['artifacts']['sample_assignments']) for rec in group}
        start=time.perf_counter()
        stability=C.stability(assignments)
        summary['stability_seconds']=time.perf_counter()-start
        summary.update({key:value for key,value in stability.items() if key!='pairs'})
        for pair in stability['pairs']:
            pair_rows.append(dict(configuration=summary['configuration'],**pair,n_evaluated=stability['n_evaluated'],
                sample_hash=prepared['samples']['development_indices_hash']))
        complete=all(rec['metrics_complete'] for rec in group)
        summary.update(eligible=complete,reason=None if complete else 'Required internal metric undefined')
        for key in ('silhouette','davies_bouldin','calinski_harabasz'):
            vals=[rec['metrics'][key] for rec in group]
            if any(x is None for x in vals):
                summary['mean_'+key]=None
            else:
                for statistic,func in [('mean',np.mean),('min',np.min),('max',np.max)]:
                    summary[statistic+'_'+key]=float(func(vals))
                summary['std_'+key]=float(np.std(vals,ddof=1))
        summary['worst_max_cluster_fraction']=max(rec['concentration']['max_cluster_fraction'] for rec in group)
        for key in group[0]['concentration']:
            if key in ('n','k'):
                continue
            vals=[rec['concentration'][key] for rec in group]
            summary.update({f'mean_{key}':float(np.mean(vals)),f'std_{key}':float(np.std(vals,ddof=1)),
                f'min_{key}':float(np.min(vals)),f'max_{key}':float(np.max(vals))})
        summary['mean_total_seconds']=float(np.mean([rec['total_seconds'] for rec in group]))
        summary['flagged_seeds']=[rec['seed'] for rec in group if any(v['triggered'] is True for v in rec['degeneracy_flags'].values())]
        summaries.append(summary)
    rows=[{'configuration':s['configuration'],**{key:s[key] for key in C.PARETO_COLUMNS}} for s in summaries if s['eligible']]
    front=C.pareto_front(rows)
    # Secondary diagnostic front omits concentration, not configurations with flags.
    geometry=C.pareto_front([{**row,'worst_max_cluster_fraction':0.0} for row in rows])
    for summary in summaries:
        summary['pareto']=summary['configuration'] in front
        summary['pareto_geometry_stability']=summary['configuration'] in geometry
    evidence=C.sealed(dict(stage=stage,preparation_digest=prepared['digest'],expected_runs=len(expected),
        recorded_runs=len(records),successful_runs=sum(x['status']=='VALID' for x in runs),
        run_digests={name:rec['digest'] for name,rec in records.items()},configurations=summaries,
        pareto=front,external_metrics_used=False))
    folder=root/f'stage_{stage.lower()}'
    for name,data in [('runs',runs),('stability',pair_rows),('cluster_distribution',distributions),
                      ('configuration_summary',summaries),('pareto_candidates',[x for x in summaries if x['pareto']])]:
        write_csv(folder/(name+'.csv'),data)
    C.atomic_json(folder/'internal_evidence.json',evidence)
    meta=C.read_json(root/C.MANIFEST)
    meta[f'expected_runs_{stage}']=len(expected)
    meta[f'successful_runs_{stage}']=evidence['successful_runs']
    meta[f'recorded_runs_{stage}']=len(records)
    meta[f'excluded_or_pending_runs_{stage}']=len(expected)-evidence['successful_runs']
    meta['status']=f'{stage}_COMPLETE_AWAITING_SELECTION' if len(records)==len(expected) else f'{stage}_INCOMPLETE'
    C.atomic_json(root/C.MANIFEST,meta)
    return evidence


def freeze(c,stage,candidates,approved_by,rationale,internal_only):
    C.require(internal_only is True,'Explicit --approve-internal-only required')
    C.require(approved_by.strip() and rationale.strip(),'Reviewer and rationale required')
    root=Path(c['output_dir'])
    with C.lock(root):
        prepared=C.load_prepared(c)
        C.require(not C.decision_path(root,stage).exists(),'Decision is already frozen; cannot overwrite')
        C.require(not (root/'stage_b/external_metrics.csv').exists(),'Cannot change decisions after external metrics')
        # Freeze exactly the evidence reviewed, without changing timing fields or
        # recalculating stability while the reviewer approves the candidates.
        evidence=C.check_seal(C.read_json(root/f'stage_{stage.lower()}/internal_evidence.json'))
        C.require(evidence['preparation_digest']==prepared['digest'],'Reviewed evidence preparation differs')
        previous=C.verify_decision(c,'A',prepared) if stage=='B' else None
        expected={C.run_id(r,k,s) for r,k,s in C.expected_runs(c,stage,previous)}
        C.require(set(evidence['run_digests'])==expected,'All planned executions must have reviewed records')
        C.require(evidence['stage']==stage and evidence['external_metrics_used'] is False,'Wrong/non-internal evidence')
        for name,digest in evidence['run_digests'].items():
            rec=C.check_seal(C.read_json(root/f'stage_{stage.lower()}/runs/{name}.json'))
            C.require(rec['digest']==digest,'Runs changed; regenerate internal report before approval')
            if rec['validity']['status']=='VALID':
                C.validate_run(root,rec,prepared)
        C.require(evidence['recorded_runs']==evidence['expected_runs'],'All planned executions must have records before selection')
        C.require(1<=len(candidates)<=3 and len(set(candidates))==len(candidates),'Choose one to three distinct configurations')
        lookup={row['configuration']:row for row in evidence['configurations']}
        C.require(all(name in evidence['pareto'] for name in candidates),'Only valid internal Pareto candidates may be selected')
        configs=[dict(representation=lookup[name]['representation'],k=lookup[name]['k']) for name in candidates]
        decision=C.sealed(dict(stage=stage,preparation_digest=prepared['digest'],
            internal_evidence_digest=evidence['digest'],run_digests=evidence['run_digests'],
            configurations=configs,seeds=c['clustering_seeds'],external_metrics_used=False,
            previous_decision_digest=previous['digest'] if previous else None,
            approved_by=approved_by,rationale=rationale,frozen_at=C.utcnow(),
            degeneracy_policy=c['degeneracy_flags'],method='pareto_then_documented_approval'))
        C.atomic_json(C.decision_path(root,stage),decision,exclusive=True)
        meta=C.read_json(root/C.MANIFEST)
        meta['status']='A_FROZEN' if stage=='A' else 'B_FROZEN_EXTERNAL_ALLOWED'
        meta['finalists' if stage=='A' else 'final_configurations']=configs
        meta['stage_'+stage.lower()+'_decision_hash']=decision['digest']
        if stage=='A':
            meta['expected_runs_B']=len(configs)*3
        C.atomic_json(root/C.MANIFEST,meta)
        return decision


def main():
    p=C.parser(__doc__)
    p.add_argument('--stage',required=True,choices=['A','B'])
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--candidate',action='append',default=[],help='For example R20:k30; never a seed')
    p.add_argument('--approved-by',default='')
    p.add_argument('--rationale',default='')
    p.add_argument('--approve-internal-only',action='store_true')
    args=p.parse_args()
    c=C.load_config(args.config)
    if args.freeze:
        result=freeze(c,args.stage,args.candidate,args.approved_by,args.rationale,args.approve_internal_only)
        print('Frozen:',result['digest'])
    else:
        with C.lock(c['output_dir']):
            prepared=C.load_prepared(c)
            evidence=summarize(c,args.stage,prepared)
            print('Internal Pareto:',evidence['pareto'])


if __name__=='__main__':
    main()
