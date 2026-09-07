"""Versioned split contracts. No model results participate in these decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ('train', 'val', 'test')
VERSIONS = {s: s + '/2.0.0' for s in ('group_stratified', 'temporal', 'temporal_per_class')}
QUESTIONS = {
    'group_stratified': 'Generalization to unseen combinations of configured group attributes; not necessarily unseen devices.',
    'temporal': 'Training only in the past, evaluation in globally later periods.',
    'temporal_per_class': 'Earlier examples of each class precede its later examples; auxiliary analysis only.',
    'random_stratified': 'Random holdout within the observed population.',
}


class SplitValidationError(ValueError):
    def __init__(self, message, diagnostics, splits=None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.splits = splits


def parameters(config):
    extra = config.extra
    result = dict(test_size=config.test_size, val_size=config.val_size,
                  val_size_definition='fraction of non-test population',
                  group_cols=config.group_cols, timestamp_col=config.timestamp_col,
                  timestamp_unit=config.timestamp_unit, temporal_bucket_freq=config.temporal_bucket_freq,
                  label_col=config.label_col, type_col=config.type_col,
                  group_tolerance=float(extra.get('group_split_tolerance', .02)),
                  group_candidates=int(extra.get('group_split_candidates', 16)),
                  minimum_class_support=int(extra.get('split_min_class_support', 1)),
                  small_support_threshold=int(extra.get('split_small_support_threshold', 30)))
    if not 0 < config.test_size < 1 or not 0 < config.val_size < 1:
        raise ValueError('test_size and conditional val_size must be between zero and one')
    if not 0 <= result['group_tolerance'] < 1 or result['group_candidates'] < 1 or result['minimum_class_support'] < 1:
        raise ValueError('Invalid split search parameters')
    return result


def targets(config):
    parameters(config)
    return np.array([(1-config.test_size)*(1-config.val_size),
                     (1-config.test_size)*config.val_size, config.test_size])


def group_codes(df, cols):
    if not cols or len(set(cols)) != len(cols):
        raise ValueError('group_cols must be nonempty and unique')
    missing = [c for c in cols if c not in df]
    if missing:
        raise ValueError(f'Missing required group columns: {missing}')
    # Group compact per-column codes, never construct a Python tuple for every row.
    # -1 is a separate null code, never a literal textual sentinel.
    encoded = pd.DataFrame({c: pd.factorize(df[c],sort=False)[0].astype(np.int32) for c in cols})
    codes = encoded.groupby(cols,sort=True,observed=True,dropna=False).ngroup().to_numpy(dtype=np.int64)
    return codes, int(codes.max()+1) if len(codes) else 0


def overlap_from_codes(codes, splits):
    sets = {s: np.unique(codes[idx]) for s, idx in splits.items()}
    out = {f'n_{s}_groups': len(sets[s]) for s in NAMES}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        out[f'{a}_{b}_overlap'] = len(np.intersect1d(sets[a], sets[b], assume_unique=True))
    out['ok'] = all(out[f'{a}_{b}_overlap'] == 0 for a,b in [('train','val'),('train','test'),('val','test')])
    return out


def group_split(df, config):
    codes, ng = group_codes(df, config.group_cols)
    p, goal = parameters(config), targets(config)
    counts = np.bincount(codes, minlength=ng)
    blocks, class_info = [], {}
    for col in (config.label_col, config.type_col):
        if col not in df or df[col].isna().any():
            raise ValueError(f'group_stratified requires complete target {col!r}')
        y, classes = pd.factorize(df[col], sort=True)
        table = np.bincount(codes * len(classes) + y, minlength=ng*len(classes)).reshape(ng, -1)
        blocks.append(table)
        class_info[col] = {str(c): {'rows': int(table[:,j].sum()), 'groups': int((table[:,j]>0).sum()),
                                  'largest_group_class_fraction': float(table[:,j].max()/table[:,j].sum())}
                           for j,c in enumerate(classes)}
    hist = np.column_stack(blocks)
    totals = hist.sum(axis=0)
    # Equal importance per target family, including rare classes, plus population size.
    features = np.column_stack([counts/len(df)] + [b/b.sum(axis=0)/np.sqrt(b.shape[1]) for b in blocks])
    desired = goal[:,None] * features.sum(axis=0)
    diagnostics = dict(unique_groups=ng, group_size_quantiles={str(q): float(np.quantile(counts,q)) for q in [0,.25,.5,.75,.9,.99,1]},
                       largest_group=int(counts.max()), largest_group_fraction=float(counts.max()/len(df)),
                       top_group_fractions=(np.sort(counts)[-20:][::-1]/len(df)).tolist(),
                       class_group_support=class_info, target_proportions=goal.tolist(), tolerance=p['group_tolerance'])
    reasons = []
    if ng < 3: reasons.append('Fewer than three indivisible groups')
    if counts.max()/len(df) > (goal+p['group_tolerance']).max():
        reasons.append('Largest indivisible group exceeds every allowed split upper bound')
    for col, classes in class_info.items():
        for c, info in classes.items():
            if info['groups'] < 3 or info['rows'] < 3*p['minimum_class_support']:
                reasons.append(f'{col}={c}: insufficient groups/rows for support in all three splits')
    diagnostics['proven_infeasibility_reasons'] = reasons
    best = None
    candidates = []
    for candidate in range(p['group_candidates']):
        rng = np.random.default_rng(config.random_state + candidate)
        # Descending large groups, deterministic seed-specific jitter diversifies candidates.
        priority = np.max(features, axis=1) * (1 if candidate == 0 else rng.uniform(.7,1.3,ng))
        order = np.lexsort((rng.random(ng), -priority))
        accumulated = np.zeros_like(desired)
        assignment = np.empty(ng, dtype=np.int8)
        used = np.zeros(3, dtype=int)
        for step,g in enumerate(order):
            delta = ((accumulated + features[g] - desired)**2 - (accumulated-desired)**2).sum(axis=1)
            if ng-step <= int((used==0).sum()):
                delta[used>0] = np.inf
            split = int(np.argmin(delta))
            assignment[g] = split
            used[split] += 1
            accumulated[split] += features[g]
        sizes = np.bincount(assignment, weights=counts, minlength=3)
        observed = np.array([hist[assignment==s].sum(axis=0) for s in range(3)])
        missing = int((observed < p['minimum_class_support']).sum())
        deviation = np.abs(sizes/len(df)-goal)
        distribution_error = float(np.mean(np.abs(observed/np.maximum(sizes[:,None],1)-totals/len(df))))
        score = (missing, float(np.maximum(deviation-p['group_tolerance'],0).sum()),
                 float(deviation.sum()) + distribution_error)
        candidates.append(dict(candidate=candidate, counts=sizes.astype(int).tolist(), proportions=(sizes/len(df)).tolist(),
                               inadequate_class_cells=missing, distribution_error=distribution_error, score=list(score)))
        if best is None or score < best[0]: best = score, assignment.copy(), candidate
    diagnostics.update(candidates=candidates, selected_candidate=best[2],
                       search='seeded greedy whole-group allocation; lexicographic support, tolerance excess, size + distribution error')
    splits = {name: np.flatnonzero(best[1][codes]==s) for s,name in enumerate(NAMES)}
    diagnostics['feasible_found'] = best[0][0] == 0 and best[0][1] <= 1e-12 and all(len(v)>0 for v in splits.values())
    diagnostics['feasibility_status'] = ('feasible_witness' if diagnostics['feasible_found'] else
                                          'proven_infeasible' if reasons else 'search_did_not_find_feasible_solution')
    if not diagnostics['feasible_found']:
        raise SplitValidationError('No acceptable group split; tolerance was not relaxed. See diagnostics.', diagnostics, splits)
    return splits, diagnostics


def time_values(df, config):
    from .splitting import parse_timestamp_series
    if config.timestamp_col not in df:
        raise ValueError(f'Missing timestamp column {config.timestamp_col!r}')
    times = parse_timestamp_series(df[config.timestamp_col], unit=config.timestamp_unit)
    if times.isna().any():
        raise ValueError(f'Temporal protocol requires valid timestamps; invalid/missing={int(times.isna().sum())}')
    bucket = times.dt.floor(config.temporal_bucket_freq) if config.temporal_bucket_freq else times
    return times, bucket


def ordered_bucket_split(bucket, positions, config):
    # Integer timestamp storage avoids materializing millions of Python Timestamp objects.
    values = bucket.iloc[positions].array.asi8
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) < 3:
        raise ValueError(f'At least three distinct time buckets required; observed {len(unique)}')
    prefix = counts.cumsum()
    goal = targets(config)
    # Optimize L1 count deviation across ALL admissible pairs in linear time.
    # For fixed first cut, convex cost in second cut attains a minimum at either ideal endpoint.
    first = np.arange(1,len(unique)-1)
    a = prefix[first-1]
    ideal = len(positions)*goal
    best = None
    for wanted in (a+ideal[1], np.full(len(a),ideal[0]+ideal[1])):
        for shift in (0,-1):
            second = np.clip(np.searchsorted(prefix,wanted)+1+shift, first+1,len(unique)-1)
            b = prefix[second-1]
            cost = abs(a-ideal[0])+abs(b-a-ideal[1])+abs(len(positions)-b-ideal[2])
            j = int(np.argmin(cost))
            choice = (float(cost[j]),int(first[j]),int(second[j]))
            if best is None or choice < best: best = choice
    _, i,j = best
    membership = np.where(inverse<i,0,np.where(inverse<j,1,2))
    return {name: np.sort(positions[membership==s]) for s,name in enumerate(NAMES)}


def temporal_split(df, config, per_class=False):
    _, bucket = time_values(df, config)
    if not per_class: return ordered_bucket_split(bucket,np.arange(len(df)),config)
    target = config.type_col if config.type_col in df else config.label_col
    if target not in df or df[target].isna().any(): raise ValueError('Per-class temporal requires complete classes')
    parts = {s: [] for s in NAMES}
    infeasible = {}
    for c, pos in df.groupby(target, observed=True, sort=True).indices.items():
        try:
            split = ordered_bucket_split(bucket,np.asarray(pos),config)
        except ValueError as exc:
            infeasible[str(c)] = str(exc)
            continue
        for s in NAMES: parts[s].append(split[s])
    if infeasible:
        raise SplitValidationError('Per-class temporal infeasible; no row fallback or class exclusion.', {'infeasible_classes': infeasible})
    return {s: np.sort(np.concatenate(parts[s])) for s in NAMES}


def row_validation(splits, n):
    if set(splits) != set(NAMES): raise ValueError('Exactly train/val/test are required')
    seen = np.zeros(n,dtype=np.uint8)
    invalid = 0
    duplicate = 0
    for s in NAMES:
        idx = np.asarray(splits[s])
        if idx.ndim != 1 or idx.dtype.kind not in 'iu': raise ValueError('Split indices must be one-dimensional integers')
        bad = (idx<0)|(idx>=n)
        invalid += int(bad.sum())
        valid = idx[~bad]
        unique = np.unique(valid)
        duplicate += len(valid)-len(unique)
        seen[unique] += 1
    overlap = int((seen>1).sum())
    missing = int((seen==0).sum())
    return dict(out_of_range=invalid, duplicates_within_splits=duplicate, overlapping_rows=overlap, missing_rows=missing,
                ok=not(invalid or duplicate or overlap or missing) and all(len(splits[s])>0 for s in NAMES))


def chronology(df, splits, config):
    times,bucket = time_values(df, config)
    intervals = {}
    sets = {}
    for s,idx in splits.items():
        t,b = times.iloc[idx],bucket.iloc[idx]
        sets[s] = pd.Index(b.unique())
        intervals[s] = dict(start_utc=str(t.min()) if len(t) else None, end_utc=str(t.max()) if len(t) else None,
                            buckets=len(sets[s]), records=len(idx))
    strict = all(len(splits[s]) for s in NAMES) and bool(times.iloc[splits['train']].max()<times.iloc[splits['val']].min() and times.iloc[splits['val']].max()<times.iloc[splits['test']].min())
    overlaps = {f'{a}_{b}_bucket_overlap': len(sets[a].intersection(sets[b])) for a,b in [('train','val'),('train','test'),('val','test')]}
    return dict(intervals=intervals, **overlaps, global_chronology_preserved=bool(strict and not any(overlaps.values())),
                ok=bool(strict and not any(overlaps.values())))


def distribution_report(df, splits, config):
    out = {}
    p = parameters(config)
    for col in (config.label_col,config.type_col):
        if col not in df: continue
        classes = sorted(str(c) for c in df[col].dropna().unique())
        entries = {}
        for s,idx in splits.items():
            counts = {str(k):int(v) for k,v in df[col].iloc[idx].value_counts().items()}
            counts = {c:counts.get(c,0) for c in classes}
            entries[s] = dict(counts=counts, missing=[c for c,n in counts.items() if n==0],
                              small_support={c:n for c,n in counts.items() if 0<n<p['small_support_threshold']},
                              below_minimum=[c for c,n in counts.items() if n<p['minimum_class_support']])
        known = {c for c,n in entries['train']['counts'].items() if n>0}
        val = {c for c,n in entries['val']['counts'].items() if n>0}
        test = {c for c,n in entries['test']['counts'].items() if n>0}
        out[col] = dict(splits=entries, known_in_train=sorted(known), first_seen_validation=sorted(val-known),
                        first_seen_test=sorted(test-known-val), unseen_from_train_in_test=sorted(test-known),
                        closed_set_test_classes=sorted(test&known))
    return out


def digest_json(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def array_hash(arr):
    return 'sha256:'+hashlib.sha256(np.asarray(arr,dtype='<i8').tobytes()).hexdigest()


def population_identity(df, config):
    # Bind order and identities, including a possible sampled subset of the eligible population.
    h = hashlib.sha256()
    columns = list(dict.fromkeys([c for c in config.group_cols+[config.timestamp_col,config.label_col,config.type_col] if c in df]))
    for start in range(0,len(df),100000):
        block = df.iloc[start:start+100000]
        h.update(pd.util.hash_pandas_object(block[columns],index=True).to_numpy(dtype='<u8').tobytes())
    return dict(source_population_id=df.attrs.get('data_quality_population_id'),
                eligible_input_rows=len(df), original_rows=df.attrs.get('data_quality_report',{}).get('original_rows',len(df)),
                ordered_split_input_hash='sha256:'+h.hexdigest(), columns_hashed=columns,
                identity_definition='source population + ordered pandas index and split-column content; indices are eligible iloc positions')


def manifest(df, splits, config, search=None):
    rows = row_validation(splits,len(df))
    if not rows['ok']: raise SplitValidationError('Invalid split row partition',rows)
    out = dict(protocol_version=VERSIONS.get(config.split_strategy,config.split_strategy+'/1.0.0'),
               scientific_question=QUESTIONS[config.split_strategy], auxiliary=config.split_strategy=='temporal_per_class',
               population=population_identity(df,config), parameters=parameters(config),
               seed=config.random_state if config.split_strategy in ('group_stratified','random_stratified') else None,
               counts={s:len(splits[s]) for s in NAMES}, proportions={s:len(splits[s])/len(df) for s in NAMES},
               index_hashes={s:array_hash(splits[s]) for s in NAMES},
               identity_hashes={s:digest_json([df.attrs.get('data_quality_population_id'),array_hash(df.index.to_numpy()[splits[s]])]) for s in NAMES}
                 if df.index.dtype.kind in 'iu' else {s:digest_json(df.index[splits[s]].astype(str).tolist()) for s in NAMES},
               row_validation=rows, distributions=distribution_report(df,splits,config))
    out['group_overlap'] = overlap_from_codes(group_codes(df,config.group_cols)[0],splits) if config.group_cols else None
    out['global_chronology'] = chronology(df,splits,config) if config.timestamp_col in df else None
    ok = rows['ok']
    if config.split_strategy == 'group_stratified':
        ok = ok and out['group_overlap']['ok'] and bool(np.all(abs(np.array(list(out['proportions'].values()))-targets(config))<=parameters(config)['group_tolerance']+1e-12))
        ok = ok and all(not d['splits'][s]['below_minimum'] for d in out['distributions'].values() for s in NAMES)
    elif config.split_strategy == 'temporal': ok = ok and out['global_chronology']['ok']
    elif config.split_strategy == 'temporal_per_class':
        from .splitting import temporal_per_class_report
        report, _ = temporal_per_class_report(df,splits,config)
        out['per_class_validation'] = report
        ok = ok and report['ok']
    out['validation_ok'] = bool(ok)
    out['split_hash'] = digest_json({k:out[k] for k in ['protocol_version','population','parameters','seed','index_hashes']})
    if search is not None: out['group_search'] = search
    return out


def load_frozen_splits(directory, df, config):
    root = Path(directory)
    saved = json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
    if not saved['validation_ok']: raise ValueError('Frozen split was not approved by validation')
    splits = {s:np.load(root/'splits'/f'{s}_indices.npy',allow_pickle=False) for s in NAMES}
    current = manifest(df,splits,config)
    if not current['validation_ok'] or current['split_hash'] != saved['split_hash']:
        raise ValueError('Frozen split mismatch: population, order, indices, configuration or protocol changed')
    return splits
