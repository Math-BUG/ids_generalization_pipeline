"""Six selectors with narrow train-only inputs and a shared exact allocator."""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
import numpy as np
import pandas as pd

from .selection_budget import Selection, resolve_budget, hash_indices

METHODS = ('random','stratified_label','stratified_type','cluster_uniform','cluster_proportional','cluster_sqrt')
VERSION = 'selection_methods/1.0.0'
ALLOCATOR_VERSION = 'bounded_largest_remainder/1.0.0'


def allocate_budget(capacities, weights, B, *, minimum_coverage=True):
    """Reserve one where possible, water-fill capacities, then largest remainders.

    Ties use input position. Callers provide strata in canonical sorted order.
    Empty strata receive zero; every nonempty stratum must have positive weight.
    """
    cap=np.asarray(capacities)
    w=np.asarray(weights,dtype=np.float64)
    if cap.ndim!=1 or cap.dtype.kind not in 'iu' or len(cap)==0 or np.any(cap<0):
        raise ValueError('capacities must be a nonempty vector of nonnegative integers')
    total=sum(int(v) for v in cap)
    b=resolve_budget(B,total)
    if total>np.iinfo('int64').max or w.shape!=cap.shape or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Invalid capacities or weights')
    cap=cap.astype(np.int64)
    active=cap>0
    if np.any(w[active]<=0):
        raise ValueError('Each nonempty stratum requires a positive weight')
    quotas=np.zeros(len(cap),dtype=np.int64)
    if b==total: return cap.copy()
    if minimum_coverage and b>=int(active.sum()): quotas[active]=1
    remaining=b-int(quotas.sum())
    while remaining:
        available=cap-quotas
        ids=np.flatnonzero(available>0)
        scaled=w[ids]/w[ids].max()
        shares=remaining*scaled/scaled.sum()
        saturated=shares>=available[ids]
        if saturated.any():
            filled=ids[saturated]
            quotas[filled]+=available[filled]
            remaining-=int(available[filled].sum())
            continue
        floors=np.floor(shares).astype(np.int64)
        quotas[ids]+=floors
        remaining-=int(floors.sum())
        if remaining:
            order=np.lexsort((ids,-(shares-floors)))
            if remaining>len(ids): raise RuntimeError('Allocator numerical residual exceeds strata count')
            quotas[ids[order[:remaining]]]+=1
            remaining=0
    if int(quotas.sum())!=b or np.any(quotas>cap): raise RuntimeError('Allocator violated exact budget')
    return quotas


@dataclass(frozen=True)
class TrainStrata:
    """One permitted attribute aligned to immutable frozen train IDs."""
    train_population_hash: str
    train_index_hash: str
    kind: str
    codes: np.ndarray
    names: tuple[str,...]

    @classmethod
    def from_values(cls,population,kind,train_ids,values):
        if kind not in ('label','type','cluster'): raise ValueError('Unknown stratum kind')
        if not np.array_equal(train_ids,population.train_indices):
            raise ValueError('Strata IDs must match frozen train exactly, including order')
        series=pd.Series(values)
        if len(series)!=len(train_ids) or series.isna().any():
            raise ValueError('Strata require exactly one nonmissing value per training row')
        if kind=='cluster':
            arr=np.asarray(values)
            if arr.dtype.kind not in 'iu' or np.any(arr<0): raise ValueError('Cluster IDs must be nonnegative integers')
            codes,names=pd.factorize(arr,sort=True)
        else:
            codes,names=pd.factorize(series.astype('string'),sort=True)
        codes=np.frombuffer(codes.astype(np.int32).tobytes(),dtype=np.int32)
        return cls(population.train_population_hash,hash_indices(train_ids),kind,codes,tuple(str(v) for v in names))


@dataclass(frozen=True)
class SelectionResult:
    selection: Selection
    diagnostics: dict

    def save(self,population,directory):
        from pathlib import Path
        from .utils import write_json
        directory=Path(directory)
        self.selection.save_summary(population,directory/'selection_summary.json')
        write_json(directory/'method_summary.json',self.diagnostics)


def select_random(population,B,selection_seed):
    """Uniform IDs only: this signature cannot receive targets or clusters."""
    b=resolve_budget(B,len(population.train_indices))
    rng=_rng(selection_seed)
    ids=None if b==len(population.train_indices) else population.train_indices[rng.choice(len(population.train_indices),size=b,replace=False)]
    selection=Selection.accept(population,B,selection_seed,ids)
    return SelectionResult(selection,dict(method='random',method_version=VERSION,information_used=['train_ids','selection_seed'],quotas=[]))


def select_stratified_label(population,B,selection_seed,labels):
    return _stratified(population,B,selection_seed,labels,'stratified_label','label','proportional')


def select_stratified_type(population,B,selection_seed,types):
    return _stratified(population,B,selection_seed,types,'stratified_type','type','proportional')


def select_cluster_uniform(population,B,selection_seed,clusters):
    return _stratified(population,B,selection_seed,clusters,'cluster_uniform','cluster','uniform')


def select_cluster_proportional(population,B,selection_seed,clusters):
    return _stratified(population,B,selection_seed,clusters,'cluster_proportional','cluster','proportional')


def select_cluster_sqrt(population,B,selection_seed,clusters):
    return _stratified(population,B,selection_seed,clusters,'cluster_sqrt','cluster','sqrt')


def _rng(seed):
    if isinstance(seed,bool) or not isinstance(seed,Integral) or not 0<=seed<2**32: raise ValueError('Invalid selection_seed')
    return np.random.default_rng(seed)


def _stratified(population,B,seed,strata,method,kind,weighting):
    b=resolve_budget(B,len(population.train_indices));rng=_rng(seed)
    if (strata.kind!=kind or strata.train_population_hash!=population.train_population_hash
            or strata.train_index_hash!=hash_indices(population.train_indices) or len(strata.codes)!=len(population.train_indices)):
        raise ValueError('Wrong information kind or frozen training population for selector')
    capacities=np.bincount(strata.codes,minlength=len(strata.names))
    if weighting=='uniform':weights=np.ones(len(capacities))
    elif weighting=='sqrt':weights=np.sqrt(capacities)
    else:weights=capacities.astype(float)
    quotas=allocate_budget(capacities,weights,b)
    parts=[]
    if b!=len(population.train_indices):
        # Only one stratum's candidate positions is materialized at a time.
        for code,q in enumerate(quotas):
            if q:
                positions=np.flatnonzero(strata.codes==code)
                chosen=positions if q==len(positions) else positions[rng.choice(len(positions),size=int(q),replace=False)]
                parts.append(population.train_indices[chosen])
    ids=None if b==len(population.train_indices) else np.concatenate(parts)
    selection=Selection.accept(population,B,seed,ids)
    quota_rows=[dict(stratum=name,capacity=int(n),weight=float(w),quota=int(q)) for name,n,w,q in zip(strata.names,capacities,weights,quotas)]
    return SelectionResult(selection,dict(method=method,method_version=VERSION,allocator_version=ALLOCATOR_VERSION,
           information_used=['train_ids',kind,'selection_seed'],weighting=weighting,quotas=quota_rows,
           stratum_assignment_hash=hash_indices(strata.codes),
           minimum_coverage_applied=b>=int((capacities>0).sum()),
           omitted_strata=[name for name,q in zip(strata.names,quotas) if q==0],
           within_stratum='uniform_without_replacement',tie_break='canonical stratum order'))


def select_from_training_frame(population,B,seed,method,df,*,label_col='label',type_col='type',clusters=None):
    """Integration boundary: pass only the required attribute, sliced to train first."""
    if method=='random': return select_random(population,B,seed)
    if method in ('stratified_label','stratified_type'):
        kind='label' if method=='stratified_label' else 'type'
        col=label_col if kind=='label' else type_col
        values=df[col].iloc[population.train_indices].to_numpy()
        strata=TrainStrata.from_values(population,kind,population.train_indices,values)
        function=select_stratified_label if kind=='label' else select_stratified_type
        return function(population,B,seed,strata)
    functions={'cluster_uniform':select_cluster_uniform,'cluster_proportional':select_cluster_proportional,'cluster_sqrt':select_cluster_sqrt}
    if method not in functions: raise ValueError(f'Unknown selection method: {method}')
    if clusters is None: raise ValueError('Cluster selector requires current train-only cluster assignments')
    return functions[method](population,B,seed,clusters)
