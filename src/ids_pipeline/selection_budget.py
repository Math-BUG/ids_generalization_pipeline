"""Real-row selection contract; deliberately contains no compression selectors."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import numpy as np

from .split_protocols import NAMES, digest_json, load_frozen_splits
from .utils import write_json

CONTRACT_VERSION = 'selection_budget/1.0.0'
BLOCK = 100000


def resolve_budget(requested, train_n):
    if isinstance(train_n, bool) or not isinstance(train_n, Integral) or train_n < 1:
        raise ValueError('Training population must contain at least one row')
    if isinstance(requested, str) and requested == 'full':
        return int(train_n)
    if isinstance(requested, bool) or not isinstance(requested, Integral):
        raise ValueError('selection_budget must be an integer or exactly full')
    if not 1 <= requested <= train_n:
        raise ValueError(f'Impossible selection_budget={requested}; require 1 <= B <= N={train_n}')
    return int(requested)


def validate_budget_config(config):
    for stage in ('split', 'clustering', 'selection', 'model'):
        config.seed_for(stage)
    if config.selection_budget is None:
        return
    # Validate syntax without capping or pretending the actual train size is known.
    value = config.selection_budget
    if not (isinstance(value, str) and value == 'full'):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError('selection_budget must be a positive integer or full')
    if (config.use_representatives_for_supervised or config.representative_strategy != 'full'
            or config.representatives_per_cluster != 20 or config.boundary_per_cluster != 5):
        raise ValueError('selection_budget is incompatible with legacy representatives parameters; '
                         'use selection_budget: null for legacy mode, or remove legacy overrides')


def hash_indices(indices):
    h = hashlib.sha256()
    for offset in range(0, len(indices), BLOCK):
        h.update(np.asarray(indices[offset:offset+BLOCK], dtype='<i8').tobytes())
    return 'sha256:' + h.hexdigest()


def integer_indices(values):
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.dtype.kind not in 'iu' or (len(arr) and arr.max() > np.iinfo('int64').max):
        raise ValueError('Real row identities must be one-dimensional integer eligible-population positions')
    if len(arr) and arr.min() < 0:
        raise ValueError('Negative row identities are invalid')
    return arr


@dataclass(frozen=True)
class FrozenTrainingPopulation:
    directory: Path
    train_indices: np.ndarray
    train_population_hash: str
    split_hash: str
    manifest: dict

    def close(self):
        mapping = getattr(self.train_indices, '_mmap', None)
        if mapping is not None and not mapping.closed:
            mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def load(cls, directory):
        """Read existing split arrays in bounded blocks; never choose new rows."""
        directory = Path(directory)
        report = json.loads((directory/'split_manifest.json').read_text(encoding='utf-8'))
        if report.get('validation_ok') is not True:
            raise ValueError('Selection requires a validated frozen split')
        digest = digest_json({k: report[k] for k in ('protocol_version','population','parameters','seed','index_hashes')})
        if digest != report['split_hash']:
            raise ValueError('Frozen split manifest hash mismatch')
        n = report['population']['eligible_input_rows']
        seen = np.zeros(n, dtype=np.uint8)
        arrays = {}
        for name in NAMES:
            # Tiny fixtures do not need a persistent OS mapping; large splits stay mapped.
            mode = 'r' if report['counts'][name] > BLOCK else None
            arr = np.load(directory/'splits'/f'{name}_indices.npy', mmap_mode=mode, allow_pickle=False)
            arr.setflags(write=False)
            integer_indices(arr)
            if len(arr) != report['counts'][name] or not len(arr) or hash_indices(arr) != report['index_hashes'][name]:
                raise ValueError(f'Frozen {name} count/hash mismatch')
            previous = -1
            for start in range(0, len(arr), BLOCK):
                block = arr[start:start+BLOCK]
                if block[0] <= previous or np.any(np.diff(block.astype('int64')) <= 0) or block[-1] >= n:
                    raise ValueError('Frozen indices must be increasing, unique and in range')
                if np.any(seen[block]):
                    raise ValueError('Frozen splits overlap')
                seen[block] = 1
                previous = int(block[-1])
            arrays[name] = arr
        if not np.all(seen):
            raise ValueError('Frozen splits do not cover the eligible population')
        population_hash = digest_json(dict(format='frozen-training-population/1.0.0',
                                           population=report['population'],
                                           train_index_hash=report['index_hashes']['train'],
                                           train_identity_hash=report['identity_hashes']['train']))
        return cls(directory, arrays['train'], population_hash, report['split_hash'], report)

    def budget_report(self, requested, seed):
        realized = resolve_budget(requested, len(self.train_indices))
        return dict(contract_version=CONTRACT_VERSION, requested_budget=requested, resolved_budget=realized,
                    train_rows=len(self.train_indices), retained_fraction=realized/len(self.train_indices),
                    selection_seed=seed, train_population_hash=self.train_population_hash,
                    frozen_split_hash=self.split_hash, status='budget_validated_only',
                    realized_budget=None, selection_hash=None)

    def bind(self, df, splits, config):
        """Use the frozen Increment 3 validator against the actual pipeline input."""
        loaded = load_frozen_splits(self.directory, df, config.for_stage('split'))
        if set(splits) != set(NAMES) or any(not np.array_equal(loaded[s], splits[s]) for s in NAMES):
            raise ValueError('Training arrays do not use the specified frozen split')


@dataclass(frozen=True)
class Selection:
    indices: np.ndarray
    requested_budget: int | str
    selection_seed: int
    train_population_hash: str
    frozen_split_hash: str
    selection_hash: str

    @classmethod
    def accept(cls, population, requested, seed, indices=None):
        """Validate IDs supplied by a future selector; full needs no selection algorithm."""
        b = resolve_budget(requested, len(population.train_indices))
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError('Invalid selection_seed')
        if indices is None:
            if b != len(population.train_indices):
                raise ValueError('Compressed budget requires explicit selected row IDs; no selector is implemented yet')
            indices = population.train_indices
        values = integer_indices(indices)
        if len(values) != b:
            raise ValueError(f'Selection must realize exactly B={b}; received {len(values)}')
        # Canonical order makes selection a set and B=N equivalent to full.
        values = np.sort(np.asarray(values, dtype=np.int64))
        _validate_membership(population.train_indices, values, b)
        digest = _selection_hash(population, values)
        # Immutable backing bytes prevent accidental mutation of the common selection.
        values = np.frombuffer(values.tobytes(), dtype=np.int64)
        return cls(values, requested, int(seed), population.train_population_hash, population.split_hash, digest)

    def validate(self, population, requested, seed):
        b = resolve_budget(requested, len(population.train_indices))
        if (resolve_budget(self.requested_budget, len(population.train_indices)) != b
                or self.selection_seed != seed or self.train_population_hash != population.train_population_hash
                or self.frozen_split_hash != population.split_hash):
            raise ValueError('Selection budget, seed or population binding mismatch')
        _validate_membership(population.train_indices, self.indices, b)
        if _selection_hash(population, self.indices) != self.selection_hash:
            raise ValueError('Selection hash mismatch')
        return np.searchsorted(population.train_indices, self.indices)

    def report(self, population):
        self.validate(population, self.requested_budget, self.selection_seed)
        out = population.budget_report(self.requested_budget, self.selection_seed)
        out.update(realized_budget=len(self.indices), selection_hash=self.selection_hash,
                   status='selection_validated', identity_definition='sorted eligible iloc IDs bound to frozen train and split')
        return out

    def save_summary(self, population, path):
        report = self.report(population)
        path = Path(path)
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved.get('selection_hash') != report['selection_hash']:
                raise ValueError('Refusing to replace the selection of this experiment; use another output directory')
        write_json(path, report)


def _validate_membership(train, selected, b):
    integer_indices(selected)
    if len(selected) != b or (len(selected)>1 and np.any(np.diff(selected.astype('int64')) <= 0)):
        raise ValueError('Selection must contain exactly B distinct IDs in canonical order')
    positions = np.searchsorted(train, selected)
    if np.any(positions >= len(train)) or not np.array_equal(train[positions], selected):
        raise ValueError('Selection contains rows outside the frozen training population')


def _selection_hash(population, indices):
    # Content identity excludes seed and spelling of B; equal sets must have equal hashes.
    return digest_json(dict(contract_version=CONTRACT_VERSION, train_population_hash=population.train_population_hash,
                            frozen_split_hash=population.split_hash, selected_index_hash=hash_indices(indices)))


def prepare_fit_selection(X, y_train, splits, population, selection, config):
    """Final shared guard before each RF fit; slice X and y with the same row mapping."""
    validate_budget_config(config)
    if not np.array_equal(splits['train'], population.train_indices):
        raise ValueError('Current training row order differs from frozen train')
    positions = selection.validate(population, config.selection_budget, config.seed_for('selection'))
    if X['train'].shape[0] != len(population.train_indices) or len(y_train) != len(population.train_indices):
        raise ValueError('Training feature/target matrices do not match the frozen row mapping')
    x_selected, y_selected = X['train'][positions], np.asarray(y_train)[positions]
    b = resolve_budget(config.selection_budget, len(population.train_indices))
    if x_selected.shape[0] != b or len(y_selected) != b:
        raise ValueError('Fit matrix does not realize B')
    return x_selected, y_selected
