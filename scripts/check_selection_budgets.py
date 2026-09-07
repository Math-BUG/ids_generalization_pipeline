"""Read frozen split artifacts and validate B only; never select or fit rows."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ids_pipeline.selection_budget import FrozenTrainingPopulation
from ids_pipeline.utils import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir',default='reports/increment_3/real/group_stratified')
    parser.add_argument('--output',default='reports/increment_4/real_budget_validation.json')
    args=parser.parse_args()
    population=FrozenTrainingPopulation.load(args.split_dir)
    budgets=[population.budget_report(b,42) for b in [1000,10000,100000,1000000,'full']]
    root=Path(args.output).parent
    before=json.loads((root/'frozen_before.json').read_text(encoding='utf-8'))
    unchanged={name:hashlib.sha256(Path(f'src/ids_pipeline/{name}.py').read_bytes()).hexdigest()==digest
               for name,digest in before.items()}
    assert all(unchanged.values())
    report=dict(train_rows=len(population.train_indices),train_population_hash=population.train_population_hash,
                frozen_split_hash=population.split_hash, frozen_files_unchanged=unchanged,
                budgets=budgets, selected_rows_created=0, model_fits=0,
                validation='Frozen manifests, membership, counts and array hashes verified; no dataset or assignments read')
    write_json(args.output,report)
    population.close()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
