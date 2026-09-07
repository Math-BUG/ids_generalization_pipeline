"""Split-only checkpoint on the frozen eligible TON_IoT population; no models."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ids_pipeline.config import PipelineConfig
from ids_pipeline.data_quality_policy import DataQualityPopulation, source_fingerprint, assert_source_unchanged
from ids_pipeline.dataset_schema import normalize_dataset_schema
from ids_pipeline.splitting import create_splits, save_splits
from ids_pipeline.split_protocols import SplitValidationError, manifest
from ids_pipeline.utils import write_json


def read_population(checkpoint, output, chunk_size):
    reference=json.loads(Path(checkpoint).read_text(encoding='utf-8'))
    population=DataQualityPopulation()
    columns=['src_ip','dst_ip','service','proto','ts','label','type','src_bytes']
    strings=['src_ip','dst_ip','service','proto','type']
    tables=[]; identities=[]; global_offset=0
    for item in reference['files']:
        path=Path(item['source_path']); source_id=item['source_id']
        fingerprint=source_fingerprint(path)
        if fingerprint['sha256']!=item['source_fingerprint']['sha256']:
            raise ValueError(f'Source differs from frozen Increment 1: {source_id}')
        population.register_file(source_id,fingerprint,source_path=str(path))
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_size,columns=columns):
            raw=batch.to_pandas();n=len(raw)
            raw.index=pd.RangeIndex(global_offset+offset,global_offset+offset+n)
            eligible=population.apply(raw,source_id=source_id,row_offset=offset)
            normalized,_=normalize_dataset_schema(eligible)
            identities.append(normalized.index.to_numpy(dtype='int64'))
            normalized=normalized.drop(columns='src_bytes')
            for col in strings: normalized[col]=normalized[col].astype('string')
            table=pa.Table.from_pandas(normalized,preserve_index=False)
            for col in strings:
                table=table.set_column(table.schema.get_field_index(col),col,table[col].dictionary_encode())
            tables.append(table)
            offset+=n
        global_offset+=offset
        assert_source_unchanged(path,fingerprint)
        print(f'Read {source_id}: {offset:,} original rows',flush=True)
    quality=population.save(output/'data_quality_population.json')
    if quality['population_id']!=reference['population_id'] or quality['eligible_rows']!=22338152:
        raise ValueError('Eligible population differs from frozen checkpoint')
    table=pa.concat_tables(tables)
    del tables
    df=table.to_pandas(categories=strings)
    del table
    df.index=np.concatenate(identities)
    df.attrs['data_quality_population_id']=quality['population_id']
    df.attrs['data_quality_report']=quality
    gc.collect()
    return df


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='reports/ton_iot_quarantine_checkpoint/parquet/data_quality_report.json')
    parser.add_argument('--output',default='reports/increment_3/real')
    parser.add_argument('--chunk-size',type=int,default=100000)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    df=read_population(args.checkpoint,output,args.chunk_size)
    print(f'Eligible split-only frame: {len(df):,} rows; {df.memory_usage(deep=True).sum()/2**20:.1f} MiB',flush=True)
    # Persist the exact identity mapping once, shared across all protocols.
    np.save(output/'eligible_original_positions.npy',df.index.to_numpy(dtype='<i8'),allow_pickle=False)
    summary={}
    for strategy in ('group_stratified','temporal','temporal_per_class'):
        print(f'Generating {strategy}',flush=True)
        config=PipelineConfig(split_strategy=strategy,group_cols=['src_ip','dst_ip','service','proto'],
                              timestamp_col='ts',timestamp_unit='s',temporal_bucket_freq='1s',
                              test_size=.25,val_size=.20,random_state=42,
                              extra={'split_export_csv':False})
        root=output/strategy;root.mkdir(exist_ok=True)
        try:
            splits=create_splits(df,config)
            save_splits(splits,root,df,config)
            report=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
            summary[strategy]={'status':'validated','split_hash':report['split_hash'],'counts':report['counts']}
        except SplitValidationError as exc:
            write_json(root/'failure_diagnostics.json',exc.diagnostics)
            summary[strategy]={'status':'infeasible_or_search_failed','reason':str(exc)}
            if exc.splits is not None:
                report=manifest(df,exc.splits,config,exc.diagnostics)
                write_json(root/'best_candidate_not_approved.json',report)
                summary[strategy]['counts']=report['counts']
        print(json.dumps(summary[strategy]),flush=True)
        write_json(output/'summary.json',summary)
        df.attrs.pop('group_split_search',None)
        gc.collect()


if __name__=='__main__': main()
