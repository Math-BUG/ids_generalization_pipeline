"""Optional small real-CUDA smoke. Never reads the dataset or runs supervised IDS."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def run(output_dir):
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    report={'status':'skipped','requested_rows':20000,'k':30,'real_cuda_executed':False}
    missing=[name for name in ['cupy','cuml'] if importlib.util.find_spec(name) is None]
    if missing:
        report['reason']='Required GPU backend packages unavailable: '+', '.join(missing)
    else:
        try:
            import cupy as cp
            from cuml.cluster import KMeans  # noqa: F401
            count=cp.cuda.runtime.getDeviceCount()
            if count < 1:
                raise RuntimeError('No accessible CUDA device')
            cp.zeros(1,dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
        except Exception as exc:
            report['reason']='GPU runtime unavailable: '+repr(exc)
        else:
            import numpy as np
            import pandas as pd
            from ids_pipeline.config import PipelineConfig
            from ids_pipeline.feature_policy import BEHAVIORAL_STRICT_FEATURES,FeaturePolicy
            from ids_pipeline.preprocessing import fit_transform_preprocessing
            from ids_pipeline.clustering import _fit_gpu_kmeans
            numeric=[c for c in BEHAVIORAL_STRICT_FEATURES if c!='conn_state']
            rng=np.random.default_rng(42)
            frame=pd.DataFrame({c:rng.integers(0,10000,20000).astype(str) for c in numeric})
            frame['conn_state']=rng.choice(['S0','SF','REJ','OTH'],20000)
            frame.loc[12000:15999,'conn_state']='ONLY_VALIDATION'
            frame.loc[16000:,'conn_state']='ONLY_TEST'
            split={'train':np.arange(12000),'val':np.arange(12000,16000),'test':np.arange(16000,20000)}
            cfg=PipelineConfig(feature_policy='behavioral_strict',compute_backend='gpu',selected_k=30,
                               svd_components=0,cluster_max_iter=20)
            cols=FeaturePolicy.from_name(cfg.feature_policy).select_features(frame)
            X,bundle=fit_transform_preprocessing(frame,split,cols,cfg,output)
            assert bundle.backend=='gpu','Backend override prevented actual GPU preprocessing'
            assert bundle.numeric_cols==numeric and bundle.categorical_cols==['conn_state']
            metadata=bundle.preprocessor['conn_state_encoding']
            width=len(metadata['vocabulary'])
            assert width==4 and {x.shape[1] for x in X.values()}=={8+width}
            assert bool(cp.all(X['val'][:,-width:]==0)) and bool(cp.all(X['test'][:,-width:]==0))
            assert bool(cp.all(X['train'][:,-width:].sum(axis=1)==1))
            _,labels,distances=_fit_gpu_kmeans(X,cfg)
            cp.cuda.Stream.null.synchronize()
            assert all(len(labels[k])==len(split[k]) for k in split)
            assert all(np.isfinite(d).all() for d in distances.values())
            report.update(status='passed',real_cuda_executed=True,numeric_columns=bundle.numeric_cols,
                          categorical_columns=bundle.categorical_cols,conn_state_encoding=metadata,
                          shapes={k:list(v.shape) for k,v in X.items()},clustering='cuML KMeans completed')
    (output/'cuda_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='reports/increment_2/cuda_smoke')
    args=parser.parse_args()
    run(args.output_dir)
