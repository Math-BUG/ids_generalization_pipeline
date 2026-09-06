import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_pipeline import data_loading
from ids_pipeline.data_quality_policy import DataQualityPopulation, known_corruption_mask, source_fingerprint
from ids_pipeline.dataset_schema import SchemaValidationError, normalize_dataset_schema


def fixture_frame():
    return pd.DataFrame({'ts':[1554220429]*6, 'src_ip':['0','0.0.0.0','0','host','x','x'],
                         'dst_ip':['ff02::2']*6, 'src_bytes':['0.0.0.0','10','20','0.0.0.0','30','30'],
                         'dns_qtype':[0]*6, 'label':[0,0,1,1,0,0],
                         'type':['normal','normal','ddos','ddos','normal','normal']}, index=[8,8,3,2,1,1])


def apply(frame, path, chunk_size=100):
    pop = DataQualityPopulation()
    pop.register_file(path.name, source_fingerprint(path))
    chunks = [pop.apply(frame.iloc[i:i+chunk_size],source_id=path.name,row_offset=i) for i in range(0,len(frame),chunk_size)]
    return pd.concat(chunks), pop.report()


def test_exact_rule_preserves_ips_and_does_not_read_targets():
    frame=fixture_frame()
    assert known_corruption_mask(frame).tolist() == [True,False,False,True,False,False]
    assert known_corruption_mask(frame.drop(columns=['label','type','src_ip'])).tolist() == [True,False,False,True,False,False]
    changed=frame.copy()
    changed['label']=[1,1,0,0,1,1]
    changed['type']='arbitrary'
    pd.testing.assert_series_equal(known_corruption_mask(frame),known_corruption_mask(changed))


@pytest.mark.parametrize('token',['0','0.0.0.0 ', ' 0.0.0.0', 'other-invalid', '-', '', None, np.inf, -1])
def test_no_other_cleaning_or_extended_sentinels(token):
    assert not known_corruption_mask(pd.DataFrame({'src_bytes':[token]})).iloc[0]


def test_remaining_schema_counts_identity_duplicates_and_source_preserved(tmp_path):
    frame=fixture_frame()
    path=tmp_path/'Network_dataset_1.csv'
    frame.to_csv(path,index=False)
    original=path.read_bytes()
    kept, report=apply(frame,path,2)
    pd.testing.assert_frame_equal(kept,frame.iloc[[1,2,4,5]])
    with pytest.raises(SchemaValidationError) as exc:
        normalize_dataset_schema(frame)
    assert report['quarantined_rows'] == exc.value.report['columns']['src_bytes']['invalid_count'] == 2
    normalized, schema_report=normalize_dataset_schema(kept)
    assert schema_report['ok'] and normalized.index.equals(kept.index)
    assert report['original_rows'] == 6 and report['eligible_rows'] == 4
    assert report['files'][0]['quarantined_row_positions_0based'] == [0,3]
    assert report['eligible_target_distributions']['label']['values'] == {'0':3,'1':1}
    assert report['eligible_target_distributions']['type']['values'] == {'ddos':1,'normal':3}
    assert path.read_bytes() == original


def test_population_id_stable_across_chunks_and_sensitive_to_source_and_membership(tmp_path):
    frame=fixture_frame()
    path=tmp_path/'Network_dataset_1.csv'
    frame.to_csv(path,index=False)
    _, full=apply(frame,path)
    _, chunked=apply(frame,path,1)
    assert full['population_id'] == chunked['population_id']
    changed=frame.copy()
    changed.iloc[0,changed.columns.get_loc('src_bytes')]='0'
    _, membership=apply(changed,path)
    assert membership['population_id'] != full['population_id']
    changed.to_csv(path,index=False)
    _, content=apply(frame,path)
    assert content['population_id'] != full['population_id']


def test_repeated_or_out_of_order_chunk_rejected(tmp_path):
    path=tmp_path/'one.csv'; path.write_text('x')
    pop=DataQualityPopulation(); pop.register_file(path.name,source_fingerprint(path))
    pop.apply(fixture_frame(),source_id=path.name,row_offset=0)
    with pytest.raises(ValueError,match='contiguous'):
        pop.apply(fixture_frame(),source_id=path.name,row_offset=0)


def test_all_quarantined_chunk_and_empty_eligible_schema(tmp_path):
    path=tmp_path/'one.csv'
    frame=fixture_frame().iloc[[0,3]]
    frame.to_csv(path,index=False)
    kept, report=apply(frame,path,1)
    assert kept.empty and report['quarantined_rows']==2 and report['eligible_rows']==0
    _, schema=normalize_dataset_schema(kept)
    assert schema['ok']


def test_source_relocation_does_not_change_population_identifier(tmp_path):
    first=tmp_path/'first'; second=tmp_path/'second'
    first.mkdir(); second.mkdir()
    a=first/'one.csv'; b=second/'one.csv'
    frame=fixture_frame(); frame.to_csv(a,index=False); b.write_bytes(a.read_bytes())
    assert apply(frame,a)[1]['population_id']==apply(frame,b)[1]['population_id']


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_loader_quarantines_before_schema_same_cpu_gpu(tmp_path,monkeypatch,backend):
    path=tmp_path/'Network_dataset_1.parquet'
    fixture_frame().to_parquet(path,index=False)
    # Exercise GPU ingestion orchestration without requiring GPU hardware.
    monkeypatch.setattr(data_loading,'_read_parquet_with_cudf',lambda files,require_cudf: pd.read_parquet(files[0]))
    report_path=tmp_path/'quality.json'
    loaded=data_loading.load_dataset(str(path),compute_backend=backend,data_quality_report_path=report_path)
    assert loaded.index.tolist() == [1,2,4,5]
    assert loaded.src_bytes.tolist() == [10,20,30,30]
    assert loaded.src_ip.tolist()[:2] == ['0.0.0.0','0']
    report=json.loads(report_path.read_text())
    assert report['quarantined_rows']==2
    assert report['population_id']==loaded.attrs['data_quality_population_id']
    _, expected=apply(pd.read_parquet(path),path,2)
    assert expected['population_id']==report['population_id']


def test_unrecognized_invalid_value_still_aborts_loader_after_quarantine(tmp_path):
    frame=fixture_frame(); frame.iloc[1,frame.columns.get_loc('src_bytes')]='bad'
    path=tmp_path/'one.csv'; frame.to_csv(path,index=False)
    with pytest.raises(SchemaValidationError) as exc:
        data_loading.load_dataset(str(path),schema_report_path=tmp_path/'schema.json')
    assert exc.value.report['columns']['src_bytes']['invalid_count']==1
    assert json.loads((tmp_path/'data_quality_report.json').read_text())['quarantined_rows']==2


def test_full_audit_matches_loader_manifest_and_retains_residual_diagnostics(tmp_path):
    spec=importlib.util.spec_from_file_location('quality_audit',Path(__file__).parents[1]/'scripts/audit_ton_iot_quality.py')
    audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    source=tmp_path/'source'; source.mkdir()
    for i in [1,2]:
        fixture_frame().to_parquet(source/f'Network_dataset_{i}.parquet',index=False)
    result=audit.run_audit(source,tmp_path/'audit',chunk_size=2,expected_files=2)
    loaded=data_loading.load_dataset(str(source),data_quality_report_path=tmp_path/'quality.json')
    assert result['complete'] and result['compatible']
    assert result['invalid_cells_after_quarantine']==0
    assert (result['original_rows'],result['quarantined_rows'],result['eligible_rows'])==(12,4,8)
    assert result['population_id']==loaded.attrs['data_quality_population_id']
    assert loaded.index.tolist()==[1,2,4,5,7,8,10,11]
    frame=fixture_frame(); frame.iloc[4,frame.columns.get_loc('src_bytes')]='bad'
    frame.to_parquet(source/'Network_dataset_1.parquet',index=False)
    failed=audit.run_audit(source,tmp_path/'failed',chunk_size=2,expected_files=2)
    assert not failed['compatible'] and failed['invalid_cells_after_quarantine']==1
    assert failed['files'][0]['issues']['src_bytes']['conversion_failure']['examples'][0]['row_0based']==4
