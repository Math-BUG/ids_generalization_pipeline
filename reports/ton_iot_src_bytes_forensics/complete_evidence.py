"""Read-only checks supplementary to investigate.py; outputs stay in this folder."""
from collections import Counter
from dataclasses import asdict
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import pandas as pd
from investigate import ROOT, REPO, extract_csv, signature, TON_IOT_SCHEMA

cached = [json.loads(x) for x in (ROOT / 'records_evidence.jsonl').read_text(encoding='utf-8').splitlines()]
controls = []
for part, count in [(1, 175), (22, 690), (23, 4)]:
    records, info = extract_csv(Path(f'D:/IC/Dataset/Network_dataset_{part}.csv'), count)
    assert records == [r for r in cached if r['partition'] == part]
    info['all_cached_records_identical_to_reinspection'] = True
    controls.append(info)
    print(f'Rechecked original CSV {part}; zero IP controls: {info["zero_ip_counts_in_scanned_partition"]}', flush=True)
(ROOT / 'csv_reinspection.json').write_text(json.dumps(controls, indent=2), encoding='utf-8')

# Only filenames are inspected for upstream evidence, never unrelated data contents.
inventory = []
for folder, dirs, files in os.walk('D:/IC/Dataset'):
    dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '.git', '__pycache__'}]
    for name in files:
        if name == 'cluster_assignments.csv':
            continue
        lower = name.lower()
        if lower.endswith(('.pcap', '.pcapng', '.log', '.zeek', '.bro', '.zip', '.7z', '.gz', '.rar', '.pdf', '.docx')) or any(t in lower for t in ['groundtruth', 'description', 'readme']):
            path = Path(folder) / name
            inventory.append({'path': str(path), 'bytes': path.stat().st_size})

sample_path = Path('D:/IC/Dataset/train_test_network.csv')
before = signature(sample_path)
sample_info = {'path': str(sample_path), 'signature': before, 'rows': 0, 'src_bytes_bad': 0, 'src_ip_zero_string': 0, 'dst_ip_in_target_set': 0, 'examples': []}
destinations = {r['fields']['dst_ip'] for r in cached}
for chunk in pd.read_csv(sample_path, dtype=str, keep_default_na=False, chunksize=50000):
    sample_info['columns'] = chunk.columns.tolist()
    sample_info['rows'] += len(chunk)
    sample_info['src_bytes_bad'] += int(chunk.src_bytes.eq('0.0.0.0').sum())
    sample_info['src_ip_zero_string'] += int(chunk.src_ip.eq('0').sum())
    mask = chunk.dst_ip.isin(destinations)
    sample_info['dst_ip_in_target_set'] += int(mask.sum())
    sample_info['examples'].extend(chunk.loc[mask].head(max(0, 5-len(sample_info['examples']))).to_dict('records'))
assert before == signature(sample_path)
sample_info['source_unchanged'] = True
(ROOT / 'recovery_sources.json').write_text(json.dumps({'local_upstream_filename_candidates': inventory, 'sample_csv': sample_info}, indent=2), encoding='utf-8')

df = pd.DataFrame([{**r['fields'], 'partition': r['partition'], 'row_0based': r['row_0based'], 'physical_line': r['physical_line_start']} for r in cached])
findings = json.loads((ROOT/'findings.json').read_text(encoding='utf-8'))
domain_rows = []
for col in df.columns:
    if col not in TON_IOT_SCHEMA:
        continue
    spec = asdict(TON_IOT_SCHEMA[col])
    info = findings['schema_report']['columns'][col]
    domain_rows.append({'column': col, 'semantic_type': spec['semantic_type'], 'schema_invalid': info['invalid_count'],
                        'missing_sentinel': info['sentinel_missing_count'], 'extra_ip_syntax_invalid': 869 if col == 'src_ip' else 0,
                        'distinct_values': df[col].nunique(), 'values_top5': json.dumps(df[col].value_counts().head(5).to_dict()),
                        'minimum': spec['minimum'], 'maximum': spec['maximum'], 'integral': spec['integral']})
pd.DataFrame(domain_rows).to_csv(ROOT/'domains_by_column.csv', index=False)
patterns = []
for (sport, dport), group in df.groupby(['src_port', 'dst_port'], sort=True):
    times = pd.to_datetime(pd.to_numeric(group.ts), unit='s', utc=True)
    patterns.append({'pattern': f'{sport}/{dport}', 'count': len(group), 'destinations': group.dst_ip.nunique(),
        'first_utc': str(times.min()), 'last_utc': str(times.max()),
        'partitions': group.partition.value_counts().to_dict(),
        'src_pkts': group.src_pkts.value_counts().to_dict(), 'src_ip_bytes': group.src_ip_bytes.value_counts().to_dict(),
        'example': group.iloc[0].to_dict()})
partition_times = []
for part, group in df.groupby('partition'):
    times = pd.to_datetime(pd.to_numeric(group.ts), unit='s', utc=True)
    partition_times.append({'partition': int(part), 'count': len(group), 'first_ts': int(pd.to_numeric(group.ts).min()), 'last_ts': int(pd.to_numeric(group.ts).max()), 'first_utc': str(times.min()), 'last_utc': str(times.max())})
extra = {'patterns': patterns, 'partition_times': partition_times,
    'exact_duplicate_rows_beyond_first': int(df[list(cached[0]['fields'])].duplicated().sum()),
    'destination_ipv6_multicast_count': sum(ipaddress.ip_address(v).version == 6 and ipaddress.ip_address(v).is_multicast for v in df.dst_ip),
    'duration_range': [float(pd.to_numeric(df.duration).min()), float(pd.to_numeric(df.duration).max())],
    'src_ip_bytes_minus_fixed_ipv6_header': 'Not applied: no packet headers, versioned extraction provenance, or extension-header accounting available.'}
(ROOT/'extra_findings.json').write_text(json.dumps(extra, indent=2), encoding='utf-8')
for dimension, rows in findings['summaries'].items():
    pd.DataFrame(rows).to_csv(ROOT/f'summary_{dimension}.csv', index=False)
print(json.dumps({'recovery': sample_info, 'upstream_candidates': inventory, 'extra': extra}, indent=2))
