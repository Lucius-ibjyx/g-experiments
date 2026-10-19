"""Read-only comparison of the fork against the recorded original and its results."""
from pathlib import Path
import hashlib,json
import pandas as pd
root=Path(__file__).resolve().parent
snapshot=json.loads((root/'original_snapshot.json').read_text())
source=Path(snapshot['source'])
changed=[name for name,checksum in snapshot['hashes'].items()
         if not (source/name).is_file() or hashlib.sha256((source/name).read_bytes()).hexdigest()!=checksum]
checks=[]
for task in sorted((root/'output').glob('P*')):
 if not task.is_dir():continue
 p=task/'gaze_stage_aoi_fused.csv'
 if not p.exists():continue
 old=source/'output'/task.name/p.name
 checks.append({'file':str(p.relative_to(root/'output')),'identical':old.exists() and p.read_bytes()==old.read_bytes()})
relative=['group_reports/coverage_regions.csv',
          '_group_spatial_v033/eccentricity_group_stage.csv',
          '_group_spatial_v033/eccentricity_participant_stage.csv']
relative += [str(p.relative_to(root/'output')) for p in sorted((root/'output/candidate_positions').glob('*/*__selected.csv'))]
for name in relative:
 a=root/'output'/name;b=source/'output'/name
 checks.append({'file':name,'identical':a.exists() and b.exists() and a.read_bytes()==b.read_bytes()})
pd.DataFrame(checks).to_csv(root/'主要结果对照.csv',index=False,encoding='utf-8-sig')
result={'original_files_checked':len(snapshot['hashes']), 'original_files_changed':changed,
        'result_files_compared':len(checks),'identical_result_files':sum(c['identical'] for c in checks),
        'different_result_files':[c['file'] for c in checks if not c['identical']],
        'qc':pd.read_csv(root/'output/_spatial_coordinate_qc/spatial_coordinate_qc_all_datasets.csv').qc_status.value_counts().to_dict(),
        'run_status':json.loads((root/'output/run_status.json').read_text()),
        'numeric_validity_test':json.loads((root/'numeric_validity_verification.json').read_text()),
        'qc_fault_test':json.loads((root/'qc_fault_verification.json').read_text())}
(root/'试跑验证记录.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
if changed or result['different_result_files']:raise SystemExit(1)
