"""Read-only acceptance checks for the three recordings added on 2026-10-18."""
from pathlib import Path
import json
import hashlib
import pandas as pd

root = Path(__file__).resolve().parent
output = root/'output'
state = json.loads((output/'run_status.json').read_text())
assert state.get('aggregation_finished') is True, state
summary = pd.read_csv(output/'pilot_run_summary.csv')
assert len(summary) == 28 and summary.status.eq('OK').all(), summary.to_dict('records')
qc = pd.read_csv(output/'result_quality.csv')
coverage = pd.read_csv(output/'group_reports/coverage_regions.csv')
new_names = ['P05_潘佳怡_T2.4', 'P05_潘佳怡_T3.2', 'P06_张洛宜_T3.2']
required = ['analysis_results.xlsx', 'stage_summary.csv', 'aoi_summary.csv',
            'gaze_stage_aoi_fused.csv', 'RESULT_STATUS.txt',
            'report_figures/whole_task__full.png',
            'report_figures/whole_task__visual_field.png']
new_rows = []
for name in new_names:
    folder = output/name
    for filename in required:
        assert (folder/filename).is_file(), str(folder/filename)
    task_summary = pd.read_csv(folder/'task_summary.csv').iloc[0]
    q = qc.loc[qc.dataset.eq(name)].iloc[0]
    stages = pd.read_csv(folder/'stages_standardized.csv')
    aoi = pd.read_csv(folder/'aoi_summary.csv')
    assert not aoi.empty, name
    new_rows.append({'dataset': name, 'task_id': task_summary.task_id,
                     'samples': int(task_summary.samples),
                     'valid_gaze_samples': int(task_summary.valid_gaze_samples),
                     'qc_status': q.qc_status, 'stages': len(stages),
                     'aoi_summary_rows': len(aoi)})
for task, n in [('T2.4', 1), ('T3.2', 2), ('T3.1', 4)]:
    rows = coverage.loc[coverage.dataset.eq('group_reports/'+task) & coverage.scope.eq('whole_task')]
    assert len(rows) == 2 and rows.participant_n.eq(n).all(), (task, rows.to_dict('records'))
    for filename in ['whole_task__selected.csv', 'whole_task__selection.json', 'whole_task__map.png']:
        assert (output/'candidate_positions'/task/filename).is_file(), (task, filename)
assert not (output/'group_reports/T3').exists()
assert not list(output.glob('P05_张洛宜*'))
snapshot = json.loads((root/'original_snapshot.json').read_text())
source = Path(snapshot['source'])
changed = [name for name, checksum in snapshot['hashes'].items()
           if not (source/name).is_file() or hashlib.sha256((source/name).read_bytes()).hexdigest() != checksum]
assert not changed, changed
result = {'recordings': 20, 'tasks': len(summary), 'task_statuses': summary.status.value_counts().to_dict(),
          'qc_statuses': qc.qc_status.value_counts().to_dict(), 'run_status': state,
          'new_tasks': new_rows, 'original_files_checked': len(snapshot['hashes']),
          'original_files_changed': changed, 'acceptance_checks_passed': True}
(root/'新增三份数据试跑核对.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
lines = ['# 新增三份数据完整试跑', '',
         '已确认潘佳怡为P05、张洛宜为P06；张洛宜文件夹已改名为P06_张洛宜_T3.2。', '',
         f'20份录制、{len(summary)}个任务已通过计算处理，完整流程包括任务结果、坐标QC、空间底图、热力图、覆盖区域和候选位置。', '',
         '坐标QC：'+str(result['qc_statuses'])+'。待复核状态见output/result_quality.csv；不把计算结束等同研究结果全部已核实。', '',
         '| 新增任务 | 样本数 | 有效二维眼动样本数 | 阶段数 | QC |',
         '|---|---:|---:|---:|---|']
for row in new_rows:
    lines.append(f"| {row['dataset']} | {row['samples']} | {row['valid_gaze_samples']} | {row['stages']} | {row['qc_status']} |")
lines += ['', '群体结果确认：T3.1为4人，T3.2为2人，T2.4为1人；T3.1与T3.2独立输出。', '',
          '桌面旧程序核对的979份文件均未改变。本次没有修改分析公式、QC阈值或已有汇报文档。']
(root/'新增三份数据试跑结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
