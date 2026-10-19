"""Isolated full rebuild with archived outputs and explicit QC completion status."""
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import json
import uuid
import pandas as pd
from g_analysis.run_quality import result_state

ROOT = Path(__file__).resolve().parent


def prepare_output(root):
    output = root/'output'
    if output.is_symlink():
        raise RuntimeError('output不能是符号链接；请使用本副本内独立的结果目录。')
    archive = None
    if output.exists():
        history = root/'output_history'
        history.mkdir(exist_ok=True)
        archive = history/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        output.rename(archive)
    output.mkdir()
    return output, archive


def write_status(output, status, **details):
    payload = {'status': status, 'automatic_qc_only': True,
               'manual_sync_and_aoi_review': 'not_verified_by_program', **details}
    (output/'run_status.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def quality_table(output, qc):
    qc = qc.copy()
    qc['included_in_spatial_aggregation'] = qc.qc_status.isin(['PASS', 'WARN'])
    qc['result_status'] = qc.qc_status.map({
        'PASS': '自动坐标检查通过；仍需人工抽查同步及AOI',
        'WARN': '试算结果待复核；纳入汇总会使相关汇总待复核',
        'FAIL': '仅供排查；已排除空间汇总和候选位置计算'})
    qc.to_csv(output/'result_quality.csv', index=False, encoding='utf-8-sig')
    for _, row in qc.iterrows():
        task = output/row.dataset
        if task.is_dir():
            (task/'RESULT_STATUS.txt').write_text(row.result_status+'\n', encoding='utf-8')
    return qc


def main():
    data = ROOT/'data'
    if not data.is_dir() or not any(p.is_dir() and not p.name.startswith('.') for p in data.iterdir()):
        print('请先在data里放入P01_姓名_T4等录制文件夹，再运行。')
        return 1
    output, archive = prepare_output(ROOT)
    logs = output/'run_logs'
    logs.mkdir()
    write_status(output, 'running', previous_output_archive=str(archive) if archive else None)
    if archive:
        print(f'旧结果已备份：{archive}', flush=True)
    print('从空output重新计算；信息框为Pilot占位尺寸，候选位置仍为探索性结果。', flush=True)
    steps = ['00_check_environment.py', '01_run_pilot.py', '04_spatial_coordinate_qc.py',
             '03_build_spatial_baseline.py', '07_generate_reports.py', '08_candidate_positions.py']
    qc = None
    for i, step in enumerate(steps, 1):
        print(f'[{i}/{len(steps)}] {step}', flush=True)
        result = subprocess.run([sys.executable, str(ROOT/step)], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        log = logs/f'{step}.log'
        log.write_text(result.stdout, encoding='utf-8')
        qc_path = output/'_spatial_coordinate_qc'/'spatial_coordinate_qc_all_datasets.csv'
        if step == '04_spatial_coordinate_qc.py' and result.returncode in [0, 1, 2] and qc_path.exists():
            qc = quality_table(output, pd.read_csv(qc_path))
            state, _ = result_state(qc)
            write_status(output, state, aggregation_finished=False,
                         previous_output_archive=str(archive) if archive else None)
            print('自动坐标检查完成：'+str(qc.qc_status.value_counts().to_dict()), flush=True)
            if state == 'blocked' or result.returncode == 1:
                write_status(output, 'blocked', failed_step=step, aggregation_finished=False,
                             previous_output_archive=str(archive) if archive else None)
                print('没有通过或可复核的空间数据，停止生成空间汇总；排查结果见result_quality.csv。', flush=True)
                return 1
            continue
        if result.returncode:
            write_status(output, 'blocked', failed_step=step, exit_code=result.returncode,
                         previous_output_archive=str(archive) if archive else None)
            print(result.stdout[-6000:])
            print(f'已停止，不能作为完整结果。完整记录：{log}', flush=True)
            return 1
        print('该步骤已运行', flush=True)
    state, code = result_state(qc)
    counts = {str(k): int(v) for k, v in qc.qc_status.value_counts().items()}
    write_status(output, state, aggregation_finished=True, qc_counts=counts,
                 previous_output_archive=str(archive) if archive else None)
    note = ('计算结束，自动坐标检查通过。仍需人工抽查同步和AOI；候选位置为探索性结果。'
            if code == 0 else
            '试算结束，存在待复核或排除任务，不能标为正式完成。请先查看result_quality.csv。')
    (output/'先看结果状态.txt').write_text(note+'\n状态详情：run_status.json\n', encoding='utf-8')
    print(note, flush=True)
    print(f'结果目录：{output}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
