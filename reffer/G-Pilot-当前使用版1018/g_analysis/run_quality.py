"""Keep spatial QC failures out of aggregate outputs without changing QC thresholds."""
from pathlib import Path
import pandas as pd


def eligible_datasets(output: Path) -> set[str]:
    path = output / '_spatial_coordinate_qc' / 'spatial_coordinate_qc_all_datasets.csv'
    if not path.exists():
        raise RuntimeError('请先运行04_spatial_coordinate_qc.py，再生成空间汇总或候选位置。')
    qc = pd.read_csv(path)
    if qc.empty or not {'dataset', 'qc_status'}.issubset(qc.columns):
        raise RuntimeError('空间QC表为空或缺少状态，不能生成汇总。')
    if qc.dataset.duplicated().any() or not qc.qc_status.isin(['PASS', 'WARN', 'FAIL']).all():
        raise RuntimeError('空间QC表存在重复任务或未知状态，请重新运行QC。')
    return set(qc.loc[qc.qc_status.isin(['PASS', 'WARN']), 'dataset'])


def result_state(qc: pd.DataFrame) -> tuple[str, int]:
    """WARN outputs remain provisional; PASS means automatic checks only."""
    if qc.empty or not qc.qc_status.isin(['PASS', 'WARN', 'FAIL']).all():
        return 'blocked', 1
    if qc.qc_status.eq('FAIL').all():
        return 'blocked', 1
    if qc.qc_status.eq('FAIL').any():
        return 'partial_results_need_review', 2
    if qc.qc_status.eq('WARN').any():
        return 'computed_needs_review', 2
    return 'automatic_checks_passed', 0
