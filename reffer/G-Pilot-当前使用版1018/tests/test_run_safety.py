import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g_analysis.pipeline import _to_bool
from g_analysis.run_quality import eligible_datasets, result_state
from g_analysis.reporting import read_groups

spec = importlib.util.spec_from_file_location('launcher', ROOT/'run_all.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class RunSafety(unittest.TestCase):
    def test_exported_validity_encodings(self):
        for values in [[True, False], [1, 0], [1., 0.], [' True ', 'FALSE'], ['1', '0']]:
            self.assertEqual(_to_bool(pd.Series(values)).tolist(), [True, False])
        self.assertEqual(_to_bool(pd.Series([1., None, 0.])).tolist(), [True, False, False])
        with self.assertRaisesRegex(ValueError, '无法识别'):
            _to_bool(pd.Series(['valid', 'invalid']))

    def test_clean_rebuild_archives_stale_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/'output'/'removed_participant'
            old.mkdir(parents=True)
            (old/'old.csv').write_text('old results')
            output, archive = launcher.prepare_output(root)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual((archive/'removed_participant'/'old.csv').read_text(), 'old results')
            self.assertNotIn(output, archive.parents)

    def test_qc_states_do_not_claim_completion_on_warning_or_failure(self):
        for values, expected in [(['PASS'], ('automatic_checks_passed', 0)),
                                 (['PASS', 'WARN'], ('computed_needs_review', 2)),
                                 (['PASS', 'FAIL'], ('partial_results_need_review', 2)),
                                 (['FAIL'], ('blocked', 1)),
                                 (['unexpected'], ('blocked', 1))]:
            self.assertEqual(result_state(pd.DataFrame({'qc_status': values})), expected)

    def test_aggregation_omits_failed_and_unchecked_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root/'output'
            output.mkdir()
            with self.assertRaises(RuntimeError):
                eligible_datasets(output)
            qc_root = output/'_spatial_coordinate_qc'
            qc_root.mkdir()
            pd.DataFrame({'dataset': ['passed', 'warning', 'failed'],
                          'qc_status': ['PASS', 'WARN', 'FAIL']}).to_csv(qc_root/'spatial_coordinate_qc_all_datasets.csv', index=False)
            for name in ['passed', 'warning', 'failed', 'unchecked']:
                task = output/name
                task.mkdir()
                pd.DataFrame({'participant_id': [name], 'spatial_valid': [True],
                              'horizontal_visual_angle_deg': [0.], 'vertical_visual_angle_deg': [0.]}).to_csv(task/'gaze_stage_aoi_fused.csv', index=False)
            self.assertEqual({p.name for p, _ in read_groups(root)}, {'passed', 'warning'})

    def test_launcher_runs_qc_before_aggregation_and_reports_review(self):
        for grades, expected_code, expected_state in [
            (['PASS'], 0, 'automatic_checks_passed'),
            (['PASS', 'WARN'], 2, 'computed_needs_review'),
            (['PASS', 'FAIL'], 2, 'partial_results_need_review'),
            (['FAIL'], 1, 'blocked')]:
            with self.subTest(grades=grades), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'data'/'recording').mkdir(parents=True)
                steps = []

                def fake_step(args, **kwargs):
                    step = Path(args[1]).name
                    steps.append(step)
                    code = 0
                    if step == '04_spatial_coordinate_qc.py':
                        qc_path = root/'output'/'_spatial_coordinate_qc'
                        qc_path.mkdir()
                        table = pd.DataFrame({'dataset': [f'task{i}' for i in range(len(grades))], 'qc_status': grades})
                        table.to_csv(qc_path/'spatial_coordinate_qc_all_datasets.csv', index=False)
                        _, code = result_state(table)
                    return SimpleNamespace(returncode=code, stdout='test boundary')

                with patch.object(launcher, 'ROOT', root), patch.object(launcher.subprocess, 'run', side_effect=fake_step):
                    self.assertEqual(launcher.main(), expected_code)
                import json
                status = json.loads((root/'output'/'run_status.json').read_text())
                self.assertEqual(status['status'], expected_state)
                self.assertTrue((root/'output'/'result_quality.csv').exists())
                if expected_state == 'blocked':
                    self.assertNotIn('03_build_spatial_baseline.py', steps)
                else:
                    self.assertLess(steps.index('04_spatial_coordinate_qc.py'), steps.index('03_build_spatial_baseline.py'))


if __name__ == '__main__':
    unittest.main()
