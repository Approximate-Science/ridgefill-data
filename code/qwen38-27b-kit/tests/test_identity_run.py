import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, atomic, load
import identity_run
import report


class IdentityRun(unittest.TestCase):
    def test_default_sequence_includes_longbench_last(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)
            atomic(root/'IDENTITY-MANIFEST.json',dict(frozen=True,gpu_count=4))
            calls=[]
            with patch.object(sys,'argv',['identity_run.py','--out',str(root)]), \
                 patch.object(identity_run.signal,'signal'), \
                 patch.object(identity_run,'require_reproduction'), \
                 patch.object(identity_run,'run_step',side_effect=lambda out,step,command:calls.append((step,command))):
                identity_run.main()
            self.assertEqual([step for step,_ in calls],['ruler','kl','ttft','longbench','report'])
            self.assertIn('longbench_identity.py',calls[3][1])
            self.assertIn('--identity-only',calls[3][1])

    def test_completion_covers_all_frozen_measurements(self):
        self.assertEqual([len(identity_run.result_files(Path('/unused'),step)) for step in identity_run.STEPS],
                         [520,200,17,503])
        self.assertEqual(identity_run.marker_path(Path('/unused'),'longbench').name,'04-longbench.complete.json')

    def test_completed_step_writes_marker_and_skips_on_resume(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp); result=root/'one.json'
            atomic(root/'IDENTITY-MANIFEST.json',dict(frozen=True))
            def execute(command): atomic(result,dict(finished=True))
            with patch.object(identity_run,'result_files',return_value=[result]), \
                 patch.object(identity_run,'run_command',side_effect=execute) as run:
                identity_run.run_step(root,'ruler',['fake'])
                before=result.read_bytes()
                marker=load(identity_run.marker_path(root,'ruler'))
                self.assertEqual(marker['count'],1)
                self.assertIn('one.json',marker['files'])
                identity_run.run_step(root,'ruler',['fake'])
                self.assertEqual(run.call_count,1)
                self.assertEqual(result.read_bytes(),before)

    def test_interrupted_step_preserves_items_and_resumes_without_marker(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp); first,second=root/'one.json',root/'two.json'
            atomic(root/'IDENTITY-MANIFEST.json',dict(frozen=True))
            def interrupted(command):
                atomic(first,dict(finished=True))
                raise subprocess.CalledProcessError(130,command)
            with patch.object(identity_run,'result_files',return_value=[first,second]), \
                 patch.object(identity_run,'run_command',side_effect=interrupted):
                with self.assertRaises(subprocess.CalledProcessError):
                    identity_run.run_step(root,'kl',['fake'])
            self.assertFalse(identity_run.marker_path(root,'kl').exists())
            before=first.read_bytes()
            def resume(command):
                self.assertEqual(first.read_bytes(),before)
                atomic(second,dict(finished=True))
            with patch.object(identity_run,'result_files',return_value=[first,second]), \
                 patch.object(identity_run,'run_command',side_effect=resume):
                identity_run.run_step(root,'kl',['fake'])
            self.assertEqual(first.read_bytes(),before)
            self.assertEqual(load(identity_run.marker_path(root,'kl'))['count'],2)

    def test_marker_rejects_changed_finished_items(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp); result=root/'one.json'
            atomic(root/'IDENTITY-MANIFEST.json',dict(frozen=True))
            atomic(result,dict(finished=True))
            with patch.object(identity_run,'result_files',return_value=[result]),patch.object(identity_run,'run_command'):
                identity_run.run_step(root,'ttft',['fake'])
                atomic(result,dict(changed=True))
                with self.assertRaisesRegex(AssertionError,'results or manifest changed'):
                    identity_run.run_step(root,'ttft',['fake'])

    def test_successful_exit_with_missing_results_does_not_mark_complete(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)
            atomic(root/'IDENTITY-MANIFEST.json',dict(frozen=True))
            with patch.object(identity_run,'result_files',return_value=[root/'missing.json']),patch.object(identity_run,'run_command'):
                with self.assertRaises(FileNotFoundError):
                    identity_run.run_step(root,'longbench',['fake'])
            self.assertFalse(identity_run.marker_path(root,'longbench').exists())

    def test_default_identity_report_requires_longbench_before_it_starts(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            with patch.object(sys,'argv',['report.py','--identity-only','--out',tmp]), \
                 patch.object(report,'longbench_report',side_effect=[{},FileNotFoundError('missing LongBench')]):
                with self.assertRaisesRegex(FileNotFoundError,'missing LongBench'):
                    report.main()

    def test_interruption_terminates_active_child(self):
        proc=Mock()
        proc.wait.side_effect=[KeyboardInterrupt(),0]
        proc.poll.return_value=None
        with patch.object(identity_run.subprocess,'Popen',return_value=proc) as start:
            with self.assertRaises(KeyboardInterrupt):
                identity_run.run_command(['fake-child'])
        start.assert_called_once_with(['fake-child'],start_new_session=True)
        proc.terminate.assert_called_once_with()
        self.assertEqual(proc.wait.call_args_list,[call(),call(timeout=15)])


if __name__=='__main__':
    unittest.main()
