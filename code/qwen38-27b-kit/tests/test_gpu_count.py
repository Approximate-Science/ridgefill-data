from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import HERE, atomic, load
import common
import drive
import identity_run
import preflight
import reproduction
import test_reproduction as reproduction_tests


class GPUCount(unittest.TestCase):
    def test_hardware_accepts_two_or_four_and_rejects_wrong_inventory(self):
        def hardware(n):return [[str(i),'NVIDIA A100-PCIE-40GB','40960',f'uuid{i}'] for i in range(n)]
        for n in (2,4):preflight.validate_hardware(hardware(n),n)
        for actual,requested in ((2,4),(4,2),(3,3)):
            with self.assertRaises(AssertionError):preflight.validate_hardware(hardware(actual),requested)

    def test_two_gpu_hardware_can_reuse_four_gpu_baseline(self):
        original=dict(model='model',projector='projector',sources={},data_manifest='hash',
                      hardware=[[str(i),'A100','40960','old'] for i in range(4)],binaries={})
        current=dict(model='model',projector='projector',sources={},gpu_count=2,
                     hardware=[[str(i),'A100','40960','new'] for i in range(2)],binaries={})
        with patch.object(preflight,'identity_baseline',return_value=(original,{})), \
             patch.object(preflight,'sha',return_value='hash'),patch.object(preflight.subprocess,'run'):
            preflight.bind_identity(current,Path('/unused'),Path('/unused/projector'))
        self.assertEqual(current['gpu_count'],2)

    def test_driver_uses_requested_shard_count_for_every_component(self):
        for n in (2,4):
            for component in ('reproduction.py','ruler_replay.py','kld_run.py','longbench_identity.py'):
                with self.subTest(n=n,component=component),tempfile.TemporaryDirectory(dir=HERE) as tmp:
                    procs=[Mock(pid=10000+i,returncode=0) for i in range(n)]
                    for proc in procs:proc.poll.return_value=0
                    with patch.object(sys,'argv',['drive.py',component,tmp,'--identity-only','--gpus',str(n)]), \
                         patch.object(drive.signal,'signal'),patch.object(drive.os,'killpg'), \
                         patch.object(drive.subprocess,'Popen',side_effect=procs) as launch:
                        drive.main()
                    self.assertEqual(launch.call_count,n)
                    for i,call in enumerate(launch.call_args_list):
                        command=call.args[0]
                        self.assertEqual(command[command.index('--gpu')+1],str(i))
                        self.assertEqual(command[command.index('--workers')+1],str(n))
                        self.assertEqual(command[command.index('--gpus')+1],str(n))

    def test_reproduction_shards_cover_all_samples_and_finalize_once(self):
        for n in (2,4):
            with self.subTest(n=n),tempfile.TemporaryDirectory(dir=HERE) as tmp:
                root=Path(tmp);args=reproduction_tests.Reproduction().fixture(root)
                args.gpus=args.workers=n
                atomic(root/'IDENTITY-MANIFEST.json',dict(gpu_count=n))
                response=dict(choices=[dict(message=dict(content='exact'),finish_reason='stop')])
                def score(args,item,attempt):return reproduction.reference(root,item['reference'])
                with patch.object(reproduction,'DATA',root),patch.object(reproduction,'assert_idle') as idle, \
                     patch.object(reproduction,'server',return_value=nullcontext()), \
                     patch.object(reproduction,'post',return_value=response) as post, \
                     patch.object(reproduction,'run_item',side_effect=score) as native:
                    for gpu in range(n):
                        args.gpu=gpu;reproduction.run(args)
                    self.assertFalse((root/'identity-reproduction/PASS.json').exists())
                    reproduction.finalize(root)
                    reproduction.require_reproduction(root)
                    self.assertEqual(post.call_count,16)
                    self.assertEqual(native.call_count,4)
                    idle.assert_not_called() # other shards intentionally occupy other GPUs

    def test_orchestrator_passes_n_to_all_steps(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp);atomic(root/'IDENTITY-MANIFEST.json',dict(gpu_count=2))
            with patch.object(sys,'argv',['identity_run.py','--out',tmp,'--gpus','2']), \
                 patch.object(identity_run,'require_reproduction'),patch.object(identity_run.signal,'signal'), \
                 patch.object(identity_run,'run_step') as step:
                identity_run.main()
            for call in step.call_args_list:
                command=call.args[2]
                self.assertEqual(command[command.index('--gpus')+1],'2')

    def test_idle_check_uses_present_n_gpus(self):
        with patch.object(common.subprocess,'check_output',side_effect=['0, uuid0\n1, uuid1\n','uuid0, 123\n']):
            common.assert_idle(except_gpu=0,gpu_count=2)
        with patch.object(common.subprocess,'check_output',side_effect=['0, uuid0\n1, uuid1\n','uuid1, 123\n']):
            with self.assertRaisesRegex(AssertionError,'Other GPUs'):
                common.assert_idle(except_gpu=0,gpu_count=2)


if __name__=='__main__':unittest.main()
