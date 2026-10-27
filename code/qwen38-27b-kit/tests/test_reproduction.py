import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import DATA, HERE, atomic, load, request, rows, sha
import identity_run
import kld_run
import longbench_identity
import preflight
import reproduction
import ruler_replay
import timing


class Reproduction(unittest.TestCase):
    def test_frozen_samples_match_all_stored_requests_and_document_hashes(self):
        root=HERE/'results-20261025/results'
        # The release does not ship the full results; structural checks still run
        # on a box, while exact historical-reference checks run when available.
        selected=reproduction.plan()
        self.assertEqual(len(selected['ruler']),8)
        self.assertEqual({n:sum(x['length']==n for x in selected['ruler']) for n in (16384,32768)},
                         {16384:4,32768:4})
        self.assertEqual(len({x['task'] for x in selected['ruler']}),4)
        self.assertEqual(len(selected['kld']),4)
        self.assertEqual(len({x['id'] for x in selected['kld']}),4)
        items={x['id']:x for x in rows(DATA/'ruler.jsonl')}
        if root.exists():
            for item in selected['ruler']:
                for arm in ('stock','e3'):
                    stored=reproduction.reference(root,item['references'][arm])
                    self.assertEqual(request(items[item['id']]['request'],arm),stored['request'])
            for item in selected['kld']:
                stored=reproduction.reference(root,item['reference'])
                self.assertEqual(sha(DATA/item['file']),stored['input_sha256'])

    def test_text_is_byte_exact_without_normalization(self):
        reproduction.text_equal('A\r\nB','A\r\nB')
        for a,b in [('A\nB','A\r\nB'),('Answer ','Answer'),('é','e\u0301')]:
            with self.assertRaisesRegex(AssertionError,'byte-for-byte'):
                reproduction.text_equal(a,b)

    def test_all_scientific_fields_are_exact_build_and_wall_can_change(self):
        original=dict(build=dict(commit='old'),wall_s=2,nll=1.25,kl=[0.,.125],logprobs=[-1.,-2.],
                      exact_rows=[1,7],vtail=dict(scales=[1.]))
        changed=copy.deepcopy(original);changed.update(build=dict(commit='rebuilt'),wall_s=8)
        reproduction.score_equal(changed,original)
        for key,value in [('nll',1.2500000000000002),('kl',[0.,.12500000000000003]),
                          ('logprobs',[-1.,-2.0000000000000004]),('exact_rows',[1,8])]:
            wrong=copy.deepcopy(changed);wrong[key]=value
            with self.assertRaisesRegex(AssertionError,key): reproduction.score_equal(wrong,original)

    def fixture(self,root):
        atomic(root/'IDENTITY-MANIFEST.json',dict(rebuilt=True,gpu_count=4))
        selected=dict(ruler=[],kld=[]); items=[]
        def pin(path,value):
            atomic(path,value)
            return dict(file=str(path.relative_to(root)),sha256=sha(path))
        for i in range(8):
            item=dict(id=f'item{i}',length=16384 if i<4 else 32768,task='qa_1')
            body=dict(messages=[dict(role='user',content=str(i))],max_tokens=32)
            refs={a:pin(root/'ruler'/a/(item['id']+'.json'),
                        dict(request=request(body,a),text='exact',finish_reason='stop')) for a in ('stock','e3')}
            selected['ruler'].append(dict(**item,references=refs));items.append(dict(**item,request=body))
        scientific=dict(nll=1.,logprobs=[-1.],kl=[0.],kl_mean=0.,target_positions=[16384],build={},wall_s=1.)
        for i in range(4):
            item=dict(id=f'doc{i}',length=16384 if i<2 else 32768,file='unused')
            stored=dict(stock=scientific,e3=scientific,dnll=0.,input_sha256='input')
            selected['kld'].append(dict(**item,reference=pin(root/'kld'/str(item['length'])/(item['id']+'.json'),stored)))
        atomic(root/'reproduction-plan.json',selected)
        (root/'ruler.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items))
        return SimpleNamespace(out=root,identity_only=True,gpu=0,port=18200)

    def test_gate_pass_is_resumable_and_bound_to_rebuilt_manifest(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp);args=self.fixture(root)
            response=dict(choices=[dict(message=dict(content='exact'),finish_reason='stop')])
            def score(args,item,attempt):
                self.assertFalse(args.identity_only)
                return reproduction.reference(root,item['reference'])
            with patch.object(reproduction,'DATA',root),patch.object(reproduction,'assert_idle'), \
                 patch.object(reproduction,'server',return_value=nullcontext()), \
                 patch.object(reproduction,'post',return_value=response) as post, \
                 patch.object(reproduction,'run_item',side_effect=score) as run:
                reproduction.run(args)
                reproduction.run(args)
                self.assertEqual(post.call_count,16)
                self.assertEqual(run.call_count,4)
                self.assertEqual(reproduction.require_reproduction(root)['counts'],dict(stock_ruler=8,e3_ruler=8,e3_kl=4))
                atomic(root/'IDENTITY-MANIFEST.json',dict(changed=True))
                with self.assertRaisesRegex(AssertionError,'artifacts or identity manifest changed'):
                    reproduction.require_reproduction(root)

    def test_mismatch_is_recorded_and_blocks_all_identity_work(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp);args=self.fixture(root)
            response=dict(choices=[dict(message=dict(content='DIFFERENT'),finish_reason='stop')])
            with patch.object(reproduction,'DATA',root),patch.object(reproduction,'assert_idle'), \
                 patch.object(reproduction,'server',return_value=nullcontext()),patch.object(reproduction,'post',return_value=response), \
                 patch.object(reproduction,'run_item') as run:
                with self.assertRaisesRegex(AssertionError,'byte-for-byte'): reproduction.run(args)
                run.assert_not_called()
                self.assertTrue((root/'identity-reproduction/FAIL.json').exists())
                self.assertFalse((root/'identity-reproduction/PASS.json').exists())
                with self.assertRaisesRegex(RuntimeError,'REPRODUCTION failed'):
                    reproduction.require_reproduction(root)
                with patch.object(sys,'argv',['identity_run.py','--out',str(root)]),patch.object(identity_run,'run_step') as step:
                    with self.assertRaises(RuntimeError): identity_run.main()
                    step.assert_not_called()

    def test_new_binary_hashes_are_accepted_and_original_hashes_retained(self):
        baseline=dict(model='model',projector='ridge',sources={'pinned':True},data_manifest='data',
                      hardware=[['0','A100','40960','old']],binaries={'engine':'old-hash'})
        fresh=dict(model='model',projector='ridge',sources={'pinned':True},
                   hardware=[['0','A100','40960','new']],binaries={'engine':'rebuilt-hash'})
        with patch.object(preflight,'identity_baseline',return_value=(baseline,{})), \
             patch.object(preflight,'sha',return_value='data'),patch.object(preflight.subprocess,'run'):
            preflight.bind_identity(fresh,Path('/unused'),Path('/unused/projector'))
        self.assertEqual(fresh['binaries'],{'engine':'rebuilt-hash'})
        self.assertEqual(fresh['baseline_binaries'],{'engine':'old-hash'})
        self.assertIn('mandatory exact REPRODUCTION',fresh['rebuild_policy'])

    def test_pure_sources_must_match_the_pinned_archive_and_build_manifest(self):
        with patch.object(preflight,'verify_sources',return_value={'src/file':'hash'}) as verify:
            preflight.verify_pure_sources('e3',dict(source_files={'src/file':'hash'}))
            self.assertEqual(verify.call_args.args[-1],'pure')
            with self.assertRaisesRegex(AssertionError,'source manifest differs'):
                preflight.verify_pure_sources('e3',dict(source_files={'src/file':'different'}))

    def test_every_identity_runner_refuses_to_start_without_gate(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            with self.assertRaises(FileNotFoundError):
                ruler_replay.run(SimpleNamespace(identity_only=True,out=Path(tmp)))
            for module in (kld_run,timing,longbench_identity):
                with self.subTest(runner=module.__name__), \
                     patch.object(sys,'argv',[module.__name__+'.py','--identity-only','--out',tmp]):
                    with self.assertRaises(FileNotFoundError): module.main()


if __name__=='__main__':
    unittest.main()
