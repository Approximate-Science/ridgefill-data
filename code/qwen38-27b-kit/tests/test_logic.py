import sys
import struct
import copy
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import E3_FIELDS, HERE, atomic, bootstrap, canonical, env, parser, request, server_binary, sha, validate_binary
from pack_data import document, prompts, select_windows
from ruler_replay import budget, score
from checks import nll_equal, same_bytes, same_generation_and_topk
from mix import token_metrics
import kld_run
import common
from build_manifest import HOOK_PATH, verify_sources
from instrument_sources import ANCHOR, instrument_text
from finalize_sources import validate_changes
from release_manifest import release_files


class Logic(unittest.TestCase):
    def test_request_identity_and_budgets(self):
        for task,expected in [('niah_multivalue',128),('vt',30),('cwe',120),('fwe',50),('qa_2',32)]:
            value=dict(messages=[dict(role='user',content='Keep this exact.\n')],max_tokens=budget(task),temperature=0,top_k=1,
                       chat_template_kwargs=dict(enable_thinking=False),seed=0,stream=False,cache_prompt=False)
            self.assertEqual(budget(task),expected)
            stock,e3=request(value,'stock'),request(value,'e3')
            self.assertEqual(stock,value)
            self.assertEqual(canonical(e3),stock)
            self.assertEqual({k:v for k,v in e3.items() if k not in stock},E3_FIELDS)

    def test_substring_scorer_controls(self):
        item=dict(id='test',task='niah_multiquery',answers=['abc','DEF'],length_bucket='16k')
        self.assertEqual(score(item,'abc'),50)
        self.assertEqual(score(item,'ABC def'),100)
        self.assertEqual(score(item,'unrelated'),0)
        item.update(task='qa_1')
        self.assertEqual(score(item,'def'),100)

    def test_document_and_window_selection(self):
        def item(i,text):
            return dict(id=i,question='Q?',messages=[dict(role='user',content='Please read the following text and answer the questions below.\n\n<text>\n'+text+'\n</text>\nQ?')])
        values=[item('c','c'*520),item('a','a'*519),item('b','b'*521)]
        selected=select_windows(values,lambda t:[ord(x) for x in t],8,count=2)
        self.assertEqual([x['name'] for x in selected],['b','c'])
        self.assertEqual([len(x['ids']) for x in selected],[520,520])
        self.assertEqual(selected[0]['ids'],[ord('b')]*520)
        self.assertEqual(document(item('embedded','x\n</text>\ny')),'x\n</text>\ny')
        with self.assertRaises(ValueError):select_windows(values,list,1000,count=1)

    def test_bootstrap_equal_task_weights_and_pairing(self):
        v=bootstrap([[1.0],[-1.0]*20],repeats=100)
        self.assertEqual(v['mean'],0)
        self.assertEqual(v['ci95'],[0,0])
        self.assertEqual(bootstrap([[1,2,5]],repeats=100),bootstrap([[1,2,5]],repeats=100))
        self.assertEqual(bootstrap([[7]*9],repeats=100)['ci95'],[7,7])
        with self.assertRaises(AssertionError):bootstrap([[]],repeats=100)

    def test_timing_prefix_shuffles_and_partial_block(self):
        for length in (16384,32768,65536,120000):
            ids=list(range(length));result=prompts(ids)
            self.assertEqual(len(result),5)
            self.assertEqual(len({r['ids'][0] for r in result}),5)
            for row in result:
                self.assertEqual(len(row['ids']),length)
                self.assertEqual(row['ids'][-4096:],ids[-4096:])
                self.assertEqual(sorted(row['ids']),ids)
            self.assertEqual(prompts(ids),result)

    def test_gates_reject_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';a.write_bytes(struct.pack('f',1));b.write_bytes(struct.pack('f',1))
            same_bytes([a,b]);b.write_bytes(struct.pack('f',2))
            with self.assertRaises(AssertionError):same_bytes([a,b])
            a.write_bytes(struct.pack('f',float('nan')));b.write_bytes(a.read_bytes())
            with self.assertRaises(AssertionError):same_bytes([a,b])
        nll_equal(dict(nll=1,logprobs=[-1]),dict(nll=1,logprobs=[-1]))
        with self.assertRaises(AssertionError):nll_equal(dict(nll=1,logprobs=[-1]),dict(nll=1,logprobs=[-2]))

    def test_mix_stream_count_validation(self):
        events=[dict(t=1,data=dict(tokens_predicted=1,tokens=[8])),
                dict(t=2,data=dict(tokens_predicted=2,tokens=[9],stop=True,timings=dict(predicted_n=2)))]
        self.assertEqual(token_metrics(events,2)['decode_tps'],1)
        with self.assertRaises(AssertionError):token_metrics(events,3)

    def test_kld_reference_removed_before_result_processing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root=Path(tmp)
            (root/'input.json').write_text('{"ids":'+str([1]*520)+'}')
            folder=root/'attempt'
            reference=folder/'reference.logprobs'
            calls=[]
            def fake_score(args,input_path,dest,mode,extra):
                calls.append((mode,extra))
                if mode=='off':
                    self.assertFalse(reference.exists())
                    reference.write_bytes(b'dtype-tagged-float32-reference')
                    return dict(nll=1,target_positions=list(range(8,520)))
                self.assertTrue(reference.exists())
                return dict(nll=1.25,target_positions=list(range(8,520)),selector_excludes_targets=True,
                            kl=[.1]*512,top1=[1]*512,kl_mean=.1,kl_median=.1,kl_p99=.1,kl_max=.1,
                            top1_agree=1,dp_target_mean=0)
            def hash_after_cleanup(path):
                self.assertFalse(reference.exists())
                return 'input-sha'
            with patch.object(kld_run,'DATA',root),patch.object(kld_run,'score_once',side_effect=fake_score), \
                 patch.object(kld_run,'sha',side_effect=hash_after_cleanup):
                result=kld_run.run_item(SimpleNamespace(gpu=0),dict(id='doc',file='input.json',length=8),folder)
            self.assertEqual(result['dnll'],.25)
            self.assertEqual(calls,[('off',['--save-logprobs',str(reference)]),('e3',['--kl-against',str(reference)])])
            self.assertFalse(reference.exists())

    def test_kld_reference_removed_after_comparison_failure(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root=Path(tmp)
            (root/'input.json').write_text('{"ids":'+str([1]*520)+'}')
            folder=root/'attempt'
            reference=folder/'reference.logprobs'
            def fail_comparison(args,input_path,dest,mode,extra):
                if mode=='off':
                    reference.write_bytes(b'dtype-tagged-float32-reference')
                    return dict(nll=1)
                self.assertTrue(reference.exists())
                raise RuntimeError('comparison failed')
            with patch.object(kld_run,'DATA',root),patch.object(kld_run,'score_once',side_effect=fail_comparison):
                with self.assertRaisesRegex(RuntimeError,'comparison failed'):
                    kld_run.run_item(SimpleNamespace(gpu=0),dict(id='doc',file='input.json',length=8),folder)
            self.assertFalse(reference.exists())

    def test_measurement_defaults_and_variant_selection(self):
        args=parser('test').parse_args([])
        self.assertEqual(args.stock_bin,HERE/'build/pure/stock/bin/llama-server')
        self.assertEqual(args.e3_bin,HERE/'build/pure/e3/bin/llama-server')
        self.assertEqual(args.scorer,HERE/'build/pure/e3/bin/llama-ridgefill-score')
        args.stock_instrumented_bin=HERE/'build/instrumented/stock/bin/llama-server'
        args.e3_instrumented_bin=HERE/'build/instrumented/e3/bin/llama-server'
        with patch.object(common,'validate_binary') as validate:
            self.assertEqual(server_binary(args,'stock'),args.stock_bin)
            validate.assert_called_with(args.stock_bin,'pure','stock')
            self.assertEqual(server_binary(args,'resident'),args.e3_bin)
            validate.assert_called_with(args.e3_bin,'pure','e3')
            self.assertEqual(server_binary(args,'stock',instrumented=True),args.stock_instrumented_bin)
            validate.assert_called_with(args.stock_instrumented_bin,'instrumented','stock')
            self.assertEqual(server_binary(args,'loaded-off',instrumented=True),args.e3_instrumented_bin)
            validate.assert_called_with(args.e3_instrumented_bin,'instrumented','e3')
        with patch.dict('os.environ',{'RIDGEFILL_AUDIT_LOGITS':'inherited-dump'}):
            self.assertNotIn('RIDGEFILL_AUDIT_LOGITS',env(0,args.stock_bin))

    def test_build_sources_reject_hook_in_pure_tree(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)
            source=root/'source';source.mkdir()
            original='void example() {\n'+ANCHOR+'}\n'
            path=source/HOOK_PATH;path.parent.mkdir(parents=True);path.write_text(original)
            archive=root/'fixture.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                member=tarfile.TarInfo(HOOK_PATH)
                payload=original.encode();member.size=len(payload)
                tar.addfile(member,io.BytesIO(payload))
            verify_sources(source,archive,'pure')
            path.write_text(instrument_text(original))
            with self.assertRaisesRegex(AssertionError,'pure source differs'):
                verify_sources(source,archive,'pure')
            verify_sources(source,archive,'instrumented')
            (source/'unexpected.cpp').write_text('not in archive')
            with self.assertRaisesRegex(AssertionError,'Unexpected/missing source files'):
                verify_sources(source,archive,'instrumented')

    def test_measurements_reject_instrumented_or_changed_binary(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)
            build=root/'build';binary=build/'bin/llama-server'
            binary.parent.mkdir(parents=True);binary.write_bytes(b'compiled-artifact')
            manifest=dict(arm='stock',variant='instrumented',source_archive_sha256='archive-hash',
                          files={'bin/llama-server':sha(binary)})
            atomic(build/'BUILD-MANIFEST.json',manifest)
            with self.assertRaisesRegex(AssertionError,'Wrong binary build variant/arm'):
                validate_binary(binary,'pure','stock')
            manifest['variant']='pure';atomic(build/'BUILD-MANIFEST.json',manifest)
            atomic(root/'sources/LOCK.json',dict(archives={'stock-1242be66f.tar.gz':'archive-hash'}))
            with patch.object(common,'HERE',root):
                validate_binary(binary,'pure','stock')
                binary.write_bytes(b'changed-compiled-artifact')
                with self.assertRaisesRegex(AssertionError,'Build artifact changed'):
                    validate_binary(binary,'pure','stock')

    def test_pure_instrumented_tokens_and_topk_must_match(self):
        row=dict(id=7,logprob=-.5,top_logprobs=[dict(id=i,logprob=-i-.1) for i in range(10)])
        pure=dict(tokens=[7]*128,content='generated output',completion_probabilities=[copy.deepcopy(row) for _ in range(128)])
        observed=copy.deepcopy(pure)
        self.assertEqual(len(same_generation_and_topk(pure,observed)),64)
        observed['completion_probabilities'][50]['top_logprobs'][9]['logprob']-=.001
        with self.assertRaisesRegex(AssertionError,'top-k probabilities differ'):
            same_generation_and_topk(pure,observed)
        observed=copy.deepcopy(pure);observed['tokens'][0]=8
        with self.assertRaises(AssertionError):same_generation_and_topk(pure,observed)
        observed=copy.deepcopy(pure);observed['completion_probabilities'].pop()
        with self.assertRaisesRegex(AssertionError,'Missing per-token probability rows'):
            same_generation_and_topk(pure,observed)

    def test_scorer_commit_scope_allows_documentation_and_rejects_runtime(self):
        validate_changes(['NOTES-A.md','tools/ridgefill/score.cpp','tools/ridgefill/score-logprobs.h',
                          'tools/ridgefill/README.md','tests/ridgefill/test_score_kl.cpp'])
        for path in ('src/llama-context.cpp','ggml/src/ggml.cpp','common/common.cpp','tools/server/server-context.cpp'):
            with self.assertRaises(AssertionError):validate_changes(['tools/ridgefill/score.cpp',path])

    def test_transfer_list_keeps_inputs_and_excludes_build_scratch(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)
            keep=('setup.sh','common.py','README.md','sources/e3.tar.gz','sources/LOCK.json',
                  'data/window.json','models/projector.gguf','tests/test_logic.py','vendor/score_ruler.py')
            skip=('.cpu-build-check/build/bin/llama-server','vendor/__pycache__/score.pyc',
                  'results/stock/item.json','cpu-preview/SCORECARD.json','data/incomplete.tmp')
            for name in (*keep,*skip):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
            self.assertEqual({str(path.relative_to(root)) for path in release_files(root)},set(keep))

if __name__=='__main__':unittest.main()
