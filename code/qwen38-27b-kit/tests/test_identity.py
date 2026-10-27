import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common
import kld_run
import timing
import report
import preflight
from common import DATA, HERE, atomic, load, request, rows
from ruler_replay import summarize
sys.path.insert(0,str(HERE/'vendor'))
import longbench_run_lb as frozen


class Identity(unittest.TestCase):
    def test_all_520_requests_and_official_longbench_fields(self):
        items=rows(DATA/'ruler.jsonl')
        self.assertEqual(len(items),520)
        for item in items:
            self.assertEqual(request(item['request'],'identity'),request(item['request'],'e3'))
        item=dict(seed=17)
        messages=[dict(role='user',content='unchanged official prompt')]
        for budget in (1024,128):
            e3=frozen.body(item,'e3',messages,budget)
            self.assertEqual(request(e3,'identity'),e3)

    def test_projector_evidence_covers_all_layers_and_neutral_components(self):
        evidence=load(HERE/'validation/identity-projector.json')
        self.assertEqual(sorted(evidence['native_identity_layers']+evidence['exact_F16_identity_layers']),list(range(16,64)))
        self.assertEqual(len(evidence['exact_F16_identity_layers']),12)
        self.assertEqual(evidence['learned_fill_maps'],0)
        self.assertTrue(evidence['metadata_identical'])
        self.assertTrue(evidence['row_scores_byte_identical'])

    def test_identity_kl_regenerates_stock_and_never_runs_e3(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp); folder=root/'attempt'; reference=folder/'reference.logprobs'
            atomic(root/'input.json',dict(ids=[1]*520))
            stock=dict(nll=1.,target_positions=list(range(8,520)),logprobs=[-1.]*512)
            atomic(root/'kld/8/doc.json',dict(stock=stock,input_sha256='same'))
            calls=[]
            def score(args,input_path,dest,mode,extra):
                calls.append(mode)
                if mode=='off':
                    reference.write_bytes(b'new reference')
                    return stock
                self.assertEqual(mode,'identity')
                self.assertTrue(reference.exists())
                return dict(nll=1.5,target_positions=list(range(8,520)),selector_excludes_targets=True,
                            kl=[.1]*512,top1=[1]*512,kl_mean=.1,kl_median=.1,kl_p99=.1,kl_max=.1,
                            top1_agree=1,dp_target_mean=0)
            with patch.object(kld_run,'DATA',root),patch.object(kld_run,'score_once',side_effect=score),patch.object(kld_run,'sha',return_value='same'):
                result=kld_run.run_item(SimpleNamespace(identity_only=True,out=root,gpu=0),dict(file='input.json',length=8,id='doc'),folder)
            self.assertEqual(calls,['off','identity'])
            self.assertNotIn('e3',result)
            self.assertEqual(result['dnll'],.5)
            self.assertFalse(reference.exists())

    def test_identity_kl_rejects_changed_stock_and_cleans_reference(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp); folder=root/'attempt'; reference=folder/'reference.logprobs'
            atomic(root/'input.json',dict(ids=[1]*520))
            atomic(root/'kld/8/doc.json',dict(stock=dict(target_positions=[8],logprobs=[-1]),input_sha256='same'))
            def score(*args):
                reference.write_bytes(b'new reference')
                return dict(target_positions=[8],logprobs=[-2])
            with patch.object(kld_run,'DATA',root),patch.object(kld_run,'score_once',side_effect=score),patch.object(kld_run,'sha',return_value='same'):
                with self.assertRaisesRegex(AssertionError,'Regenerated stock'):
                    kld_run.run_item(SimpleNamespace(identity_only=True,out=root,gpu=0),dict(file='input.json',length=8,id='doc'),folder)
            self.assertFalse(reference.exists())

    def test_timing_identity_reruns_all_four_arms_and_preserves_old_timing(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)
            atomic(root/'IDENTITY-MANIFEST.json',dict(gpu_count=2,hardware=[['0','A100','40960','uuid0'],['1','A100','40960','uuid1']]))
            atomic(root/'timing/16384/stock/result.json',dict(original=True))
            original=(root/'timing/16384/stock/result.json').read_bytes()
            calls=[]
            def run(args,length,arm,folder):
                calls.append((length,arm)); return dict(rows=[])
            with patch.object(sys,'argv',['timing.py','--identity-only','--gpus','2','--out',tmp]), \
                 patch('reproduction.require_reproduction'),patch.object(timing,'run_arm',side_effect=run):
                timing.main()
                timing.main()
            self.assertEqual(len(calls),16)
            self.assertEqual(set(calls),{(n,a) for n in (16384,32768,65536,120000) for a in ('stock','resident','streamed','identity')})
            self.assertEqual((root/'timing/16384/stock/result.json').read_bytes(),original)
            session=load(root/'timing-identity/SESSION.json')
            for n,a in calls:
                value=load(root/'timing-identity'/str(n)/a/'result.json')
                self.assertEqual(value['session_id'],session['session_id'])
                self.assertEqual(value['gpu_uuid'],'uuid0')

    def test_ruler_paired_identity_confidence_intervals(self):
        import score_ruler
        records=[dict(id=f'{n}-{task}-{i}',length=n,task=task,stock=50,e3=40,identity=60)
                 for n in (16384,32768) for task in score_ruler.TASKS for i in range(20)]
        # Degenerate differences make pairing/direction exact, independent of RNG.
        with patch('ruler_replay.bootstrap',side_effect=lambda groups:common.bootstrap(groups,repeats=10)):
            result=summarize(records)
        for v in result.values():
            self.assertEqual(v['identity_vs_stock']['ci95'],[10,10])
            self.assertEqual(v['identity_vs_e3']['ci95'],[20,20])

    def test_kl_report_pairs_identity_by_document(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp); index=[]
            for length in (16384,32768):
                for i in range(100):
                    item=dict(length=length,id=f'doc{i}'); index.append(item)
                    base=dict(id=item['id'],input_sha256=f'hash{i}',stock=dict(nll=1.),dnll=.25,
                              e3=dict(nll=1.25,kl=[.125]*512,top1=[1]*512,kl_mean=.125,dp_target_mean=0))
                    added=dict(id=item['id'],input_sha256=f'hash{i}',stock=dict(nll=1.),dnll=.5,
                               identity=dict(nll=1.5,kl=[.25]*512,top1=[0]*512,kl_mean=.25))
                    atomic(root/'kld'/str(length)/(item['id']+'.json'),base)
                    atomic(root/'kld-identity'/str(length)/(item['id']+'.json'),added)
            atomic(root/'kld-index.json',index)
            with patch.object(report,'DATA',root),patch.object(report,'bootstrap',side_effect=lambda g:common.bootstrap(g,repeats=10)):
                result=report.kld_report(root,True)
            for v in result.values():
                self.assertEqual(v['identity']['dnll']['ci95'],[.5,.5])
                self.assertEqual(v['identity_vs_e3']['dnll']['ci95'],[.25,.25])
                self.assertEqual(v['identity_vs_e3']['delta_kl_to_stock']['ci95'],[.125,.125])
                self.assertEqual(v['identity']['top1_agreement'],0)

    def test_baseline_manifest_binds_reused_results_without_mutating_them(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)
            atomic(root/'RUN-MANIFEST.json',dict(original=True))
            atomic(root/'checks/PASS.json',dict(ok=True))
            before=(root/'RUN-MANIFEST.json').read_bytes()
            with patch.object(preflight,'sha',return_value='hash'):
                baseline,files=preflight.identity_baseline(root)
            self.assertEqual(baseline,dict(original=True))
            self.assertEqual(len(files),1254)
            self.assertEqual((root/'RUN-MANIFEST.json').read_bytes(),before)
            self.assertFalse(any('identity' in p for p in files))

    def test_timing_report_pairs_prompts_and_renders_both_comparisons(self):
        def fake_load(path):
            if path.name=='SESSION.json':return dict(session_id='same',gpu_uuid='uuid0')
            arm=path.parent.name
            base=dict(stock=1000,resident=800,identity=900,streamed=850)[arm]
            records=[dict(phase=phase,prompt_ms=base+i*10,client_ttft_s=(base+i*10)/1000+.2)
                     for i,phase in enumerate(('cold','warmup','timed-0','timed-1','timed-2'))]
            return dict(session_id='same',gpu_uuid='uuid0',gpu=0,rows=records,peak_mib=100,warm_prompt_s=common.distribution([r['prompt_ms']/1000 for r in records[2:]]),
                        warm_client_ttft_s=common.distribution([r['client_ttft_s'] for r in records[2:]]))
        with patch.object(report,'load',side_effect=fake_load),patch.object(report,'bootstrap',side_effect=lambda g:common.bootstrap(g,repeats=10)):
            values=report.timing_report(Path('/unused'),True)
        for row in values:
            if row['arm']=='identity':
                self.assertAlmostEqual(row['identity_vs_stock']['prompt_s']['mean'],-.1)
                self.assertAlmostEqual(row['identity_vs_e3']['prompt_s']['mean'],.1)
        result=dict(status='SYNTHETIC CPU TEST',identity_only=True,identity_protocol='Synthetic test.',timing=values,
                    longbench=dict(overall=dict(n=1,stock=50,e3=40,delta=dict(mean=-10,ci95=[-10,-10])),
                                   diagnostics={},workload_prefill_speedup=1))
        text=report.markdown(result)
        self.assertIn('Identity−stock prompt s [95% CI]',text)
        self.assertIn('Identity−E3 client TTFT s [95% CI]',text)


if __name__=='__main__':
    unittest.main()
