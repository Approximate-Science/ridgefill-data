"""Mandatory CHECK-A/B gates. Every mismatch, missing dump or engine error exits nonzero."""
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from common import DATA, HERE, assert_idle, atomic, env, load, parser, post, provenance, request, server, sha, validate_binary
from kld_run import score_once


def same_bytes(paths):
    contents=[Path(p).read_bytes() for p in paths]
    assert contents and contents[0] and all(x==contents[0] for x in contents),'Raw float32 logits differ'
    for data in contents:
        values=array.array('f');values.frombytes(data)
        assert all(math.isfinite(x) for x in values),'Nonfinite raw logits in gate'


def nll_equal(a,b):
    # Compare all 512 target logprobs and the reduction, not just a rounded mean.
    assert a['logprobs']==b['logprobs'] and a['nll']==b['nll'],'Exact NLL mismatch'


def completion(args,folder,ids,arm,predict=1,n_probs=1):
    folder.mkdir(parents=True,exist_ok=True)
    body=request(dict(prompt=ids,n_predict=predict,temperature=0,top_k=1,seed=0,ignore_eos=True,
                      return_tokens=True,cache_prompt=False,n_probs=n_probs,post_sampling_probs=False),arm)
    atomic(folder/'request.json',body)
    result=post(args.port,'/completion',body)
    atomic(folder/'response.json',result)
    assert len(result['tokens'])==predict and result['timings']['prompt_n']==len(ids)
    assert result['timings']['cache_n']==0
    return result


def same_generation_and_topk(pure,instrumented,predict=128,n_probs=10):
    for result in (pure,instrumented):
        assert len(result['tokens'])==predict,'Incomplete generated token sequence'
        probabilities=result['completion_probabilities']
        assert len(probabilities)==predict,'Missing per-token probability rows'
        for token,row in zip(result['tokens'],probabilities):
            assert row['id']==token and math.isfinite(row['logprob'])
            top=row['top_logprobs']
            assert len(top)==n_probs and len({entry['id'] for entry in top})==n_probs,'Missing/duplicate top-k candidates'
            assert all(math.isfinite(entry['logprob']) for entry in top),'Nonfinite top-k probability'
    assert pure['tokens']==instrumented['tokens'] and pure['content']==instrumented['content'],'Pure/instrumented generated output differs'
    assert pure['completion_probabilities']==instrumented['completion_probabilities'],'Pure/instrumented top-k probabilities differ'
    return hashlib.sha256(json.dumps(pure['completion_probabilities'],sort_keys=True).encode()).hexdigest()


def pure_instrumented_gate(args,root):
    # Same frozen input and request for both variants; the observer is active in
    # the instrumented variant, matching its use in the raw-logit gates below.
    prompts=[dict(name=f'bridge-{length}',ids=load(DATA/f'timing/{length}.json')[0]['ids'])
             for length in (16384,32768)]
    comparisons=[]
    for arm in ('stock','e3'):
        outputs={}
        for variant in ('pure','instrumented'):
            folder=root/arm/variant
            dump=(folder/'logits.f32').resolve()
            dump.unlink(missing_ok=True)
            observed=variant=='instrumented'
            extra_env={'RIDGEFILL_AUDIT_LOGITS':str(dump)} if observed else None
            outputs[variant]=[]
            assert_idle()
            with server(args,arm,folder,ctx=33792,extra_env=extra_env,instrumented=observed):
                for item in prompts:
                    outputs[variant].append(completion(args,folder/item['name'],item['ids'],arm,predict=128,n_probs=10))
            if observed:
                assert dump.stat().st_size==2*128*248320*4,'Incomplete instrumented bridge logits'
        for index,item in enumerate(prompts):
            digest=same_generation_and_topk(outputs['pure'][index],outputs['instrumented'][index])
            comparisons.append(dict(arm=arm,prompt=item['name'],predict=128,n_probs=10,
                                    tokens_identical=True,topk_probabilities_identical=True,topk_sha256=digest))
    atomic(root/'result.json',dict(pass_gate=True,comparisons=comparisons))
    return comparisons


def gate_a(args,root):
    bridge=pure_instrumented_gate(args,root/'pure-vs-instrumented')
    base=load(DATA/'timing/32768.json')[0]['ids']
    prompts=[dict(name=f'g0-{n}',ids=base[:n]) for n in (8192,16384,32768)]
    prompts+=load(DATA/'checks-longbench.json')
    outputs={}
    for arm in ('stock','fork-off','loaded-off'):
        folder=root/arm
        dump=(folder/'logits.f32').resolve()
        dump.unlink(missing_ok=True)
        outputs[arm]=[]
        assert_idle()
        with server(args,arm,folder,ctx=121101,extra_env={'RIDGEFILL_AUDIT_LOGITS':str(dump)},instrumented=True):
            for item in prompts:
                before=dump.stat().st_size if dump.exists() else 0
                result=completion(args,folder/item['name'],item['ids'],arm)
                assert dump.exists() and dump.stat().st_size>before,'Missing raw logits hook'
                data=dump.read_bytes()[before:]
                raw=folder/item['name']/'logits.f32'
                raw.write_bytes(data)
                values=array.array('f');values.frombytes(data)
                assert len(values)==248320,'Unexpected Qwen vocabulary/dump row count'
                outputs[arm].append(dict(name=item['name'],dump=raw,response=result))
    comparisons=[]
    for index,item in enumerate(prompts):
        matched=[outputs[a][index] for a in outputs]
        same_bytes([x['dump'] for x in matched])
        assert all(x['response']['tokens']==matched[0]['response']['tokens'] for x in matched)
        comparisons.append(dict(name=item['name'],sha256=sha(matched[0]['dump']),byte_identical=True))
    docs=[load(DATA/'kld-index.json')[i] for i in (0,100)]
    for item in docs:
        input_path=DATA/item['file']
        folder=root/'scorer'/str(item['length'])
        fork=score_once(args,input_path,folder/'fork','off')
        upstream_bin=HERE/'build/pure/stock/bin/exact-score'
        validate_binary(upstream_bin,'pure','stock')
        provenance(folder/'upstream',args,[upstream_bin])
        target=folder/'upstream/result.json'
        command=[str(upstream_bin),str(args.model),str(input_path),str(target)]
        atomic(folder/'upstream/command.json',command)
        assert_idle()
        with (folder/'upstream/native.log').open('w') as log:
            subprocess.run(command,env=env(0,upstream_bin),stdout=log,stderr=subprocess.STDOUT,check=True)
        nll_equal(fork,load(target))
        comparisons.append(dict(name=f'scorer-off-{item["length"]}',nll=fork['nll'],all512_logprobs_identical=True))
    atomic(root/'result.json',dict(pass_gate=True,comparisons=comparisons,pure_vs_instrumented=bridge))


def gate_b(args,root):
    comparisons=[]
    for length in (16384,65536):
        ids=load(DATA/f'timing/{length}.json')[0]['ids']
        results=[]
        dumps=[]
        for arm in ('resident','streamed'):
            folder=root/str(length)/arm
            dump=(folder/'logits.f32').resolve()
            dump.unlink(missing_ok=True)
            assert_idle()
            with server(args,arm,folder,ctx=length+1024,extra_env={'RIDGEFILL_AUDIT_LOGITS':str(dump)},instrumented=True):
                results.append(completion(args,folder/'completion',ids,arm,predict=128))
            assert dump.stat().st_size==128*248320*4,'Missing or incomplete dense raw logits'
            dumps.append(dump)
        same_bytes(dumps)
        assert results[0]['tokens']==results[1]['tokens'] and results[0]['content']==results[1]['content'],'Resident/streamed output mismatch'
        input_path=root/str(length)/'input.json'
        # Exactly L prompt tokens followed by 512 document tokens for NLL; 65K gets a separate full document prefix.
        if length==16384:
            input_path=DATA/load(DATA/'kld-index.json')[0]['file']
        else:
            input_path=DATA/'checks-65k.json'
        resident=score_once(args,input_path,root/str(length)/'nll-resident','resident')
        streamed=score_once(args,input_path,root/str(length)/'nll-streamed','streamed')
        nll_equal(resident,streamed)
        comparisons.append(dict(length=length,output128_identical=True,logits128_byte_identical=True,
                                nll=resident['nll'],all512_logprobs_identical=True))
    atomic(root/'result.json',dict(pass_gate=True,comparisons=comparisons))


def main():
    p=parser(__doc__)
    p.add_argument('--stock-instrumented-bin',type=Path,default=HERE/'build/instrumented/stock/bin/llama-server')
    p.add_argument('--e3-instrumented-bin',type=Path,default=HERE/'build/instrumented/e3/bin/llama-server')
    args=p.parse_args()
    assert args.gpu==0
    root=args.out/'checks'/str(time.time_ns())
    try:
        gate_a(args,root/'CHECK-A')
        gate_b(args,root/'CHECK-B')
        atomic(args.out/'checks/PASS.json',dict(pass_gate=True,provenance=str(root.relative_to(args.out))))
    except BaseException as error:
        atomic(args.out/'checks/STOP.json',dict(error=repr(error),provenance=str(root.relative_to(args.out))))
        raise

if __name__=='__main__':
    main()
