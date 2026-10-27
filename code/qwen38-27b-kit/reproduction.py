"""Mandatory exact stock/E3 reproduction gate for freshly rebuilt identity kits."""
import copy
from pathlib import Path
import time

import sys
from common import DATA, HERE, assert_idle, atomic, load, parser, post, request, rows, server, sha, run_command
from kld_run import run_item


def plan():
    value=load(DATA/'reproduction-plan.json')
    assert len(value['ruler'])==8 and len(value['kld'])==4
    assert {x['length'] for x in value['ruler']}=={16384,32768}
    assert {x['length'] for x in value['kld']}=={16384,32768}
    assert len({x['id'] for x in value['kld']})==4, 'Require four distinct KL documents'
    return value


def receipt_files(out, selected):
    out=Path(out)/'identity-reproduction'
    return ([out/'ruler'/arm/(x['id']+'.json') for arm in ('stock','e3') for x in selected['ruler']]+
            [out/'kld'/str(x['length'])/(x['id']+'.json') for x in selected['kld']])


def signature(out):
    out=Path(out)
    files=receipt_files(out,plan())
    return dict(status='PASS',counts=dict(stock_ruler=8,e3_ruler=8,e3_kl=4),
                manifest_sha256=sha(out/'IDENTITY-MANIFEST.json'),plan_sha256=sha(DATA/'reproduction-plan.json'),
                files={str(p.relative_to(out)):sha(p) for p in files})


def require_reproduction(out):
    out=Path(out)
    if (out/'identity-reproduction/FAIL.json').exists():
        raise RuntimeError('REPRODUCTION failed; inspect identity-reproduction/FAIL.json. No identity measurement is permitted.')
    saved=load(out/'identity-reproduction/PASS.json')
    current=signature(out)
    assert {k:saved[k] for k in current}==current, 'REPRODUCTION artifacts or identity manifest changed'
    return saved


def reference(out, pin):
    path=Path(out)/pin['file']
    assert sha(path)==pin['sha256'], f'Reproduction reference changed: {pin["file"]}'
    return load(path)


def text_equal(actual, expected):
    assert isinstance(actual,str) and isinstance(expected,str)
    assert actual.encode('utf-8')==expected.encode('utf-8'), 'RULER response text differs byte-for-byte'


def score_equal(actual, expected):
    # Build identity and elapsed time necessarily change on a rebuilt box. All
    # other fields, including every NLL/KL value, target array, selector row and
    # vtail scale, must remain exactly equal. No tolerance or normalization.
    ignored={'build','wall_s'}
    a={k:v for k,v in actual.items() if k not in ignored}
    b={k:v for k,v in expected.items() if k not in ignored}
    assert a.keys()==b.keys(), 'Scorer result fields differ'
    for key in sorted(a):
        assert a[key]==b[key], f'Scorer field differs exactly: {key}'


def verify_ruler(receipt, stored, manifest):
    assert receipt['manifest_sha256']==manifest, 'RULER reproduction uses a different identity manifest'
    assert receipt['request']==stored['request'], 'RULER reproduction request differs'
    text_equal(receipt['text'],stored['text'])
    assert receipt['finish_reason']==stored['finish_reason'], 'RULER finish reason differs'


def verify_kl(receipt, stored, manifest):
    assert receipt['manifest_sha256']==manifest, 'KL reproduction uses a different identity manifest'
    assert receipt['result']['input_sha256']==stored['input_sha256'], 'KL reproduction input differs'
    for arm in ('stock','e3'):
        score_equal(receipt['result'][arm],stored[arm])
    assert receipt['result']['dnll']==stored['dnll'], 'KL reproduction dNLL differs exactly'


def run(args):
    workers=getattr(args,'workers',None) or 1
    assert args.identity_only and 0<=args.gpu<workers, 'Invalid REPRODUCTION shard'
    out=Path(args.out)
    folder=out/'identity-reproduction'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'FAIL.json').exists():
        raise RuntimeError('REPRODUCTION has a recorded failure; owner review is required before resuming')
    if (folder/'PASS.json').exists():
        require_reproduction(out)
        print('REPRODUCTION PASS (verified saved gate)',flush=True)
        return
    selected=plan()
    selected={key:[x for i,x in enumerate(selected[key]) if i%workers==args.gpu] for key in ('ruler','kld')}
    manifest=sha(out/'IDENTITY-MANIFEST.json')
    try:
        if workers==1:
            assert_idle()
        items={x['id']:x for x in rows(DATA/'ruler.jsonl')}
        for arm in ('stock','e3'):
            todo=[]
            for item in selected['ruler']:
                path=folder/'ruler'/arm/(item['id']+'.json')
                stored=reference(out,item['references'][arm])
                assert request(items[item['id']]['request'],arm)==stored['request']
                if path.exists():
                    verify_ruler(load(path),stored,manifest)
                else:
                    todo.append((item,path,stored))
            if not todo:
                continue
            attempt=folder/'attempts'/f'ruler-{arm}-{time.time_ns()}'
            with server(args,arm,attempt):
                for item,path,stored in todo:
                    if workers==1:
                        assert_idle(except_gpu=args.gpu)
                    body=request(items[item['id']]['request'],arm)
                    response=post(args.port,'/v1/chat/completions',body)
                    choice=response['choices'][0]
                    value=dict(id=item['id'],arm=arm,request=body,response=response,
                               text=choice['message'].get('content') or '',finish_reason=choice['finish_reason'],
                               manifest_sha256=manifest,reference_sha256=item['references'][arm]['sha256'],
                               provenance=str(attempt.relative_to(out)))
                    atomic(path,value)
                    verify_ruler(value,stored,manifest)
                    print('REPRODUCTION RULER',arm,item['id'],'exact',flush=True)
        # Match the original per-document KL protocol: new E3 scorer off/save,
        # followed by E3/compare. References are regenerated and deleted per pair.
        scorer_args=copy.copy(args)
        scorer_args.identity_only=False
        for item in selected['kld']:
            path=folder/'kld'/str(item['length'])/(item['id']+'.json')
            stored=reference(out,item['reference'])
            if path.exists():
                value=load(path)
            else:
                if workers==1:
                    assert_idle()
                attempt=folder/'attempts'/f'kl-{item["length"]}-{item["id"]}-{time.time_ns()}'
                value=dict(result=run_item(scorer_args,item,attempt),manifest_sha256=manifest,
                           reference_sha256=item['reference']['sha256'],provenance=str(attempt.relative_to(out)))
                atomic(path,value)
            verify_kl(value,stored,manifest)
            print('REPRODUCTION KL',item['length'],item['id'],'exact',flush=True)
        if workers==1:
            finalize(out)
        else:
            print('REPRODUCTION shard complete',args.gpu,flush=True)
    except Exception as error:
        atomic(folder/'FAIL.json',dict(status='FAIL',reason=str(error),manifest_sha256=manifest,time=time.time()))
        raise


def finalize(out):
    out=Path(out)
    manifest=sha(out/'IDENTITY-MANIFEST.json')
    selected=plan()
    for item in selected['ruler']:
        for arm in ('stock','e3'):
            path=out/'identity-reproduction/ruler'/arm/(item['id']+'.json')
            verify_ruler(load(path),reference(out,item['references'][arm]),manifest)
    for item in selected['kld']:
        path=out/'identity-reproduction/kld'/str(item['length'])/(item['id']+'.json')
        verify_kl(load(path),reference(out,item['reference']),manifest)
    atomic(out/'identity-reproduction/PASS.json',dict(**signature(out),completed_at=time.time()))
    print('REPRODUCTION PASS: stock RULER 8, E3 RULER 8, E3 KL 4; all exact',flush=True)


def main():
    p=parser(__doc__)
    p.add_argument('--workers',type=int,choices=(2,4))
    args=p.parse_args()
    if args.workers is not None:
        assert args.workers==args.gpus
        run(args)
        return
    out=Path(args.out)
    manifest=load(out/'IDENTITY-MANIFEST.json')
    assert manifest['gpu_count']==args.gpus
    folder=out/'identity-reproduction'
    if (folder/'FAIL.json').exists():
        raise RuntimeError('REPRODUCTION has a recorded failure; owner review is required before resuming')
    if (folder/'PASS.json').exists():
        require_reproduction(out)
        return
    assert_idle(gpu_count=args.gpus)
    try:
        run_command([sys.executable,str(HERE/'drive.py'),'reproduction.py',str(out),
                     '--identity-only','--gpus',str(args.gpus)])
        finalize(out)
    except Exception as error:
        if not (folder/'FAIL.json').exists():
            atomic(folder/'FAIL.json',dict(status='FAIL',reason=str(error),time=time.time()))
        raise


if __name__=='__main__':
    main()
