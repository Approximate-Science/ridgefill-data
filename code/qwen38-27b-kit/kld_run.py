"""Paired post-prompt KL on 100 frozen LongBench document windows per length."""
import math
from pathlib import Path
import subprocess
import time
from common import DATA, atomic, env, load, parser, provenance, rows, sha, validate_binary, VRAM

# score.cpp uses shorter names for the core selector options. Every value matches run_lb.py E3_FLAGS.
SCORE_E3 = ['--mode','quality+qattn','--tail','4096','--min-prompt','8192','--share','0.5','--digits','on',
            '--special-exact','on','--alpha','0','--qattn-share','0.2','--qattn-window','64','--header-rows','0',
            '--qattn-window-mode','last','--qattn-turn-cap','256','--ridgefill-s1-window','256',
            '--ridgefill-s1-factor','1.0','--ridgefill-vtail','on','--ridgefill-vtail-rows','512',
            '--ridgefill-lexical','on','--ridgefill-lexical-window','64','--ridgefill-lexical-min-words','4',
            '--ridgefill-lexical-span-rows','80','--ridgefill-lexical-max-occurrences','2',
            '--ridgefill-trim-kv','on','--qattn-window-end','known']


def score_once(args, input_path, folder, mode, extra=(), scored=512):
    validate_binary(args.scorer,'pure','e3')
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    provenance(folder,args,[args.scorer])
    output=folder/'native.jsonl'
    command=[str(args.scorer),'--model',str(args.model),'--input',str(input_path),'--output',str(output),
             '--scored',str(scored),'--gpu-layers','99','--threads','4','--batch','2048','--ubatch','512',
             '--cache-type','f16','--h-type','f16','--flash-attn','on']
    if mode=='off':
        command+=['--mode','off']
    else:
        command+=['--projector',str(args.identity_projector if mode=='identity' else args.projector),*SCORE_E3,
                  '--ridgefill-stream-projector','on' if mode=='streamed' else 'off']
    command+=list(extra)
    atomic(folder/'command.json',dict(command=command,input_sha256=sha(input_path),gpu=args.gpu))
    with (folder/'native.log').open('w') as log:
        proc=subprocess.Popen(command,env=env(args.gpu,args.scorer),stdout=log,stderr=subprocess.STDOUT)
        try:
            with VRAM(folder,proc.pid):
                rc=proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait()
        if rc:
            raise RuntimeError(f'Scorer exited {rc}: {folder}/native.log')
    records=rows(output)
    assert len(records)==1
    result=records[0]
    assert result['scored']==scored and math.isfinite(result['nll'])
    assert len(result['logprobs'])==scored and all(math.isfinite(x) for x in result['logprobs'])
    return result


def run_item(args, item, folder):
    input_path=DATA/item['file']
    assert len(load(input_path)['ids'])==item['length']+512
    # Opaque dtype-tagged float32 reference: about 0.5 GB at 512 scored positions.
    # drive.py starts one worker per GPU; each worker completes pairs serially.
    reference=folder/'reference.logprobs'
    reference.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    try:
        stock=score_once(args,input_path,folder/'stock','off',['--save-logprobs',str(reference)])
        assert reference.stat().st_size>0
        if getattr(args,'identity_only',False):
            previous=load(args.out/'kld'/str(item['length'])/(item['id']+'.json'))
            assert previous['input_sha256']==sha(input_path), 'KL baseline input differs'
            assert stock['target_positions']==previous['stock']['target_positions']
            assert stock['logprobs']==previous['stock']['logprobs'], 'Regenerated stock reference differs from original baseline'
        arm='identity' if getattr(args,'identity_only',False) else 'e3'
        e3=score_once(args,input_path,folder/arm,arm,['--kl-against',str(reference)])
        reference.unlink()  # Release the large payload immediately after comparison.
        assert len(e3['kl'])==len(e3['top1'])==512
        assert all(x in (0,1,False,True) for x in e3['top1'])
        for key in ('kl_mean','kl_median','kl_p99','kl_max','top1_agree','dp_target_mean'):
            assert math.isfinite(e3[key]),key
        assert all(math.isfinite(x) for x in e3['kl'])
        assert stock['target_positions']==e3['target_positions']==list(range(item['length'],item['length']+512))
        assert e3['selector_excludes_targets'], 'Selector read scored targets'
        return dict(id=item['id'],length=item['length'],gpu=args.gpu,stock=stock,**{arm:e3},
                    dnll=e3['nll']-stock['nll'],wall_s=time.monotonic()-started,input_sha256=sha(input_path))
    finally:
        reference.unlink(missing_ok=True)


def main():
    p=parser(__doc__)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.identity_only:
        from reproduction import require_reproduction
        require_reproduction(args.out)
    # Do not overlap document pairs: at most one saved reference per GPU.
    for index,item in enumerate(load(DATA/'kld-index.json')):
        if index%args.workers!=args.gpu:
            continue
        component='kld-identity' if args.identity_only else 'kld'
        output=args.out/component/str(item['length'])/(item['id']+'.json')
        if output.exists():
            continue
        folder=args.out/(component+'-work')/str(item['length'])/item['id']/str(time.time_ns())
        atomic(output,run_item(args,item,folder))
        print('KLD',args.gpu,item['length'],item['id'],flush=True)

if __name__=='__main__':
    main()
