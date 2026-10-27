"""TIMING-WARM: fresh server, cold, full warm-up, three distinct timed prompts."""
import re
import time
from common import DATA, assert_idle, atomic, distribution, load, parser, request, server, stream, sha


def log_records(text):
    # Frozen harness log parser; compares server log prompt_eval with the SSE timing.
    records={}
    for number,line in enumerate(text.splitlines(),1):
        match=re.search(r'\| task\s+(\d+) \|',line)
        if not match:
            continue
        task=int(match[1])
        if 'new prompt,' in line:
            assert task not in records
            records[task]=dict(task=task,start_line=number)
        if task not in records:
            continue
        row=records[task]
        clear=re.search(r'clearing prompt with (\d+) tokens',line)
        if clear:
            row['cleared_previous_tokens']=int(clear[1])
        fill=re.search(r'RidgeFill: filling prompt tokens \[0, (\d+)\) of (\d+)',line)
        if fill:
            row.update(ridgefill_from_zero=True,ridgefill_prefix=int(fill[1]),ridgefill_prompt_n=int(fill[2]))
        cache=re.search(r'cached n_tokens =\s*(\d+)',line)
        if cache and not row.get('ridgefill_from_zero') and 'initial_cached_n' not in row:
            row['initial_cached_n']=int(cache[1])
        timing=re.search(r'prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens',line)
        if timing:
            row.update(prompt_ms=float(timing[1]),prompt_n=int(timing[2]))
    return list(records.values())


def run_arm(args,length,arm,folder):
    assert_idle(gpu_count=args.gpus)
    prompts=load(DATA/f'timing/{length}.json')
    assert len(prompts)==5 and len({x['ids'][0] for x in prompts})==5
    results=[]
    with server(args,arm,folder,ctx=length+1024) as (proc,vram):
        for item in prompts:
            assert_idle(except_gpu=0,gpu_count=args.gpus)
            assert len(item['ids'])==length
            body=request(dict(prompt=item['ids'],n_predict=1,ignore_eos=True,temperature=0,top_k=1,
                              seed=0,cache_prompt=False,return_tokens=True),arm)
            atomic(folder/(item['phase']+'-request.json'),body)
            events,ttft,wall=stream(args.port,body)
            response=events[-1]['data']
            timings=response['timings']
            assert timings['prompt_n']==length and timings['cache_n']==0 and timings['predicted_n']==1
            assert (timings.get('prompt_filled_n',0)>0)==(arm!='stock')
            atomic(folder/(item['phase']+'-events.json'),events)
            row=dict(phase=item['phase'],client_ttft_s=ttft,wall_s=wall,prompt_ms=timings['prompt_ms'],response=response)
            atomic(folder/(item['phase']+'-result.json'),row)
            results.append(row)
            assert proc.poll() is None
    logs=log_records((folder/'server.log').read_text())
    assert len(logs)==len(results)==5, ('Unrecognized server timing logs',len(logs))
    for row,log in zip(results,logs):
        assert log['prompt_n']==length and abs(log['prompt_ms']-row['prompt_ms'])<=.02
        if arm=='stock':
            assert log['initial_cached_n']==0
        else:
            assert log.get('ridgefill_from_zero') and 'cleared_previous_tokens' in log
            assert log['ridgefill_prompt_n']==length and log['ridgefill_prefix']==length-4096
        row['server_log']=log
    warm=[r for r in results if r['phase'].startswith('timed-')]
    return dict(length=length,arm=arm,rows=results,peak_mib=vram.peak,
                warm_prompt_s=distribution([r['prompt_ms']/1000 for r in warm]),
                warm_client_ttft_s=distribution([r['client_ttft_s'] for r in warm]))


def main():
    args=parser(__doc__).parse_args()
    if args.identity_only:
        from reproduction import require_reproduction
        require_reproduction(args.out)
    assert args.gpu==0,'TIMING-WARM requires physical GPU 0'
    root=args.out/('timing-identity' if args.identity_only else 'timing')
    session=None
    if args.identity_only:
        manifest=load(args.out/'IDENTITY-MANIFEST.json')
        assert manifest['gpu_count']==args.gpus
        binding=dict(manifest_sha256=sha(args.out/'IDENTITY-MANIFEST.json'),gpu=0,
                     gpu_uuid=manifest['hardware'][0][3],gpu_count=args.gpus)
        path=root/'SESSION.json'
        if path.exists():
            session=load(path)
            assert {k:session[k] for k in binding}==binding,'Timing session hardware/manifest changed'
        else:
            session=dict(**binding,session_id=str(time.time_ns()))
            atomic(path,session)
    for length in (16384,32768,65536,120000):
        order=('stock','resident','streamed') if length in (16384,65536) else ('streamed','resident','stock')
        if args.identity_only:
            order=('stock','resident','identity','streamed') if length in (16384,65536) else ('streamed','identity','resident','stock')
        for arm in order:
            target=root/str(length)/arm/'result.json'
            if target.exists():
                if session:
                    assert load(target)['session_id']==session['session_id']
                continue
            # An interrupted arm restarts from cold on a fresh server.
            folder=target.parent/str(time.time_ns())
            value=run_arm(args,length,arm,folder)
            value['provenance']=str(folder.relative_to(args.out))
            if session:
                value.update(session_id=session['session_id'],gpu_uuid=session['gpu_uuid'],gpu=0)
            atomic(target,value)
        values={a:load(root/str(length)/a/'result.json') for a in order}
        for resident,streamed in zip(values['resident']['rows'],values['streamed']['rows']):
            assert resident['response']['tokens']==streamed['response']['tokens']
            assert resident['response']['content']==streamed['response']['content']

if __name__=='__main__':
    main()
