"""CUDA port of the frozen fixed-arrival concurrency mix; schedules are unchanged."""
import json
import math
import threading
import time
import urllib.request
from common import DATA, assert_idle, atomic, distribution, load, parser, request, server
save = atomic

def token_metrics(events, expected):
    delivered, times, coalesced = 0, [], 0
    final = None
    for event in events:
        value, at = event['data'], event['t']
        if 'error' in value:
            raise RuntimeError('Server stream error: ' + str(value['error']))
        count = value.get('tokens_predicted', delivered)
        assert type(count) is int and count >= delivered, 'Nonmonotone token count'
        if count > delivered:
            assert value.get('tokens') or value.get('stop'), 'Count advanced without delivered tokens'
            coalesced += max(0, count - delivered - 1)
            times.extend([at] * (count - delivered))
            delivered = count
        if value.get('stop'):
            assert final is None, 'Multiple final events'
            final = value
    assert final is not None and delivered == expected, ('Incomplete stream', delivered, expected)
    assert final['timings']['predicted_n'] == expected and len(times) >= 2
    assert not final.get('truncated'), 'Context truncated'
    gaps = [dict(begin=a, end=b, ms=1000 * (b - a)) for a, b in zip(times, times[1:])]
    assert all(g['ms'] >= 0 for g in gaps)
    return dict(first=times[0], last=times[-1], token_times=times, gaps=gaps, coalesced_tokens=coalesced,
                itl_ms=distribution([g['ms'] for g in gaps]), decode_tps=(expected - 1) / (times[-1] - times[0]), final=final)

def request_json(url, path, body=None, timeout=180):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    assert 'error' not in result, result
    return result

def stream(url, item, body, epoch, dest, output, errors):
    try:
        payload = json.dumps(body).encode()
        deadline = epoch + item['arrival_s']
        threading.Event().wait(max(0, deadline - time.monotonic()))
        sent = time.monotonic() - epoch
        req = urllib.request.Request(url + '/completion', data=payload, headers={'Content-Type': 'application/json'})
        events = []
        with urllib.request.urlopen(req, timeout=240) as response:
            for raw in response:
                if not raw.startswith(b'data:'):
                    continue
                raw = raw[5:].strip()
                if raw == b'[DONE]':
                    break
                value = json.loads(raw)
                events.append(dict(t=time.monotonic() - epoch, data=value))
        save(dest / (item['id'] + '-events.json'), events)
        metrics = token_metrics(events, item['predict'])
        final = metrics['final']
        assert final['tokens_evaluated'] == len(item['ids'])
        assert final['timings']['prompt_n'] == len(item['ids']), 'Prompt cache reuse or incomplete prompt'
        assert sent - item['arrival_s'] < .1, ('Arrival missed', item['id'], sent)
        report = final['timings'].get('ridgefill')
        expected_fill = body['ridgefill_mode'] != 'off'
        assert (final['timings'].get('prompt_filled_n', 0) > 0) == expected_fill
        if expected_fill:
            assert report and report['n_filled'] > 0 and report['n_cached_exact_prefix'] == 0
        metrics.update(id=item['id'], role=item['role'], n_prompt=len(item['ids']), predict=item['predict'], sent=sent,
                       planned=item['arrival_s'], arrival_lateness_ms=1000 * (sent - item['arrival_s']),
                       ttft_s=metrics['first'] - sent, scheduled_ttft_s=metrics['first'] - item['arrival_s'])
        save(dest / (item['id'] + '-result.json'), metrics)
        output.append(metrics)
    except Exception as error:
        errors.append(dict(id=item['id'], error=repr(error)))

def summarize(requests):
    injector = next(r for r in requests if r['role'] == 'injector')
    other = [r for r in requests if r['role'] == 'background']
    for row in other:
        assert row['first'] < injector['sent'] < row['last'], 'Background was not decoding at injection'
        assert sum(t < injector['sent'] for t in row['token_times']) >= 4
    gaps = [g for r in other for g in r['gaps']]
    prefill_gaps = [g for g in gaps if g['end'] >= injector['sent'] and g['begin'] <= injector['first']]
    pre = [g for g in gaps if g['end'] <= injector['sent']]
    post = [g for g in gaps if g['begin'] >= injector['first']]
    wall = max(r['last'] for r in requests) - min(r['sent'] for r in requests)
    return dict(other_ttft_s=distribution([r['ttft_s'] for r in other]),
                other_itl_ms=distribution([g['ms'] for g in gaps]),
                other_prefill_itl_ms=distribution([g['ms'] for g in prefill_gaps]),
                other_pre_itl_ms=distribution([g['ms'] for g in pre]), other_post_itl_ms=distribution([g['ms'] for g in post]),
                other_aggregate_tps=sum(r['predict'] for r in other) / (max(r['last'] for r in other) - min(r['sent'] for r in other)) if other else None,
                aggregate_tps=sum(r['predict'] for r in requests) / wall, injector_ttft_s=injector['ttft_s'],
                injector_decode_tps=injector['decode_tps'], injector_server_decode_tps=injector['final']['timings']['predicted_per_second'],
                coalesced_tokens=sum(r['coalesced_tokens'] for r in requests), workload_seconds=wall,
                arrival_lateness_ms=distribution([r['arrival_lateness_ms'] for r in requests]),
                background_overlap_checked=len(other), background_active_at_injector_first=sum(r['last'] > injector['first'] for r in other),
                other_prefill_tps=sum(injector['sent'] <= t <= injector['first'] for r in other for t in r['token_times']) / (injector['first'] - injector['sent']))


def request_body(item, arm):
    active=item['role']=='injector' and arm!='stock'
    body=request(dict(prompt=item['ids'],n_predict=item['predict'],id_slot=item['slot'],seed=0,temperature=0,top_k=1,
                      ignore_eos=True,stream=True,return_tokens=True,cache_prompt=False,post_sampling_probs=False),
                 'streamed' if active else 'stock')
    if not active:
        body.update(ridgefill=False,ridgefill_mode='off')
    return body


def run_one(args,spec,run,folder):
    concurrency=run['concurrency']
    plan=spec['schedules'][str(concurrency)]
    arm='stock' if run['arm']=='stock' else 'streamed'
    ctx=concurrency*(math.ceil(max(len(i['ids'])+i['predict'] for i in plan)/512)*512+512)
    assert_idle()
    started=time.monotonic()
    with server(args,arm,folder,ctx=ctx,slots=concurrency,unified=spec['kv_unified']) as (proc,vram):
        expected_kv="kv_unified = '"+('true' if spec['kv_unified'] else 'false')+"'"
        assert expected_kv in (folder/'server.log').read_text(),'Server KV layout differs from frozen mix spec'
        warm=request_body(dict(ids=plan[0]['ids'][:32],slot=0,predict=2,role='background'),arm)
        warm['stream']=False
        atomic(folder/'warmup.json',request_json(f'http://127.0.0.1:{args.port}','/completion',warm))
        output,errors,threads=[],[],[]
        # write every request file before the clock starts: at C8 the fsynced writes took ~3.5 s and made every arrival late
        bodies=[]
        for item in plan:
            body=request_body(item,arm)
            atomic(folder/(item['id']+'-request.json'),body)
            bodies.append(body)
        epoch=time.monotonic()+.5
        for item,body in zip(plan,bodies):
            thread=threading.Thread(target=stream,args=(f'http://127.0.0.1:{args.port}',item,body,epoch,folder,output,errors),daemon=True)
            threads.append(thread)
            thread.start()
        deadline=time.monotonic()+600
        while any(t.is_alive() for t in threads):
            assert proc.poll() is None and time.monotonic()<deadline,'Server died/workload timed out'
            assert_idle(except_gpu=0)
            for thread in threads:
                thread.join(timeout=.1)
        atomic(folder/'errors.json',errors)
        assert not errors and len(output)==len(plan),errors
        metrics=summarize(output)
    return dict(run,kv_unified=spec['kv_unified'],metrics=metrics,requests=sorted(output,key=lambda x:x['id']),
                wall_seconds=time.monotonic()-started,peak_mib=vram.peak,provenance=str(folder.relative_to(args.out)))


def main():
    args=parser(__doc__).parse_args()
    assert args.gpu==0,'Concurrency mix requires physical GPU 0'
    for path in sorted((DATA/'mix').glob('*.json')):
        spec=load(path)
        for run in spec['runs']:
            target=args.out/'mix'/path.stem/run['name']/'result.json'
            if target.exists():
                continue
            folder=target.parent/str(time.time_ns())
            atomic(target,run_one(args,spec,run,folder))
            print(path.stem,run['name'],'PASS',flush=True)

if __name__=='__main__':
    main()
