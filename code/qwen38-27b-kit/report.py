"""One complete paper scorecard, with strict item counts and paired confidence intervals."""
import json
from pathlib import Path
import statistics
from common import DATA, atomic, bootstrap, distribution, load, parser, rows
from ruler_replay import summarize


def ruler_report(out,identity=False):
    records=[]
    for item in rows(DATA/'ruler.jsonl'):
        pair={a:load(out/'ruler'/a/(item['id']+'.json')) for a in (('stock','e3','identity') if identity else ('stock','e3'))}
        records.append(dict(id=item['id'],length=item['length'],task=item['task'],
                            **{a:r['score'] for a,r in pair.items()},
                            **{a+'_finish':r['finish_reason'] for a,r in pair.items()}))
    result=summarize(records)
    for length,values in result.items():
        for task,v in values['per_task'].items():
            part=[r for r in records if str(r['length'])==length and r['task']==task]
            for arm in (('stock','e3','identity') if identity else ('stock','e3')):
                v[arm+'_ci']=bootstrap([[r[arm] for r in part]])['ci95']
        for arm in (('stock','e3','identity') if identity else ('stock','e3')):
            values[arm+'_ci']=bootstrap([[r[arm] for r in records if str(r['length'])==length and r['task']==task]
                                         for task in values['per_task']])['ci95']
    return result


def kld_report(out,identity=False):
    result={}
    for length in (16384,32768):
        index=[x for x in load(DATA/'kld-index.json') if x['length']==length]
        assert len(index)==100
        items=[load(out/'kld'/str(length)/(x['id']+'.json')) for x in index]
        kl=[x for r in items for x in r['e3']['kl']]
        top1=[x for r in items for x in r['e3']['top1']]
        assert len(kl)==len(top1)==51200
        stats=distribution(kl)
        result[str(length)]=dict(n_docs=100,n_positions=51200,mean_kl=bootstrap([[r['e3']['kl_mean'] for r in items]]),
                                median_kl=stats['median'],p99_kl=stats['p99'],max_kl=stats['maximum'],
                                top1_agreement=statistics.mean(top1),dnll=bootstrap([[r['dnll'] for r in items]]),
                                median_dnll=statistics.median(r['dnll'] for r in items),
                                dp_target_mean=statistics.mean(r['e3']['dp_target_mean'] for r in items))
        if identity:
            added=[load(out/'kld-identity'/str(length)/(x['id']+'.json')) for x in index]
            assert all(a['id']==b['id'] and a['input_sha256']==b['input_sha256'] for a,b in zip(items,added))
            stats=distribution([v for r in added for v in r['identity']['kl']])
            assert stats['n']==51200
            result[str(length)].update(stock_nll=statistics.mean(r['stock']['nll'] for r in items),
                e3_nll=statistics.mean(r['e3']['nll'] for r in items),
                identity=dict(nll=statistics.mean(r['identity']['nll'] for r in added),
                    mean_kl=bootstrap([[r['identity']['kl_mean'] for r in added]]),
                    median_kl=stats['median'],p99_kl=stats['p99'],max_kl=stats['maximum'],
                    top1_agreement=statistics.mean(v for r in added for v in r['identity']['top1']),
                    dnll=bootstrap([[r['dnll'] for r in added]])),
                identity_vs_e3=dict(dnll=bootstrap([[b['identity']['nll']-a['e3']['nll'] for a,b in zip(items,added)]]),
                    delta_kl_to_stock=bootstrap([[b['identity']['kl_mean']-a['e3']['kl_mean'] for a,b in zip(items,added)]])))
    return result


def longbench_report(out=None,identity=False):
    items=rows(DATA/'longbench-saved.jsonl')
    assert len(items)==503
    if identity:
        gold={x['id']:x['answers'][0] for x in rows(DATA/'longbench-items.jsonl')}
        for item in items:
            value=load(out/'longbench/identity'/(item['id']+'.json'))
            assert value['id']==item['id'] and value['gold']==gold[item['id']]
            assert value['judge']==(value['pred']==value['gold'])
            item['arms']['identity']=value
    result={}
    groups={'overall':items}
    for field in ('domain','difficulty','length'):
        groups.update({field+': '+v:[r for r in items if r[field]==v] for v in sorted({r[field] for r in items})})
    for key,part in groups.items():
        scores={a:100*statistics.mean(r['arms'][a]['judge'] for r in part) for a in ('stock','e3')}
        result[key]=dict(n=len(part),**scores,delta=bootstrap([[100*(int(r['arms']['e3']['judge'])-int(r['arms']['stock']['judge'])) for r in part]]))
        if identity:
            result[key].update(identity=100*statistics.mean(r['arms']['identity']['judge'] for r in part),
                **{'identity_vs_'+arm:bootstrap([[100*(int(r['arms']['identity']['judge'])-int(r['arms'][arm]['judge']))
                    for r in part]]) for arm in ('stock','e3')})
    result['diagnostics']={a:dict(no_answer=sum(r['arms'][a]['pred'] is None for r in items),
                                 pass1_length_stops=sum(r['arms'][a]['pass1_finish']=='length' for r in items),
                                 pass2_length_stops=sum(r['arms'][a]['pass2_finish']=='length' for r in items)) for a in ('stock','e3')}
    if identity:
        result['diagnostics']['identity']=dict(no_answer=sum(r['arms']['identity']['pred'] is None for r in items),
            pass1_length_stops=sum(r['arms']['identity']['pass1_finish']=='length' for r in items),
            pass2_length_stops=sum(r['arms']['identity']['pass2_finish']=='length' for r in items))
    times={a:sum(r['arms'][a]['pass1_timings']['prompt_ms'] for r in items) for a in ('stock','e3')}
    result['workload_prefill_speedup']=times['stock']/times['e3']
    return result


def timing_report(out,identity=False):
    result=[]
    root=out/('timing-identity' if identity else 'timing')
    session=load(root/'SESSION.json') if identity else None
    for length in (16384,32768,65536,120000):
        arms={a:load(root/str(length)/a/'result.json') for a in (('stock','resident','identity','streamed') if identity else ('stock','resident','streamed'))}
        for arm,row in arms.items():
            if session:
                assert row['session_id']==session['session_id'] and row['gpu_uuid']==session['gpu_uuid'] and row['gpu']==0
            assert len(row['rows'])==5 and row['warm_prompt_s']['n']==3
            result.append(dict(length=length,arm=arm,cold_prompt_s=row['rows'][0]['prompt_ms']/1000,
                               cold_client_ttft_s=row['rows'][0]['client_ttft_s'],warm_prompt_s=row['warm_prompt_s'],
                               warm_client_ttft_s=row['warm_client_ttft_s'],peak_gib=row['peak_mib']/1024,
                               delta_stock_mib=row['peak_mib']-arms['stock']['peak_mib'],
                               speedup=arms['stock']['warm_prompt_s']['median']/row['warm_prompt_s']['median']))
            if arm=='identity':
                for reference,label in (('stock','stock'),('resident','e3')):
                    added={r['phase']:r for r in row['rows'] if r['phase'].startswith('timed-')}
                    base={r['phase']:r for r in arms[reference]['rows'] if r['phase'].startswith('timed-')}
                    assert added.keys()==base.keys() and len(added)==3
                    result[-1]['identity_vs_'+label]=dict(
                        prompt_s=bootstrap([[(added[k]['prompt_ms']-base[k]['prompt_ms'])/1000 for k in added]]),
                        client_ttft_s=bootstrap([[added[k]['client_ttft_s']-base[k]['client_ttft_s'] for k in added]]))
    return result


def mix_report(out):
    result=[]
    for path in sorted((DATA/'mix').glob('*.json')):
        spec=load(path)
        values=[load(out/'mix'/path.stem/r['name']/'result.json') for r in spec['runs']]
        for arm in ('stock','e3-streamed'):
            part=[r for r in values if r['arm']==arm]
            assert len(part)==2
            def avg(key):
                return statistics.mean(r['metrics'][key] for r in part)
            def itl(key):
                return statistics.mean(r['metrics']['other_itl_ms'][key] for r in part) if part[0]['metrics']['other_itl_ms'] else None
            result.append(dict(schedule=path.stem,arm=arm,reps=2,injector_ttft_s=avg('injector_ttft_s'),
                               injector_decode_tps=avg('injector_decode_tps'),
                               other_aggregate_tps=avg('other_aggregate_tps') if part[0]['metrics']['other_aggregate_tps'] is not None else None,
                               other_itl_mean_ms=itl('mean'),other_itl_p99_ms=itl('p99'),
                               other_itl_max_ms=max(r['metrics']['other_itl_ms']['maximum'] for r in part) if part[0]['metrics']['other_itl_ms'] else None,
                               peak_gib=max(r['peak_mib'] for r in part)/1024,raw_replicates=part))
    return result


def markdown(result):
    lines=['# E3 vs stock — A100 CUDA scorecard','',f"Status: {result['status']}. All new measurements use pure builds of unmodified upstream 1242be66f and the committed E3 scorer source in RUN-MANIFEST.json. Separate instrumented servers are used only for correctness gates; CHECK-A ties them to the pure builds with identical 128-token outputs and top-10 probabilities on two prompts per arm.",
           '', 'RULER deviations: owner instruct format, study lengths, thinking off. Official answer budgets are unchanged. LongBench uses official CoT with thinking off. KL is KL(stock || E3), over the 512 document tokens after the prompt; bootstrap units are documents. dNLL is E3 minus the engine exact forward, tied to upstream by CHECK-A.',
           '', 'RULER CIs resample paired items within task and average the 13 task means (10,000 resamples, seed 1). KL median/p99 and top-1 agreement pool the equal-length per-position arrays. Timing shows median [min,max] of three distinct warm requests; cold and warm-up are excluded. Mix means average two fresh replicates; maximum ITL is the maximum over both.', '']
    if result.get('identity_protocol'):
        lines[0]='# Stock, E3 and identity — A100 CUDA scorecard'
        lines[2]=f"Status: {result['status']}. Identity uses rebuilt pure sources pinned in IDENTITY-MANIFEST.json. Mandatory REPRODUCTION ties the new stock/E3 builds to the stored baseline with 16 exact RULER texts and four exact E3 KL/NLL documents; original CHECK-A/B remain evidence for the original baseline builds."
        lines += [result['identity_protocol'], '']
    if 'ruler' in result:
        lines+=['| RULER length/task | n | Stock % | E3 % | Delta pp [95% CI] | Length stops S/E |','|---|---:|---:|---:|---:|---:|']
        for length,v in result['ruler'].items():
            for task,row in [('overall',v),*v['per_task'].items()]:
                d=row['delta'];lo,hi=d['ci95']
                stops=row.get('length_stops',{})
                lines.append(f"| {length}/{task} | {row['n']} | {row['stock']:.2f} | {row['e3']:.2f} | {d['mean']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {stops.get('stock','—')}/{stops.get('e3','—')} |")
        lines+=['','Official vt remains a 30-token-budget cell; if both arms stop in the preamble it carries no retrieval information.','']
        if result.get('identity_only'):
            def ci(v):
                return f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
            lines+=['| RULER length/task | Stock % | E3 % | Identity % | Identity−stock pp [95% CI] | Identity−E3 pp [95% CI] |',
                    '|---|---:|---:|---:|---:|---:|']
            for length,v in result['ruler'].items():
                for task,row in [('overall',v),*v['per_task'].items()]:
                    lines.append(f"| {length}/{task} | {row['stock']:.2f} | {row['e3']:.2f} | {row['identity']:.2f} | {ci(row['identity_vs_stock'])} | {ci(row['identity_vs_e3'])} |")
            lines+=['']
    if 'kld' in result:
        lines+=['| Prompt tokens | Docs | Mean KL [95% CI] | Median KL | p99 KL | Top-1 agreement | dNLL [95% CI] |','|---:|---:|---:|---:|---:|---:|---:|']
        for length,v in result['kld'].items():
            k,d=v['mean_kl'],v['dnll']
            lines.append(f"| {length} | 100 | {k['mean']:.6g} [{k['ci95'][0]:.6g}, {k['ci95'][1]:.6g}] | {v['median_kl']:.6g} | {v['p99_kl']:.6g} | {v['top1_agreement']:.4%} | {d['mean']:+.6g} [{d['ci95'][0]:+.6g}, {d['ci95'][1]:+.6g}] |")
        lines+=['']
        if result.get('identity_only'):
            def ci(v):
                return f"{v['mean']:+.6g} [{v['ci95'][0]:+.6g}, {v['ci95'][1]:+.6g}]"
            lines+=['| KL prompt | Stock NLL | E3 NLL | Identity NLL | Identity−stock NLL [95% CI] | Identity−E3 NLL [95% CI] |',
                    '|---|---:|---:|---:|---:|---:|']
            for length,v in result['kld'].items():
                lines.append(f"| {length} | {v['stock_nll']:.6g} | {v['e3_nll']:.6g} | {v['identity']['nll']:.6g} | {ci(v['identity']['dnll'])} | {ci(v['identity_vs_e3']['dnll'])} |")
            lines+=['','| KL prompt | E3 mean KL to stock [95% CI] | Identity mean KL to stock [95% CI] | Paired difference [95% CI] | Identity median / p99 KL | Identity top-1 |',
                    '|---|---:|---:|---:|---:|---:|']
            for length,v in result['kld'].items():
                i=v['identity']
                lines.append(f"| {length} | {ci(v['mean_kl'])} | {ci(i['mean_kl'])} | {ci(v['identity_vs_e3']['delta_kl_to_stock'])} | {i['median_kl']:.6g} / {i['p99_kl']:.6g} | {i['top1_agreement']:.4%} |")
            lines+=['','The paired KL difference subtracts KL(stock || E3) from KL(stock || identity); it is not KL(E3 || identity). Stock references were regenerated per document and E3 was reused.','']
    lines+=['| LongBench group | n | Stock % | E3 % | Delta pp |','|---|---:|---:|---:|---:|']
    for name,v in result['longbench'].items():
        if isinstance(v,dict) and 'n' in v:
            lines.append(f"| {name} | {v['n']} | {v['stock']:.2f} | {v['e3']:.2f} | {v['delta']['mean']:+.2f} |")
    lb=result['longbench'];overall=lb['overall']['delta']
    lines+=['',f"LongBench paired delta 95% CI: [{overall['ci95'][0]:+.2f}, {overall['ci95'][1]:+.2f}] pp. Extraction/cap diagnostics: `{json.dumps(lb['diagnostics'],sort_keys=True)}`.",f"LongBench workload prefill speedup: {lb['workload_prefill_speedup']:.3f}×; arms ran on different physical GPUs in the completed benchmark.",'']
    if 'identity' in lb['overall']:
        lines+=['| LongBench group | Stock % | E3 % | Identity % | Identity−stock pp [95% CI] | Identity−E3 pp [95% CI] |',
                '|---|---:|---:|---:|---:|---:|']
        for name,v in lb.items():
            if isinstance(v,dict) and 'n' in v:
                s,e=v['identity_vs_stock'],v['identity_vs_e3']
                lines.append(f"| {name} | {v['stock']:.2f} | {v['e3']:.2f} | {v['identity']:.2f} | {s['mean']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}] | {e['mean']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}] |")
        lines+=['']
    if 'timing' in result:
        lines+=['| Tokens | Arm | Warm prompt s [min,max] | Client TTFT s [min,max] | Peak GiB | Delta stock MiB | Speedup |','|---:|---|---:|---:|---:|---:|---:|']
        for v in result['timing']:
            p,t=v['warm_prompt_s'],v['warm_client_ttft_s']
            lines.append(f"| {v['length']} | {v['arm']} | {p['median']:.3f} [{p['minimum']:.3f},{p['maximum']:.3f}] | {t['median']:.3f} [{t['minimum']:.3f},{t['maximum']:.3f}] | {v['peak_gib']:.3f} | {v['delta_stock_mib']:+.0f} | {v['speedup']:.3f}× |")
        lines+=['']
    if 'timing' in result and result.get('identity_only'):
        def ci(v): return f"{v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
        lines+=['| Tokens | Identity−stock prompt s [95% CI] | Identity−E3 prompt s [95% CI] | Identity−stock client TTFT s [95% CI] | Identity−E3 client TTFT s [95% CI] |',
                '|---|---:|---:|---:|---:|']
        for v in result['timing']:
            if v['arm']=='identity':
                s,e=v['identity_vs_stock'],v['identity_vs_e3']
                lines.append(f"| {v['length']} | {ci(s['prompt_s'])} | {ci(e['prompt_s'])} | {ci(s['client_ttft_s'])} | {ci(e['client_ttft_s'])} |")
        lines+=['','Identity timing uses only the new timing-identity session: stock, E3 resident, E3 streamed and identity share physical GPU 0. Old timing files are unchanged. Timing CIs pair three timed prompts; n=3 gives limited precision.','']
    if 'mix' in result:
        def fmt(x):return '—' if x is None else f'{x:.3f}'
        lines+=['| Mix | Arm | Injector TTFT s | Other tok/s | ITL mean/p99/max ms | Injector tok/s | Peak GiB |','|---|---|---:|---:|---:|---:|---:|']
        for v in result['mix']:
            lines.append(f"| {v['schedule']} | {v['arm']} | {v['injector_ttft_s']:.3f} | {fmt(v['other_aggregate_tps'])} | {fmt(v['other_itl_mean_ms'])}/{fmt(v['other_itl_p99_ms'])}/{fmt(v['other_itl_max_ms'])} | {v['injector_decode_tps']:.3f} | {v['peak_gib']:.3f} |")
    if result.get('missing'):
        lines+=['','Missing measurements (this is an incomplete preview): '+', '.join(result['missing'])]
    return '\n'.join(lines)+'\n'


def main():
    p=parser(__doc__)
    p.add_argument('--allow-partial',action='store_true',help='Explicit incomplete preview; never a completed scorecard')
    args=p.parse_args()
    identity_lb=args.identity_only
    result=dict(status='COMPLETE',longbench=longbench_report(),missing=[],identity_only=args.identity_only)
    if identity_lb:
        try:
            result['longbench']=longbench_report(args.out,True)
        except FileNotFoundError:
            if not args.allow_partial: raise
            result['missing'].append('longbench_identity')
    if args.identity_only:
        result['identity_protocol']='Same flags/data and S1 algorithm as E3; S1 rows depend on fill maps. This is not a fixed-row or guaranteed equal-speed ablation.'
        from reproduction import require_reproduction
        try:
            result['reproduction']=require_reproduction(args.out)
        except FileNotFoundError:
            if not args.allow_partial: raise
            result['missing'].append('REPRODUCTION')
    for name,fn in [('ruler',ruler_report),('kld',kld_report),('timing',timing_report),('mix',mix_report)]:
        try:
            result[name]=fn(args.out,args.identity_only) if name!='mix' else fn(args.out)
        except FileNotFoundError:
            if not args.allow_partial:
                raise
            result['missing'].append(name)
    gate=args.out/'checks/PASS.json'
    if gate.exists():
        result['checks']=load(gate)
        result['run_manifest']=load(args.out/'RUN-MANIFEST.json')
        if args.identity_only:
            result['identity_manifest']=load(args.out/'IDENTITY-MANIFEST.json')
    elif not args.allow_partial:
        raise RuntimeError('Mandatory CUDA gates have not passed')
    else:
        result['missing'].append('CHECK-A/B')
    if result['missing']:
        result['status']='INCOMPLETE'
    stem='SCORECARD-IDENTITY' if args.identity_only else 'SCORECARD'
    atomic(args.out/(stem+'.json'),result)
    target=args.out/(stem+'.md')
    temp=target.with_suffix('.md.tmp');temp.write_text(markdown(result));temp.replace(target)
    print(target)

if __name__=='__main__':
    main()
