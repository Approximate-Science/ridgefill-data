"""Frozen official-budget RULER replay; --cpu-gate rescoring needs no GPU."""
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

from common import DATA, HERE, atomic, bootstrap, canonical, load, parser, post, request, rows, server
sys.path.insert(0, str(HERE / 'vendor'))
import score_ruler

BUDGETS = dict(niah=128, vt=30, cwe=120, fwe=50, qa=32)


def budget(task):
    return BUDGETS['niah' if task.startswith('niah') else 'qa' if task.startswith('qa') else task]


def score(item, text):
    # Call the frozen validate-harness scorer, including its unparseable handling.
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        items, predictions = Path(tmp) / 'items', Path(tmp) / 'predictions'
        items.write_text(json.dumps(item) + '\n')
        predictions.write_text(json.dumps(dict(id=item['id'], text=text)) + '\n')
        return score_ruler.score(items, predictions)['overall']['score']


def summarize(records):
    result = {}
    identity=all('identity' in r for r in records)
    for length in (16384, 32768):
        per, all_diffs = {}, []
        # Match the auditor's alphabetical run traversal, including RNG draw order.
        for task in sorted(score_ruler.TASKS):
            part = [r for r in records if r['length'] == length and r['task'] == task]
            assert len(part) == 20 and len({r['id'] for r in part}) == 20, (length, task, len(part))
            diffs = [r['e3']-r['stock'] for r in part]
            per[task] = dict(n=20, stock=statistics.mean(r['stock'] for r in part),
                             e3=statistics.mean(r['e3'] for r in part), delta=bootstrap([diffs]),
                             length_stops={a:sum(r.get(a+'_finish') == 'length' for r in part) for a in ('stock','e3')})
            if identity:
                per[task].update(identity=statistics.mean(r['identity'] for r in part),
                                 identity_vs_stock=bootstrap([[r['identity']-r['stock'] for r in part]]),
                                 identity_vs_e3=bootstrap([[r['identity']-r['e3'] for r in part]]))
                per[task]['length_stops']['identity']=sum(r.get('identity_finish')=='length' for r in part)
            all_diffs.append(diffs)
        result[str(length)] = dict(n=260, stock=statistics.mean(v['stock'] for v in per.values()),
                                  e3=statistics.mean(v['e3'] for v in per.values()), delta=bootstrap(all_diffs), per_task=per)
        if identity:
            result[str(length)].update(identity=statistics.mean(v['identity'] for v in per.values()),
                **{'identity_vs_'+arm:bootstrap([[r['identity']-r[arm] for r in records
                    if r['length']==length and r['task']==task] for task in per]) for arm in ('stock','e3')})
    return result


def cpu_gate():
    items = rows(DATA / 'ruler.jsonl')
    saved = {r['id']:r for r in rows(DATA / 'ruler-saved.jsonl')}
    expected = load(DATA / 'ruler-audit.json')
    records = []
    for item in items:
        assert item['request']['max_tokens'] == budget(item['task'])
        assert canonical(request(item['request'], 'stock')) == canonical(request(item['request'], 'e3')) == item['request']
        assert request(item['request'],'identity') == request(item['request'],'e3')
        v = saved[item['id']]
        records.append(dict(id=item['id'], task=item['task'], length=item['length'],
                            stock=score(item, v['stock']), e3=score(item, v['e3']),
                            stock_finish=v['stock_finish'], e3_finish=v['e3_finish']))
    result = summarize(records)
    for length, values in result.items():
        for task, v in values['per_task'].items():
            ref = expected[length][task]
            for arm in ('stock','e3'):
                assert abs(v[arm] - ref[arm]) < 2e-6, (length, task, arm, v[arm], ref[arm])
        target = {'16384':(84.27,83.47), '32768':(85.73,85.19)}[length]
        assert tuple(round(values[a],2) for a in ('stock','e3')) == target
        d = values['delta']
        print(f"{length}: stock {values['stock']:.2f} / E3 {values['e3']:.2f}; delta {d['mean']:+.2f} pp, 95% CI [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]")
        for task, v in values['per_task'].items():
            print(f"  {task:20s} {v['stock']:.6f} / {v['e3']:.6f}")
    print('CPU GATE PASS: all 520 canonical requests, 1040 saved responses, 26 audited per-task pairs')
    atomic(HERE / 'cpu-gate.json', result)


def run(args):
    if args.identity_only:
        from reproduction import require_reproduction
        require_reproduction(args.out)
    items = rows(DATA / 'ruler.jsonl')
    # Static disjoint shards: crash recovery has no stale claims and preserves identical GPU pairing.
    items = [x for i,x in enumerate(items) if i % args.workers == args.gpu]
    arms = ('identity',) if args.identity_only else (('stock','e3') if args.gpu < 2 else ('e3','stock'))
    for arm in arms:
        todo = [x for x in items if not (args.out / 'ruler' / arm / (x['id']+'.json')).exists()]
        if not todo:
            continue
        folder = args.out / 'ruler' / f'server-gpu{args.gpu}-{arm}-{time.time_ns()}'
        with server(args, arm, folder):
            measured, estimated = [], []
            for index, item in enumerate(todo):
                body = request(item['request'],arm)
                assert canonical(body) == item['request']
                t = time.monotonic()
                response = post(args.port, '/v1/chat/completions', body)
                elapsed = time.monotonic()-t
                choice = response['choices'][0]
                text = choice['message'].get('content') or ''
                record = dict(id=item['id'], length=item['length'], task=item['task'], arm=arm, gpu=args.gpu,
                              request=body, response=response, text=text, score=score(item,text), wall_s=elapsed,
                              finish_reason=choice['finish_reason'], provenance=str(folder.relative_to(args.out)))
                atomic(args.out / 'ruler' / arm / (item['id']+'.json'), record)
                measured.append(elapsed)
                estimated.append({16384:{'stock':14.4,'e3':12.3},32768:{'stock':28.6,'e3':23.8}}[item['length']]['e3' if arm=='identity' else arm])
                if index+1 == max(1, round(.1*len(todo))) and sum(measured)>1.5*sum(estimated):
                    atomic(args.out / 'RATE-STOP.json', dict(component='ruler', gpu=args.gpu, arm=arm,
                           measured_s=sum(measured), expected_s=sum(estimated), projected_factor=sum(measured)/sum(estimated)))
                    raise RuntimeError('RULER rate >1.5x estimate: stop and report revised cost before resuming')
                print(f"gpu{args.gpu} {arm} {item['id']} {record['score']:.2f} {elapsed:.1f}s", flush=True)


def main():
    p = parser(__doc__)
    p.add_argument('--cpu-gate', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.cpu_gate:
        cpu_gate()
    else:
        run(args)

if __name__ == '__main__':
    main()
