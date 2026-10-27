"""Final default identity step: 503 LongBench items, frozen official CoT requests."""
import sys
import time
from common import DATA, HERE, atomic, parser, post, rows, server

sys.path.insert(0, str(HERE/'vendor'))
import longbench_run_lb as frozen


def main():
    p=parser(__doc__)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    assert args.identity_only, 'This runner measures only identity'
    from reproduction import require_reproduction
    require_reproduction(args.out)
    items=rows(DATA/'longbench-items.jsonl')
    assert len(items)==503 and len({x['id'] for x in items})==503
    # Same long-first order, with disjoint shards and no stale claims.
    items.sort(key=lambda x:(-x['n_prompt_tokens'],x['id']))
    todo=[x for i,x in enumerate(items) if i%args.workers==args.gpu and
          not (args.out/'longbench/identity'/(x['id']+'.json')).exists()]
    if not todo:
        return
    folder=args.out/'longbench'/f'identity-gpu{args.gpu}-{time.time_ns()}'
    frozen.post=lambda port,body,timeout:post(port,'/v1/chat/completions',body,timeout=timeout)
    with server(args,'identity',folder,ctx=121101):
        for item in todo:
            # The frozen E3 HTTP fields are also the identity fields. Only the
            # server's loaded projector differs. Keep both official passes.
            result=frozen.run_item(args,item,'e3')
            result.update(arm='identity',provenance=str(folder.relative_to(args.out)),max_len=120000)
            atomic(args.out/'longbench/identity'/(item['id']+'.json'),result)
            print('LongBench identity',args.gpu,item['id'],result['judge'],flush=True)


if __name__=='__main__':
    main()
