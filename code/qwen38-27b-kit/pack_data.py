"""CPU-only reproducible data pack. Local input paths are explicit CLI arguments."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from common import DATA, HERE, atomic, canonical, load, rows, sha
from ruler_replay import budget

PHASES = ('cold','warmup','timed-0','timed-1','timed-2')


def document(item):
    text = item['messages'][0]['content']
    prefix = 'Please read the following text and answer the questions below.\n\n<text>\n'
    assert text.startswith(prefix), item['id']
    # rsplit preserves any literal </text> embedded in a document.
    context, suffix = text[len(prefix):].rsplit('\n</text>',1)
    assert item['question'].strip() in suffix, item['id']
    return context


def select_windows(items, tokenize, length, count=100):
    selected = []
    for item in sorted(items,key=lambda x:x['id']):
        ids = tokenize(document(item))
        if len(ids) >= length+512:
            selected.append(dict(name=item['id'], ids=ids[:length+512], source_tokens=len(ids)))
            if len(selected) == count:
                return selected
    raise ValueError(f'Only {len(selected)} documents at {length}, need {count}')


def prompts(ids, seed=0):
    # Exact TIMING-WARM rule. At 120000 the final prefix block has 192 tokens;
    # shuffle it with all the full 512-token blocks, leaving the tail fixed.
    prefix, tail = ids[:-4096], ids[-4096:]
    blocks = [prefix[i:i+512] for i in range(0,len(prefix),512)]
    rng = random.Random(seed)
    result = [dict(phase='cold', ids=ids)]
    for phase in PHASES[1:]:
        for _ in range(1000):
            order = list(range(len(blocks)))
            rng.shuffle(order)
            tokens = [t for i in order for t in blocks[i]] + tail
            if all(tokens[0] != x['ids'][0] for x in result):
                break
        else:
            raise ValueError('Cannot obtain distinct leading tokens')
        result.append(dict(phase=phase,ids=tokens,prefix_block_order=order))
    return result


def write_rows(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for value in values:
            f.write(json.dumps(value,ensure_ascii=False)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate',type=Path,required=True)
    p.add_argument('--items',type=Path,required=True)
    p.add_argument('--audit-rows',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    args = p.parse_args()
    fg = args.validate/'results/final-e3/533120111f'
    rulers, saved, audit = [], [], {'16384':{},'32768':{}}
    for path in sorted((fg/'runs').glob('ruler-*/spec.json')):
        spec = load(path)
        length, task = int(spec['length']), spec['task']
        if length not in (16384,32768) or path.parent.name != f'ruler-{length}-{task}':
            continue
        for rec in spec['records']:
            item = rec['item']
            folder = path.parent/item['id']
            req = canonical(load(folder/'candidate-request.json'))
            assert req['max_tokens'] == budget(task)
            sreq = load(rec['stock']['request']) if rec['stock'] else load(folder/'stock-request.json')
            assert req == canonical(sreq), item['id']
            st = load(rec['stock']['source']) if rec['stock'] else load(folder/'stock.json')
            ca = load(folder/'candidate.json')
            finish = st.get('finish_reason')
            if rec['stock']:
                finish = load(rec['stock']['response'])['choices'][0]['finish_reason']
            rulers.append(dict(id=item['id'],task=task,length=length,length_bucket=item['length_bucket'],
                               answers=item['answers'],request=req))
            saved.append(dict(id=item['id'],stock=st['text'],e3=ca['text'],stock_finish=finish,e3_finish=ca['finish_reason']))
    assert len(rulers)==520 and len({x['id'] for x in rulers})==520
    for row in load(args.audit_rows):
        if row['run'] == f"ruler-{row['length']}-{row['task']}" and str(row['length']) in audit:
            audit[str(row['length'])][row['task']] = {k:row[k] for k in ('stock','e3')}
    write_rows(DATA/'ruler.jsonl',rulers)
    write_rows(DATA/'ruler-saved.jsonl',saved)
    atomic(DATA/'ruler-audit.json',audit)
    cache_dir = DATA/'token-cache'
    cache_dir.mkdir(exist_ok=True)
    def tokenize(text):
        h=hashlib.sha256(text.encode()).hexdigest()
        cache=cache_dir/(h+'.json')
        if not cache.exists():
            result=subprocess.run([str(args.tokenizer),'-m',str(args.model),'--stdin','--ids','--no-bos','--no-escape'],
                                  input=text,text=True,capture_output=True,check=True)
            atomic(cache,json.loads(result.stdout))
        return load(cache)
    items=rows(args.items)
    cache={}
    def get_tokens(text):
        h=hashlib.sha256(text.encode()).hexdigest()
        if h not in cache:
            cache[h]=tokenize(text)
        return cache[h]
    index=[]
    for length in (16384,32768):
        for d in select_windows(items,get_tokens,length):
            name=f"kld/{length}/{d['name']}.json"
            atomic(DATA/name,dict(name=d['name'],ids=d['ids']))
            index.append(dict(id=d['name'],length=length,file=name,source_tokens=d['source_tokens'],n_tokens=length+512))
        print('packed KLD',length,flush=True)
    atomic(DATA/'kld-index.json',index)
    for length in (16384,32768):
        values=[]
        for phase in PHASES:
            req=load(fg/f'headline-final/card03/{length}/stock/{phase}-request.json')
            values.append(dict(phase=phase,ids=req['prompt']))
        assert [x['ids'] for x in values] == [x['ids'] for x in prompts(values[0]['ids'])]
        atomic(DATA/f'timing/{length}.json',values)
    for length in (65536,120000):
        sources=[]
        combined=[]
        for item in sorted(items,key=lambda x:x['id']):
            ids=get_tokens(document(item))
            combined.extend(ids)
            sources.append(dict(id=item['id'],tokens=len(ids),context_sha256=hashlib.sha256(document(item).encode()).hexdigest()))
            if len(combined)>=length+512:
                atomic(DATA/f'timing/{length}.json',prompts(combined[:length]))
                atomic(DATA/f'timing/{length}-source.json',dict(sources=sources,concatenation='document token arrays in ascending item id; no separators'))
                if length==65536:
                    atomic(DATA/'checks-65k.json',dict(name='check-b-65536',ids=combined[:length+512]))
                break
        else:
            raise ValueError(f'Insufficient document tokens for {length}')
    probes=[dict(name='special-token-control',text='hello <|im_start|>user\nworld<|im_end|>')]
    probes += [dict(name=x['id'],text=document(x)[:16384]) for x in sorted(items,key=lambda x:x['id'])[:3]]
    for probe in probes:
        probe['ids']=get_tokens(probe['text'])
    write_rows(DATA/'tokenizer-probes.jsonl',probes)
    for path in sorted((fg/'mix-final-80331d2380').glob('*/spec.json')):
        spec=load(path)
        atomic(DATA/f'mix/{path.parent.name}.json',{k:spec[k] for k in ('length','kv_unified','schedules','runs')})
    # Freeze two actual LongBench prompts, including the benchmark question/template.
    checks=[]
    for name,target in [('lb32',32768),('lb120',120000)]:
        item=min(items,key=lambda x:(abs(x['n_prompt_tokens']-target),x['id']))
        checks.append(dict(name=name,id=item['id'],ids=get_tokens(item['messages'][0]['content'])))
    atomic(DATA/'checks-longbench.json',checks)
    lb=[]
    for item in items:
        pair={a:load(HERE.parent/f'cloud-longbench/results-20261025/full/{a}/{item["id"]}.json') for a in ('stock','e3')}
        lb.append(dict(id=item['id'],domain=item['domain'],difficulty=item['difficulty'],length=item['length_bucket'],
                       arms={a:{k:r.get(k) for k in ('judge','pred','pass1_finish','pass2_finish','pass1_timings','n_prompt_tokens')} for a,r in pair.items()}))
    write_rows(DATA/'longbench-saved.jsonl',lb)
    atomic(DATA/'PROVENANCE.json',dict(longbench_items_sha256=sha(args.items),tokenizer_sha256=sha(args.tokenizer),
           model_sha256=sha(args.model),ruler_scorer_sha256=sha(HERE/'vendor/score_ruler.py'),
           document_rule='Verbatim text between the frozen <text> and final </text> delimiters; no question; no added BOS; parse_special=true (server default and llama-tokenize default).',
           timing_rule='16K/32K exact saved prompts; 65K/120K first L tokens from document arrays concatenated in ascending LongBench id without separators; seed-0 512-token prefix-block shuffles, last4096 fixed; partial final prefix block at120000.'))
    import shutil
    shutil.rmtree(cache_dir)
    sums=[f'{sha(path)}  {path.relative_to(DATA)}' for path in sorted(DATA.rglob('*')) if path.is_file() and path.name!='SHA256SUMS']
    (DATA/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    print('packed',len(rulers),'RULER items,',len(index),'KLD windows')

if __name__=='__main__':
    main()
