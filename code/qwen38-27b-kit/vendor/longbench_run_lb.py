"""LongBench v2, official CoT protocol (THUDM/LongBench pred.py rev 2e00731f), one worker per GPU.

Pass 1: items.jsonl messages (official 0shot_cot prompt, official middle cut at max_len 120000 on the HF tokenizer,
prepared locally), temperature 0.1, max_tokens 1024. Pass 2: official 0shot_cot_ans template with the pass-1 output
inserted, temperature 0.1, max_tokens 128. Score: official extract_answer on the pass-2 output. Request bodies match
the local lane-Q harness (same seed per item for both arms, thinking off, cache_prompt off).

Each worker claims items for its arm with O_EXCL claim files, writes one JSON per item-arm atomically, and switches to
the other arm when its own queue is empty. A rerun skips finished item-arms.
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from official_extract import extract_answer

HERE = Path(__file__).resolve().parent
TEMPLATE_ANS = (HERE / '0shot_cot_ans.txt').read_text(encoding='utf-8')
COMMON = ['-ngl', '99', '-t', '4', '-b', '2048', '-ub', '512', '-ctk', 'f16', '-ctv', 'f16', '-fa', 'on',
          '-c', '121101', '-np', '1', '--jinja', '--cache-ram', '0', '--host', '127.0.0.1']
E3_FLAGS = ['--ridgefill-mode', 'quality+qattn', '--ridgefill-tail', '4096', '--ridgefill-min-prompt', '8192',
            '--ridgefill-share', '0.5', '--ridgefill-digits', 'on', '--ridgefill-special-exact', 'on',
            '--ridgefill-alpha', '0', '--ridgefill-qattn-share', '0.2', '--ridgefill-qattn-window', '64',
            '--ridgefill-header-rows', '0', '--ridgefill-qattn-window-mode', 'last',
            '--ridgefill-qattn-turn-cap', '256', '--ridgefill-s1-window', '256', '--ridgefill-s1-factor', '1.0',
            '--ridgefill-vtail', 'on', '--ridgefill-vtail-rows', '512', '--ridgefill-lexical', 'on',
            '--ridgefill-lexical-window', '64', '--ridgefill-lexical-min-words', '4',
            '--ridgefill-lexical-span-rows', '80', '--ridgefill-lexical-max-occurrences', '2',
            '--ridgefill-stream-projector', 'off', '--ridgefill-trim-kv', 'on']
E3_FIELDS = dict(ridgefill=True, ridgefill_mode='quality+qattn', ridgefill_tail=4096, ridgefill_min_prompt=8192,
                 ridgefill_share=0.5, ridgefill_digits=True, ridgefill_special_exact=True, ridgefill_alpha=0,
                 ridgefill_qattn_share=0.2, ridgefill_qattn_window=64, ridgefill_header_rows=0,
                 ridgefill_qattn_window_mode='last', ridgefill_qattn_turn_cap=256, ridgefill_s1_window=256,
                 ridgefill_s1_factor=1.0, ridgefill_vtail=True, ridgefill_vtail_rows=512, ridgefill_lexical=True,
                 ridgefill_lexical_window=64, ridgefill_lexical_min_words=4, ridgefill_lexical_span_rows=80,
                 ridgefill_lexical_max_occurrences=2)


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def post(port, body, timeout):
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def start_server(args, arm, log):
    binary = args.stock_bin if arm == 'stock' else args.e3_bin
    cmd = [binary, '-m', args.model, *COMMON, '--port', str(args.port)]
    if arm == 'e3':
        cmd += ['--ridgefill', args.projector, *E3_FLAGS]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f'{arm} server exited rc={proc.returncode} during startup')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as r:
                if json.loads(r.read()).get('status') == 'ok':
                    return proc, cmd
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f'{arm} server not ready after 600 s')


def choice(item, letter):
    choices = item['choices']
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    return choices[letter]


def body(item, arm, messages, max_tokens):
    value = dict(messages=messages, temperature=0.1, max_tokens=max_tokens, seed=item['seed'], cache_prompt=False,
                 chat_template_kwargs=dict(enable_thinking=False))
    if arm == 'e3':
        value.update(E3_FIELDS)
    return value


def run_item(args, item, arm):
    start = time.monotonic()
    r1 = post(args.port, body(item, arm, item['messages'], 1024), timeout=3600)
    cot = (r1['choices'][0]['message'].get('content') or '').strip()
    prompt2 = (TEMPLATE_ANS.replace('$DOC$', '').replace('$Q$', item['question'].strip())
               .replace('$C_A$', choice(item, 'A').strip()).replace('$C_B$', choice(item, 'B').strip())
               .replace('$C_C$', choice(item, 'C').strip()).replace('$C_D$', choice(item, 'D').strip())
               .replace('$COT$', cot))
    r2 = post(args.port, body(item, arm, [dict(role='user', content=prompt2)], 128), timeout=600)
    answer = (r2['choices'][0]['message'].get('content') or '').strip()
    pred = extract_answer(answer)
    return dict(id=item['id'], arm=arm, gpu=args.gpu, pass1=cot, pass1_finish=r1['choices'][0]['finish_reason'],
                pass1_usage=r1.get('usage'), pass1_timings=r1.get('timings'), pass1_ridgefill=r1.get('ridgefill'),
                pass2=answer, pass2_finish=r2['choices'][0]['finish_reason'], pred=pred, gold=item['answers'][0],
                judge=pred == item['answers'][0], empty_pass1=cot == '', difficulty=item['difficulty'],
                length=item['length_bucket'], n_prompt_tokens=item['n_prompt_tokens'],
                wall_s=time.monotonic() - start, saved_at=time.time())


def claim(out, arm, item_id):
    path = out / 'claims' / arm / item_id
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--arm', choices=['stock', 'e3'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--items', type=Path, default=HERE / 'items.jsonl')
    p.add_argument('--model', required=True)
    p.add_argument('--projector', required=True)
    p.add_argument('--stock-bin', required=True)
    p.add_argument('--e3-bin', required=True)
    p.add_argument('--limit', type=int, default=0, help='only the first N items (pilot)')
    p.add_argument('--switch', action='store_true', help='take the other arm when this arm is done')
    args = p.parse_args()
    items = [json.loads(line) for line in args.items.open()]
    items.sort(key=lambda x: (-x['n_prompt_tokens'], x['id']))  # long first, so the tail is short items
    if args.limit:
        items = sorted(items, key=lambda x: x['id'])[:args.limit]
    arms = [args.arm] + ([a for a in ('stock', 'e3') if a != args.arm] if args.switch else [])
    for arm in arms:
        (args.out / arm).mkdir(parents=True, exist_ok=True)
        (args.out / 'claims' / arm).mkdir(parents=True, exist_ok=True)
        todo = [x for x in items if not (args.out / arm / (x['id'] + '.json')).exists()]
        if not todo:
            continue
        log = open(args.out / f'server-gpu{args.gpu}-{arm}.log', 'ab')
        proc, cmd = start_server(args, arm, log)
        atomic(args.out / f'server-gpu{args.gpu}-{arm}.json', dict(cmd=cmd, started=time.time()))
        try:
            for item in todo:
                if (args.out / arm / (item['id'] + '.json')).exists() or not claim(args.out, arm, item['id']):
                    continue
                try:
                    result = run_item(args, item, arm)
                except Exception as error:  # record and release the claim; a rerun retries it
                    (args.out / 'errors').mkdir(exist_ok=True)
                    atomic(args.out / 'errors' / f"{arm}-{item['id']}-{int(time.time())}.json",
                           dict(id=item['id'], arm=arm, gpu=args.gpu, error=repr(error)))
                    (args.out / 'claims' / arm / item['id']).unlink(missing_ok=True)
                    if proc.poll() is not None:
                        raise RuntimeError(f'{arm} server died rc={proc.returncode}') from error
                    continue
                atomic(args.out / arm / (item['id'] + '.json'), result)
                print(f"gpu{args.gpu} {arm} {item['id']} judge={result['judge']} {result['wall_s']:.0f}s", flush=True)
        finally:
            proc.terminate()
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
            atomic(args.out / f'server-gpu{args.gpu}-{arm}-exit.json', dict(rc=rc, time=time.time()))
            log.close()


if __name__ == '__main__':
    sys.exit(main())
