"""Summarise the cloud LongBench v2 run (official CoT protocol): accuracy by group, paired difference with a bootstrap
CI, exact sign test on discordant items, answer-extraction failures and prefill timing. Usage: report_lb.py RESULTS_DIR"""
import json
import math
import random
import sys
from pathlib import Path

out = Path(sys.argv[1])
arms = {arm: {p.stem: json.loads(p.read_text()) for p in (out / arm).glob('*.json')} for arm in ('stock', 'e3')}
ids = sorted(set(arms['stock']) & set(arms['e3']))
print(f'paired items {len(ids)} (stock {len(arms["stock"])}, e3 {len(arms["e3"])})\n')


def accuracy(rows):
    return 100 * sum(r['judge'] for r in rows) / len(rows) if rows else float('nan')


groups = [('overall', lambda r: True)]
groups += [(g, lambda r, g=g: r['difficulty'] == g) for g in ('easy', 'hard')]
groups += [(g, lambda r, g=g: r['length'] == g) for g in ('short', 'medium', 'long')]
print('| Group | n | Stock % | E3 % | Delta pp |\n|---|---:|---:|---:|---:|')
for name, keep in groups:
    sel = [i for i in ids if keep(arms['stock'][i])]
    s, e = accuracy([arms['stock'][i] for i in sel]), accuracy([arms['e3'][i] for i in sel])
    print(f'| {name} | {len(sel)} | {s:.1f} | {e:.1f} | {e - s:+.1f} |')

diff = [int(arms['e3'][i]['judge']) - int(arms['stock'][i]['judge']) for i in ids]
rng = random.Random(20261025)
boot = sorted(100 * sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(10000))
only_s, only_e = diff.count(-1), diff.count(1)
n = only_s + only_e
p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_s, only_e) + 1)) / 2 ** n) if n else 1.0
print(f'\npaired E3 - stock: {100 * sum(diff) / len(diff):+.2f} pp, 95% bootstrap CI '
      f'[{boot[250]:+.2f}, {boot[9750]:+.2f}] (10,000 resamples)')
print(f'only stock correct {only_s}, only E3 correct {only_e}, exact sign test p = {p:.3f}')

for arm in ('stock', 'e3'):
    rows = [arms[arm][i] for i in ids]
    print(f'{arm}: no extracted answer {sum(r["pred"] is None for r in rows)}, pass-1 length stops '
          f'{sum(r["pass1_finish"] == "length" for r in rows)}, pass-2 length stops '
          f'{sum(r["pass2_finish"] == "length" for r in rows)}, empty pass 1 {sum(r["empty_pass1"] for r in rows)}')

prefill = [(arms['stock'][i]['pass1_timings']['prompt_ms'], arms['e3'][i]['pass1_timings']['prompt_ms'],
            arms['stock'][i]['n_prompt_tokens']) for i in ids
           if arms['stock'][i].get('pass1_timings') and arms['e3'][i].get('pass1_timings')]
for name, low, high in (('all', 0, 10 ** 9), ('<32K', 0, 32768), ('32-100K', 32768, 100000), ('>=100K', 100000, 10 ** 9)):
    sel = [(s, e) for s, e, t in prefill if low <= t < high]
    if sel:
        print(f'prefill {name}: n {len(sel)}, stock {sum(s for s, _ in sel) / 1000:.0f} s total, '
              f'E3 {sum(e for _, e in sel) / 1000:.0f} s total, speedup {sum(s for s, _ in sel) / sum(e for _, e in sel):.3f}x')
