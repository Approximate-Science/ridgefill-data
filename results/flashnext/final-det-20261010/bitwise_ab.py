"""Bitwise check of two passes of one arm: results/<arm>/ppl-<a>/lp-*.json vs ppl-<b>. Numbers only, no text.
Usage: bitwise_ab.py <results dir> <arm> <pass a> <pass b>"""
import glob, json, os, sys

R, arm, pa, pb = sys.argv[1:5]
A = {os.path.basename(f): f for f in glob.glob(f"{R}/{arm}/ppl-{pa}/lp-*.json")}
B = {os.path.basename(f): f for f in glob.glob(f"{R}/{arm}/ppl-{pb}/lp-*.json")}
common = sorted(set(A) & set(B))
equal, worst = 0, (0.0, None)
for k in common:
    a, b = json.load(open(A[k]))["logprobs"], json.load(open(B[k]))["logprobs"]
    if a == b:
        equal += 1
        continue
    d = max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else float("inf")
    worst = max(worst, (d, k))
print(f"{arm} ppl-{pa} vs ppl-{pb}: {len(common)} docs in both, {equal} bitwise equal; worst max|d| {worst[0]:.3g} {worst[1] or ''}")
