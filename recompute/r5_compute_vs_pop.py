"""R5 and VERIFY check 7: RidgeFill's prefill multiply-accumulates against POP's, from the project's cost model
(code/kva/compute.py: Dims, layer_exact, layer_fill, the projector term and kva(); ported here, dense attention models
only). RidgeFill = kva(dims, n, S, 4096) with the maps; POP = kva(dims, n, L - floor(L/3), 1, projector=False). Dims
are the hand-entered config.json fields in recompute/cfg/. Prints Qwen3-8B's S22/S24 ratios at 8K-64K, the 0.759
ratio of S13/S24 at 32,768 tokens, and the crossovers (the smallest prompt length at which RidgeFill costs less)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, num, parse, say            # noqa: E402


class Dims:
    def __init__(self, c):
        self.L, self.H = c["num_hidden_layers"], c["hidden_size"]
        self.I, self.V = c["intermediate_size"], c["vocab_size"]
        self.hq, self.hkv = c["num_attention_heads"], c["num_key_value_heads"]
        self.d = c.get("head_dim") or self.H // self.hq

    def layer_exact(self, first, last):
        n = last - first
        linear = self.H * self.hq * self.d + 2 * self.H * self.hkv * self.d + self.hq * self.d * self.H
        ctx = 2 * self.hq * self.d * n * (first + last + 1) / 2
        return n * (linear + 3 * self.H * self.I) + ctx

    def layer_fill(self, n):
        return n * 2 * self.H * self.hkv * self.d


def kva(dm, n, split, tail, projector=True):
    p = max(0, n - tail)
    c = split * dm.layer_exact(0, n)
    for _ in range(split, dm.L):
        c += dm.layer_fill(p) + dm.layer_exact(p, n) + (p * dm.H * dm.H if projector else 0)
    return c + dm.H * dm.V


def ratio(dm, n, split):
    pop = dm.L - dm.L // 3
    return kva(dm, n, split, 4096) / kva(dm, n, pop, 1, projector=False)


def crossover(dm, split, hi=1 << 20):
    """Smallest n > 4096 with ratio < 1 (the ratio falls with n)."""
    lo = 4097
    if ratio(dm, hi, split) >= 1:
        raise SystemExit(f"split {split}: RidgeFill never costs less than POP below {hi} tokens")
    while lo < hi:
        mid = (lo + hi) // 2
        if ratio(dm, mid, split) < 1:
            hi = mid
        else:
            lo = mid + 1
    return lo


args = parse(__doc__)
cfg = Path(__file__).resolve().parent / "cfg"
llama, qwen = Dims(load(cfg / "llama-3.1-8b.json")), Dims(load(cfg / "qwen3-8b.json"))
for n, key in ((8192, "8k"), (16384, "16k"), (32768, "32k"), (65536, "64k")):
    r = ratio(qwen, n, 22)
    say(f"Qwen3-8B RidgeFill S22 / POP S24 at {n}: {r:.4f}")
    claim(f"R5.qwen3.s22_over_pop.{key}", num(r, 3))
r = ratio(qwen, 32768, 13)
say(f"Qwen3-8B RidgeFill S13 / POP S24 at 32768: {r:.4f}")
claim("R5.qwen3.s13_over_pop.32k", num(r, 3))
for key, dm, split, nd in (("llama.s16", llama, 16, 0), ("llama.s20", llama, 20, 0), ("qwen3.s13", qwen, 13, 1)):
    n = crossover(dm, split)
    say(f"crossover {key} vs POP S{dm.L - dm.L // 3}: RidgeFill cheaper from {n} tokens ({n / 1000:.2f}K)")
    claim(f"R5.crossover_k.{key}", num(n / 1000, nd))
