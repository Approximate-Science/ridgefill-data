"""PROPOSED substitute for the registered random-25%-rows control (not run unless the coordinator chooses it).

The engine has one exact-row rule (vllm/v1/ced.py XrowsSelector, class56): among the prompt rows whose token is in the
classes cap / mixed / piece, keep round(share x count) by rarity (-logfreq of the unigram table). A random-rows rule
is not expressible. The closest substitute that needs no engine change is a unigram table whose values are a seeded
permutation of the real table's values over the eligible ids: the engine then keeps the same number of rows of the
same classes, chosen by a random order of token identities instead of by rarity. It is count- and class-matched and
rarity-blind; it is NOT position-random (every occurrence of a token id is in or out together) and NOT drawn from all
rows (only class56 rows). It answers "does ranking by rarity matter at a fixed class budget", not "do these classes
matter". Ineligible ids keep 0.0 (never read).

Usage: python -m c5.randctl --freq <projector-tcc>/freq.safetensors --out <dir>/freq-shuffled-<seed>.safetensors
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load, save

from kva.integrity import read_direct

SEED = 20261008


def shuffled(logfreq, eligible, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    idx = eligible.nonzero().flatten()
    perm = idx[torch.randperm(idx.numel(), generator=g)]
    out = logfreq.clone()
    out[idx] = logfreq[perm]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", required=True)
    ap.add_argument("--src-score", required=True, help="the release's rowsel.safetensors (eligible = finite score)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    lf = load(read_direct(a.freq))["logfreq"]
    elig = torch.isfinite(load(read_direct(a.src_score))["score"])
    out = shuffled(lf, elig, a.seed)
    b = save({"logfreq": out}, metadata={"c5": json.dumps({"rule": "seeded permutation of logfreq over the class56-"
                                                           "eligible ids (c5/randctl.py)", "seed": a.seed,
                                                           "from": Path(a.freq).name}, sort_keys=True)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(b)
    if read_direct(a.out) != b:
        raise SystemExit("read-back differs")
    print(hashlib.sha256(b).hexdigest(), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
