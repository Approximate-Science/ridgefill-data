"""T: fresh-document key-token results in paper section 6.7.

Use only the nll-fresh cell, in its own dataset scope. Sum exact-arm n_key for the
token count and count documents with n_key > 0. At half and 5/8 depth, pair RidgeFill
with identity on those documents and average document key_nll differences equally.
Tree.contrast reproduces the finals' domain-stratified BCa (5,000 resamples, seed
20261004, 95%); the paper prints only the point estimates for these key-token claims.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, parse, say, sgn                  # noqa: E402
from reftree import Tree                                   # noqa: E402

args = parse(__doc__)
model = "llama-3.1-8b-instruct"
cell = model + "-nll-fresh"
t = Tree(args.root, [cell])
scope = "dataset:" + t.cells[cell]["dataset"]
records = t.records(scope, model, "longppl")
counts = [r["arms"]["exact"]["-"]["n_key"] for r in records.values()]
claim("T.llama.fresh.key.tokens", sum(counts))
claim("T.llama.fresh.key.documents", sum(n > 0 for n in counts))
with redirect_stdout(sys.stderr):
    say("Fresh exact key mask:", sum(counts), "tokens in", sum(n > 0 for n in counts), "documents")

for frac, layer in (("1/2", 16), ("5/8", 20)):
    c = t.contrast(scope, model, "nll_key", ("kva", frac), ("noproj", frac))
    claim(f"T.llama.fresh.s{layer}.key.ridgefill_vs_identity.delta", sgn(c["est"], 3))
    with redirect_stdout(sys.stderr):
        say("Fresh key-token RidgeFill - identity at", frac, ":", c["est"], c["lo"], c["hi"], "n=", c["n"])
