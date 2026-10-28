"""Paper section 2.3, Figure 8 (section 3), and Table 5 (section 3.2): Llama S12 key-rescaled novel-token loss
above exact, realized continuation-token ranges, total exact-defined key tokens, and key-bearing document counts.

The loss is the document-weighted mean of paired kva+ktail[3/8] minus exact[-] novel_nll from the superseding
nll-causal cell, via the same Tree.mean routing as finals/tables.py. No interval is requested. Population counts
use original nll cells, exclude declared skipped records, and use the exact arm's n_key mask. Continuation counts
come from the raw work/results longppl records' cont_len, checked against the exact logprobs array length and the
converted population's document ids. No pooling of fresh sets and no token weighting of loss.
Claim lines go to stdout; human-readable evidence goes to stderr.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, parse, say, sgn       # noqa: E402
from reftree import MODELS, Tree                           # noqa: E402

args = parse(__doc__)
tree = Tree(args.root, [model + "-nll" for model in MODELS.values()] + [MODELS["llama"] + "-nll-causal"])
loss = tree.mean("battery", MODELS["llama"], "nll_novel", "kva+ktail", "3/8", vs_exact=True)
with redirect_stdout(sys.stderr):
    say(f"Llama S12 causal key rescale minus exact: {loss:+.12g} nats, document-weighted")
claim("M.llama.s12.key_rescale.novel_nll_vs_exact", sgn(loss, 3))

for short, model in MODELS.items():
    records = tree.records("battery", model, "longppl")
    counts = [r["arms"]["exact"]["-"]["n_key"] for r in records.values()]
    raw_dir = need(args.root / "results" / "reference" / (model + "-nll") / "work" / "results" /
                   ("longppl-" + model + "-nll"))
    lengths = {}
    for path in sorted(raw_dir.glob("lp-*.json")):
        raw = load(path)
        if "skipped" in raw:
            continue
        length = raw["cont_len"]
        if length != len(raw["runs"]["exact"]["logprobs"]):
            raise SystemExit(f"{path}: cont_len differs from exact logprobs length")
        doc = raw["name"].removeprefix("lp-")
        if doc in lengths:
            raise SystemExit(f"{path}: duplicate continuation document {doc}")
        lengths[doc] = length
    if not lengths or set(lengths) != {r["doc"] for r in records.values()}:
        raise SystemExit(f"{model}: raw continuation documents do not match the non-skipped sample population")
    low, high = min(lengths.values()), max(lengths.values())
    with redirect_stdout(sys.stderr):
        say(f"{model}: {len(records)} scored documents, continuation tokens {low} to {high}, "
            f"{sum(counts)} exact-defined key tokens in {sum(n > 0 for n in counts)} documents")
    if short == "qwen35":
        if low != high:
            raise SystemExit(f"{model}: continuation length is not constant ({low} to {high})")
        claim(f"M.{short}.continuation_tokens", low)
    else:
        claim(f"M.{short}.continuation_tokens.min", low)
        claim(f"M.{short}.continuation_tokens.max", high)
    claim(f"M.{short}.key_tokens.total", sum(counts))
    if short in ("qwen35", "qwen3"):
        claim(f"M.{short}.key_documents", sum(n > 0 for n in counts))
