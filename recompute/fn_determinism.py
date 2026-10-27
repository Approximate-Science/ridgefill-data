"""S6: Qwen3.8-Flash-Next determinism. Exact's two perplexity passes on its server start (ppl-a, ppl-b): documents
whose stored log-probs are bitwise identical. Quality mode's start with the fill switched off (graded/control):
documents whose log-probs equal exact's ppl-a bitwise."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, parse, say           # noqa: E402

args = parse(__doc__)
F = args.root / "results" / "flashnext" / "final-det-20261010" / "results"


def logprobs(folder):
    return {f.name: load(f)["logprobs"] for f in sorted(need(folder).glob("lp-*.json"))}


a = logprobs(F / "exact" / "ppl-a")
for key, other in (("exact_ab", F / "exact" / "ppl-b"), ("control", F / "graded" / "control")):
    b = logprobs(other)
    common = sorted(set(a) & set(b))
    equal = sum(a[k] == b[k] for k in common)
    say(f"{key}: {equal} of {len(common)} documents bitwise identical")
    claim(f"S6.{key}.identical", equal)
    claim(f"S6.{key}.docs", len(common))
