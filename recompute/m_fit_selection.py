"""Paper section 2.2: the largest runner-up penalty's held-out cache-read cosine gap across reference fits.

For each released reference fit, exclude the identity control, rank numeric penalty candidates by mean_read_cos,
subtract the runner-up score from the winning score, and take the maximum over models and splits. These are the
fit_sweep.py held-document and layer means already recorded by the capture/fit run; no new weighting or bootstrap.
Claim lines go to stdout; human-readable evidence goes to stderr so the recompute gate can read stdout directly.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say       # noqa: E402
from reftree import MODELS                                 # noqa: E402

args = parse(__doc__)
fits = need(args.root / "provenance" / "fits" / "reference")
gaps = []
for model in MODELS.values():
    files = sorted(fits.glob(f"fit-sweep-{model}-s*.json"))
    if not files:
        raise SystemExit(f"missing input: no fit-sweep records for {model} under {fits}")
    for path in files:
        record = load(path)
        candidates = sorted((float(score), float(penalty))
                            for penalty, score in record["mean_read_cos"].items() if penalty != "identity")
        if len(candidates) < 2 or candidates[-1][1] != record["lambda_star"]:
            raise SystemExit(f"{path}: fewer than two penalties or lambda_star is not the best score")
        best, runner_up = candidates[-1], candidates[-2]
        gap = best[0] - runner_up[0]
        gaps.append((gap, path.name))
        with redirect_stdout(sys.stderr):
            say(f"{path.name}: best lambda {best[1]:g}, runner-up {runner_up[1]:g}, gap {gap:.12g}")
maximum, filename = max(gaps)
with redirect_stdout(sys.stderr):
    say(f"Maximum runner-up gap {maximum:.12g} from {filename}, across {len(gaps)} reference fits")
claim("M.reference.fit.runner_up_cosine_gap.max", num(maximum, 4))
