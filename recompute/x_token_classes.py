"""Paper section 2.4: token-class enrichment of absolute first-order damage.

Read the public aggregate only. Each model pools every indexed window after
dropping row 0. E_c is the class's share of |impact| divided by its token share.
The protected range spans four classes on two models; the number range spans
the two models. Lower-case word-start enrichments are diagnostic claims for
the qualitative sentence. Six decimals retain the paper's endpoint rounding.
"""
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, num, parse, say             # noqa: E402

MODELS = ("llama-3.1-8b-instruct", "qwen3.5-9b")
CLASSES = {"special", "newline", "space", "number", "punct", "word-start",
           "word-start-cap", "word-piece", "word-piece-cap", "mixed"}
PROTECTED = ("word-start-cap", "word-piece-cap", "word-piece", "mixed")

args = parse(__doc__)
data = load(args.root / "results" / "development" / "token-classes" / "impact-corpus-classes.json")
if data["schema"] != "impact-corpus-classes-v1" or set(data["models"]) != set(MODELS):
    raise SystemExit("unexpected token-class aggregate schema or model set")
if set(data["method"]["protected_classes"]) != set(PROTECTED):
    raise SystemExit("unexpected protected classes")

enrichments = {}
for model in MODELS:
    record = data["models"][model]
    classes, totals = record["classes"], record["totals"]
    if set(classes) != CLASSES:
        raise SystemExit(f"{model}: unexpected class set")
    if record["n_documents"] != 300 or record["n_windows"] != record["provenance"]["npz_count"]:
        raise SystemExit(f"{model}: inconsistent document/window provenance")
    if not isinstance(totals["N"], int) or totals["N"] <= 0:
        raise SystemExit(f"{model}: invalid total token count")
    if not math.isfinite(totals["sum_abs_impact"]) or totals["sum_abs_impact"] <= 0:
        raise SystemExit(f"{model}: invalid total absolute impact")
    for label, row in classes.items():
        if (not isinstance(row["N_c"], int) or row["N_c"] < 0 or
                not math.isfinite(row["sum_abs_impact"]) or row["sum_abs_impact"] < 0 or
                (row["N_c"] == 0 and row["sum_abs_impact"] != 0)):
            raise SystemExit(f"{model}/{label}: invalid class count or absolute impact")
    if sum(row["N_c"] for row in classes.values()) != totals["N"]:
        raise SystemExit(f"{model}: class counts do not sum to N")
    if not math.isclose(math.fsum(row["sum_abs_impact"] for row in classes.values()),
                        totals["sum_abs_impact"], rel_tol=1e-12, abs_tol=0.0):
        raise SystemExit(f"{model}: class absolute impacts do not sum to the total")
    enrichments[model] = {}
    for label in PROTECTED + ("number", "word-start"):
        row = classes[label]
        if row["N_c"] == 0:
            raise SystemExit(f"{model}/{label}: no tokens")
        enrichments[model][label] = ((row["sum_abs_impact"] / totals["sum_abs_impact"])
                                    / (row["N_c"] / totals["N"]))
    with redirect_stdout(sys.stderr):
        say(f"{model}: {record['n_documents']} documents, {record['n_windows']} windows, "
            f"{totals['N']} tokens after excluding row 0")
        for label, value in enrichments[model].items():
            say(f"  {label}: E={value:.12g}")

protected = [enrichments[m][c] for m in MODELS for c in PROTECTED]
numbers = [enrichments[m]["number"] for m in MODELS]
claim("X.token_classes.protected.min", num(min(protected), 6))
claim("X.token_classes.protected.max", num(max(protected), 6))
claim("X.token_classes.number.min", num(min(numbers), 6))
claim("X.token_classes.number.max", num(max(numbers), 6))
for model, tag in zip(MODELS, ("llama", "qwen")):
    claim(f"X.token_classes.word_start.{tag}", num(enrichments[model]["word-start"], 6))
