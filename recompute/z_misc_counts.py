"""Z: Appendix H's source/log counts and Appendix I's development numerators.

Count theorem/lemma declarations in KvaLean/*.lean, including private helpers
and SoftmaxShift (comments and string literals removed). Parse the expert plan
for host-resident routed experts per rank. Reconstruct development fill correct
count from acc_projected times n_projected, count exact hexadecimal responses,
and check the two models' special-token occurrence counts agree.

Also report diagnostic per-request protected-row percentage endpoints, joining
quality-mode CED request ids to graded/ppl-a response_id and dividing by prefix
positions, excluding the exact tail. These cannot establish Appendix H's
per-chunk range; its two rows stay in still-needs-data.tsv.
"""
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, need, num, parse, say        # noqa: E402

args = parse(__doc__)


def lean_code(text):
    """Mask Lean nested block comments, line comments and strings, retaining newlines."""
    out, i, depth, string = [], 0, 0, False
    while i < len(text):
        c = text[i]
        pair = text[i:i + 2]
        if depth:
            if pair in ("/-", "-/"):
                depth += 1 if pair == "/-" else -1
                out.extend("  ")
                i += 2
                continue
        elif string:
            if c == "\\":
                out.extend("  ")
                i += 2
                continue
            if c == '"':
                string = False
        elif pair == "/-":
            depth = 1
            out.extend("  ")
            i += 2
            continue
        elif pair == "--":
            end = text.find("\n", i)
            end = len(text) if end < 0 else end
            out.extend(" " * (end - i))
            i = end
            continue
        elif c == '"':
            string = True
        else:
            out.append(c)
            i += 1
            continue
        out.append("\n" if c == "\n" else " ")
        i += 1
    if depth or string:
        raise ValueError("unterminated Lean comment or string")
    return "".join(out)


sources = sorted(need(args.root / "code" / "lean" / "kva_lean" / "KvaLean").glob("*.lean"))
if not sources:
    raise SystemExit("missing input: no KvaLean sources")
declarations = {f.name: len(re.findall(r"\b(?:theorem|lemma)\s+[\w'.]+", lean_code(f.read_text()))) for f in sources}
claim("Z.formal.theorem_lemma_declarations", sum(declarations.values()))

G = args.root / "results" / "flashnext" / "final-det-20261010" / "results" / "graded"
ansi = re.compile(r"\x1b\[[0-9;]*m")
boot = ansi.sub("", need(G / "boot" / "server-boot.log").read_text())
host, = {int(x) for x in re.findall(r"host rows \(per rank\).*?\b(\d+) experts \+", boot)}
claim("Z.flashnext.host_experts_per_gpu", f"{host:,}")

log = ansi.sub("", need(G / "boot" / "server.log").read_text())
requests = re.findall(r"CED request (\S+): (\d+) of (\d+) prompt tokens approximated from layer \d+, "
                      r"(\d+) prefix rows exact", log)
shares = []
for f in sorted(need(G / "ppl-a").glob("*.json")):
    if f.name.startswith("_"):
        continue
    r = load(f)
    matches = [x for x in requests if x[0] == r["response_id"] or x[0].startswith(r["response_id"] + "-")]
    if len(matches) != 1:
        raise ValueError(f"{f.name}: expected one CED request, got {len(matches)}")
    _, prefix, total, exact = matches[0]
    if int(total) != r["n"] or int(prefix) != r["expected_tail_start"]:
        raise ValueError(f"{f.name}: CED log and response disagree on prefix/total length")
    shares.append(100 * int(exact) / int(prefix))
if not shares:
    raise SystemExit("missing input: no quality-mode perplexity requests")
claim("Z.flashnext.protected_rows.per_prefix.min_pct", num(min(shares), 0))
claim("Z.flashnext.protected_rows.per_prefix.max_pct", num(max(shares), 0) + " %")

D = args.root / "results" / "development"
k28 = load(D / "needles-dev" / "k28_niah.json")
n = k28["n_projected"]
correct = k28["arms"]["base"]["acc_projected"] * n
if abs(correct - round(correct)) > 1e-8:
    raise ValueError("non-integral corrected-format fill numerator")
claim("Z.development.corrected_format.fill_correct", round(correct))
claim("Z.development.corrected_format.filled_needles", n)
# exact_projected concerns only n_projected; it is insufficient for the paper's
# exact-all correct count. Do not multiply it by k28['n'].
hex_files = sorted(need(D / "needles-reference" / "niahg-bypass-llama").glob("ng-hexid-*.json"))
if not hex_files:
    raise SystemExit("missing input: no hexadecimal needle responses")
claim("Z.development.llama.hexid.exact_correct", sum(load(f)["runs"]["exact"]["correct"] for f in hex_files))
special = {m: load(D / "token-classes" / f"{m}-s12.json")["by_class"]["special"]["n"]
           for m in ("llama", "qwen")}
if len(set(special.values())) != 1:
    raise ValueError(f"models disagree on special-token occurrence count: {special}")
claim("Z.development.special_token_occurrences", next(iter(special.values())))
with redirect_stdout(sys.stderr):
    say(f"Lean: {sum(declarations.values())} declarations over {len(sources)} files, {declarations}")
    say(f"Host experts per rank/GPU: {host}; quality-mode perplexity requests joined: {len(shares)}")
    say(f"Protected rows: {min(shares):.9f}% to {max(shares):.9f}% per prefix; per-chunk counts are absent")
    say(f"Corrected-format fill: {round(correct)}/{n}; exact hex responses: {len(hex_files)}; "
        f"special-token occurrences: {special}")
