"""Appendix G: capture/fit elapsed times and raw-token calibration overlap maxima.

Read four small timing excerpts and the redacted token-overlap measurements under
--root. Sum nonoverlapping active-stage brackets for Flash-Next, not the idle gaps
between them. Emit minutes for the reference models and hours for Flash-Next, to
six decimals; the paper rounds those four quantities to whole units.

Overlap maxima are aggregated from every released per-calibration-document raw
measurement across all four tokenizers and both evaluation sets. Include the
Flash-Next terminal-correction prompts among fitted documents; exclude every
lambda-selection document. The 80-token held-out witness is checked against the
raw heldc0 records. The historical scanner measures contiguous shared-shingle
positions, an upper bound on a common string with any single evaluation document.
Re-running the original text scan requires withheld texts; this script uses only
the published measurements. Diagnostics go to stderr; stdout contains only claims.
"""
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, num, parse, say             # noqa: E402

args = parse(__doc__)


def describe(*parts):
    with redirect_stdout(sys.stderr):
        say(*parts)


def instant(endpoint):
    timestamp = endpoint["timestamp"]
    value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    # Check the derived full timestamp against the preserved machine log clock.
    if timestamp not in endpoint["line"] and not endpoint["line"].startswith(value.strftime("%H:%M:%S ")):
        raise SystemExit("timestamp differs from its preserved machine event")
    return value


def elapsed(folder, model, expected_stages):
    data = load(args.root / "provenance" / "fits" / folder / "timing" / (model + ".json"))
    stages = data["stages"]
    if data["model"] != model or [stage["name"] for stage in stages] != expected_stages:
        raise SystemExit(f"{model}: missing or reordered timing stages")
    seconds = 0
    previous_end = None
    for stage in stages:
        start, end = instant(stage["start"]), instant(stage["end"])
        if end <= start or (previous_end is not None and start < previous_end):
            raise SystemExit(f"{model}: nonpositive or overlapping timing bracket")
        duration = (end - start).total_seconds()
        seconds += duration
        previous_end = end
        describe(f"{model} {stage['name']}: {duration:.0f} seconds ({duration / 60:.6f} min)")
    if "fit_record" in data:
        written = datetime.fromisoformat(data["fit_record"]["time_utc"].replace("Z", "+00:00"))
        lag = (previous_end - written).total_seconds()
        if not 0 <= lag <= 2:
            raise SystemExit(f"{model}: final map event does not agree with S8 fit metadata")
    return seconds


reference = (
    ("llama-3.1-8b-instruct", "capture_and_fit", "X.G.minutes.llama31_8b"),
    ("qwen3.5-9b", "download_capture_and_fit", "X.G.minutes.qwen35_9b"),
    ("qwen3-8b", "capture_and_fit", "X.G.minutes.qwen3_8b"),
)
for model, stage, key in reference:
    claim(key, num(elapsed("reference", model, [stage]) / 60, 6))

seconds = elapsed("flashnext", "qwen3.8-flash-next", [
    "initial_capture", "batch0_accumulate", "batch1_capture_and_accumulate", "split24_fit_and_report",
])
describe(f"Flash-Next active stages: {seconds / 60:.6f} min; {seconds / 3600:.6f} elapsed hours on two GPUs")
claim("X.G.hours.flashnext", num(seconds / 3600, 6))

report = load(args.root / "provenance" / "checks" / "calib-token-overlap.json")
plan = {
    "llama-3.1-8b-instruct": ("ref_fit", "ref_held"),
    "qwen3.5-9b": ("ref_fit", "ref_held"),
    "qwen3-8b": ("ref_fit", "ref_held"),
    "qwen3.8-flash-next": ("fn_fit", "fn_held", "fn_sterm"),
}
evaluations = ("locked126", "fresh30")
expected_scans = {f"{model}/{cohort}/raw/{evaluation}"
                  for model, cohorts in plan.items() for cohort in cohorts for evaluation in evaluations}
if set(report["shingles"]) != expected_scans or set(report["tokenizers"]) != set(plan):
    raise SystemExit("missing, unexpected or non-raw overlap scan/tokenizer")
if report["evaluation_documents"] != {"locked126": 126, "fresh30": 30}:
    raise SystemExit("overlap report covers different evaluation sets")

fit_values, held_values = [], []
for key in sorted(expected_scans):
    model, cohort, variant, evaluation = key.split("/")
    measurements = report["shingles"][key]["per_calib_doc"]
    if len(measurements) > report["sha256"][cohort]["docs"]:
        raise SystemExit(f"{key}: more measured documents than cohort documents")
    values = [row["longest_run_tokens"] for row in measurements.values()]
    if any(type(value) is not int or value < 8 for value in values):
        raise SystemExit(f"{key}: invalid shared-shingle run measurement")
    maximum = max(values, default=0)
    (held_values if cohort in ("ref_held", "fn_held") else fit_values).append(maximum)
    describe(f"{key}: maximum {maximum} tokens")

fit_max = max(fit_values)
held_max = max(held_values)
if fit_max >= 50:
    raise SystemExit("a fit document has a raw shared-shingle run of at least 50 tokens")

witnesses = []
for row in report["long_runs"]:
    key = f"{row['model']}/{row['set']}/{row['variant']}/{row['eval_set']}"
    measurement = report["shingles"][key]["per_calib_doc"][row["calib_doc"]]
    if row["run_tokens"] != measurement["longest_run_tokens"]:
        raise SystemExit("long-run witness differs from its per-document measurement")
    if (row["variant"] == "raw" and row["set"] in ("ref_held", "fn_held")
            and row["eval_set"] == "fresh30" and row["calib_doc"].split("/")[-1] == "heldc0"
            and "source_code_128000_0.txt" in row["eval_docs"]):
        witnesses.append(row["run_tokens"])
if not witnesses or max(witnesses) != held_max:
    raise SystemExit("missing heldc0/fresh source-code witness for the held-out maximum")

claim("X.G.overlap.fit_raw_max_tokens", fit_max)
claim("X.G.overlap.held_header_raw_tokens", max(witnesses))
