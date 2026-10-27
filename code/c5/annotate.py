"""The real tail of every request (protocol: "the real tail is recorded per request"), from the engine's own log line
(vllm/v1/ced.py CedPolicy.admit):
    CED request <request id>: <tail_start> of <N> prompt tokens approximated from layer <S>[, <k> prefix rows exact]
matched to the arm's sample files by their response id (the engine's request id starts with it). Writes
<results>/<arm>/tails.json: {sample: {tail_start, n, tail, xrows, expected_tail_start, match}} plus a summary; a sample
of a RidgeFill request with no log line, or a tail that differs from the planned one, is listed under "problems".
Sample files are not modified (their .sha256 sidecars stay valid).
"""
import argparse
import json
import re
from pathlib import Path

from c5.client import write_json

LINE = re.compile(r"CED request (?P<rid>\S+): (?P<ts>\d+) of (?P<n>\d+) prompt tokens approximated from layer "
                   r"(?P<split>\d+)(?:, (?P<x>\d+) prefix rows exact)?")


def parse(log_text):
    return [dict(rid=m["rid"], tail_start=int(m["ts"]), n=int(m["n"]), split=int(m["split"]),
                 xrows=int(m["x"]) if m["x"] else 0) for m in LINE.finditer(log_text)]


def annotate(log_text, results_dir):
    lines = parse(log_text)
    out, problems = {}, []
    for f in sorted(Path(results_dir).glob("*/*.json")):
        if f.name.startswith("_") or f.parent.name == "speed" or f.name == "tails.json":
            continue
        rec = json.loads(f.read_text())
        rid = rec.get("response_id")
        if not rid:
            continue
        hits = [x for x in lines if x["rid"].startswith(rid)]
        key = f"{f.parent.name}/{rec['name']}"
        want = rec.get("expected_tail_start")
        if not hits:
            if want is not None:
                problems.append(f"{key}: RidgeFill request with no CED log line (not approximated?)")
            out[key] = dict(approximated=False, n=rec["n"], expected_tail_start=want)
            continue
        h = hits[-1]
        ok = want is not None and h["tail_start"] == want and h["n"] == rec["n"]
        if not ok:
            problems.append(f"{key}: tail start {h['tail_start']} of {h['n']}, planned {want} of {rec['n']}")
        out[key] = dict(approximated=True, tail_start=h["tail_start"], n=h["n"], tail=h["n"] - h["tail_start"],
                        xrows=h["xrows"], split=h["split"], expected_tail_start=want, match=ok)
    return dict(samples=out, problems=problems, log_lines=len(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, action="append", help="engine log; repeat it for every boot of the arm")
    ap.add_argument("--results", required=True, help="results/<arm>")
    a = ap.parse_args(argv)
    rep = annotate("\n".join(Path(f).read_text(errors="replace") for f in a.log), a.results)
    write_json(Path(a.results) / "tails.json", rep)
    print(f"tails: {len(rep['samples'])} samples, {len(rep['problems'])} problems, {rep['log_lines']} CED lines")
    for p in rep["problems"][:20]:
        print("  PROBLEM", p)
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
