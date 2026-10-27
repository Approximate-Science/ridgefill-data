#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Positive and negative controls for tools/strip_prompts.py, tools/scrub_check.py and tools/verify_ledger.py.

Runs in a temporary directory; exits 1 on the first failed control. Planted values are assembled at run time so
this file itself contains none of the patterns it plants.
Usage: python3 tools/selftest.py
"""
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PY = sys.executable


def run(*args):
    return subprocess.run([PY, "-I", *map(str, args)], capture_output=True, text=True)


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        sys.exit(1)


def strip_controls(tmp):
    src, dst, log = tmp / "src", tmp / "dst", tmp / "log.tsv"
    (src / "a").mkdir(parents=True)
    planted = "third-party haystack text"
    (src / "a" / "with.json").write_text(json.dumps({"prompt": planted, "keep": 1}, indent=1) + "\n")
    (src / "a" / "without.json").write_text('{"x": 1}\n')
    (src / "a" / "rows.jsonl").write_text(json.dumps({"messages": [planted]}) + "\n" + '{"y": 2}\n')
    (src / "a" / "server.log").write_text("GPU-" + "0123abcd" * 2 + " started\n")
    (src / "a" / "big.f32").write_bytes(b"\0" * 8)
    r = run(TOOLS / "strip_prompts.py", "--src", src, "--dst", dst, "--dst-name", "results/x", "--origin", "ctl",
            "--fields", "prompt,messages", "--exclude", "*.f32", "--gzip", "*.log",
            "--redact", "GPU_UUID=GPU-[0-9a-f]{16}", "--log", log)
    check("strip: runs", r.returncode == 0)
    out = (dst / "a" / "with.json").read_text()
    check("strip: planted prompt field replaced by its hash (positive)", planted not in out and '"stripped": true' in out)
    check("strip: planted messages field in jsonl replaced (positive)", planted not in (dst / "a" / "rows.jsonl").read_text())
    check("strip: file without the fields byte-identical (negative)",
          (dst / "a" / "without.json").read_bytes() == (src / "a" / "without.json").read_bytes())
    check("strip: excluded file absent", not (dst / "a" / "big.f32").exists())
    log_text = gzip.decompress((dst / "a" / "server.log.gz").read_bytes()).decode()
    check("strip: GPU id redacted inside gzipped log", "0123abcd" not in log_text and "GPU_UUID-1" in log_text)
    dst2 = tmp / "dst2"
    r = run(TOOLS / "strip_prompts.py", "--src", src, "--dst", dst2, "--dst-name", "results/y", "--origin", "ctl",
            "--fields", "prompt,messages", "--strip-only", "rows.jsonl", "--exclude", "*.f32", "--log", tmp / "log2.tsv")
    check("strip: --strip-only strips the matching file (positive)",
          r.returncode == 0 and planted not in (dst2 / "a" / "rows.jsonl").read_text())
    check("strip: --strip-only leaves other files byte-identical (negative)",
          (dst2 / "a" / "with.json").read_bytes() == (src / "a" / "with.json").read_bytes())
    (src / "b").mkdir()
    (src / "b" / "item.json").write_text(json.dumps({"gold": "C", "judge": True}) + "\n")
    (src / "b" / "item.json.sha256").write_text("0" * 64 + "  item.json\n")
    dst3, log3 = tmp / "dst3", tmp / "log3.tsv"
    r = run(TOOLS / "strip_prompts.py", "--src", src / "b", "--dst", dst3, "--dst-name", "results/z", "--origin", "ctl",
            "--drop-fields", "gold", "--log", log3)
    rec = json.loads((dst3 / "item.json").read_text())
    check("strip: --drop-fields removes the value with no hash (positive)",
          r.returncode == 0 and rec["gold"] == {"stripped": True, "n": 1} and rec["judge"] is True)
    check("strip: --drop-fields leaves out the sidecar and withholds the original hash",
          not (dst3 / "item.json.sha256").exists() and "withheld" in log3.read_text())
    check("strip: refuses a destination inside the source",
          run(TOOLS / "strip_prompts.py", "--src", src, "--dst", src / "in", "--dst-name", "x", "--origin", "y",
              "--log", tmp / "l2.tsv").returncode == 2)
    check("strip: refuses a non-empty destination",
          run(TOOLS / "strip_prompts.py", "--src", src, "--dst", dst, "--dst-name", "x", "--origin", "y",
              "--log", tmp / "l3.tsv").returncode == 2)


def scrub_controls(tmp):
    root = tmp / "tree"
    (root / "results").mkdir(parents=True)
    planted = {
        "hf.txt": "token " + "hf" + "_" + "A" * 34,
        "gh.txt": "gh" + "p_" + "B" * 36,
        "key.txt": "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
        "assign.txt": "HF" + "_TOKEN=" + "x" * 20,
        "path.txt": "/var/" + "home/someone/file",
        "uuid.txt": "GPU-" + "1234abcd-" + "1234-" * 3 + "123456789abc",
        "mail.txt": "someone" + "@" + "example.org",
        "results/rec.json": '{"' + 'prompt": "text"}',
    }
    for name, text in planted.items():
        (root / name).write_text(text + "\n")
    (root / "results" / "logits.f32").write_bytes(b"\0")
    (root / "clean.txt").write_text("nothing to see\n")
    allow = tmp / "allow.md"
    allow.write_text("| path glob | rule | reason |\n|---|---|---|\n| mail.txt | EMAIL | synthetic address in a control |\n")
    r = run(TOOLS / "scrub_check.py", root, "--allowlist", allow, "--report", tmp / "report.txt")
    report = (tmp / "report.txt").read_text()
    check("scrub: planted files make the scan fail (positive)", r.returncode == 1)
    for name, rule in [("hf.txt", "HF_TOKEN"), ("gh.txt", "GITHUB_TOKEN"), ("key.txt", "PRIVATE_KEY"),
                       ("assign.txt", "SECRET_ASSIGN"), ("path.txt", "HOME_PATH"), ("uuid.txt", "GPU_UUID"),
                       ("results/rec.json", "UNSTRIPPED_PROMPT"), ("results/logits.f32", "FORBIDDEN_FILE")]:
        check(f"scrub: {rule} caught in {name}", f"HIT      {name}" in report and rule in report)
    check("scrub: allowlisted hit shown as ALLOWED, not hidden", "ALLOWED  mail.txt:1  EMAIL" in report)
    check("scrub: report never contains a planted value", "A" * 34 not in report and "x" * 20 not in report)
    clean = tmp / "clean"
    clean.mkdir()
    shutil.copy(root / "clean.txt", clean / "clean.txt")
    empty_allow = tmp / "empty.md"
    empty_allow.write_text("| path glob | rule | reason |\n|---|---|---|\n")
    check("scrub: clean tree passes (negative)", run(TOOLS / "scrub_check.py", clean, "--allowlist", empty_allow).returncode == 0)
    check("scrub: stale allowlist row fails", run(TOOLS / "scrub_check.py", clean, "--allowlist", allow).returncode == 2)
    private = tmp / "private.tsv"
    private.write_text("# local only\nMACHINE_NAME\t\\bexample" + "host7\\b\n")
    (clean / "host.txt").write_text("built on example" + "host7 yesterday\n")
    r = run(TOOLS / "scrub_check.py", clean, "--allowlist", empty_allow, "--private-patterns", private, "--report", tmp / "r2.txt")
    check("scrub: a pattern from a private file is applied (positive)",
          r.returncode == 1 and "HIT      host.txt:1  MACHINE_NAME" in (tmp / "r2.txt").read_text())
    check("scrub: without the private file that hit is not reported (negative)",
          run(TOOLS / "scrub_check.py", clean, "--allowlist", empty_allow).returncode == 0)


def ledger_controls(tmp):
    import hashlib
    lines, prev = [], None
    for seq in range(3):
        line = json.dumps({"seq": seq, "prev": prev, "type": "record", "utc": "2026-01-01T00:00:00+00:00", "label": f"r{seq}"})
        lines.append(line)
        prev = hashlib.sha256(line.encode()).hexdigest()
    folder = tmp / "ledger"
    folder.mkdir()
    rows = ["seq\tutc\ttype\tlabel\tsha256\tprev"]
    for line in lines:
        e = json.loads(line)
        rows.append("\t".join([str(e["seq"]), e["utc"], e["type"], e["label"], hashlib.sha256(line.encode()).hexdigest(), e["prev"] or ""]))
    (folder / "index.tsv").write_text("\n".join(rows) + "\n")
    (folder / "cited.jsonl").write_text(lines[1] + "\n")
    check("ledger: intact chain passes (negative)", run(TOOLS / "verify_ledger.py", folder).returncode == 0)
    (folder / "cited.jsonl").write_text(lines[1].replace("r1", "r9") + "\n")
    check("ledger: edited cited line fails (positive)", run(TOOLS / "verify_ledger.py", folder).returncode == 1)


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for part, fn in (("s", strip_controls), ("c", scrub_controls), ("l", ledger_controls)):
            (tmp / part).mkdir()
            fn(tmp / part)
    print("all controls passed")


if __name__ == "__main__":
    main()
