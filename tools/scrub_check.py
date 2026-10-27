#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Secret and privacy scan of a release tree. Reports path:line and rule only, never the matched value.

Every text file is scanned line by line (gzip files are decompressed first; .git is skipped). Rules cover credential
forms, private-key headers, operator and private-machine identifiers, unstripped prompt fields under results/
and derived/, forbidden file kinds and files over the size limit. A hit is ALLOWED only when an allowlist row names
its path glob and rule with a reason; allowed hits are still listed. An allowlist row that matches no hit is stale
and fails the scan, so exemptions cannot outlive their reason.

Allowlist rows (markdown table): | path glob | RULE | reason |

Machine-specific patterns (host names, home paths, internal project names) are not published. Maintainers keep them
in a local file and pass it with --private-patterns FILE; each line is RULE<TAB>REGEX (lines starting with # are
comments). Without it the scan still runs every rule below, including a generic home-directory rule.

Exit 0: no unallowed hit. Exit 1: unallowed hits. Exit 2: bad arguments, malformed or stale allowlist.
Usage: python3 tools/scrub_check.py ROOT --allowlist tools/ALLOWLIST.md [--report FILE] [--max-size-mib 50]
       [--private-patterns FILE ...]
"""
import argparse
import fnmatch
import gzip
import re
import sys
from pathlib import Path

RULES = {
    "PRIVATE_KEY": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "HF_TOKEN": r"\bhf_[A-Za-z0-9]{30,}\b",
    "GITHUB_TOKEN": r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    "OPENAI_KEY": r"\bsk-[A-Za-z0-9_-]{20,}",
    "SLACK_TOKEN": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "AWS_KEY": r"\bAKIA[0-9A-Z]{16}\b",
    "GOOGLE_KEY": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "SECRET_ASSIGN": r"\b[A-Z][A-Z0-9_]*(?:TOKEN|API_KEY|APIKEY|SECRET|PASSWORD|PASSWD)\s*[=:]\s*['\"]?[^\s'\"$<{]{6,}",
    "PASSWORD": r"(?i:\bpassword)\s*[=:]\s*['\"]?[^\s'\"]{4,}",
    "HOME_PATH": r"/(?:var/)?home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/",
    "GPU_UUID": r"\bGPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b|\bGPU-[0-9a-fA-F]{16}\b",
    "PCI_ID": r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b",
    "EMAIL": r"\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
    "RADIANCE": r"(?i:radiance)",
    "OPERATOR": r"(?i:\bvast\.ai\b|\bvastai\b|runpod)|\bVAST_[A-Z_]+",
}
PROMPT_FIELD = re.compile(r'"(?:prompt|messages)": *(?:"|\[)')
FORBIDDEN_NAMES = [".env", "*.env", "id_rsa*", "id_ed25519*", "*.pem", "*.key", ".netrc", "*.receipt",
                   "nvidia-smi-q.txt", "*.f32", "*.pt", "*.pkl", "*.pickle"]
# One alternation with a named group per rule: a single pass per line. A line that matches two rules at one position
# reports the first; the line is flagged either way.
COMBINED = re.compile("|".join(f"(?P<{name}>{rx})" for name, rx in RULES.items()))


def compile_rules():
    return re.compile("|".join(f"(?P<{name}>{rx})" for name, rx in RULES.items()))


def load_private(paths):
    """Add RULE<TAB>REGEX lines from unpublished local files to RULES; return how many rules were added."""
    added = 0
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            rule, sep, rx = line.partition("\t")
            if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", rule) or rule in RULES:
                sys.exit(f"{path}:{number}: expected a new RULE<TAB>REGEX line")
            re.compile(rx)
            RULES[rule] = rx
            added += 1
    return added


def load_allowlist(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) != 3 or set(cells[0]) <= set("-: ") or cells[0] == "path glob":
            continue
        glob, rule, reason = cells
        glob = glob.strip("`")
        if rule not in RULES and rule not in ("UNSTRIPPED_PROMPT", "FORBIDDEN_FILE", "BIG_FILE"):
            sys.exit(f"allowlist line {number}: unknown rule {rule!r}")
        if len(reason) < 10:
            sys.exit(f"allowlist line {number}: reason too short for {glob} {rule}")
        rows.append({"glob": glob, "rule": rule, "reason": reason, "used": 0, "line": number})
    return rows


def text_lines(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def scan_file(rel, path, max_bytes):
    hits = []
    name = path.name
    if any(fnmatch.fnmatch(name, g) for g in FORBIDDEN_NAMES):
        hits.append((0, "FORBIDDEN_FILE"))
    if path.stat().st_size > max_bytes:
        hits.append((0, "BIG_FILE"))
    lines = text_lines(path)
    if lines is None:
        return hits
    check_prompt = rel.startswith(("results/", "derived/")) and (".json" in name)
    for number, line in enumerate(lines, 1):
        for rule in sorted({m.lastgroup for m in COMBINED.finditer(line)}):
            hits.append((number, rule))
        if check_prompt and PROMPT_FIELD.search(line):
            hits.append((number, "UNSTRIPPED_PROMPT"))
    return hits


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("--allowlist", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p.add_argument("--max-size-mib", type=float, default=50)
    p.add_argument("--private-patterns", type=Path, action="append", default=[])
    args = p.parse_args(argv)
    global COMBINED
    private_rules = load_private(args.private_patterns)
    COMBINED = compile_rules()
    if not args.root.is_dir():
        print(f"root {args.root} is not a directory", file=sys.stderr)
        return 2
    allow = load_allowlist(args.allowlist)
    allow_file = args.allowlist.resolve()
    files = sorted(f for f in args.root.rglob("*") if f.is_file() and ".git" not in f.relative_to(args.root).parts)
    if not files:
        print(f"no files under {args.root}", file=sys.stderr)
        return 2
    out, bad, allowed = [], 0, 0
    for path in files:
        rel = path.relative_to(args.root).as_posix()
        if path.resolve() in (allow_file, Path(__file__).resolve()):
            continue   # the scanner's own rules and the allowlist's reasons name the patterns
        for number, rule in scan_file(rel, path, args.max_size_mib * 2**20):
            row = next((a for a in allow if a["rule"] == rule and fnmatch.fnmatch(rel, a["glob"])), None)
            where = f"{rel}:{number}" if number else rel
            if row:
                row["used"] += 1
                allowed += 1
                out.append(f"ALLOWED  {where}  {rule}  (allowlist line {row['line']})")
            else:
                bad += 1
                out.append(f"HIT      {where}  {rule}")
    stale = [a for a in allow if a["used"] == 0]
    for a in stale:
        out.append(f"STALE    allowlist line {a['line']}: {a['glob']} {a['rule']} matched nothing")
    summary = (f"files {len(files)}; unallowed hits {bad}; allowed {allowed}; stale allowlist rows {len(stale)}; "
               f"private rules loaded {private_rules}")
    text = "\n".join(out + [summary]) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(summary)
    if stale:
        return 2
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
