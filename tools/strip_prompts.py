#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Copy a results tree, replacing named JSON fields by their hash.

Every file under --src is copied to the same relative path under --dst, except:
  * files matching --exclude are skipped (a pattern is matched against the
    relative path, the file name and every directory name on the path);
  * in .json and .jsonl files (only those whose file name matches a --strip-only
    glob, when any is given), every value whose key is in --fields (at any
    depth) is replaced by {"stripped": true, "sha256": <hex>, "n": <len>},
    where <hex> is the sha256 of the value's canonical JSON (sorted keys,
    separators (",", ":"), UTF-8, no ASCII escaping);
  * every value whose key is in --drop-fields is replaced by {"stripped": true,
    "n": <len>} with no hash (for answer keys, where a hash of a short value
    would reveal it); such a file's original sha256 is logged as "withheld",
    and the runner's sidecar <file>.sha256, which would reveal it too, is left out;
  * every match of a --redact NAME=REGEX is replaced by "<NAME>-<k>", with k
    numbering the distinct matched values (sorted) across the whole run, so
    equal values stay equal and nothing of the value is kept;
  * files matching --gzip are written gzip-compressed (mtime 0) as <name>.gz.
A file that none of these touch is copied byte for byte. Every transformed
file gets one row in --log (TSV, appended). Nothing is written in place:
--dst must not exist or be empty, and must not lie inside --src.
"""
import argparse
import fnmatch
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

LOG_HEADER = "path\torigin\torigin_sha256\tsha256\ttransform\tdetail\n"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def stripped_record(value, keep_hash=True):
    n = len(value) if isinstance(value, (str, list, dict)) else None
    if not keep_hash:
        return {"stripped": True, "n": n}
    return {"stripped": True, "sha256": sha256_bytes(canonical(value)), "n": n}


DROP = set()   # keys replaced without a hash (--drop-fields)


def strip_fields(obj, fields, counts):
    """Return obj with every value under a key in `fields` replaced; count replacements per key."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in fields and not (isinstance(value, dict) and value.get("stripped") is True):
                out[key] = stripped_record(value, keep_hash=key not in DROP)
                counts[key] = counts.get(key, 0) + 1
            else:
                out[key] = strip_fields(value, fields, counts)
        return out
    if isinstance(obj, list):
        return [strip_fields(item, fields, counts) for item in obj]
    return obj


def json_style(text):
    """Guess the indent and separators the original writer used, to keep diffs small."""
    lines = text.split("\n", 2)
    indent = None
    if len(lines) > 1:
        lead = len(lines[1]) - len(lines[1].lstrip(" "))
        indent = lead if lead > 0 else None
    ascii_only = all(ord(c) < 128 for c in text)
    separators = (",", ":") if indent is None and '": ' not in text[:4000] else None
    return {"indent": indent, "ensure_ascii": ascii_only, "separators": separators}


def dump(obj, style, trailing_newline):
    text = json.dumps(obj, indent=style["indent"], ensure_ascii=style["ensure_ascii"],
                      separators=style["separators"])
    return text + ("\n" if trailing_newline else "")


def transform_json(text, fields):
    counts = {}
    obj = json.loads(text)
    new = strip_fields(obj, fields, counts)
    if not counts:
        return text, counts
    return dump(new, json_style(text), text.endswith("\n")), counts


def transform_jsonl(text, fields):
    counts = {}
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if not body.strip():
            out.append(line)
            continue
        line_counts = {}
        new = strip_fields(json.loads(body), fields, line_counts)
        if line_counts:
            sep = None if '": ' in body else (",", ":")
            out.append(json.dumps(new, ensure_ascii=all(ord(c) < 128 for c in body), separators=sep)
                       + line[len(body):])
            for key, value in line_counts.items():
                counts[key] = counts.get(key, 0) + value
        else:
            out.append(line)
    return "".join(out), counts


def excluded(rel, patterns):
    parts = PurePosixPath(rel).parts
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern) or any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
    return False


def collect(src, patterns):
    files = []
    for root, dirs, names in os.walk(src):
        dirs.sort()
        for name in sorted(names):
            path = Path(root) / name
            rel = path.relative_to(src).as_posix()
            if not excluded(rel, patterns):
                files.append((rel, path))
    return files


def redaction_labels(files, redactions):
    """Number every distinct matched value per rule, in sorted order, over all files."""
    seen = {name: set() for name, _ in redactions}
    for _, path in files:
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        for name, regex in redactions:
            seen[name].update(regex.findall(text))
    return {name: {value: f"{name}-{k}" for k, value in enumerate(sorted(values), 1)}
            for name, values in seen.items()}


def apply_redactions(text, redactions, labels, counts):
    for name, regex in redactions:
        def label(match, name=name):
            counts[name] = counts.get(name, 0) + 1
            return labels[name][match.group(0)]
        text = regex.sub(label, text)
    return text


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-strip")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def process(rel, path, args, redactions, labels, dropped):
    if rel.endswith(".sha256") and rel[:-len(".sha256")] in dropped:
        return (f"{args.dst_name}/{rel}", f"{args.origin}/{rel}", "withheld", "", "omitted",
                "runner sidecar of a file with dropped answer fields")
    data = path.read_bytes()
    detail = {}
    out = data
    if b"\0" not in data[:8192]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is not None:
            new = text
            strip_here = args.fields and (not args.strip_only or
                                          any(fnmatch.fnmatch(PurePosixPath(rel).name, g) for g in args.strip_only))
            if strip_here and rel.endswith(".json"):
                new, counts = transform_json(text, args.fields)
                detail.update({f"{'drop' if k in DROP else 'strip'}:{k}": v for k, v in counts.items()})
            elif strip_here and rel.endswith(".jsonl"):
                new, counts = transform_jsonl(text, args.fields)
                detail.update({f"{'drop' if k in DROP else 'strip'}:{k}": v for k, v in counts.items()})
            red_counts = {}
            new = apply_redactions(new, redactions, labels, red_counts)
            detail.update({f"redact:{k}": v for k, v in red_counts.items()})
            if new != text:
                out = new.encode("utf-8")
    dst_rel = rel
    if any(fnmatch.fnmatch(PurePosixPath(rel).name, g) for g in args.gzip):
        out = gzip.compress(out, compresslevel=9, mtime=0)
        dst_rel = rel + ".gz"
        detail["gzip"] = 1
    write_atomic(args.dst / dst_rel, out)
    origin_sha = sha256_bytes(data)
    if any(k.startswith("drop:") for k in detail):
        dropped.add(rel)
        origin_sha = "withheld"
    if detail:
        return (f"{args.dst_name}/{dst_rel}", f"{args.origin}/{rel}", origin_sha, sha256_bytes(out),
                ",".join(sorted(detail)), ";".join(f"{k}={v}" for k, v in sorted(detail.items())))
    return None


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--dst-name", required=True, help="repository path of --dst, used in the log")
    p.add_argument("--origin", required=True, help="origin alias for --src, used in the log (no absolute paths)")
    p.add_argument("--fields", default="", help="comma-separated JSON keys to replace by their hash")
    p.add_argument("--drop-fields", default="", help="comma-separated JSON keys to replace without a hash")
    p.add_argument("--strip-only", action="append", default=[], help="file-name glob limiting --fields")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--redact", action="append", default=[], help="NAME=REGEX")
    p.add_argument("--gzip", action="append", default=[], help="file-name glob to store gzip-compressed")
    p.add_argument("--log", type=Path, required=True)
    args = p.parse_args(argv)
    DROP.update(f for f in args.drop_fields.split(",") if f)
    args.fields = {f for f in args.fields.split(",") if f} | DROP
    args.src = args.src.resolve()
    args.dst = args.dst.resolve()
    if not args.src.is_dir():
        p.error(f"--src {args.src} is not a directory")
    if args.dst == args.src or args.src in args.dst.parents:
        p.error(f"--dst {args.dst} lies inside --src {args.src}; refusing")
    if args.dst.exists() and any(args.dst.iterdir()):
        p.error(f"--dst {args.dst} exists and is not empty; refusing to write over it")
    rules = []
    for item in args.redact:
        name, sep, regex = item.partition("=")
        if not sep or not name:
            p.error(f"--redact expects NAME=REGEX, got {item!r}")
        rules.append((name, re.compile(regex)))
    args.redact = rules
    return args


def main(argv):
    args = parse_args(argv)
    files = collect(args.src, args.exclude)
    if not files:
        sys.exit(f"no files to copy under {args.src} after exclusions")
    labels = redaction_labels(files, args.redact) if args.redact else {}
    rows, dropped = [], set()
    for i, (rel, path) in enumerate(files, 1):
        row = process(rel, path, args, args.redact, labels, dropped)
        if row:
            rows.append(row)
        if i % 500 == 0:
            print(f"{i}/{len(files)} files", file=sys.stderr)
    new_log = not args.log.exists()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as log:
        if new_log:
            log.write(LOG_HEADER)
        for row in rows:
            log.write("\t".join(row) + "\n")
    print(f"copied {len(files)} files, transformed {len(rows)}; "
          f"distinct redacted values: {sum(len(v) for v in labels.values())}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
