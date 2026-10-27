#!/usr/bin/env python3
"""Offline structural check of finals/Dockerfile against the assembled build context (build.sh
--check runs this; no docker daemon is needed): every COPY source must exist, the base image must be
pinned (no :latest), the pinned-requirements install must be there, and the entrypoint must exist.
Fails with named errors on any of: missing COPY source, a floating base tag, a missing
'requirements.txt' pin, a missing entrypoint, or an unpinned torch.
"""
import re
import sys
from pathlib import Path

PINS_REQUIRED = ("requirements.txt",)      # the harness deps; torch is pinned via ARG TORCH_VERSION


def check(dockerfile, ctx):
    problems = []
    lines = [l.split("#", 1)[0].rstrip() for l in Path(dockerfile).read_text().splitlines()]
    instr = []
    for i, l in enumerate(lines, 1):
        s = l.strip()
        if s:
            instr.append((i, s))

    seen = {"FROM": set(), "COPY": set(), "ARG": set()}
    for i, s in instr:
        words = s.split()
        op = words[0].upper()
        if op == "FROM":
            tag = words[1].split(":")[-1]
            if tag == "latest" or ":" not in words[1]:
                problems.append(f"line {i}: base image {words[1]!r} is not pinned to a version tag")
            seen["FROM"].add(words[1])
        elif op in ("COPY", "ADD"):
            for src in words[1:-1]:
                if src.startswith("--"):
                    continue
                p = Path(ctx) / src.rstrip("/")
                if not p.exists():
                    problems.append(f"line {i}: COPY source {src!r} does not exist in the context {ctx}")
                seen["COPY"].add(src.rstrip("/"))
        elif op == "ARG":
            seen["ARG"].add(words[1].split("=")[0])
        elif op == "RUN":
            if "pip install" in l and "-r requirements.txt" in l and "--no-cache-dir" not in l:
                problems.append(f"line {i}: pip install without --no-cache-dir (image bloat)")

    joined = "\n".join(s for _, s in instr)
    for pin in PINS_REQUIRED:
        if not re.search(rf"pip install[^\n]*-r {pin}", joined):
            problems.append(f"no 'pip install -r {pin}': the harness deps are not pinned in the image")
    if "TORCH_VERSION" not in seen["ARG"]:
        problems.append("no ARG TORCH_VERSION: the torch wheel version must be pinned, not floating")
    if not re.search(r"pip install[^\n]*torch==\$\{?TORCH_VERSION\}?", joined):
        problems.append("torch is not installed from the pinned TORCH_VERSION")
    if not any(s.split()[0].upper() == "ENTRYPOINT" for _, s in instr):
        problems.append("no ENTRYPOINT: the image does not run run_final.sh by default")
    else:
        ep = [s.split() for _, s in instr if s.split()[0].upper() == "ENTRYPOINT"][-1]
        target = ep[-1].strip('[]"').removeprefix("/opt/kva/")
        if not (Path(ctx) / target).exists():
            problems.append(f"ENTRYPOINT {ep[-1]!r} does not exist in the context")
    for needed in ("kva/", "scripts/", "configs/", "tests/", "requirements.txt", "finals/"):
        if needed.rstrip("/") not in seen["COPY"]:
            problems.append(f"{needed} is never COPYed into the image (the reference harness must ride along)")
    if "datasets/test-set-locked" in str(seen["COPY"]):
        problems.append("datasets/test-set-locked is COPYed: locked-set texts must never be baked in")
    return problems


def main():
    if len(sys.argv) != 3:
        print("usage: check_dockerfile.py DOCKERFILE CONTEXT_DIR", file=sys.stderr)
        return 2
    problems = check(sys.argv[1], sys.argv[2])
    for p in problems:
        print(f"DOCKERFILE FAIL: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"check_dockerfile: {sys.argv[1]} OK against {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())