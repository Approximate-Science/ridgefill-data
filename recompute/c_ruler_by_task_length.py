"""C: Appendix C, Tables C.1-C.3, filled-region RULER counts and arm means by task and length.

Read results/reference/*-ruler/samples/ruler through reftree.Tree; retain the four needle tasks' records with
needle_filled=true at the primary split 3/8 and tail 4096. Match code/finals/tables.py PerSampleFiller.rows:
count the selected records and report 100 * numpy.mean(score) over records carrying each arm (exact at '-').
These are sample-weighted descriptive means, without bootstrap intervals. Qwen3.5's single_1 row pools 8K-64K;
its no-fill mean uses only records carrying that arm (8K-32K). Other rows are separate task/length cells.
Human-readable summaries go to stderr so stdout contains only gate-compatible claim lines.
"""
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, num, parse, say                   # noqa: E402
from reftree import MODELS, Tree                            # noqa: E402

TASKS = ("niah_multikey_1", "niah_multiquery", "niah_multivalue", "niah_single_1")
ARMS = (("exact", "exact", "-"), ("ridgefill", "kva", "3/8"), ("identity", "noproj", "3/8"),
        ("nofill", "skip", "3/8"), ("ridgefill_rows", "kva+qattn10", "3/8"),
        ("identity_rows", "noproj+qattn10", "3/8"))


def length_label(lengths):
    """A single length or the endpoints of the pooled row, in claim-id notation."""
    lengths = sorted(set(lengths))
    if not lengths:
        raise SystemExit("no lengths in the selected population")
    ends = lengths if len(lengths) == 1 else (lengths[0], lengths[-1])
    return "-".join(f"{length // 1024}k" for length in ends)


args = parse(__doc__)
t = Tree(args.root, [model + "-ruler" for model in MODELS.values()])
for short, model in MODELS.items():
    cell = t.cells[model + "-ruler"]
    if cell["split"] != "3/8" or cell["tail"] != 4096:
        raise SystemExit(f"{model}: Appendix C requires split 3/8 and tail 4096")
    records = t.records("battery", model, "ruler")
    groups = {}
    for unit in sorted(records):
        r = records[unit]
        if not r.get("needle_filled") or r["task"] not in TASKS:
            continue
        # The paper pools only Qwen3.5's single-needle row; every other row retains its length.
        group_length = 0 if short == "qwen35" and r["task"] == "niah_single_1" else r["length"]
        groups.setdefault((r["task"], group_length), []).append(r)
    if not groups:
        raise SystemExit(f"{model}: no filled-region needle records")
    for (task, _), selected in sorted(groups.items()):
        lengths = sorted({r["length"] for r in selected})
        row_length = length_label(lengths)
        prefix = f"C.{short}.{task}"
        claim(f"{prefix}.{row_length}.n", len(selected))
        summary = []
        for key, arm, split in ARMS:
            carrying = [r for r in selected if split in r["arms"].get(arm, {})]
            if not carrying:
                if key == "nofill" and lengths == [65536]:
                    continue                              # declared not run at 64K: the paper prints n/a
                raise SystemExit(f"{model} {task} {row_length}: no record carries {arm}@{split}")
            scores = np.array([r["arms"][arm][split]["score"] for r in carrying], dtype=float)
            if not np.isfinite(scores).all():
                raise SystemExit(f"{model} {task} {row_length}: non-finite {arm}@{split} score")
            value = num(100.0 * np.mean(scores), 1)
            arm_length = length_label(r["length"] for r in carrying)
            claim(f"{prefix}.{arm_length}.{key}", value)
            summary.append(f"{key}={value} (n={len(carrying)}, {arm_length})")
        with redirect_stdout(sys.stderr):
            say(f"{model} {task} {row_length}: filled-region n={len(selected)}; " + "; ".join(summary))
