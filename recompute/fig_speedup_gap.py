# SPDX-License-Identifier: Apache-2.0
"""Figure 8 (Qwen3.8-27B row shares and compute ceiling), drawn from the release data.

Usage: python fig_speedup_gap.py <ridgefill-data root> <output dir>
Per length, every quantity is the median over the three timed requests of the resident arm, the same requests whose
median prompt time gives Table 8: exact prefix rows (n_exact_prefix) and the analytic skipped share
(mac_share_skipped). Speedups are stock median / arm median prompt time. Style follows the paper's other figures."""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
T = root / "results" / "qwen38-27b" / "results" / "timing"
LENGTHS, LABELS, TAIL_ROWS = [16384, 32768, 65536, 120000], ["16K", "32K", "64K", "120K"], 4096


def timed(length, arm):
    rows = json.loads((T / str(length) / arm / "result.json").read_text())["rows"]
    rows = [r for r in rows if r["phase"].startswith("timed-")]
    assert len(rows) == 3, (length, arm, len(rows))
    return rows


exact, share, resident, streamed = [], [], [], []
for n in LENGTHS:
    rf = [r["response"]["timings"]["ridgefill"] for r in timed(n, "resident")]
    exact.append(st.median(x["n_exact_prefix"] for x in rf))
    share.append(st.median(x["mac_share_skipped"] for x in rf))
    stock = st.median(r["prompt_ms"] for r in timed(n, "stock"))
    resident.append(stock / st.median(r["prompt_ms"] for r in timed(n, "resident")))
    streamed.append(stock / st.median(r["prompt_ms"] for r in timed(n, "streamed")))
ceiling = [1 / (1 - s) for s in share]
f_pct = [100 * (n - TAIL_ROWS - e) / n for n, e in zip(LENGTHS, exact)]
e_pct = [100 * e / n for n, e in zip(LENGTHS, exact)]
t_pct = [100 * TAIL_ROWS / n for n in LENGTHS]
for row in zip(LABELS, exact, [f"{100 * e / (n - TAIL_ROWS):.1f}" for n, e in zip(LENGTHS, exact)],
               [f"{v:.1f}" for v in f_pct], [f"{s:.5f}" for s in share], [f"{c:.3f}" for c in ceiling],
               [f"{v:.3f}" for v in resident], [f"{v:.3f}" for v in streamed]):
    print("length {} exact {} ({} % of prefix) filled {} % share {} ceiling {} resident {} streamed {}".format(*row))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})
EXACT, FILLED, TAILC, INK, MUTED = "#2f5c8a", "#f2dcae", "#6f98c2", "#222222", "#8a8a8a"
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1, 1.25]})
x = range(4)
a.bar(x, f_pct, color=FILLED, edgecolor=INK, lw=0.4, width=0.62, label="filled")
a.bar(x, e_pct, bottom=f_pct, color=EXACT, edgecolor=INK, lw=0.4, width=0.62, label="exact prefix rows")
a.bar(x, t_pct, bottom=[f + e for f, e in zip(f_pct, e_pct)], color=TAILC, edgecolor=INK, lw=0.4, width=0.62,
      label="tail (4,096)")
for i, v in enumerate(f_pct):
    a.text(i, v / 2, f"{v:.0f} %", ha="center", va="center", fontsize=6.4)
a.set_xticks(list(x), LABELS)
a.set_ylim(0, 100)
a.set_ylabel("share of prompt tokens (%)")
a.set_xlabel("prompt length")
a.set_title("a  Where the prompt's rows go", loc="left", fontsize=8, weight="bold")
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, handlelength=1.0,
         columnspacing=1.0)
b.fill_between(LENGTHS, resident, ceiling, color="#e6e6e6", lw=0)
b.plot(LENGTHS, ceiling, color=MUTED, ls=(0, (4, 2)), marker="o", ms=3, lw=1.0,
       label="ceiling from skipped MACs, 1 / (1 − share)")
b.plot(LENGTHS, resident, color=EXACT, marker="o", ms=3.2, lw=1.3, label="measured, resident")
b.plot(LENGTHS, streamed, color=TAILC, marker="s", ms=3, lw=1.1, label="measured, streamed")
b.text(46000, 1.44, "not explained by\nskipped compute (§4.4)", fontsize=6.2, color=INK, ha="center", va="center")
b.set_xscale("log")
b.set_xticks(LENGTHS, LABELS)
b.minorticks_off()
b.set_ylim(1.0, 1.9)
b.set_ylabel("speedup over stock (×)")
b.set_xlabel("prompt length")
b.set_title("b  Measured speedup against the compute ceiling", loc="left", fontsize=8, weight="bold")
b.spines[["top", "right"]].set_visible(False)
b.legend(frameon=False, fontsize=6.2, loc="upper left")
fig.subplots_adjust(wspace=0.32, bottom=0.3)
for ext in ("pdf", "svg", "png"):
    fig.savefig(out / f"fig_speedup_gap.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
