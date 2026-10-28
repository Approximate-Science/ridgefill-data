"""Paper sections 4.2 and 4.4: Qwen3.8-27B exact-prefix and filled-token shares, net analytic MAC shares, and the
range of client/server TTFT offsets. Follow recompute/fig_speedup_gap.py: use the resident arm's three timed
requests at each of the four timing lengths, median exact-prefix row count and median mac_share_skipped (which
includes map, probe and value-rescale costs). Divide exact rows by the prefix length, filled rows by the whole
prompt length. Offsets are median(client_ttft_s) - median(prompt_ms / 1000) within each of all twelve timing
cells, then the minimum and maximum in milliseconds; cold and warm-up requests are excluded throughout.
Human-readable diagnostics go to stderr so stdout can be passed directly to recompute/gate.py.
"""
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from statistics import median

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import claim, load, num, parse, say             # noqa: E402

args = parse(__doc__)
T = args.root / "results" / "qwen38-27b" / "results" / "timing"
LENGTHS = (16384, 32768, 65536, 120000)
prefix_pct, filled_pct, mac_pct, offsets = {}, {}, {}, []
for length in LENGTHS:
    for arm in ("stock", "resident", "streamed"):
        path = T / str(length) / arm / "result.json"
        result = load(path)
        timed = [r for r in result["rows"] if r["phase"].startswith("timed-")]
        if sorted(r["phase"] for r in timed) != ["timed-0", "timed-1", "timed-2"]:
            raise SystemExit(f"{path}: expected three distinct timed requests")
        server = median(r["prompt_ms"] / 1000 for r in timed)
        client = median(r["client_ttft_s"] for r in timed)
        offsets.append(1000 * (client - server))
        with redirect_stdout(sys.stderr):
            say(f"{length} {arm}: client median minus server median {offsets[-1]:.9f} ms")
        if arm != "resident":
            continue
        rf = [r["response"]["timings"]["ridgefill"] for r in timed]
        tails = {r["n_exact_tail"] for r in rf}
        if len(tails) != 1:
            raise SystemExit(f"{path}: exact-tail length varies across timed requests")
        tail = next(iter(tails))
        for r in rf:
            if (r["n_prompt"] != length or r["n_prefix"] != length - tail
                    or r["n_exact_prefix"] + r["n_filled"] != r["n_prefix"]):
                raise SystemExit(f"{path}: inconsistent prompt row counts")
            derived_share = 1 - r["macs_ridgefill"] / r["macs_exact"]
            if not math.isclose(derived_share, r["mac_share_skipped"], rel_tol=1e-12, abs_tol=1e-12):
                raise SystemExit(f"{path}: net MAC share differs from engine MAC counters")
        exact = median(r["n_exact_prefix"] for r in rf)
        prefix_pct[length] = 100 * exact / (length - tail)
        filled_pct[length] = 100 * (length - tail - exact) / length
        mac_pct[length] = 100 * median(r["mac_share_skipped"] for r in rf)
        with redirect_stdout(sys.stderr):
            say(f"{length} resident: exact prefix {prefix_pct[length]:.9f} %, "
                f"filled prompt {filled_pct[length]:.9f} %, net skipped MACs {mac_pct[length]:.9f} %")

claim("E.timing.exact_prefix_pct.min", num(min(prefix_pct.values()), 0))
claim("E.timing.exact_prefix_pct.max", num(max(prefix_pct.values()), 0) + " %")
for length in (LENGTHS[0], LENGTHS[-1]):
    claim(f"E.timing.{length}.filled_prompt_pct", num(filled_pct[length], 0) + " %")
    claim(f"E.timing.{length}.net_mac_skipped_pct", num(mac_pct[length], 0) + " %")
claim("E.timing.client_server_offset_ms.min", num(min(offsets), 0))
claim("E.timing.client_server_offset_ms.max", num(max(offsets), 0))
