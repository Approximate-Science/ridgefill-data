"""C5 smoke test on one booted RidgeFill server (the graded arm's boot). Never reads the locked set: the prompts are
token ids of the repo's own markdown (paper/, docs/), repeated to length, so nothing here is test data.

Checks, in order (each printed and written to <out>/smoke.json):
  1  /health and the served model
  2  RidgeFill engages: a 16K prompt_logprobs request with dyluhn_kva=1 returns every prompt position (the scored
     tail is checked by the caller against the server log's "CED request ... approximated" line)
  3  determinism: the same 16K prompt_logprobs request twice on the exact path (dyluhn_kva=0) and twice with
     RidgeFill; bitwise-equal logprobs or the max |difference| (the deterministic top-k question, protocol.json
     engine_tier.final_requires)
  4  generation: 32 greedy tokens at 16K, both paths
  5  timing: one TTFT per path at 9,216 / 16,384 / 32,768 tokens (the rates the plan re-estimates from) and one
     prompt_logprobs pass at 32K per path (the perplexity rate)
  6  the longest RULER row fits: one 65,408-token request, max_tokens 1 (exact path), unless --no-64k
Exit 0 when every request succeeded; the determinism result is reported, not judged.
"""
import argparse
import json
import time
from pathlib import Path

from c5.client import Client, write_json, parse_prompt_logprobs, ttft, tokenizer, ids_sha

REPO = Path(__file__).resolve().parents[1]


def filler_ids(tok, n):
    text = "\n\n".join(p.read_text(errors="replace") for d in ("paper", "docs") for p in sorted((REPO / d).rglob("*.md"))
                       if "test-set-locked" not in str(p))
    base = tok(text, add_special_tokens=False).input_ids
    if not base:
        raise SystemExit("no filler text")
    out = []
    while len(out) < n:
        out += base
    return out[:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8079")
    ap.add_argument("--tail", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-64k", action="store_true")
    a = ap.parse_args(argv)
    cl = Client(a.base_url, 1, a.tail)
    rep = {"t0": time.time()}
    if not cl.healthy():
        raise SystemExit("not healthy")
    tok = tokenizer()
    ids16 = filler_ids(tok, 16384)
    rep["filler_sha256"] = ids_sha(ids16)

    def logprob_pass(ids, kva, tag):
        reply, wall, _ = cl.post(cl.body(ids, 1, f"c5-smoke-{tag}-{time.time()}", kva=kva, prompt_logprobs=0))
        return parse_prompt_logprobs(reply, ids), wall, reply.get("id")

    for kva in (1, 0):
        lp1, w1, id1 = logprob_pass(ids16, kva, f"lp{kva}a")
        lp2, w2, id2 = logprob_pass(ids16, kva, f"lp{kva}b")
        diff = max(abs(x - y) for x, y in zip(lp1, lp2))
        tail = lp1[-512:]
        rep[f"logprobs_kva{kva}"] = dict(bitwise_equal=lp1 == lp2, max_abs_diff=diff, wall_s=[w1, w2], ids=[id1, id2],
                                         tail512_mean_nll=-sum(tail) / len(tail))
        print(f"logprobs dyluhn_kva={kva}: two passes bitwise equal {lp1 == lp2}, max |diff| {diff:.3g}, "
              f"walls {w1:.1f}s {w2:.1f}s, tail-512 mean NLL {-sum(tail) / len(tail):.4f}", flush=True)
    for kva in (1, 0):
        reply, wall, _ = cl.post(cl.body(ids16, 32, f"c5-smoke-gen{kva}", kva=kva))
        ch = reply["choices"][0]
        rep[f"gen_kva{kva}"] = dict(finish=ch.get("finish_reason"), tokens=reply["usage"]["completion_tokens"],
                                    wall_s=wall, id=reply.get("id"), text_sha256=ids_sha(ch["text"]))
        print(f"generation dyluhn_kva={kva}: {reply['usage']['completion_tokens']} tokens, {wall:.1f}s", flush=True)
    rep["ttft"] = {}
    for L in (9216, 16384, 32768):
        ids = filler_ids(tok, L)
        for kva in (1, 0):
            r = ttft(cl, ids, f"c5-smoke-ttft-{L}-{kva}-{time.time()}", kva=kva)
            rep["ttft"][f"{L}-kva{kva}"] = r
            print(f"ttft {L} dyluhn_kva={kva}: {r['ttft_s']:.2f}s ({L / r['ttft_s']:.0f} tok/s)", flush=True)
    ids32 = filler_ids(tok, 32768)
    for kva in (1, 0):
        _, wall, rid = logprob_pass(ids32, kva, f"lp32-{kva}")
        rep[f"ppl32_kva{kva}"] = dict(wall_s=wall, id=rid, tok_s=32768 / wall)
        print(f"prompt_logprobs 32K dyluhn_kva={kva}: {wall:.1f}s ({32768 / wall:.0f} tok/s)", flush=True)
    if not a.no_64k:
        ids64 = filler_ids(tok, 65408)
        r = ttft(cl, ids64, f"c5-smoke-64k-{time.time()}", kva=0)
        rep["ttft_64k_exact"] = r
        print(f"64K exact path: ttft {r['ttft_s']:.1f}s", flush=True)
    rep["t1"] = time.time()
    write_json(Path(a.out) / "smoke.json", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
