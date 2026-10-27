"""C5 endpoint client: the locked-set tests of one arm against one booted server (tcclaviger's vLLM, OpenAI API).

Sets (prompts from the repo's own builders, sent as token ids, never as text):
  ppl     kva.locked.longppl_docs: every locked long-prompt document of nominal length <= --ppl-max-len (all domains);
          prompt_logprobs=0 on the whole document; scored = the 512-token continuation, all and novel tokens
          (kva.data.novel_mask over the whole prompt, as longppl_docs computes it). The engine skips RidgeFill on
          prompt_logprobs requests unless vllm_xargs dyluhn_kva=1 asks for it (vllm/v1/ced.py CedPolicy.plan): a
          RidgeFill arm sends it on every request.
  ruler   kva.locked.ruler_prompts, raw format, the 4 NIAH tasks at --ruler-lengths, protocol.json ruler_max_new caps;
          greedy, kva.locked.score; needle_filled = every scored needle ends at or before N - T (finals/convert.py).
  speed   time to first token (streamed, max_tokens 1) at 9,216 / 16,384 / 32,768 tokens: prefixes of one locked
          32K document, 1 warm-up + --speed-runs timed per length; on a RidgeFill boot each round runs the arm and
          the same server's exact path (dyluhn_kva=0) back to back, interleaved (protocol section 2, Speed v1.1).
  control the positive control on a RidgeFill boot: the first --control-docs ppl documents with dyluhn_kva=0 (no row
          approximated: the exact path with the projector loaded), compared with the exact boot by c5.summarize.
Each sample is one JSON file <out>/<arm>/<set>[-<pass>]/<name>.json with a .sha256 sidecar, written atomically
and read back; a sample whose file and sidecar agree is skipped on resume. Nothing prints document text.

Usage: python -m c5.client --arm NAME --kva 0|1 --tail 2048 --out DIR --sets ppl,ruler,speed [--pass a]
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_DIR = os.environ.get("C5_MODEL_DIR")             # the Flash-Next checkpoint directory (its tokenizer)
MODEL = "qwen3.8-flash-next"
SERVED = "qwen3.8-flash-next"
NIAH = ("niah_single_1", "niah_multikey_1", "niah_multivalue", "niah_multiquery")
SPEED_LENGTHS = (9216, 16384, 32768)
TIMEOUT = 3600


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def ids_sha(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode())


# ---- files

def write_json(path, obj):
    """Atomic write + .sha256 sidecar, then read back and compare (host RAM: kva/integrity.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    h = sha256(b)
    side = path.with_name(path.name + ".sha256")
    side.write_text(f"{h}  {path.name}\n")
    if sha256(path.read_bytes()) != h:
        raise SystemExit(f"{path}: read-back differs from what was written (host RAM?)")
    return h


def done(path):
    path = Path(path)
    side = path.with_name(path.name + ".sha256")
    if not (path.is_file() and side.is_file()):
        return False
    return side.read_text().split()[0] == sha256(path.read_bytes())


# ---- server

class Client:
    def __init__(self, base_url, kva, tail, pace=0.0):
        self.base = base_url.rstrip("/")
        self.kva = kva
        self.tail = tail
        self.pace = pace

    def healthy(self):
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
            return False

    def body(self, ids, max_tokens, salt, kva=None, **more):
        """kva: None = the arm's default (RidgeFill arms force it on, exact sends nothing); 0 / 1 = this request."""
        # greedy; the checkpoint's generation_config (top_k 20, top_p 0.95) is moot at temperature 0; no penalties;
        # stops at its eos ids [248046, 248044]; text decoded without special tokens, as kva's evaluator scores
        b = {"model": SERVED, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "cache_salt": salt,
             "repetition_penalty": 1.0, "skip_special_tokens": True}
        on = self.kva if kva is None else kva
        if self.kva:
            b["vllm_xargs"] = {"dyluhn_kva": "1" if on else "0", "dyluhn_kva_tail": self.tail}
        b.update(more)
        return b

    def post(self, body, stream=False):
        """(reply json or SSE chunks, wall s, time to first text chunk). 5xx / connection errors retry 3 times."""
        if self.pace:
            time.sleep(self.pace)
        req = urllib.request.Request(f"{self.base}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    if not stream:
                        return json.load(r), time.perf_counter() - t0, None
                    chunks, first = [], None
                    for line in r:
                        line = line.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        if first is None and chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                            first = time.perf_counter() - t0
                        chunks.append(chunk)
                    return chunks, time.perf_counter() - t0, first
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")[:500]
                if e.code < 500:
                    raise SystemExit(f"request rejected ({e.code}): {text}")
                print(f"  server error {e.code}, retry {attempt + 1}/3: {text}", file=sys.stderr, flush=True)
            except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
                print(f"  connection error, retry {attempt + 1}/3: {e}", file=sys.stderr, flush=True)
            if not self.healthy():
                raise SystemExit("server unhealthy after a failed request: stopping (check the watchdog / kernel log)")
            time.sleep(10 * (attempt + 1))
        raise SystemExit("request failed three times")


# ---- prompts

def tokenizer():
    from transformers import AutoTokenizer
    if not MODEL_DIR:
        raise SystemExit("c5.client: C5_MODEL_DIR is not set: name the Flash-Next checkpoint directory (its tokenizer)")
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def protocol():
    from kva import data
    return json.loads((data.DATASETS.parent / "protocol.json").read_text())


def ppl_docs(tok, max_len, dataset="", domains=()):
    """The locked long prompts, or (dataset) a frozen set beside them (kva.locked.DATASETS); domains: only these."""
    from kva import locked
    if dataset:
        root = locked.dataset_dir(dataset)
        docs, skipped = locked.longppl_docs(tok, locked.manifest(root, None, dataset), MODEL, max_len, root)
    else:
        docs, skipped = locked.longppl_docs(tok, locked.manifest(), MODEL, max_len)
    if domains:
        docs = [q for q in docs if q["task"] in domains]
    return docs, skipped


def ruler_rows(tok, lengths, tasks=NIAH, fmt="raw", prefix="as-is"):
    """fmt raw: BOS + input + answer prefix as one text (the kit's first choice, copied from Qwen3.5's protocol entry);
    chat: the model's chat template with thinking off and the answer prefix opening the assistant turn. Flash-Next is a
    thinking chat model: on raw prompts it opens a reasoning block, stops at once or continues the text (exact only
    ~73% on RULER 8K-32K, 2026-10-08)."""
    from kva import locked
    p = protocol()
    rows, skipped = locked.ruler_prompts(tok, fmt, locked.manifest(), None, tasks=list(tasks), lengths=list(lengths),
                                         max_new=p["ruler_max_new"]["tasks"], prefix=prefix)
    return rows, skipped


def speed_prompts(tok):
    """{length: ids}: prefixes (raw, no BOS on this tokenizer) of the first locked 32K long-prompt document in
    manifest path order whose ids reach 32,768 tokens."""
    from kva import locked
    docs, _ = locked.longppl_docs(tok, locked.manifest(), MODEL, 32768)
    for d in sorted(docs, key=lambda d: d["source"]):
        if d["length"] == 32768 and len(d["ids"]) >= max(SPEED_LENGTHS):
            return {L: d["ids"][:L] for L in SPEED_LENGTHS}, d["source"]
    raise SystemExit("no locked 32K document reaches 32,768 tokens")


# ---- tests

def parse_prompt_logprobs(reply, ids):
    """Logprobs of every prompt position after the first; each entry must hold exactly the prompt's token."""
    entries = reply["choices"][0]["prompt_logprobs"]
    n = reply["usage"]["prompt_tokens"]
    if n != len(ids) or len(entries) != n or entries[0] is not None:
        raise SystemExit(f"prompt_logprobs: {len(entries)} entries, usage {n}, sent {len(ids)}")
    out = []
    for i, e in enumerate(entries[1:], 1):
        if str(ids[i]) not in e:
            raise SystemExit(f"prompt_logprobs position {i}: token {ids[i]} not in the entry")
        out.append(e[str(ids[i])]["logprob"])
    return out


def run_ppl(cl, q, salt, kva=None):
    ids = q["ids"]
    reply, wall, _ = cl.post(cl.body(ids, 1, salt, kva=kva, prompt_logprobs=0))
    lp = parse_prompt_logprobs(reply, ids)[-q["cont_len"]:]
    novel = q["novel"]
    xs = [-x for x in lp]
    nov = [x for x, m in zip(xs, novel) if m]
    return dict(name=q["name"], set="longppl", task=q["task"], length=q["length"], source=q["source"],
                source_sha256=q["source_sha256"], n=len(ids), ids_sha256=ids_sha(ids), cont_len=q["cont_len"],
                tail=cl.tail if (cl.kva and kva != 0) else None,
                expected_tail_start=(len(ids) - cl.tail) if (cl.kva and kva != 0) else None,
                logprobs=lp, novel=novel, novel_count=len(nov), nll_all=sum(xs) / len(xs),
                nll_novel=(sum(nov) / len(nov)) if nov else None, response_id=reply.get("id"),
                usage=reply.get("usage"), wall_s=wall)


def run_ruler(cl, q, salt):
    from kva import locked
    ids = q["ids"]
    reply, wall, _ = cl.post(cl.body(ids, q["max_new"], salt))
    ch = reply["choices"][0]
    sc = locked.score(q, ch["text"])
    n = len(ids)
    return dict(name=q["name"], set="ruler", task=q["task"], length=q["length"], source=q["source"],
                source_sha256=q["source_sha256"], row=q["row"], n=n, ids_sha256=ids_sha(ids), max_new=q["max_new"],
                refs=q["refs"], text=ch["text"], finish_reason=ch.get("finish_reason"), **sc,
                needles=q["needles"], tail=cl.tail if cl.kva else None,
                needle_filled=all(e <= n - cl.tail for _, e in q["needles"]),
                expected_tail_start=(n - cl.tail) if cl.kva else None, response_id=reply.get("id"),
                usage=reply.get("usage"), wall_s=wall)


def ttft(cl, ids, salt, kva=None):
    chunks, wall, first = cl.post(cl.body(ids, 1, salt, kva=kva, stream=True,
                                          stream_options={"include_usage": True}), stream=True)
    usage = next((c["usage"] for c in chunks if c.get("usage")), None)
    if usage and usage["prompt_tokens"] != len(ids):
        raise SystemExit(f"speed: server counted {usage['prompt_tokens']} prompt tokens, sent {len(ids)}")
    return dict(ttft_s=first, wall_s=wall, id=chunks[0].get("id") if chunks else None)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run_speed(cl, prompts, source, runs, arm, out_dir):
    """Per length: 1 warm-up + runs rounds; a RidgeFill boot alternates the arm and its exact path per round."""
    modes = ("arm", "exact-switch") if cl.kva else ("arm",)
    for L, ids in prompts.items():
        path = out_dir / f"speed-{L}.json"
        if done(path):
            continue
        rec = dict(length=L, n=len(ids), source=source, ids_sha256=ids_sha(ids), arm=arm, tail=cl.tail if cl.kva else None,
                   runs={m: [] for m in modes})
        for i in range(1 + runs):
            for j, m in enumerate(modes if i % 2 == 0 else modes[::-1]):        # alternate which goes first
                r = ttft(cl, ids, f"c5-{arm}-speed-{L}-{m}-{i}", kva=(0 if m == "exact-switch" else None))
                r["warmup"] = i == 0
                rec["runs"][m].append(r)
                print(f"  speed {L} {m} run {i}{' (warm-up)' if i == 0 else ''}: ttft {r['ttft_s']:.3f}s", flush=True)
        for m in modes:
            t = [r["ttft_s"] for r in rec["runs"][m] if not r["warmup"]]
            rec[f"{m}_ttft_median_s"] = median(t)
            q1, q3 = sorted(t)[len(t) // 4], sorted(t)[(3 * len(t)) // 4]
            rec[f"{m}_ttft_iqr_s"] = [q1, q3]
        if cl.kva:
            rec["speedup_same_boot"] = rec["exact-switch_ttft_median_s"] / rec["arm_ttft_median_s"]
        write_json(path, rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--kva", type=int, choices=(0, 1), required=True, help="1: a RidgeFill server (force it per request)")
    ap.add_argument("--tail", type=int, default=2048)
    ap.add_argument("--base-url", default="http://127.0.0.1:8079")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="ppl,ruler,speed")
    ap.add_argument("--pass", dest="pass_", default="a", help="pass label (exact runs twice on one boot: a, b)")
    ap.add_argument("--ppl-max-len", type=int, default=32768)
    ap.add_argument("--ppl-dataset", default="", help="a frozen set beside the locked one (kva.locked.DATASETS)")
    ap.add_argument("--ppl-domains", default="", help="comma list: only these domains of the set")
    ap.add_argument("--ruler-lengths", default="8192,16384,32768")
    ap.add_argument("--ruler-tasks", default=",".join(NIAH))
    ap.add_argument("--ruler-format", choices=("raw", "chat"), default="raw")
    ap.add_argument("--ruler-prefix", choices=("as-is", "lstrip", "none", "instruct"), default="as-is",
                    help="lstrip: the answer prefix without its leading space (H10); none: no answer prefix (H11); "
                         "instruct: the question as an instruction with the answer format, no prefix (H12)")
    ap.add_argument("--speed-runs", type=int, default=5)
    ap.add_argument("--control-docs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke: the first N samples of each set only")
    ap.add_argument("--deadline", type=float, default=0, help="epoch seconds: stop starting samples after it")
    ap.add_argument("--pace", type=float, default=0.0)
    a = ap.parse_args(argv)
    cl = Client(a.base_url, a.kva, a.tail, a.pace)
    if not cl.healthy():
        raise SystemExit(f"{a.base_url}: not healthy")
    tok = tokenizer()
    out = Path(a.out) / a.arm
    sets = [s for s in a.sets.split(",") if s]
    lim = (lambda xs: xs[:a.limit]) if a.limit else (lambda xs: xs)

    def late():
        return a.deadline and time.time() > a.deadline

    for s in sets:
        if s in ("ppl", "control"):
            docs, skipped = ppl_docs(tok, a.ppl_max_len, a.ppl_dataset, [x for x in a.ppl_domains.split(",") if x])
            d = out / (f"ppl-{a.pass_}" + (f"-{a.ppl_dataset}" if a.ppl_dataset else "") if s == "ppl" else "control")
            write_json(d / "_set.json", dict(set=s, docs=[q["name"] for q in docs], skipped=skipped,
                                             max_len=a.ppl_max_len, dataset=a.ppl_dataset or "locked"))
            todo = lim(docs) if s == "ppl" else docs[:a.control_docs]
            for k, q in enumerate(todo):
                path = d / f"{q['name']}.json"
                if done(path):
                    continue
                if late():
                    print(f"deadline: {s} stopped before {q['name']}", flush=True)
                    return 3
                rec = run_ppl(cl, q, f"c5-{a.arm}-{s}-{a.pass_}-{q['name']}", kva=(0 if s == "control" else None))
                rec.update(arm=a.arm, pass_=a.pass_, control=(s == "control"))
                write_json(path, rec)
                print(f"  {s} {k + 1}/{len(todo)} {q['name']} n={rec['n']} nll_novel={rec['nll_novel']:.4f} "
                      f"{rec['wall_s']:.1f}s", flush=True)
        elif s == "ruler":
            lengths = [int(x) for x in a.ruler_lengths.split(",")]
            rows, skipped = ruler_rows(tok, lengths, a.ruler_tasks.split(","), a.ruler_format, a.ruler_prefix)
            d = out / f"ruler-{a.pass_}"
            write_json(d / "_set.json", dict(set="ruler", rows=[q["name"] for q in rows], skipped=skipped,
                                             tail=a.tail, prefix=a.ruler_prefix, eligible=sum(all(e <= len(q["ids"]) - a.tail for _, e in q["needles"])
                                                                       for q in rows)))
            for k, q in enumerate(lim(rows)):
                path = d / f"{q['name']}.json"
                if done(path):
                    continue
                if late():
                    print(f"deadline: ruler stopped before {q['name']}", flush=True)
                    return 3
                rec = run_ruler(cl, q, f"c5-{a.arm}-ruler-{a.pass_}-{q['name']}")
                rec.update(arm=a.arm, pass_=a.pass_, prefix=a.ruler_prefix)
                write_json(path, rec)
                print(f"  ruler {k + 1}/{len(rows)} {q['name']} score={rec['score']:.2f} filled={rec['needle_filled']} "
                      f"{rec['wall_s']:.1f}s", flush=True)
        elif s == "speed":
            prompts, source = speed_prompts(tok)
            run_speed(cl, prompts, source, a.speed_runs, a.arm, out / "speed")
        else:
            raise SystemExit(f"unknown set {s!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
