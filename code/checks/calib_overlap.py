#!/usr/bin/env python3
"""Q6/Q7/Q13 (RidgeFill preprint questions, 2026-10-11): the calibration documents behind the paper's maps,
their token counts per model, and their overlap with the 126 locked long documents and the 30 fresh documents.

CPU only, read-only (writes only beside this script). No evaluation text is printed or written: counts, names and
hashes only.

Calibration sets (rebuilt from the committed / private plan files, checked against the fit records):
  ref-fit   the 71 documents of every reference projector (Llama-3.1-8B, Qwen3.5-9B, Qwen3-8B): plan-train.json
            kind 'calib' minus datasets/calibration-exclude.json 'train' (kva.data.calib_docs('train')); checked
            against results/calib-docs-qwen3-8b-sweep.json and against the projector metadata row counts.
  ref-held  the 5 lambda-selection documents (kva.data.calib_docs('held')).
  fn-fit    Flash-Next kva-big-s24 sums = FN-BIGCAP batches b0 + b1: raw 71 train + 80 impact ('big'),
            chat 22 FN-CAP2 chat copies ('chattrain') + 80 impact chat copies ('chatbig').
  fn-held   Flash-Next lambda choice: the 5 held documents + their 5 chat copies.
  fn-sterm  the 13 plan-sterm prompts of the Flash-Next GDN terminal-state correction shipped in the release.

Checks per model tokenizer (Llama / Qwen3.5 / Qwen3-8B for ref-*, Flash-Next for fn-*):
  (a) sha256: every calibration text (raw and normalised) against every evaluation file (bytes, text, normalised
      text, prompt part raw and normalised); also the gsq-v2 source-document sha256s of the manifests.
  (b) 8-token shingles: tokens = tokenizer(text, add_special_tokens=False) (no BOS), variant 'raw' (text exactly as
      read, as the model saw it) and 'norm' (NFKC, casefold, every whitespace run -> one space, stripped). A shingle
      is 8 consecutive token ids; matches are found by 64-bit hash and then verified token for token.
      Runs: maximal stretches of consecutive calibration positions whose shingle is shared; run length in tokens =
      positions + 7 (an upper bound on the longest common token string with one evaluation document).
"""
# Release copy: six absolute paths replaced by relative input placeholders.
# This historical scanner needs withheld texts, plans, tokenizers and manifests.
# Run from a prepared input directory; released maxima are recomputed from
# provenance/checks/calib-token-overlap.json by the recompute script.
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("inputs/kva-wt-paper-final")
DS = REPO / "datasets"
PRIV = Path("inputs/kva-private")
PLANS = Path("inputs/flashnext-bigcap/plans")
MODELS = Path("models")
TOK = {"llama-3.1-8b-instruct": MODELS / "llama-3.1-8b-instruct", "qwen3.5-9b": MODELS / "qwen3.5-9b",
       "qwen3-8b": MODELS / "qwen3-8b", "qwen3.8-flash-next": MODELS / "tcc-qwen38-flash-next-mxfp4-fp8-gptq"}
ROWS = {"llama-3.1-8b-instruct": 834541, "qwen3.5-9b": 839652, "qwen3-8b": 845043}   # projector metadata 'rows'
MAX_LEN = 32768                                           # kva.data.MAX_LEN
N = 8
OUT = Path(__file__).resolve().parent
CHAT_USER_OPEN = "<|im_start|>user\n"                     # b0/scoring.py
CHAT_ASSISTANT_OPEN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CHAT_ASKS = ("Summarize the main points of the text above.", "What is this text about? Answer briefly.",
             "List three facts stated in the text above.", "Continue the text above in the same style.")
FNCAP2_CHAT_EVERY = 3                                     # qfn/steps.py: "3 on 10-04: time box"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t).casefold()).strip()


def load(p):
    return json.loads(Path(p).read_text())


# ---- calibration sets ------------------------------------------------------------------------------
def calib_sets():
    excl = load(DS / "calibration-exclude.json")
    priv = {f: load(PRIV / f) for f in ("plan-train.json", "plan-held.json", "plan-sterm.json")}
    shared = {f: {x["name"]: x for x in load(DS / "graded-and-calibration-sets" / f)} for f in priv}
    train_all = [x for x in priv["plan-train.json"] if x["kind"] == "calib"]
    ref_fit = [x for x in train_all if x["name"] not in set(excl["train"])]
    held = [x for x in priv["plan-held.json"] if x["tokens"] >= 1024 and x["name"] not in set(excl["held"]) | {"EXCLUDED-DOC"}]
    sterm_all = priv["plan-sterm.json"]
    # the committed (shared) copy the Vast refits read must hold the same texts
    for f, docs in (("plan-train.json", ref_fit), ("plan-held.json", held), ("plan-sterm.json", sterm_all)):
        for x in docs:
            assert shared[f][x["name"]]["text"] == x["text"], (f, x["name"])
    q3 = load(REPO / "results" / "calib-docs-qwen3-8b-sweep.json")
    assert q3["docs"] == [x["name"] for x in ref_fit], "Qwen3-8B fit list differs from the clean list"
    # FN-CAP2's capture plan as of 10-04 (before the exclusion list): raw = 99 train + 5 held, chat copies of
    # every 3rd train doc and all held docs, ask i % 4 (qfn.steps.capture_plan / chat_text)
    raw = [("train", x) for x in train_all] + [("held", x) for x in held]
    chat = {(f"chat{k}", x["name"]): CHAT_USER_OPEN + x["text"] + "\n\n" + CHAT_ASKS[i % 4] + CHAT_ASSISTANT_OPEN
            for i, (k, x) in enumerate(raw) if k == "held" or i % FNCAP2_CHAT_EVERY == 0}
    b0 = load(PLANS / "PLAN-b0.json")
    b1 = load(PLANS / "PLAN-b1.json")
    by_name = {x["name"]: x for x in train_all}
    fn_fit = []
    for d in b0:
        if d["kind"] == "train":
            fn_fit.append(dict(name=f"train/{d['name']}", text=by_name[d["name"]]["text"], fmt="raw"))
        else:
            fn_fit.append(dict(name=f"chattrain/{d['name']}", text=chat[("chattrain", d["name"])], fmt="chat"))
    for d in b1:
        assert d["kind"] in ("big", "chatbig")
        fn_fit.append(dict(name=f"{d['kind']}/{d['name']}", text=d["text"], fmt="chat" if d["kind"] == "chatbig" else "raw",
                           domain=d["domain"]))
        if d["kind"] == "big":
            assert sha(d["text"]) == d["sha256"], d["name"]
    assert sum(d["fmt"] == "raw" for d in fn_fit) == 151 and sum(d["fmt"] == "chat" for d in fn_fit) == 102
    fn_held = [dict(name=f"held/{x['name']}", text=x["text"]) for x in held] + \
              [dict(name=f"chatheld/{x['name']}", text=chat[("chatheld", x["name"])]) for x in held]
    return dict(
        ref_fit=[dict(name=x["name"], text=x["text"], docs=x.get("docs"), domains=x.get("domains"),
                      fn_tokens=x.get("tokens")) for x in ref_fit],
        ref_held=[dict(name=x["name"], text=x["text"], kind=x["kind"]) for x in held],
        fn_fit=fn_fit, fn_held=fn_held,
        fn_sterm=[dict(name=x["name"], text=x["text"]) for x in sterm_all],
        excluded_train=excl["train"], excluded_sterm=excl["sterm"])


# ---- evaluation sets ------------------------------------------------------------------------------
def eval_set(root):
    man = {}
    for line in (root / "MANIFEST.tsv").read_text().splitlines()[1:]:
        f = line.split("\t")
        man[f[0]] = f[1]
    splits = {}
    for line in (root / "long-prompts" / "SPLITS.tsv").read_text().splitlines()[1:]:
        f = line.split("\t")
        splits[f[0]] = int(f[1])
    docs = []
    for p in sorted((root / "long-prompts").glob("*.txt")):
        rel = f"long-prompts/{p.name}"
        b = p.read_bytes()
        assert sha(b) == man[rel], rel
        t = p.read_text()                                   # as the harness reads it
        docs.append(dict(name=p.name, bytes_sha=sha(b), text=t, split=splits[rel]))
    return docs


# ---- shingles ----------------------------------------------------------------------------------------
P1 = np.uint64(0x9E3779B97F4A7C15)


def shingle_hashes(ids):
    """uint64 hash of every N-token window of ids (polynomial, wraps mod 2^64)."""
    a = np.asarray(ids, dtype=np.uint64)
    if len(a) < N:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(a) - N + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):
            h = h * P1 + a[k:len(a) - N + 1 + k] + np.uint64(1)
    return h


class EvalIndex:
    """Every N-token window of an evaluation set: hashes sorted, with document id and global token offset."""

    def __init__(self, eval_ids):
        self.names = list(eval_ids)
        toks, H, D, G, off = [], [], [], [], 0
        for d, n in enumerate(self.names):
            ids = eval_ids[n]
            h = shingle_hashes(ids)
            H.append(h)
            D.append(np.full(len(h), d, dtype=np.int32))
            G.append(off + np.arange(len(h), dtype=np.int64))
            toks.append(ids)
            off += len(ids)
        self.tok = np.concatenate(toks)
        H, D, G = np.concatenate(H), np.concatenate(D), np.concatenate(G)
        o = np.argsort(H, kind="stable")
        self.H, self.D, self.G = H[o], D[o], G[o]

    def lookup(self, ids):
        """(verified calibration positions, their hashes, {eval doc index}) for one calibration document."""
        h = shingle_hashes(ids)
        if len(h) == 0:
            return np.zeros(0, dtype=np.int64), h, set()
        i = np.searchsorted(self.H, h)
        i[i == len(self.H)] = 0
        cand = np.nonzero(self.H[i] == h)[0]
        if len(cand) == 0:
            return cand, h[cand], set()
        cw = np.lib.stride_tricks.sliding_window_view(ids, N)[cand]
        ew = self.tok[self.G[i[cand]][:, None] + np.arange(N)]
        ok = (cw == ew).all(1)                                   # exact token check against the first occurrence
        ver = cand[ok]
        hv = np.unique(h[ver])
        lo, hi = np.searchsorted(self.H, hv, "left"), np.searchsorted(self.H, hv, "right")
        docs = set()
        for a, b in zip(lo.tolist(), hi.tolist()):
            docs.update(np.unique(self.D[a:b]).tolist())
        return ver, hv, docs


def overlap(calib_ids, eval_ids):
    """calib_ids / eval_ids: {name: np.int64 ids}. Per calibration document: shared verified shingles, the eval
    docs it touches, and run statistics."""
    ix = EvalIndex(eval_ids)
    res = {}
    shared_global = set()
    eval_touched = Counter()
    for cn, ids in calib_ids.items():
        ver, hv, dset = ix.lookup(ids)
        if len(ver) == 0:
            continue
        shared_global |= set(hv.tolist())
        touched = sorted(ix.names[d] for d in dset)
        for en in touched:
            eval_touched[en] += 1
        runs = np.split(ver, np.nonzero(np.diff(ver) != 1)[0] + 1)
        lens = [len(r) + N - 1 for r in runs]
        res[cn] = dict(shared_positions=int(len(ver)), distinct_shared_shingles=len(hv), eval_docs=touched,
                       n_runs=len(runs), longest_run_tokens=int(max(lens)),
                       runs_ge_16=sum(x >= 16 for x in lens), runs_ge_32=sum(x >= 32 for x in lens),
                       runs_ge_50=sum(x >= 50 for x in lens))
    return dict(calib_docs_with_overlap=len(res), eval_docs_with_overlap=len(eval_touched),
                distinct_shared_shingles=len(shared_global),
                shared_positions=sum(r["shared_positions"] for r in res.values()),
                calib_docs_with_run_ge_16=sum(r["runs_ge_16"] > 0 for r in res.values()),
                calib_docs_with_run_ge_32=sum(r["runs_ge_32"] > 0 for r in res.values()),
                calib_docs_with_run_ge_50=sum(r["runs_ge_50"] > 0 for r in res.values()),
                longest_run_tokens=max([r["longest_run_tokens"] for r in res.values()] or [0]),
                per_calib_doc=res, per_eval_doc=dict(eval_touched))


def main():
    from transformers import AutoTokenizer
    t0 = time.time()
    cs = calib_sets()
    ev = {"locked126": eval_set(DS / "test-set-locked"), "fresh30": eval_set(DS / "fresh-long-20261008")}
    assert len(ev["locked126"]) == 126 and len(ev["fresh30"]) == 30
    log(f"calibration: ref_fit {len(cs['ref_fit'])}, ref_held {len(cs['ref_held'])}, fn_fit {len(cs['fn_fit'])}, "
        f"fn_held {len(cs['fn_held'])}, fn_sterm {len(cs['fn_sterm'])}; eval 126 + 30")
    report = dict(script=str(Path(__file__)), tokenizers={}, sha256={}, tokens={}, shingles={})

    # (a) sha256 ---------------------------------------------------------------------------------------
    ev_hashes = defaultdict(set)
    for es, docs in ev.items():
        for d in docs:
            for h in (d["bytes_sha"], sha(d["text"]), sha(norm(d["text"])), sha(d["text"][:d["split"]]),
                      sha(norm(d["text"][:d["split"]]))):
                ev_hashes[h].add(f"{es}/{d['name']}")
    sha_res = {}
    for cset in ("ref_fit", "ref_held", "fn_fit", "fn_held", "fn_sterm"):
        hits = [(d["name"], sorted(ev_hashes[h])) for d in cs[cset] for h in (sha(d["text"]), sha(norm(d["text"])))
                if h in ev_hashes]
        sha_res[cset] = dict(docs=len(cs[cset]), matches=hits)
    # gsq-v2 source documents (the packed calibration prompts' parts) by their manifest sha256
    srcs = {}
    for m in (Path("inputs/source-manifests/paro/manifest.json"),
              Path("inputs/source-manifests/gsq-v2/manifest.json")):
        for d in load(m)["documents"]:
            srcs.setdefault(d["id"], d)
    used = sorted({s for d in cs["ref_fit"] for s in (d["docs"] or [])})
    missing = [s for s in used if s not in srcs]
    sha_res["gsq_source_docs"] = dict(docs=len(used), not_in_manifests=missing,
                                      matches=[s for s in used if s in srcs and srcs[s]["sha256"] in ev_hashes])
    report["sha256"] = sha_res
    log("sha256 done: " + ", ".join(f"{k} {len(v['matches'])}" for k, v in sha_res.items()))

    # token counts and shingles per model --------------------------------------------------------------
    plan = {"llama-3.1-8b-instruct": ("ref_fit", "ref_held"), "qwen3.5-9b": ("ref_fit", "ref_held"),
            "qwen3-8b": ("ref_fit", "ref_held"), "qwen3.8-flash-next": ("fn_fit", "fn_held", "fn_sterm")}
    for model, sets in plan.items():
        tok = AutoTokenizer.from_pretrained(str(TOK[model]))
        report["tokenizers"][model] = dict(path=str(TOK[model]), tokenizer_json_sha256=sha((TOK[model] / "tokenizer.json").read_bytes()))
        bos = tok("a").input_ids[:1] if tok.bos_token_id is not None and tok("a").input_ids[0] == tok.bos_token_id else []
        enc = lambda texts: [np.asarray(x, dtype=np.int64) for x in tok(texts, add_special_tokens=False).input_ids]  # noqa: E731
        for variant in ("raw", "norm"):
            f = (lambda t: t) if variant == "raw" else norm
            eval_ids = {es: dict(zip([d["name"] for d in docs], enc([f(d["text"]) for d in docs]))) for es, docs in ev.items()}
            for cset in sets:
                docs = cs[cset]
                ids = dict(zip([d["name"] for d in docs], enc([f(d["text"]) for d in docs])))
                if variant == "raw":
                    n = {k: len(v) for k, v in ids.items()}
                    seen = {k: min(len(bos) + v, MAX_LEN) if cset.startswith("ref") else v for k, v in n.items()}
                    arr = np.array(list(seen.values()))
                    report["tokens"][f"{model}/{cset}"] = dict(
                        docs=len(arr), total=int(arr.sum()), min=int(arr.min()), p25=float(np.percentile(arr, 25)),
                        median=float(np.median(arr)), p75=float(np.percentile(arr, 75)), max=int(arr.max()),
                        bos=len(bos), capped=int(sum((len(bos) + v) > MAX_LEN for v in n.values())) if cset.startswith("ref") else None,
                        per_doc=seen)
                for es in ev:
                    r = overlap(ids, eval_ids[es])
                    report["shingles"][f"{model}/{cset}/{variant}/{es}"] = r
                    log(f"{model} {cset} {variant} vs {es}: calib docs {r['calib_docs_with_overlap']}/{len(docs)}, "
                        f"eval docs {r['eval_docs_with_overlap']}, distinct shingles {r['distinct_shared_shingles']}, "
                        f"longest run {r['longest_run_tokens']}, docs with run>=50 {r['calib_docs_with_run_ge_50']}")
        if model in ROWS:
            got = report["tokens"][f"{model}/ref_fit"]["total"]
            report["tokens"][f"{model}/ref_fit"]["projector_rows"] = ROWS[model]
            log(f"{model}: ref_fit tokens {got} vs projector rows {ROWS[model]} ({'MATCH' if got == ROWS[model] else 'DIFF'})")
    report["seconds"] = round(time.time() - t0, 1)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    log(f"wrote {OUT / 'report.json'} in {report['seconds']} s")


if __name__ == "__main__":
    sys.exit(main())
