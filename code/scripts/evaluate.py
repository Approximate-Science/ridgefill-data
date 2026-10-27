#!/usr/bin/env python3
"""Evaluate arms on the graded prompts (continuation NLL), the NIAH sets (greedy exact match) or the locked test set.

grade  per prompt: log-probs of the last 512 prompt tokens per arm (all inside every arm's exact
       tail), the novel-token mask, and the short-context reference (text[short_start:]). Prompt
       format per config: 'grade_format' raw | chat (those 512 tokens as the assistant's reply).
niah   per prompt: greedy continuation per arm (up to 48 tokens, stops after EOS), scored on the
       first standalone 7-digit number after removing any think block; records the token ids, the
       text with special tokens, finish_reason, P (approximated rows) and whether the needle lies
       inside them (needle_end <= P). Prompt format per config: 'niah_format' raw (K20) | chat.
niahg  the non-numeric NIAH dev set (kva/niahg.py: names, phrases, hex ids, identifiers, sentences),
       greedy as niah, scored per type; only prompts whose needle lies in every arm's approximated
       prefix (needle_end <= N - niahg_tail, default the largest arm tail; r15/GPU-PLAN.md runs the
       selector arms on projected needles only). Config: niahg_types, niahg_tail.
multiturn  (MT1; paper T1.8: multiturn_turns 2) the mk4 prompts as 4-turn chats over one carried cache: turn 1
       the prompt, turns 2-4 a question about each distractor key prefilled exactly on top of it, each answered greedily
       as niah; arms with "repair": r recompute the late layers after turn 1 (kva/multiturn.py). Needs
       niah_variant mk4; records per arm the turns (correct, text, needle_projected, ...) and the repair cost.
ruler, longbench, longppl  the locked test set (kva/locked.py; PROTOCOL.md 10.2): finals only, so the config
       needs a FINAL- registration. ruler and longbench: greedy per arm up to the task's official output
       length (stops after EOS), records gen, text, the prediction as scored (pred), its 0-1 score and
       the official metric (kva/metrics.py); the task score is 100 x the mean (RULER and LongBench round
       to 2 dp). longppl: log-probs of the continuation tokens (every arm's exact tail must hold them,
       checked before the run) and score = their mean NLL. Every record has the task, the nominal
       length and the source file's MANIFEST.tsv sha256. Config: ruler_max_len, longbench_max_len
       (required), longppl_max_len; rows over a cap are listed in the provenance's 'skipped'.
multiturn_turns (config, default 4) asks only the first k turns (k in 2..4): T1.8's second question is k = 2 (turn 1
       the target key, turn 2 the first distractor key, against the same cache: the prefix is filled once, with the
       rows chosen for turn 1's question).
multiturn_from (config, default "mk4"): "ruler" runs the multiturn set on the locked two-key RULER set (paper T1.8,
       kva/derived.py: niah_multikey_1 rows, turn 2 another key of the row in RULER's wording, every sample checked
       against datasets/derived/ruler-twokey.json): a locked set, so finals only (validated as the ruler set, with
       its ruler_max_len cap); multiturn_turns 2; up to 128 tokens a turn (RULER's niah output length).
ruler_variant (config, ruler set): "qfirst" is the question-before-document RULER prompt of T1.4 on the four NIAH
       tasks (kva/derived.py, checked against datasets/derived/ruler-qfirst.json); prompt names rulerqf-....
decode  (paper T1.7, kva/agree.py) per prompt the exact model and every arm decode decode_steps tokens (config,
       default 512) after the same prompt: the exact greedy tokens, then per arm a teacher-forced pass on them
       (per-step KL(exact || arm), forced top-1 agreement) and a free greedy pass (first divergence, share of identical
       positions). Prompts: the config's decode_from set ("grade" or "longppl", the latter a locked set: finals only)
       without its scored tokens (the last 512 / the continuation). The exact arm is recomputed with any missing arm
       (its per-step log-probs are not stored).
Key-token NLL (grade and longppl, paper T1.3, kva/keytok.py): with "keytoken_tail": T in the config, each prompt also
records rec["tail_only"] = the exact model's log-probs of the scored tokens given only [BOS] + the last T ids
(T, bos, logprobs); kva.keytok.readout() then scores key tokens (exact NLL rises by more than a threshold without the
prefix) next to the novel tokens (kva.data.novel_mask).
Paper arms (kva/kva.py header): nofill is fill "none"; pop-native is fill "identity" at kva.kva.pop_split(L) with
"tail_from": "continuation" (no "tail"): the exact tail is exactly the rows whose logits are scored, set per prompt
(grade 513, longppl the continuation + 1, generation sets 1: POP's last prompt token run as the first decode step);
recorded as run["tail_from"] and the real tail. swiftkv-native is fill "swiftkv_native" (SwiftKV's checkpoint as
released, no exact tail; on longppl its scored rows run SwiftKV's own late layers, so the exact-tail check exempts it).
An arm with a "sets" list runs on those sets only, else on every set.
Layers < split run once per prompt and are shared by the arms (the exact arm continues them).
Bypass arms ("select": {"kind": digits | regex | verbatim | qattn | pred | freq | frag | blend | start_recent |
keydiff | deviation | band | qfp | random | tail_match, ...}, kva/select.py): the selected prefix rows run exactly
through the late layers (tail_match, and start_recent past its first rows: the tail grows instead); each run records
the selection (select, n_rows, tail_extra, rows as ranges; blend: overlap; niah: needle_rows_exact; runs
with a freq, frag, regex or verbatim signal: digit_rows and digit_share of every selection, for the overlap with
the digit rule). pred / blend read "predictor" (scripts/fit_surprisal.py, same split; sha256 in
file_sha256) with the h_S rows of the shared lower pass; deviation the arm's projector or the select's
"projector"; keydiff the lower pass's cached keys; freq / frag the "table" of scripts/token_freq.py (sha256 in
file_sha256).
K/V fold arms (fill 'kvfold', or 'kvfold' in fill_by_type): "kvfold": the fit_kvfold.py file.
Fill-quality arms (kva/kva.py): "context": K with a scripts/fit_context.py projector (its metadata says K; the
shared lower pass keeps the mixer outputs of layers split-K .. split-1); "refresh": R with "refresh_projector" (a
plain projector of split R + 1), also with a "select" (FQ3's stack).
K/V splice arms (KV1, kva/kva.py): "exact_part": "k" | "v" takes that half of the late attention layers' prefix K/V
from the true inputs (kept as for oracle fills) and the other from the fill; recorded in the run.
Tail-bias arms (BV3, kva/kva.py): "tail_bias_from": L adds the exact tail's mean projector residual to the prefix
fill of every 'proj' layer >= L; "tail_bias_skip_last": k keeps the last k prompt rows out of that mean (the grade
scores the last 512); both recorded in the run.
K/V rescale arms (KN1 / VN1, kva/kva.py): "rescale": knorm | kdir | vnorm (true inputs kept as for oracle fills) |
ktail | vtail rescales the late attention layers' filled prefix K or V; "rescale_skip_last": k keeps the last k
prompt rows out of ktail / vtail's tail scale; both recorded in the run.
"drop_bos": true (grade only; SINK1, off-distribution) drops each prompt's first token, the BOS, keeping the scored
tokens and their novel mask (data.grade_prompts); recorded per prompt (bos_dropped).
Oracle fills (fill 'oracle', or 'oracle' in fill_by_type) read the true late-layer inputs, kept from
the exact arm's pass (rows [0, P), layers >= the arm's split, model dtype, host RAM).
Raw output: WORK/results/<set>-<config name>/<prompt>.json (resumable: missing arms only), written atomically and
read back (prov.write_json).
Host RAM hardening (2026-10-03): the model's files are checked against their SHA256SUMS after loading (model.load)
and every projector against the sha256 recorded when it was written (Projector.load); then a canary prompt
(kva/canary.py) must reproduce its stored logits before any result is written (bitwise; within 0.02 log-prob for
models with known numeric variants, Qwen3.5). --canary accept adds this run's canary result as an accepted variant,
--canary reset replaces the stored ones (an intended engine change).
Usage: evaluate.py CONFIG.json {grade,niah,niahg,multiturn,decode,ruler,longbench,longppl} [--only NAME,...] [--limit N]
       [--canary check|accept|reset]
"""
import argparse
import dataclasses
import hashlib
import os
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kva import gpu  # noqa: E402
from kva import locked, paths

KEEP = 513           # rows whose logits score the last 512 prompt tokens
NIAH_STEPS = 48      # orchestrator 2026-10-01: >= 48 (think/special tokens can eat a small budget)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


SETS = ("grade", "niah", "niahg", "multiturn", "decode") + locked.SETS
DECODE_FROM = ("grade", "longppl")
MULTITURN_FROM = ("mk4", "ruler")    # the dev mk4 prompts, or the locked two-key RULER set (T1.8)
DECODE_STEPS = 512   # paper T1.7
TURN_ROWS = NIAH_STEPS + 128   # multiturn: bounds one later turn's rows (question ~30-60 ids, answer <= 48)


def _complete(run, which):
    return {"grade": "argmax", "longppl": "logprobs", "multiturn": "turns", "decode": "gen"}.get(which, "text") in run


def check_reuse(specs):
    """A reuse_fill arm takes its source's late-layer caches as they are, so both must fill the same
    rows the same way: same split, tail, fill and maps, and no select (tail_match grows the tail, so P
    differs; exact rows change the filled caches), tail_bias_from or rescale (finish() refuses fill reuse with
    them; ktail / vtail read the arm's own tail)."""
    by_name = {s["name"]: s for s in specs}
    for s in specs:
        src = by_name.get(s.get("reuse_fill"))
        if src is None:
            continue
        diff = [k for k in ("split", "tail", "tail_round", "tail_from", "fill", "fill_by_type", "projector", "kvfold",
                            "select", "context", "refresh", "refresh_projector", "tail_bias_from",
                            "tail_bias_skip_last", "exact_part", "rescale", "rescale_skip_last")
                if s.get(k) != src.get(k) or (k == "select" and s.get(k))
                or (k in ("tail_bias_from", "rescale") and s.get(k) is not None)]
        if diff:
            raise ValueError(f"arm {s['name']} reuses the fill of {src['name']} but its P or fill can differ "
                             f"({', '.join(diff)})")


def check_tail_round(specs):
    """tail_round (protocol.json engine_tail: the engine's block-rounded tail, used on the reference side of an engine
    comparison) is refused unless base is the arm's tail and at least the engine's minimum tail (kva.kva.ENGINE_MIN_TAIL)
    and block is a positive integer; and always with a select (the rows would be picked against the unrounded
    boundary, and the rounding would overwrite tail_match's extra tail; v1.1 draft A.1) or tail_from."""
    from kva.kva import ENGINE_MIN_TAIL
    for s in specs:
        tr = s.get("tail_round")
        if tr is None:
            continue
        bad = []
        if not (isinstance(tr, dict) and set(tr) == {"base", "block"} and type(tr["base"]) is int
                and type(tr["block"]) is int):
            bad.append(f"{tr!r} is not {{'base': int, 'block': int}}")
        else:
            if tr["base"] != s.get("tail") or tr["base"] < ENGINE_MIN_TAIL:
                bad.append(f"base {tr['base']} must equal the arm's tail ({s.get('tail')}) and be at least the "
                           f"engine's minimum tail {ENGINE_MIN_TAIL}")
            if tr["block"] < 1:
                bad.append(f"block {tr['block']} must be positive")
        if s.get("select"):
            bad.append("with a select the rows would be picked against the unrounded boundary")
        if s.get("tail_from") is not None:
            bad.append("tail_from sets the tail per prompt")
        if bad:
            raise SystemExit(f"arm {s['name']}: tail_round refused: " + "; ".join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("set", choices=SETS)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--canary", choices=("check", "accept", "reset"), default="check")
    args = ap.parse_args()
    from kva import protocol
    cfg = paths.expand(json.loads(Path(args.config).read_text()))
    source = args.set
    if args.set == "decode":
        source = cfg.get("decode_from")
        if source not in DECODE_FROM:
            raise SystemExit(f"the decode set needs 'decode_from' in {DECODE_FROM}, got {source!r}")
    mt_from = cfg.get("multiturn_from", "mk4")
    if mt_from not in MULTITURN_FROM:
        raise SystemExit(f"multiturn_from {mt_from!r}: one of {MULTITURN_FROM}")
    if args.set == "multiturn" and mt_from == "ruler":
        source = "ruler"                               # locked rows: validated as the ruler set (finals only)
    if cfg.get("dataset") is not None and args.set != "longppl":   # another frozen set (kva/locked.py DATASETS)
        raise SystemExit(f"dataset {cfg['dataset']!r} replaces the locked set for longppl only, not {args.set!r}")
    if os.environ.get("KVA_CPU_DRYRUN") == "1" and "protocol" not in cfg:
        stamp = None                                   # CPU dry runs of pre-protocol configs: code paths only
        log("dry run of a pre-protocol config: no protocol stamp")
    else:
        stamp = protocol.validate(cfg, set_name=source)     # paper/PROTOCOL.md: fails before the GPU is opened
    if cfg.get("drop_bos") and args.set != "grade":
        raise SystemExit("drop_bos runs on the grade set only (SINK1)")
    # set-specific config keys (a config may serve several sets; each key acts on its own set only)
    key_tail = cfg.get("keytoken_tail")
    if key_tail is not None and not isinstance(key_tail, int):
        raise SystemExit(f"keytoken_tail {key_tail!r}: an integer (grade and longppl)")
    key_tail = key_tail if args.set in ("grade", "longppl") else None
    turns = cfg.get("multiturn_turns", 4)
    if turns not in (2, 3, 4):
        raise SystemExit(f"multiturn_turns {turns!r}: 2, 3 or 4 (the multiturn set)")
    steps_dec = cfg.get("decode_steps", DECODE_STEPS)
    if not isinstance(steps_dec, int) or steps_dec < 1:
        raise SystemExit(f"decode_steps {steps_dec!r}: a positive integer (the decode set)")
    device, gpu_info = gpu.setup()
    import torch
    from kva import agree as AG
    from kva import canary, data, integrity, keytok, niahg, prov
    from kva import multiturn as MT
    from kva import select as SEL
    from kva import model as M
    from kva.kva import Arm, KVFold, Projector, check_finite, exclude_scored, finish, greedy, oracle_tap, run_lower, score_tokens, rounded_tail
    from kva.pred import Predictor, load_freq, prefix_scores
    from kva import impact as IM

    if device.type == "cuda" and os.environ.get("KVA_MEMLOG"):
        # diagnostics (VRAM alerts 2026-10-01): every 0.5 s, torch allocated/reserved and the device-wide
        # free memory (hipMemGetInfo), which also sees runtime pools outside torch's allocator
        import threading
        memlog = open(os.environ["KVA_MEMLOG"], "a")

        def sample():
            while True:
                free, total = torch.cuda.mem_get_info()
                memlog.write(f"{time.time():.2f} alloc {torch.cuda.memory_allocated() / 2**20:.0f} "
                             f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} device_used {(total - free) / 2**20:.0f}\n")
                memlog.flush()
                time.sleep(0.5)
        threading.Thread(target=sample, daemon=True).start()
    model_dir = Path(cfg["model"])
    out_dir = prov.WORK / "results" / f"{args.set}-{cfg['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = data.tokenizer(cfg.get("tokenizer", cfg["model"]))
    m = M.load(model_dir, device=device)
    log(f"loaded {model_dir.name}: {gpu.used_gib(device):.1f} GiB")
    ref = canary.reference_path(prov.WORK / "results", model_dir.name, stamp["version"] if stamp else "none", gpu_info)
    log(canary.check(m, tok, ref, gpu_info, args.canary, canary.tolerance(model_dir.name)))

    projectors, sterms, hashes = {}, {}, {}

    def load_proj(cls, p):
        """Projector or KVFold p (kept on the CPU: one projector on the GPU at a time, sweeps), its bytes checked
        against the sha256 recorded when it was written."""
        if p not in projectors:
            projectors[p] = cls.load(p, "cpu", sha256=integrity.recorded_sha256(p, prov.WORK / "results"))
            hashes[p] = projectors[p].sha256
        return projectors[p]

    def arm_of(spec):
        tf = spec.get("tail_from")
        if tf is not None and (tf != "continuation" or "tail" in spec or spec.get("select") or spec.get("tail_round")
                               or spec.get("reuse_fill")):
            raise SystemExit(f"arm {spec['name']}: tail_from 'continuation' sets the tail per prompt: no 'tail', "
                             f"select, tail_round or reuse_fill")
        a = Arm(spec["name"], split=spec.get("split"), tail=spec.get("tail", 0), fill=spec["fill"], tail_from=tf,
                f1=spec.get("f1", False), fill_by_type=spec.get("fill_by_type"),
                tail_bias_from=spec.get("tail_bias_from"), tail_bias_skip_last=spec.get("tail_bias_skip_last", 0),
                exact_part=spec.get("exact_part"), rescale=spec.get("rescale"),
                rescale_skip_last=spec.get("rescale_skip_last", 0), meta=spec)
        if spec.get("projector"):
            p = spec["projector"]
            a.projector = load_proj(Projector, p)
            assert a.projector.split == a.split, (spec, a.projector.split)
            a.context = spec.get("context", 0)
            if int(a.projector.meta.get("context", 0)) != a.context:
                raise SystemExit(f"arm {a.name}: context {a.context}, but {p} reads context "
                                 f"{a.projector.meta.get('context', 0)}")
        if spec.get("refresh") is not None:
            p = spec["refresh_projector"]
            a.refresh, a.refresh_projector = spec["refresh"], load_proj(Projector, p)
            if a.refresh_projector.split != a.refresh + 1 or int(a.refresh_projector.meta.get("context", 0)):
                raise SystemExit(f"arm {a.name}: refresh {a.refresh} needs a plain projector of split {a.refresh + 1}")
        if spec.get("kvfold"):
            a.kvfold = load_proj(KVFold, spec["kvfold"])
            assert a.kvfold.split == a.split, (spec, a.kvfold.split)
        if spec.get("st"):
            p = spec["st"]
            if p not in sterms:
                sterms[p] = {int(k): v.to(device) for k, v in torch.load(p).items()}
                hashes[p] = prov.sha256(p)
            a.st = sterms[p]
        return a

    specs = [s for s in cfg["arms"] if args.set in s.get("sets", SETS)]
    check_reuse(specs)
    check_tail_round(specs)
    MT.check_arms(specs, args.set)
    arms = [arm_of(s) for s in specs]
    by_name = {a.name: a for a in arms}
    sels = [a.meta["select"] for a in arms if a.meta.get("select")]
    for sel in sels:
        sig = SEL.signals(sel)
        if (sel["kind"] not in SEL.KINDS or (sel.get("match") in by_name) != (sel["kind"] in ("random", "tail_match"))
                or bool(sel.get("predictor")) != ("pred" in sig) or bool(sel.get("table")) != bool(sig & {"freq", "frag"})):
            raise SystemExit(f"bad select spec {sel}")
    predictors, freqs = {}, {}
    for a in arms:
        t = a.meta.get("select", {}).get("table")
        if t and t not in freqs:
            freqs[t] = load_freq(t)
            hashes[t] = prov.sha256(t)
        p = a.meta.get("select", {}).get("predictor")
        if p and p not in predictors:
            predictors[p] = Predictor.load(p)
            hashes[p] = prov.sha256(p)
        if p and predictors[p].split != a.split:
            raise SystemExit(f"arm {a.name}: predictor {p} is for split {predictors[p].split}, not {a.split}")
        sel = a.meta.get("select")
        if sel and "deviation" in SEL.signals(sel):
            q = sel.get("projector")
            dev_proj = load_proj(Projector, q) if q else a.projector
            if dev_proj is None or dev_proj.split != a.split:
                raise SystemExit(f"arm {a.name}: deviation needs a projector of split {a.split}")
    lexical = any(SEL.signals(sel) & {"freq", "frag", "regex", "verbatim"} for sel in sels)
    digit_set = SEL.digit_ids(tok) if lexical or any(sel["kind"] == "digits" for sel in sels) else None
    breaks = SEL.word_breaks(tok) if any("frag" in SEL.signals(sel) for sel in sels) else None
    tbytes = SEL.token_bytes(tok) if any(SEL.signals(sel) & {"regex", "verbatim"} for sel in sels) else None
    exact_split = max(a.split for a in arms if a.split is not None)
    splits = sorted({a.split for a in arms if a.split is not None})
    run_cfg = dict(config=cfg, protocol=stamp, config_file=str(Path(args.config).resolve()), set=args.set, keep=KEEP,
                   niah_steps=NIAH_STEPS, file_sha256=hashes)

    fmt = cfg.get("niah_format", "raw")
    if args.set == "decode":                       # the source set's prompts without their scored tokens
        if source == "grade":
            src = data.grade_prompts(tok, cfg.get("grade_format", "raw"))
        else:
            src, run_cfg["skipped"] = locked.prompts(source, tok, cfg, model_dir.name, protocol.load())
            run_cfg["longppl_max_len"] = cfg.get("longppl_max_len")
        prompts = []
        for q in src:
            cut = data.SCORED if source == "grade" else q["cont_len"]
            prompts.append(dict(name=q["name"], ids=q["ids"][:len(q["ids"]) - cut], decode_from=source, cut=cut,
                                **{k: q[k] for k in ("kind", "task", "length", "source", "source_sha256", "format")
                                   if k in q}))
        run_cfg.update(decode_from=source, decode_steps=steps_dec)
    elif args.set == "grade":
        prompts = data.grade_prompts(tok, cfg.get("grade_format", "raw"), drop_bos=bool(cfg.get("drop_bos")))
    elif args.set == "niah":
        prompts = data.niah_prompts(tok, fmt, cfg.get("niah_variant"))
    elif args.set == "multiturn" and mt_from == "ruler":
        from kva import derived
        if turns != 2:
            raise SystemExit("the two-key RULER set has 2 turns: multiturn_turns 2")
        fmt = locked.fmt_of(cfg, model_dir.name, protocol.load())
        prompts, run_cfg["skipped"] = derived.prompts("twokey", tok, fmt, locked.manifest(), cfg.get("ruler_max_len"))
        run_cfg.update(multiturn_from="ruler", ruler_max_len=cfg.get("ruler_max_len"),
                       derived_manifest_sha256=prov.sha256(derived.manifest_path("twokey")))
        if run_cfg["skipped"]:
            log(f"multiturn (two-key RULER): skipped over the length cap: {', '.join(run_cfg['skipped'])}")
    elif args.set == "multiturn":
        if cfg.get("niah_variant") != "mk4":
            raise SystemExit("the multiturn set needs niah_variant mk4 (the distractor keys are the later turns)")
        prompts = data.niah_prompts(tok, fmt, "mk4")
    elif args.set in locked.SETS:
        prompts, run_cfg["skipped"] = locked.prompts(args.set, tok, cfg, model_dir.name, protocol.load())
        for k in ("ruler_max_len", "longbench_max_len", "longppl_max_len", "ruler_max_new"):
            run_cfg[k] = cfg.get(k)
        if cfg.get("dataset") is not None:
            run_cfg.update(dataset=cfg["dataset"], dataset_manifest_sha256=cfg["dataset_manifest_sha256"])
        if args.set == "ruler" and cfg.get("ruler_variant"):
            from kva import derived
            run_cfg.update(ruler_variant=cfg["ruler_variant"],
                           derived_manifest_sha256=prov.sha256(derived.manifest_path(cfg["ruler_variant"])))
        if run_cfg["skipped"]:
            log(f"{args.set}: skipped over the length cap: {', '.join(run_cfg['skipped'])}")
    else:
        trim = run_cfg["niahg_tail"] = cfg.get("niahg_tail", max(a.tail for a in arms if a.split is not None))
        prompts = [p for p in niahg.prompts(tok, fmt, cfg.get("niahg_types", niahg.TYPES))
                   if p["needle_end"] <= len(p["ids"]) - trim]
    provenance = prov.provenance(model_dir, run_cfg, gpu_info)
    eos = data.eos_ids(cfg.get("tokenizer", cfg["model"]), tok)
    only = args.only or cfg.get("only", "")
    if only:
        prompts = [p for p in prompts if p["name"] in only.split(",")]
    if args.limit:
        prompts = prompts[:args.limit]
    if args.set == "longppl":
        locked.check_tails(prompts, arms)
    for q in prompts:                             # RoPE positions past the model's window are never run
        steps_q = q.get("max_new", NIAH_STEPS)
        gen = (0 if args.set in ("grade", "longppl") else steps_dec if args.set == "decode"
               else (turns - 1) * (TURN_ROWS - NIAH_STEPS + steps_q) + steps_q if args.set == "multiturn"
               else steps_q)
        if m.max_positions and len(q["ids"]) + gen > m.max_positions:
            raise SystemExit(f"PositionsExceeded: {q['name']}: {len(q['ids'])} prompt ids + up to {gen} more exceed "
                             f"{model_dir.name}'s max_position_embeddings {m.max_positions} (cap the set's length, "
                             f"e.g. ruler_max_len / longppl_max_len, or use a model with scaled RoPE)")
    tok_bos = data.bos_ids(tok)

    def bos_of(ids):                              # the prompt's own BOS, kept in front of a tail-only context
        return ids[:1] if tok_bos and ids[0] == tok_bos[0] else []
    if key_tail is not None:                      # every prompt must leave a prefix to remove and score every token
        run_cfg["keytoken_tail"] = key_tail
        for q in prompts:
            scored = data.SCORED if args.set == "grade" else q["cont_len"]
            try:
                keytok.check_tail(len(q["ids"]), scored, key_tail, bos_of(q["ids"]))
            except ValueError as e:
                raise SystemExit(f"{q['name']}: {e}")
    t_all = time.time()
    for p in prompts:
        path = out_dir / f"{p['name']}.json"
        rec = json.loads(path.read_text()) if path.exists() else None
        want = ["exact"] + [a.name for a in arms]
        ids_sha = hashlib.sha256(json.dumps(p["ids"]).encode()).hexdigest()
        if rec and rec["ids_sha256"] == ids_sha and all(n in rec["runs"] and _complete(rec["runs"][n], args.set)
                                                        for n in want) \
                and (key_tail is None or rec.get("tail_only", {}).get("T") == key_tail):
            continue
        ids = torch.tensor(p["ids"])
        n = ids.shape[0]
        if rec is None or rec["ids_sha256"] != ids_sha:
            rec = dict(name=p["name"], n=n, ids_sha256=ids_sha, runs={}, provenance=[],
                       **{k: v for k, v in p.items() if k not in ("ids", "short_ids", "name", "asks")})
        rec["provenance"].append(provenance)
        t0 = time.time()
        if args.set == "multiturn":
            with torch.no_grad():
                msg = MT.run_prompt(m, tok, p, arms, rec, fmt, eos, p.get("max_new", NIAH_STEPS), digit_set, turns)
            prov.write_json(path, rec)
            log(f"{p['name']} n={n} {time.time() - t0:.1f}s peak {gpu.peak_gib(device):.1f} GiB | {msg}")
            continue
        keep = KEEP if args.set == "grade" else p["cont_len"] + 1 if args.set == "longppl" else 1
        # v1.1.3: on the sets that score prompt tokens teacher-forced, the last keep - 1 rows are the targets; no
        # adaptive decision (qattn window, tail-measured rescale or bias) may read them
        scored = keep - 1 if args.set in ("grade", "longppl") else 0
        known = n - scored
        with torch.no_grad():
            todo = [Arm("exact", split=exact_split)] + sorted(arms, key=lambda x: -x.split)
            todo = [a for a in todo if not (a.name in rec["runs"] and _complete(rec["runs"][a.name], args.set))]
            todo = [dataclasses.replace(a, tail=keep) if a.tail_from else a for a in todo]   # pop-native's tail
            todo = [exclude_scored(a, scored) for a in todo]
            if args.set == "decode" and todo and todo[0].name != "exact":   # arms compare with the exact decode
                todo.insert(0, Arm("exact", split=exact_split))
            ref_dec = None
            # matched arms (random, tail_match) need their reference arm's selection on this prompt
            refs = [by_name[a.meta["select"]["match"]] for a in todo if "match" in a.meta.get("select", {})]
            oracle, tap = None, None
            if any(a.uses_oracle() for a in todo):
                if todo[0].name != "exact":                    # the exact pass supplies the true inputs
                    todo.insert(0, Arm("exact", split=exact_split))
                oracle = {}
                tap = oracle_tap(oracle, min(a.split for a in todo if a.uses_oracle()),
                                 max(a.prefix_len(n) for a in todo if a.uses_oracle()))
            need = sorted({a.split for a in todo + refs})
            ends = {known}                                     # window ends: known (n on generation sets), or P for window "prefix"
            for b in todo + refs:
                sel = b.meta.get("select", {})
                if sel and "qattn" in SEL.signals(sel) and sel.get("window") == "prefix":
                    ends.add(b.prefix_len(n))
            wins = {end: {} for end in ends}
            want_win = any("qattn" in SEL.signals(b.meta["select"]) for b in todo + refs if b.meta.get("select"))
            recv_P = sorted({b.prefix_len(n) for b in todo + refs if b.meta.get("select")
                             and "qfp" in SEL.signals(b.meta["select"])})
            recv = {}                                     # (P, layer) -> mass each row receives from later prefix rows
            pos_all = torch.arange(n, device=m.device)

            def lower_hook(i, h):
                if tap is not None:
                    tap(i, h)
                if want_win:
                    for end, win in wins.items():
                        win[i] = h[max(0, end - SEL.WINDOW):end].clone()
                if recv_P and isinstance(i, int) and m.layer_types[i] == "attn":
                    x = m.in_norm(h, m.W[i]["ln1"])
                    q, k = m.queries(i, x, pos_all), IM.kv_of(m, i, x, pos_all)[0]
                    for P_ in recv_P:
                        recv[(P_, i)] = IM.received_mass(q, k, m.scale, rows_below=P_,
                                                         stride=int(cfg.get("qfp_stride", 1)))[1]
            mixes = {}
            mix_layers = {a.split - k for a in todo for k in range(1, a.context + 1)}
            hs, lower = run_lower(m, ids, set(need) if need else {exact_split},
                                  hs_device="cpu" if len(need) > 2 else None,
                                  hook=lower_hook if (tap is not None or want_win or recv_P) else None,
                                  mix=(lambda i, x: mixes.__setitem__(i, x) if i in mix_layers else None)
                                  if mix_layers else None)

            def recv_of(split, P_):
                low = sum(v for (pp, i), v in recv.items() if pp == P_ and i < split)
                return SEL.per_later_token(low, P_)
            masses = {}

            def mass_of(split, layers, end):
                key = (tuple(SEL.qattn_layers(m, split, layers)), end)
                if key not in masses:
                    masses[key] = SEL.window_mass(m, wins[end], lower, key[0], end)
                return masses[key]
            scores, emb = {}, []

            def pred_of(split, path, P):
                if not emb:
                    emb.append(m.embed(ids))
                if (split, path, P) not in scores:
                    scores[split, path, P] = prefix_scores(predictors[path], hs[split], emb[0], P)
                return scores[split, path, P]

            def keys_of(split, layers, P):
                key = ("keydiff", tuple(SEL.qattn_layers(m, split, layers)), P)
                if key not in scores:
                    scores[key] = SEL.key_diff(lower, key[1], P)
                return scores[key]

            def dev_of(a, P):
                q = a.meta["select"].get("projector")
                proj = projectors[q] if q else a.projector
                key = ("deviation", id(proj), a.split, P)
                if key not in scores:
                    scores[key] = SEL.fill_deviation(m, proj, hs[a.split], a.split, P)
                return scores[key]
            picks = SEL.choose(todo + refs, n, p["ids"], p["name"], digit_set, mass_of, pred_of, keys_of, dev_of,
                               freqs.get, breaks, recv_of, tbytes, known=known)
            del emb
            fills = {}
            for j, a in enumerate(todo):
                old = rec["runs"].get(a.name)
                t1 = time.time()
                for pr in projectors.values():
                    if pr is not a.projector and pr is not a.kvfold and pr is not a.refresh_projector:
                        pr.release()
                store = {} if any(b.meta.get("reuse_fill") == a.name for b in todo[j + 1:]) else None
                reuse = fills.pop(a.meta.get("reuse_fill"), None)
                pick = picks.get(a.name)
                run_arm = dataclasses.replace(a, tail=a.tail + pick["extra_tail"]) if pick else a
                if a.meta.get("tail_round"):                     # block-rounded tail (paper/YARDSTICK.md)
                    tr = a.meta["tail_round"]
                    run_arm = dataclasses.replace(run_arm, tail=rounded_tail(n, tr["base"], tr["block"]))
                logits, cache, P = finish(m, ids, hs[a.split], lower, run_arm, keep, keep_cache=keep == 1,
                                          fill_store=store, fill_reuse=reuse, oracle=oracle,
                                          exact_rows=pick["rows"] if pick else None,
                                          hook=tap if a.name == "exact" else None, mix=mixes)
                check_finite(logits, f"{p['name']} {a.name}")
                if store is not None:
                    fills[a.name] = store
                del reuse
                run = dict(P=P, split=a.split, tail=n - P, fill=a.fill, st=bool(a.st), f1=a.f1)
                if a.tail_from:
                    run["tail_from"] = a.tail_from
                if a.context or a.refresh is not None:
                    run.update(context=a.context, refresh=a.refresh)
                if a.tail_bias_from is not None:
                    run.update(tail_bias_from=a.tail_bias_from, tail_bias_skip_last=a.tail_bias_skip_last)
                if a.exact_part is not None:
                    run["exact_part"] = a.exact_part
                if a.rescale is not None:
                    run.update(rescale=a.rescale, rescale_skip_last=a.rescale_skip_last)
                if pick:
                    run.update(select=pick["kind"], n_rows=len(pick["rows"]), tail_extra=pick["extra_tail"],
                               rows=SEL.ranges(pick["rows"]))
                    if "overlap" in pick:
                        run["overlap"] = pick["overlap"]
                    if "window_end" in pick:
                        run["window_end"] = pick["window_end"]
                    if lexical:
                        run["digit_rows"] = sum(int(p["ids"][r]) in digit_set for r in pick["rows"])
                        run["digit_share"] = round(run["digit_rows"] / len(pick["rows"]), 4) if pick["rows"] else None
                    if "needle_start" in p:
                        run["needle_rows_exact"] = sum(p["needle_start"] <= r < p["needle_end"] for r in pick["rows"])
                    if args.set == "niahg":
                        run["value_rows_exact"] = sum(p["value_start"] <= r < p["value_end"] for r in pick["rows"])
                if a.fill_by_type:
                    run["fill_by_type"] = a.fill_by_type
                if args.set in ("grade", "longppl"):
                    run["logprobs"] = [round(float(x), 6) for x in score_tokens(logits, ids, keep - 1)]
                    if args.set == "grade":
                        run["argmax"] = logits[:-1].argmax(-1).tolist()  # teacher-forced top-1 (draft/verify alpha)
                    else:
                        run.update(score=locked.nll(run["logprobs"]), metric="nll")
                    if old is not None:                                   # re-score pass: reproducibility check
                        run["rescore_max_abs_logprob_diff"] = max(abs(x - y) for x, y in zip(run["logprobs"], old["logprobs"]))
                        run["first_run"] = {k: v for k, v in old.items() if k in ("logprobs", "wall_s")}
                elif args.set == "decode":
                    if a.name == "exact":
                        y, lp = AG.decode(m, n, cache, logits, steps_dec, a, logprobs=True)
                        ref_dec = (y, lp)
                        run.update(gen=y, first_eos=next((t for t, x in enumerate(y) if x in eos), None))
                    else:
                        run.update(AG.compare(m, n, cache, logits, steps_dec, a, *ref_dec))
                else:
                    gen, finish_reason = greedy(m, ids, cache, logits, p.get("max_new", NIAH_STEPS), a, eos)
                    text = tok.decode(gen, skip_special_tokens=False)
                    run.update(gen=gen, text=text, finish_reason=finish_reason)
                    if args.set in locked.GEN_SETS:              # the official pipelines decode without specials
                        run.update(locked.score(p, tok.decode(gen, skip_special_tokens=True)))
                    elif args.set == "niah":
                        run["needle_projected"] = p["needle_end"] <= P
                        run["correct"] = data.niah_correct(text, p["value"])
                    else:
                        run["needle_projected"] = p["needle_end"] <= P
                        run.update(niahg.score(text, p["value"], p["type"], fmt))
                run["wall_s"] = round(time.time() - t1, 3)
                if device.type == "cuda":
                    run["mem_gib"] = dict(alloc_peak=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                          reserved_peak=round(torch.cuda.max_memory_reserved() / 2**30, 2))
                    if os.environ.get("KVA_MEMLOG"):
                        log(f"  {a.name}: {run['mem_gib']} reserved now {torch.cuda.memory_reserved() / 2**30:.2f} GiB")
                    torch.cuda.reset_peak_memory_stats()
                rec["runs"][a.name] = run
                del logits, cache
                # free what no remaining arm reads: lower caches at/after the largest remaining split,
                # boundary states of splits no longer used
                rest = {b.split for b in todo[j + 1:]}
                top = max(rest) if rest else 0
                for i in range(top, len(lower)):
                    lower[i] = {}
                for S in [S for S in hs if S not in rest]:
                    del hs[S]
            if key_tail is not None and rec.get("tail_only", {}).get("T") != key_tail:
                scored = KEEP - 1 if args.set == "grade" else p["cont_len"]
                bos = bos_of(p["ids"])
                rec["tail_only"] = dict(T=key_tail, bos=bos, logprobs=[
                    round(float(x), 6) for x in keytok.exact_tail_only(m, p["ids"], key_tail, bos, scored)])
            if args.set == "grade" and "short" not in rec:
                s = torch.tensor(p["short_ids"])
                lg, _ = m.forward_exact(s, keep=torch.arange(s.shape[0] - KEEP, s.shape[0]))
                rec["short"] = dict(n=s.shape[0], tail_match=p["short_tail_match"],
                                    logprobs=[round(float(x), 6) for x in score_tokens(lg, s, KEEP - 1)])
            del hs, lower, fills, oracle, ref_dec
        # no empty_cache(): memory handed back by torch stays cached in the HIP runtime outside torch's
        # pool (KFD showed ~6 GiB above torch reserved, 2026-10-01), so keep it in torch's cache instead
        prov.write_json(path, rec)
        if args.set == "grade":
            nov = p["novel"]
            ex = rec["runs"]["exact"]["logprobs"]

            def nll(lp):
                xs = [-x for x, keep_ in zip(lp, nov) if keep_]
                return sum(xs) / len(xs)
            msg = " ".join(f"{k}:{nll(r['logprobs']) - nll(ex):+.4f}" for k, r in rec["runs"].items() if k != "exact")
        elif args.set == "longppl":
            ex = rec["runs"]["exact"]["score"]
            msg = " ".join(f"{k}:{r['score'] - ex:+.4f}" for k, r in rec["runs"].items() if k != "exact")
        elif args.set in locked.GEN_SETS:
            msg = " ".join(f"{k}:{r['score']:.2f}" for k, r in rec["runs"].items())
        elif args.set == "decode":
            msg = " ".join(f"{k}: same {r['frac_identical']:.3f} div@{r['first_divergence']} kl {r['kl_mean']:.4g}"
                           for k, r in rec["runs"].items() if k != "exact")
        else:
            msg = " ".join(f"{k}:{'ok' if r['correct'] else 'MISS'}{'*' if r.get('needle_projected') else ''}"
                           for k, r in rec["runs"].items())
        log(f"{p['name']} n={n} {time.time() - t0:.1f}s peak {gpu.peak_gib(device):.1f} GiB | {msg}")
    log(f"done {len(prompts)} prompts in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
