"""Multi-turn NIAH over a carried cache, and deferred repair (protocol v1.1 draft; paper/drafts/MULTITURN-PREREG.md).

In chat use a pasted document is prefilled once and cached, and several questions follow. The question-attention
selector only sees the first; this set asks whether an approximated cache stays usable for the rest, and whether
paying the skipped late-layer work back after the first answer (deferred repair, Dylan R18b) restores it.
Per mk4 prompt (data.niah_prompts variant mk4), the first n_turns (default 4; the paper's T1.8 second question
on a cached prefix is n_turns = 2: one prefill, the rows chosen for question 1, then question 2 on the same cache).
The paper's finals run T1.8 on the locked two-key RULER set instead (kva/derived.py, evaluate.py "multiturn_from":
"ruler"): its prompts carry their turns (turn_plan: key, value, needle span) and turn 2's text (asks: RULER's
question and answer prefix for the other key, ask_text_ids); everything else below is the same.
  turn 1    the mk4 prompt as is (the question about the target key), prefilled by the arm (kva.finish);
  turn 2-4  one per distractor key, in p["distractors"] order: the same question template as a new user turn,
            prefilled exactly on top of the carried cache (those few rows are the turn's exact tail), then
            greedy decode as NIAH (steps tokens, stops after EOS), scored with data.niah_correct.
History. Each arm keeps its own answer tokens, as a chat server with a persistent cache does: the rows decoded
in a turn stay cached. The answer's last token is not cached yet (greedy never feeds it); the next turn feeds it
first unless it is an EOS, which the template's turn closer replaces.
  chat  next turn = [last token] + closer (template text between an assistant reply and the next user message)
        + question (leading blank lines dropped: the templates trim message text) + assistant header;
  raw   next turn = [last token] + question (it starts with a blank line); no template.
So histories differ between arms by their answers (<= steps tokens a turn); each turn records its new ids.
Arms. Any evaluate.py arm with a plain finish() answers turn 1 (skip+qattn5 too: FastKV-style, the selected rows
are the only prefix rows with late K/V); later turns reuse its cache as it is ("no repair"; selected rows stay
those picked for turn 1's question). An arm with "repair": r is its base arm (the same spec without name, repair
and repair_by; turn 1 runs once per base and is shared) plus one repair after turn 1's answer, of the late
layers' cache (attention K/V; GDN conv and recurrent state). Fills proj and identity only.
  repair_by "position" (default; repair@r), between turns 1 and 2, before the next question is known: rows
  [0, R) in position order, R = round(r P) (P = the approximated prefix), and every cached row at r = 1. Every
  late layer runs exactly over rows [0, R) from their exact layer-split inputs on a scratch cache, as chunked
  prefill of those rows would: a row's exact late inputs need the exact K/V of every earlier row, so the repair
  cannot skip ahead. At r = 1 that is the prefix, the turn-1 tail with its question, and the answer rows (all
  were computed against the approximated prefix), so the cache equals an exact forward over the history. For
  r < 1 the rows after R (the rest of the prefix, the tail, question and answer) keep their stale K/V, and the
  GDN states stay as they are: they summarise every cached row, so only a repair of all of them (r = 1, where
  the scratch states are exact) can replace them.
  repair_by "question" (repair-q@r, ProphetKV-style), on turn 2's arrival: turn 2's rows run the exact lower
  layers, and their attention mass (the qattn selector, kva/select.py, over turn 2's last WINDOW rows) picks
  R = round(r P) prefix rows; those run the late layers by finish(exact_rows=...)'s computation over the other
  prefix rows as filled. Out of position order a recomputed row still reads approximated rows before it, so it
  is only approximately exact. Their K/V replace the cached ones; the tail, answer and unpicked rows stay stale,
  and so do the GDN states. Turns 3-4 reuse the cache (the rows are not re-picked per turn).
  Records per repair: rows, late layers, compute (kva/compute.py) and the memory it needs: the stored layer-split
  inputs of the rows it may recompute, hidden x rows x 2 bytes (bf16): R rows for repair@r, all P prefix rows
  for repair-q (the rows are picked only once the question arrives).
Records: runs[arm] = P, split, tail, fill, selection, repair, turn1_wall_s (shared by the base's arms), wall_s
(repair and turns 2-4) and turns: [{turn, key, value, correct, text, gen, finish_reason, ctx (cached rows
before the turn), new (rows prefilled), needle_start/end, needle_projected (needle_end <= P), needle_rows_exact
(selector arms), needle_repaired (repair arms: the whole needle repaired before the turn)}].
"""
import json
import time

import torch

from kva import compute, data
from kva import select as SEL
from kva.kva import Arm, finish, run_lower

TURNS = 4
TURN_FILLS = ("proj", "identity", "none")
REPAIR_FILLS = ("proj", "identity")
HS_BYTES = 2                    # bf16 layer-split inputs kept for a repair


def _fills(s):
    return list((s.get("fill_by_type") or {}).values()) if s.get("fill") == "by_type" else [s.get("fill")]


def check_arms(specs, set_name):
    """Repair arms run on the multiturn set only; the set takes the arms whose turn 1 is a plain finish()."""
    for s in specs:
        name, r, by = s["name"], s.get("repair"), s.get("repair_by", "position")
        if set_name != "multiturn":
            if "repair" in s or "repair_by" in s:
                raise SystemExit(f"arm {name}: 'repair' runs on the multiturn set only")
            continue
        sel = s.get("select", {}).get("kind")
        if (any(f not in TURN_FILLS for f in _fills(s)) or s.get("tail_round") or sel in ("tail_match", "digits")
                or s.get("reuse_fill") or s.get("tail_from")):
            raise SystemExit(f"arm {name}: not supported on the multiturn set")
        if r is None:
            if "repair_by" in s:
                raise SystemExit(f"arm {name}: repair_by without repair")
            continue
        if not (isinstance(r, (int, float)) and 0 <= r <= 1) or by not in ("position", "question"):
            raise SystemExit(f"arm {name}: repair must be a fraction in [0, 1] by position or question")
        if (any(f not in REPAIR_FILLS for f in _fills(s)) or (by == "question" and (sel or s.get("f1")))
                or s.get("tail_bias_from") is not None or s.get("rescale") is not None):
            # fill_inputs() rebuilds fills without the tail mean or a K/V rescale
            raise SystemExit(f"arm {name}: repair needs fill proj or identity, no tail_bias_from or rescale "
                             "(repair-q: no select, no f1)")


def followup_pieces(tok, fmt):
    """Ids around a follow-up user message: (closer after an assistant reply, header that opens the next reply).
    Raw: none."""
    if fmt != "chat":
        return [], []
    u1, a1, u2 = "\x00U1\x00", "\x00A1\x00", "\x00U2\x00"
    r = tok.apply_chat_template([{"role": "user", "content": u1}, {"role": "assistant", "content": a1},
                                 {"role": "user", "content": u2}], tokenize=False, add_generation_prompt=True)
    close, opener = r.split(a1)[1].split(u2)
    return data.encode(tok, close), data.encode(tok, opener)


def ask_ids(tok, fmt, key, pieces):
    close, opener = pieces
    q = data.niah_question(fmt, key)
    return close + data.encode(tok, q.lstrip() if fmt == "chat" else q) + opener


def ask_text_ids(tok, fmt, ask, pieces):
    """A later turn given as text (the locked two-key RULER set, kva/derived.py): chat, the question as the next user
    message and the reply prefix opening the next assistant turn (as kva.locked.wrap's turn 1); raw, a new line, the
    question and the reply prefix (RULER's input puts the question on its own line)."""
    close, opener = pieces
    if fmt == "chat":
        return close + data.encode(tok, ask["question"]) + opener + data.encode(tok, ask["reply"])
    return data.encode(tok, "\n" + ask["question"] + ask["reply"])


def needle_span(tok, ids, key, value):
    """[start, end) of the needle sentence's ids in the prompt (niah_prompts splices them in as encoded alone)."""
    nd = data.encode(tok, data.niah_needle(key, value))
    for s in range(len(ids) - len(nd) + 1):
        if ids[s] == nd[0] and ids[s:s + len(nd)] == nd:
            return s, s + len(nd)
    raise ValueError(f"needle for {key} not found in the prompt")


def plan(tok, p, turns=TURNS):
    """The prompt's first `turns` turns: the target key, then the distractors, each with its needle span."""
    if not 2 <= turns <= TURNS:
        raise ValueError(f"{turns} turns: 2..{TURNS} (an mk4 prompt has {TURNS} keys)")
    if "turn_plan" in p:                           # the locked two-key RULER set: turns and needle spans built in
        if turns > len(p["turn_plan"]):
            raise ValueError(f"{turns} turns, but {p['name']} has {len(p['turn_plan'])} (the two-key set: 2)")
        return [dict(t) for t in p["turn_plan"][:turns]]
    if p.get("variant") != "mk4":
        raise ValueError("the multiturn set reads the mk4 prompts (their distractors are the later turns' keys)")
    out = [dict(key=p["key"], value=p["value"], needle_start=p["needle_start"], needle_end=p["needle_end"])]
    for d in p["distractors"]:
        s, e = needle_span(tok, p["ids"], d["key"], d["value"])
        out.append(dict(key=d["key"], value=d["value"], needle_start=s, needle_end=e))
    return out[:turns]


def extend(model, cache, ids, start, split=None, before=None):
    """Exact forward of new rows (positions start, ...) on top of cache, appending to it. before(inputs): called
    once the rows have passed the layers < split, with their inputs there {layer: rows}, before layer split runs
    (it may replace cache's late-layer entries). Returns (logits of the last row, the rows' layer-split inputs or
    None)."""
    ids = torch.as_tensor(ids, device=model.device)
    pos = torch.arange(start, start + ids.shape[0], device=model.device)
    h, hb, low = model.embed(ids), None, {}
    for i in range(model.n_layers):
        if i == split:
            hb = h
            if before is not None:
                before(low)
        if before is not None and i < split:
            low[i] = h
        h = model.layer(i, h, pos, cache)
    return model.logits(h[-1:]), hb


def decode(model, cache, logits, start, steps, eos=(), split=None):
    """Greedy as kva.greedy (stops after an EOS; the last token is not fed), feeding rows from start. Returns
    (ids, 'stop' | 'length', the fed rows' layer-split inputs [len(ids) - 1, H], or None)."""
    out, hb = [], []
    tok = int(logits[-1].argmax())
    reason = "length"
    for t in range(steps):
        out.append(tok)
        if tok in eos:
            reason = "stop"
            break
        if t == steps - 1:
            break
        lg, b = extend(model, cache, [tok], start + t, split)
        hb.append(b)
        tok = int(lg[-1].argmax())
    return out, reason, (torch.cat(hb) if hb and split is not None else None)


def cached_rows(cache, layer):
    c = cache[layer]
    return c.get("n", c.get("len", 0))


def repair_rows(r, P, H):
    """Rows repair@r recomputes, from row 0: round(r P), or all H cached rows at r = 1."""
    return H if P > 0 and r >= 1 else round(r * P)


def _put_kv(model, cache, i, k, v):
    """Replace layer i's K/V with new buffers (the old ones may be shared with other forks)."""
    new = {}
    model._append_kv(new, k, v)
    cache[i] = new


def repair(model, cache, hb, split, rows):
    """repair@r on the arm's own cache list (module docstring): layers >= split recomputed exactly over rows
    [0, rows) from their exact layer-split inputs hb[:rows]. Replaces list entries only, never a tensor in place,
    so caches it was forked from stay intact."""
    H = cached_rows(cache, split)
    if not 0 <= rows <= H or hb.shape[0] < rows:
        raise ValueError(f"repair of {rows} rows: {H} cached, {hb.shape[0]} inputs")
    if rows == 0:
        return
    pos = torch.arange(rows, device=model.device)
    tmp = model.new_cache()
    h = hb[:rows].to(model.device)
    for i in range(split, model.n_layers):
        h = model.layer(i, h, pos, tmp)
        c, t = cache[i], tmp[i]
        if model.layer_types[i] == "attn":
            _put_kv(model, cache, i, torch.cat((t["k"], c["k"][:, rows:]), 1), torch.cat((t["v"], c["v"][:, rows:]), 1))
        elif rows == c["len"]:
            cache[i] = dict(conv=t["conv"], rec=t["rec"], len=rows)
        tmp[i] = {}


def fill_inputs(model, arm, i, b):
    """Late layer i's predicted inputs for prefix rows with layer-split inputs b, as finish() fills them."""
    f = arm.fill_of(model.layer_types[i])
    if f == "proj":
        return arm.projector(i, b)
    if f == "identity":
        return b
    raise ValueError(f"no fill inputs for fill {f!r}")


def repair_selected(model, cache, arm, b, rows):
    """repair-q on the arm's own cache list: prefix rows `rows` (sorted, < P = len(b)) through layers >= split by
    finish(exact_rows=rows)'s computation (kva.py) over the other prefix rows as filled from b; their attention
    K/V replace the cached ones (new buffers). Out of order, so only approximately exact; GDN states stay."""
    if len(rows) == 0:
        return
    P = b.shape[0]
    r = torch.as_tensor(rows, dtype=torch.long, device=model.device)
    pos = torch.arange(P, device=model.device)
    b = b.to(model.device)
    hr = b[r]
    tmp = model.new_cache()
    for i in range(arm.split, model.n_layers):
        x = fill_inputs(model, arm, i, b)
        hr_next = model.layer_rows(i, x.index_copy(0, r, hr.to(x.dtype)), pos, r, tmp)
        if model.layer_types[i] == "attn":
            c, t = cache[i], tmp[i]
            _put_kv(model, cache, i, c["k"].index_copy(1, r, t["k"][:, r]), c["v"].index_copy(1, r, t["v"][:, r]))
        tmp[i] = {}
        hr = hr_next


def repair_q_hook(model, cache, arm, r, b, dm, info):
    """extend()'s hook on turn 2 of a repair-q arm: once turn 2's rows have passed the lower layers, their
    attention mass (qattn over its last WINDOW rows) picks round(r P) prefix rows (P = len(b)), which
    repair_selected() recomputes. Fills info (rows, macs, hs_bytes, picked, fixed, wall_s)."""
    P = b.shape[0]

    def hook(low):
        t0 = time.time()
        layers = SEL.qattn_layers(model, arm.split)
        end = cached_rows(cache, 0)
        w = {i: low[i][-SEL.WINDOW:] for i in layers}
        picked = SEL.top_rows(SEL.window_mass(model, w, cache, layers, end), P, round(r * P))
        repair_selected(model, cache, arm, b, picked)
        pos = sum(picked) / len(picked) / P if picked else 0.5
        window = w[layers[0]].shape[0] if layers else 0
        info.update(rows=len(picked), macs=compute.repair_selected(dm, arm.split, len(picked), P, pos)
                    + compute.qattn_select(dm, end, arm.split, window=window),
                    hs_bytes=model.hidden * P * HS_BYTES, picked=SEL.ranges(picked), fixed=set(picked),
                    wall_s=round(time.time() - t0, 3))
    return hook


def _turn(tok, t, spec, gen, reason, ctx, new, P, rows=None):
    text = tok.decode(gen, skip_special_tokens=False)
    d = dict(turn=t, key=spec["key"], value=spec["value"], correct=data.niah_correct(text, spec["value"]),
             text=text, gen=gen, finish_reason=reason, ctx=ctx, new=new, needle_start=spec["needle_start"],
             needle_end=spec["needle_end"], needle_projected=spec["needle_end"] <= P)
    if rows is not None:
        d["needle_rows_exact"] = sum(spec["needle_start"] <= r < spec["needle_end"] for r in rows)
    return d


def later_turns(model, tok, cache, H, last, asks, turns, steps, eos, P, rows=None, split=None, before=None):
    """Turns 2+ on cache (H rows; last = the previous answer's unfed token); before: extend()'s hook on turn 2.
    Appends to cache."""
    out = []
    for t, (ask, spec) in enumerate(zip(asks, turns), 2):
        new = ([] if last in eos else [last]) + ask
        hook = before if t == 2 else None
        lg, _ = extend(model, cache, new, H, split if hook else None, hook)
        ctx, H = H, H + len(new)
        gen, reason, _ = decode(model, cache, lg, H, steps, eos)
        H += len(gen) - 1
        out.append(dict(_turn(tok, t, spec, gen, reason, ctx, len(new), P, rows), new_ids=new))
        last = gen[-1]
    return out


def base_key(a):
    if a.fill == "exact":
        return "exact"
    return json.dumps({k: v for k, v in a.meta.items() if k not in ("name", "repair", "repair_by")}, sort_keys=True)


def run_prompt(model, tok, p, arms, rec, fmt, eos, steps, digit_set=None, n_turns=TURNS):
    """Every arm (plus exact) on one mk4 prompt: fills rec['turns'] (the first n_turns) and rec['runs'][arm]."""
    ids = torch.tensor(p["ids"])
    n = ids.shape[0]
    turns = rec["turns"] = plan(tok, p, n_turns)
    pieces = followup_pieces(tok, fmt)
    asks = ([ask_text_ids(tok, fmt, a, pieces) for a in p["asks"][:len(turns) - 1]] if "asks" in p
            else [ask_ids(tok, fmt, t["key"], pieces) for t in turns[1:]])
    todo = [Arm("exact", split=max(a.split for a in arms))] + list(arms)
    groups = {}
    for a in todo:
        groups.setdefault(base_key(a), []).append(a)
    # turn 1's selection, as evaluate.py: qattn windows on the last rows of the turn-1 prompt (or before P)
    sel = [a for a in todo if a.meta.get("select")]
    wins = {e: {} for e in {n} | {a.prefix_len(n) for a in sel if a.meta["select"].get("window") == "prefix"}}

    def hook(i, h):
        for e, w in wins.items():
            w[i] = h[max(0, e - SEL.WINDOW):e].clone()
    splits = {a.split for a in todo}
    hs, lower = run_lower(model, ids, splits, hs_device="cpu" if len(splits) > 2 else None,
                          hook=hook if sel else None)
    masses = {}

    def mass_of(split, layers, end):
        key = (tuple(SEL.qattn_layers(model, split, layers)), end)
        if key not in masses:
            masses[key] = SEL.window_mass(model, wins[end], lower, key[0], end)
        return masses[key]
    picks = SEL.choose(todo, n, p["ids"], p["name"], digit_set, mass_of)
    dm = compute.Dims(model.path)
    for members in groups.values():
        a = members[0]
        t0 = time.time()
        pick = picks.get(a.name)
        rows = pick["rows"] if pick else None
        logits, cache, P = finish(model, ids, hs[a.split], lower, a, 1, exact_rows=rows)
        if a.projector is not None:      # only turn 1 fills from it (repair-q copies it back): 0.6 GiB on Llama
            a.projector.release()
        need_hb = any(b.meta.get("repair") is not None for b in members)
        gen, reason, hb = decode(model, cache, logits, n, steps, eos, a.split if need_hb else None)
        H = n + len(gen) - 1
        first = _turn(tok, 1, turns[0], gen, reason, 0, n, P, rows)
        del logits
        t1 = round(time.time() - t0, 3)
        for b in members:
            t0 = time.time()
            run = dict(P=P, split=b.split, tail=n - P, fill=b.fill, st=bool(b.st), f1=b.f1, turn1_from=a.name,
                       turn1_wall_s=t1)
            if pick:
                run.update(select=pick["kind"], n_rows=len(rows), rows=SEL.ranges(rows))
            # the group's last arm carries turn 1's cache on instead of forking it: later turns outgrow the K/V
            # buffers' spare rows and regrow every layer, and a fork's source would keep each old buffer alive
            # (Llama 32K: +2.5 GiB of late-layer K/V, the MT1 OOM)
            if b is members[-1]:
                c, cache = cache, None
            else:
                c = [dict(x) for x in cache]
            r, by, before, info = b.meta.get("repair"), b.meta.get("repair_by", "position"), None, {}
            if r is not None and by == "position":
                R = repair_rows(r, P, H)
                hb_all = hs[b.split].to(model.device) if hb is None else torch.cat((hs[b.split].to(model.device), hb))
                repair(model, c, hb_all, b.split, R)
                del hb_all
                info = dict(rows=R, macs=compute.repair(dm, b.split, R), hs_bytes=model.hidden * R * HS_BYTES,
                            fixed=range(R), wall_s=round(time.time() - t0, 3))
            elif r is not None:
                before = repair_q_hook(model, c, b, r, hs[b.split][:P], dm, info)
            run["turns"] = [first] + later_turns(model, tok, c, H, gen[-1], asks, turns[1:], steps, eos, P, rows,
                                                 b.split, before)
            if r is not None:
                fixed = info.pop("fixed")
                L = model.n_layers - b.split
                run["repair"] = dict(r=r, by=by, layers=L, row_layers=info["rows"] * L,
                                     macs_exact_prompt=compute.exact(dm, n), **info)
                run["turns"] = [dict(t, needle_repaired=t["turn"] > 1 and all(
                    j in fixed for j in range(t["needle_start"], t["needle_end"]))) for t in run["turns"]]
            run["wall_s"] = round(time.time() - t0, 3)
            rec["runs"][b.name] = run
            del c
        del hb
    del hs, lower
    return " ".join(f"{k}:" + "".join("+" if t["correct"] else "-" for t in r["turns"]) for k, r in rec["runs"].items())
