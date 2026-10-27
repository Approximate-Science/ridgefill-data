"""Exact-row selectors for the bypass arms (definitions: r15/SELECTORS.md). An arm's "select" spec
picks rows inside its approximated prefix [0, P); finish(exact_rows=...) runs them exactly.

  digits      rows whose token text contains an ASCII digit (this model's tokenizer; added/special
              tokens excluded). Natural budget.
  regex       R15's rule (ii) (r15/selectors_r15.regex): digits, plus every row of a whitespace word of >= 4
              chars that holds an ASCII digit (IDs, hex, UUIDs, versions), plus capitalised names: a word
              matching NAME_WORD (wrapping punctuation stripped) and not in STOP_CAPS, kept if a neighbouring
              word is one too with no sentence punctuation between them, or if it is not sentence-initial and
              spans >= 2 tokens. Words are R15's: the prefix's bytes (token_bytes; added/special tokens read
              as a space) split on ASCII whitespace, cut at P. Natural budget.
  verbatim    regex plus URLs (scheme:// or www.), emails (EMAIL: non-space @ non-space . non-space), the
              inside of "..." or '...' on one line when it spans >= 2 tokens, and code identifiers of >= 6
              chars with an underscore or a lower-to-upper capital (snake_case, camelCase). Natural budget.
              A separate kind, so regex stays R15's.
  Capped content rules (digits, regex, verbatim with a "budget"): a seeded uniform sample of round(budget * N)
              of the rule's rows (all of them if fewer), seed sha256("<kind>|<prompt>|P"), so the cap is not
              biased by position and a smaller budget's rows are a subset of a larger one's.
  qattn       Aw (SELECTORS.md (iv-w)): per prefix row j, the attention probability the last WINDOW
              prompt rows give it (v1.1.3: on a set that scores prompt tokens, the last WINDOW rows before the
              scored rows, choose(known=...)) (window "prefix": the WINDOW rows just before P instead, so the
              selector never sees the tail or the question, as with a cached prefix), summed over the query rows, all query heads and the exact lower
              attention layers `layers` (default: those of LAYERS below the split), dense causal
              softmax over every key. The top round(budget * N) rows (N = prompt tokens, as
              SELECTORS.md's 5pct; capped at P), ties broken by position.
  pred        predictability (kva/pred.py; Dylan's R16/R18a): per prefix row j its predicted surprisal from
              h_S(j-1) (the run's lower pass) and the embedding of j; "predictor" = the fit_surprisal.py
              file of this model and split. The top round(budget * N) rows in [0, P), ties by position;
              it reads rows < P only, never the tail or the question.
  freq        rarity (a free baseline for pred: the most common words are filled first): per prefix row
              j, -log of its token's unigram frequency ("table": scripts/token_freq.py, add-one smoothed
              counts over the calibration train docs). Rarest first, ties by position.
  frag        word fragmentation (words the tokenizer splits into several pieces: rare words, names,
              identifiers, numbers): per prefix row j, the number of pieces of its word, ties broken by
              the token's freq rarity, then by position. Words (byte-level BPE): a new word starts at the
              first prompt token, at a token whose decoded text begins with whitespace or punctuation,
              and right after a token whose decoded text ends with whitespace or punctuation (else the
              first word of a line, ")\n" + "Next", would join the newline's word). Whitespace or
              punctuation: str.isspace() or a Unicode category P* / S*, except U+FFFD (a partial UTF-8
              byte, which continues the word). Words are cut at P (rows >= P are not read).
  blend       a first signal ("first": qattn, default, or qfp) and a second one at the same total budget
              b = round(budget * N) (capped at P): "second" (pred, default; or freq, frag, keydiff, deviation,
              qfp, regex, verbatim, with that kind's fields) takes round(share * b) of it (Python round;
              "share" default 0.5), the first signal (qattn: window and layers as for qattn) the rest. Rank
              fusion, kept simple: the first signal's top b - round(share * b) rows, then the second signal's
              rows in rank order, skipping rows already taken, until there are b rows (an overlap refills from
              the second signal's next-ranked rows), so the budget is exact without duplicates. A content rule
              as the second signal: its rows ranked by the first signal's score (VB1: by qfp; ties by
              position), single rows, not whole words; when it runs out of rows, the first signal's
              next-ranked rows fill the budget. share 0 is the first signal, 1 the second alone (a content
              rule with fewer than b rows: its rows plus the first signal's top rows).
  Query-free controls for the question-blind setting (same budget as pred and qblind):
  start_recent  "keep the start plus the recent window": rows [0, start) exact ("start" default 64; the
              drafts use 64, so that at 5% it differs from tail_match) and the rest of round(budget * N)
              grows the exact tail (as tail_match). Count = the budget.
  keydiff     KeyDiff-style key distinctiveness: per prefix row j, the sum over the lower attention
              layers (qattn's `layers`) and KV heads of 1 - cos(k_j, mean_{i < P} k_i), k = the cached
              post-RoPE keys of the exact lower pass. Top round(budget * N) rows in [0, P).
  deviation   CacheBlend-style fill-error proxy: per prefix row j, the sum over the late attention layers
              i >= split of |r_i(proj_i(b_j)) - r_i(b_j)|^2 / |r_i(b_j)|^2, r_i = layer i's pre-RoPE
              [K | V] (kva.kv_target), b_j = the row's layer-split state, proj = the arm's projector (or
              "projector"): how far the projected fill moves what attention reads from the identity fill.
              The rows where the two fills disagree most stand in for the rows the fill gets most
              wrong (no true late input exists at prefill). Top round(budget * N) rows in [0, P).
  random      uniform without replacement in [0, P), as many rows as the arm named in "match" picked
              on the same prompt; seed = sha256("bypass|<prompt>|P|count"). Matched to a natural-budget
              content-rule arm, it is that rule's count-matched random control (VB1).
  tail_match  no rows; the exact tail grows by the matched arm's row count (KVA-028's equal-budget
              control at row granularity).
Every kind but qattn / blend with qattn first, with the default window, reads rows < P only (never the tail or
the question).
"""
import hashlib
import re
import unicodedata

import torch

WINDOW = 64
LAYERS = (3, 7, 11, 15)
KINDS = ("digits", "regex", "verbatim", "qattn", "pred", "freq", "frag", "blend", "start_recent", "keydiff",
         "deviation", "band", "qfp", "random", "tail_match")
RULES = ("digits", "regex", "verbatim")                                         # content rules: natural budget
SECONDS = ("pred", "freq", "frag", "keydiff", "deviation", "qfp", "regex", "verbatim")   # blend's second signals
FIRSTS = ("qattn", "qfp")                                                       # blend's first signals


def signals(sel):
    """The signals a select spec reads: its kind, or for blend {first, second}."""
    if sel["kind"] == "blend":
        return {sel.get("first", "qattn"), sel.get("second", "pred")}
    return {sel["kind"]}


def digit_ids(tok):
    """Vocabulary ids whose token text contains 0-9 (byte-level BPE keeps ASCII digits as is)."""
    added = set(tok.added_tokens_decoder)
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    return {i for i, s in enumerate(pieces) if s is not None and i not in added and re.search("[0-9]", s)}


def digits(ids, P, digit_set):
    return [r for r in range(P) if int(ids[r]) in digit_set]


def _byte_map():
    """GPT-2's byte -> printable character map (the alphabet of byte-level BPE pieces), inverted."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tok):
    """The bytes of every vocabulary id ([len(tok)]; added/special tokens None). Byte-level BPE pieces (Llama-3.1,
    Qwen3.5) go back through GPT-2's byte map, so a character split over tokens is not lost to U+FFFD as with
    per-token decode; a piece outside that map is taken as UTF-8."""
    inv = _byte_map()
    added = set(tok.added_tokens_decoder)
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    return [None if s is None or i in added else bytes(inv[c] for c in s) if all(c in inv for c in s) else s.encode()
            for i, s in enumerate(pieces)]


def prefix_text(ids, P, tbytes):
    """(bytes of rows [0, P), the row of each byte); added/special tokens read as a space (R15)."""
    text, row = bytearray(), []
    for r in range(P):
        b = tbytes[int(ids[r])]
        b = b" " if b is None else b
        text += b
        row += [r] * len(b)
    return bytes(text), row


WORD = re.compile(rb"\S+")
NAME_WORD = re.compile(r"^[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*$")
STOP_CAPS = frozenset("The A An In On At Of For And But Or If When While This That These Those It Its We You He She "
                      "They I My Our Your His Her Their To From By With As Is Are Was Were Be Not No Yes So Then "
                      "There Here What Which Who How Why Where All Any Each Every Some Use See Note Returns "
                      "Return".split())
SENT_END = re.compile(r"[.!?:;]$")


def regex_rows(ids, P, digit_set, tbytes):
    """R15 rule (ii), ported from r15/selectors_r15.regex (same words, name rule and stop list): rows < P."""
    text, row = prefix_text(ids, P, tbytes)
    ws = [(m.group().decode("utf-8", errors="replace"), sorted(set(row[m.start():m.end()])),
           text[max(0, m.start() - 1):m.start()]) for m in WORD.finditer(text)]
    out = set(digits(ids, P, digit_set))
    name = []
    for w, rows, _ in ws:
        if len(w) >= 4 and re.search(r"[0-9]", w):
            out.update(rows)
        core = w.strip("\"'`([{<,.;:!?)]}>*")
        name.append(bool(NAME_WORD.match(core)) and core not in STOP_CAPS)
    for k, (w, rows, before) in enumerate(ws):
        if not name[k]:
            continue
        prev_ok = k > 0 and name[k - 1] and not SENT_END.search(ws[k - 1][0])
        next_ok = k + 1 < len(ws) and name[k + 1] and not SENT_END.search(w)
        initial = k == 0 or before in (b"\n", b"") or bool(SENT_END.search(ws[k - 1][0]))
        if prev_ok or next_ok or (not initial and len(rows) >= 2):
            out.update(rows)
    return sorted(out)


URL = re.compile(rb"(?:https?|ftp)://\S+|www\.\S+")
EMAIL = re.compile(rb"\S+@\S+\.\S+")
QUOTED = re.compile(rb"\"([^\"\n]+)\"|(?<![\w'])'([^'\n]+)'(?!\w)")    # an apostrophe in a word opens nothing
IDENT = re.compile(rb"(?<!\w)[A-Za-z_]\w{5,}(?!\w)")                       # bytes: \w is [A-Za-z0-9_]
CODE = re.compile(rb"_|[a-z][A-Z]")


def verbatim_rows(ids, P, digit_set, tbytes):
    """regex plus URLs, emails, quoted strings (the inside, >= 2 tokens) and snake_case / camelCase identifiers."""
    text, row = prefix_text(ids, P, tbytes)
    out = set(regex_rows(ids, P, digit_set, tbytes))
    for pat in (URL, EMAIL):
        for m in pat.finditer(text):
            out.update(row[m.start():m.end()])
    for m in QUOTED.finditer(text):
        g = 1 if m.group(1) is not None else 2
        inside = set(row[m.start(g):m.end(g)])
        if len(inside) >= 2:
            out.update(inside)
    for m in IDENT.finditer(text):
        if CODE.search(m.group()) and re.search(rb"[A-Za-z]", m.group()):
            out.update(row[m.start():m.end()])
    return sorted(out)


def rule_rows(kind, ids, P, digit_set, tbytes=None):
    """A content rule's rows in [0, P)."""
    if kind == "digits":
        return digits(ids, P, digit_set)
    if tbytes is None:
        raise ValueError(f"select kind {kind!r} needs token_bytes(tok)")
    return (regex_rows if kind == "regex" else verbatim_rows)(ids, P, digit_set, tbytes)


def rule_score(rows, P, name, kind):
    """A content rule's rows as a ranking [P] (fp64): its rows in a seeded uniform order (seed: kind, prompt, P;
    not the budget, so budgets nest), every other row -inf (never picked for the rule)."""
    s = torch.full((P,), float("-inf"), dtype=torch.float64)
    seed = int.from_bytes(hashlib.sha256(f"{kind}|{name}|{P}".encode()).digest()[:8], "little") >> 1
    perm = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))
    s[torch.as_tensor(rows, dtype=torch.long)[perm]] = torch.arange(len(rows), 0, -1, dtype=torch.float64)
    return s


def qattn_layers(model, split, layers=None):
    layers = LAYERS if layers is None else layers
    return [L for L in layers if L < split and model.layer_types[L] == "attn"]


@torch.no_grad()
def window_mass(model, win, cache, layers, end=None):
    """Aw attention mass [n] (fp64, CPU) of every row of an n-row prompt. win {L: the inputs of layer
    L for the window rows [end - w, end), end = n by default}; cache: the exact lower cache (post-RoPE
    keys of all rows). Causal: rows at or after end get no mass."""
    mass = None
    for L in layers:
        k = cache[L]["k"]
        hkv, n, d = k.shape
        h = win[L]
        w = h.shape[0]
        end = n if end is None else end
        qpos = torch.arange(end - w, end, device=k.device)
        q = model.queries(L, model.in_norm(h, model.W[L]["ln1"]), qpos)
        rep = q.shape[0] // hkv
        mask = torch.arange(n, device=k.device)[None, :] > qpos[:, None]
        acc = torch.zeros(n, dtype=torch.float64, device=k.device)
        for g in range(hkv):                                 # one K/V head group at a time (memory)
            s = torch.matmul(q[g * rep:(g + 1) * rep].float(), k[g].float().T).mul_(model.scale)
            s.masked_fill_(mask, float("-inf"))
            acc += torch.softmax(s, -1).sum((0, 1)).double()
        mass = acc if mass is None else mass + acc
    return mass.cpu()


def _breaks(text):
    return text != "" and text != "\ufffd" and (text.isspace() or unicodedata.category(text)[0] in "PS")


def word_breaks(tok):
    """(starts, ends) bool [len(tok)]: the id's decoded text begins / ends with whitespace or punctuation."""
    texts = [tok.decode([i]) for i in range(len(tok))]
    return (torch.tensor([_breaks(t[:1]) for t in texts]), torch.tensor([_breaks(t[-1:]) for t in texts]))


def words(ids, P, starts, ends):
    """Word index [P] of rows [0, P) (frag's rule)."""
    t = torch.as_tensor(list(ids[:P]), dtype=torch.long)
    new = starts[t].clone()
    new[1:] |= ends[t[:-1]]
    if P:
        new[0] = True
    return new.long().cumsum(0) - 1


def rarity(ids, P, logfreq):
    """freq score [P] (fp64): -log unigram frequency of each prefix row's token."""
    return -logfreq[torch.as_tensor(list(ids[:P]), dtype=torch.long)].double()


def frag_scores(ids, P, starts, ends, logfreq):
    """frag score [P]: pieces of the row's word plus rarity / (max rarity + 1), so the piece count decides and
    rarity (>= 0, the add-one table's log frequencies are < 0) only breaks its ties."""
    if P == 0:
        return torch.zeros(0, dtype=torch.float64)
    w = words(ids, P, starts, ends)
    r = rarity(ids, P, logfreq)
    return torch.bincount(w)[w].double() + r / (float(r.max()) + 1)


@torch.no_grad()
def key_diff(cache, layers, P):
    """keydiff score [P] (fp64, CPU) from the exact lower cache: rows [0, P) of the keys only."""
    score = torch.zeros(P, dtype=torch.float64)
    for L in layers:
        k = cache[L]["k"][:, :P].float()                              # [kv_heads, P, head_dim]
        mean = k.mean(1, keepdim=True)
        cos = (k * mean).sum(-1) / (k.norm(dim=-1) * mean.norm(dim=-1)).clamp_min(1e-12)
        score += (1 - cos).sum(0).double().cpu()
    return score


@torch.no_grad()
def fill_deviation(model, projector, h, split, P, block=4096):
    """deviation score [P] (fp64, CPU) from rows [0, P) of the layer-split states h."""
    from kva.kva import kv_target
    late = [i for i in range(split, model.n_layers) if model.layer_types[i] == "attn"]
    out = torch.zeros(P, dtype=torch.float64)
    for r in range(0, P, block):
        b = h[r:min(r + block, P)].to(model.device)
        acc = torch.zeros(b.shape[0], dtype=torch.float64, device=b.device)
        for i in late:
            ident = kv_target(model, i, b).float()
            d = kv_target(model, i, projector(i, b)).float() - ident
            acc += (d.pow(2).sum(-1) / ident.pow(2).sum(-1).clamp_min(1e-12)).double()
        out[r:r + b.shape[0]] = acc.cpu()
    return out


def top_rows(score, P, b):
    """The b highest-scoring rows in [0, P), ties broken by position; sorted."""
    b = max(0, min(int(b), P))
    order = torch.sort(-score[:P].double(), stable=True).indices
    return sorted(order[:b].tolist())


def blend_rows(first, second, P, b, b_second):
    """b rows in [0, P) (b capped at P): first's top b - b_second rows, then second's rows in rank order (ties
    by position) not taken yet, until there are b; second's -inf rows (off a content rule) are never taken, and
    if second runs out, first's next-ranked rows fill the rest. Returns (sorted rows, overlap = how many of
    second's top b_second rows first had already taken, each refilled from second's next-ranked rows). first
    may be None when b_second >= b and second can fill b, second when b_second = 0 (that signal is not read)."""
    b = max(0, min(int(b), P))
    b_second = min(int(b_second), b)
    head = torch.sort(-first[:P].double(), stable=True).indices.tolist() if first is not None else []
    rows = set(head[:b - b_second])
    order = []
    if b_second > 0:
        s = second[:P].double()
        idx = torch.sort(-s, stable=True).indices
        order = idx[s[idx] > float("-inf")].tolist()
    overlap = len(rows & set(order[:b_second]))
    for r in order + head[b - b_second:]:
        if len(rows) >= b:
            break
        rows.add(r)
    return sorted(rows), overlap


def random_rows(name, P, b, salt="bypass"):
    b = max(0, min(int(b), P))
    seed = int.from_bytes(hashlib.sha256(f"{salt}|{name}|{P}|{b}".encode()).digest()[:8], "little") >> 1
    return sorted(torch.randperm(P, generator=torch.Generator().manual_seed(seed))[:b].tolist())


def choose(arms, n, ids, name, digit_set=None, mass_of=None, pred_of=None, keys_of=None, dev_of=None, freq_of=None,
           breaks=None, recv_of=None, tbytes=None, known=None):
    """Per prompt, for each arm with meta["select"]: {arm name: dict(kind, rows, extra_tail, count)} (blend
    also overlap: blend_rows'). Matched arms (random, tail_match) read the named arm's count, which
    is chosen first. mass_of(split, layers, end) -> the Aw mass [n] of this prompt at qattn_layers(model,
    split, layers), windowed on the rows just before end. pred_of(split, predictor, P) -> the predicted
    surprisal [P] of rows [0, P) (kva.pred.prefix_scores). keys_of(split, layers, P) -> key_diff of
    qattn_layers(model, split, layers); dev_of(arm, P) -> fill_deviation with the arm's projector;
    freq_of(table) -> the log-frequency table [V] (kva.pred.load_freq); breaks = word_breaks(tok);
    tbytes = token_bytes(tok) (regex, verbatim). known: the rows available before anything is scored (v1.1.3): on a
    set that scores prompt tokens teacher-forced (longppl, grade) the default qattn window ends there, never inside
    the scored rows; None = n (generation sets, where the whole prompt precedes what is scored)."""
    by_name = {a.name: a for a in arms}
    if known is not None and not 0 < known <= n:
        raise ValueError(f"known rows {known} not in (0, n = {n}]")
    out = {}

    def score(a, kind, sel, P, end):
        if kind == "qattn":
            return mass_of(a.split, sel.get("layers"), end)
        if kind == "pred":
            return pred_of(a.split, sel["predictor"], P)
        if kind == "freq":
            return rarity(ids, P, freq_of(sel["table"]))
        if kind == "frag":
            return frag_scores(ids, P, *breaks, freq_of(sel["table"]))
        if kind == "keydiff":
            return keys_of(a.split, sel.get("layers"), P)
        if kind == "deviation":
            return dev_of(a, P)
        if kind == "qfp":                 # TE2 analysis: prefix attention per later token (question-free)
            return recv_of(a.split, P)
        raise ValueError(f"{a.name}: {kind!r} is not a ranking signal")

    def pick(a):
        if a.name in out:
            return out[a.name]
        sel = a.meta["select"]
        kind = sel["kind"]
        P = a.prefix_len(n)
        rows, extra, more = [], 0, {}
        b = round(sel.get("budget", 0.05) * n)
        end = P if sel.get("window") == "prefix" else (n if known is None else known)
        if "qattn" in signals(sel):
            more["window_end"] = end
        if kind in RULES:
            rows = rule_rows(kind, ids, P, digit_set, tbytes)
            if "budget" in sel:                    # capped: a seeded uniform sample of the rule's rows
                rows = top_rows(rule_score(rows, P, name, kind), P, min(b, len(rows)))
        elif kind in ("qattn",) + SECONDS:
            rows = top_rows(score(a, kind, sel, P, end), P, b)
        elif kind == "blend":
            share, first, second = sel.get("share", 0.5), sel.get("first", "qattn"), sel.get("second", "pred")
            if not 0 <= share <= 1 or first not in FIRSTS or second not in SECONDS:
                raise ValueError(f"{a.name}: blend share {share} not in [0, 1], first {first!r} not in {FIRSTS} "
                                 f"or second {second!r} not in {SECONDS}")
            b2 = round(share * min(b, P))
            # a content rule can run out of rows, and then the first signal fills the budget
            s1 = score(a, first, sel, P, end) if b2 < min(b, P) or second in RULES else None
            if b2 > 0 and second in RULES:        # the rule's rows, ranked by the first signal
                s2 = torch.full((P,), float("-inf"), dtype=torch.float64)
                r2 = torch.as_tensor(rule_rows(second, ids, P, digit_set, tbytes), dtype=torch.long)
                s2[r2] = s1[:P].double()[r2]
            else:
                s2 = score(a, second, sel, P, end) if b2 > 0 else None
            rows, more["overlap"] = blend_rows(s1, s2, P, b, b2)
        elif kind == "band":                       # DB1: a fixed distance band before the end of the prompt (the read point)
            lo, hi = int(sel["lo"]), int(sel["hi"])  # rows n-hi .. n-lo-1, clipped to the approximated prefix
            rows = list(range(max(0, n - hi), min(P, max(0, n - lo))))
        elif kind == "start_recent":
            k = min(sel.get("start", 64), b, P)
            extra = min(b, P) - k
            rows = list(range(min(k, P - extra)))
        elif kind in ("random", "tail_match"):
            ref = by_name[sel["match"]]
            if "match" in ref.meta["select"]:
                raise ValueError(f"{a.name}: matched arm {ref.name} is itself matched")
            count = pick(ref)["count"]
            if kind == "random":
                rows = random_rows(name, P, count)
            else:
                extra = count
        else:
            raise ValueError(f"{a.name}: select kind {kind!r} not in {KINDS}")
        out[a.name] = dict(kind=kind, rows=rows, extra_tail=extra, count=len(rows) + extra, **more)
        return out[a.name]

    for a in arms:
        if a.meta.get("select"):
            pick(a)
    return out


def ranges(rows):
    """Sorted rows as [[start, end), ...] runs (compact raw output)."""
    out = []
    for r in rows:
        if out and out[-1][1] == r:
            out[-1][1] = r + 1
        else:
            out.append([r, r + 1])
    return out


def per_later_token(received_low, P):
    """qfp score: attention a row receives from later PREFIX rows (exact lower attention layers, summed over heads
    and layers), divided by how many later prefix rows there are (TE2 sensitivity analysis: the one question-free
    measurement that ranks impact the same way in every document of both models)."""
    later = torch.clamp(P - 1 - torch.arange(received_low.shape[0], dtype=torch.float64), min=1.0)
    out = received_low.double() / later
    out[P:] = 0.0
    return out
