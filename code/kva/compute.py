"""Analytic prefill compute (multiply-accumulates) per configuration, from the model dimensions.

Counted: every linear map (attention q/k/v/o or GDN in/out projections, MLP), causal attention
(QK^T and PV: 2 x context x heads x head_dim per token), the GDN conv and recurrence (state decay,
read-before-write, rank-1 update and output read: 4 x value_heads x dk x dv per token), the KVA
projector (hidden^2 per prefix token per late layer) and the lm_head for the last token. Not
counted: norms, RoPE, softmax and other elementwise work. Prefill of N prompt tokens:
  exact     every layer on every token;
  kva       layers < S on every token; layers >= S: prefix tokens (N - T) projector + fill only
            (attention: k/v projections; GDN: in-projections q/k/v/beta/decay, conv, state update
            without the output read), tail tokens exact with attention over the whole context;
  noproj    as kva without the projector;
  kvfold    as kva, but each late attention layer's prefix K/V come straight from the K/V fold (H x 2 kv_heads
            head_dim per token: the K/V projections' MACs, and no projector); late GDN layers keep the projector
            fill (the configured fallback, configs/rmsfold-qwen.json);
  swiftkv   SwiftKV as released: layers < 16 on every token, late layers' k/v via kv_proj_swiftkv for
            every token, the full late layers for the last token only;
  bypass    kva plus r exact prefix rows R (finish(exact_rows=R), r <= P = N - T). In every late layer the
            r rows run the full layer and the other P - r rows fill and get the projector. Assumptions:
            a row in R attends to pos x (P + 1) keys (pos = 0.5: R spread evenly over [0, P), the mean
            exact cost of a prefix row; scripts/frontier_bypass.py passes the measured mean); on GDN layers an efficient engine reads the state at R
            from the chunked recurrence, which already holds the chunk-boundary states, so a row in R
            costs what an exact row costs (the output read, nv x dk x dv; intra-chunk work is not counted,
            as in exact). gdn="reference" charges finish/layer_rows as written, an upper bound: the delta
            rule computes outputs at every prefix row (+ (P - r) output reads per late GDN layer) and the
            projector runs on all P rows. Selection: qattn recomputes the queries of the last WINDOW rows
            at its exact lower attention layers and their QK^T scores against every earlier key (no PV;
            softmax not counted); digits and random cost nothing; tail_match is kva with tail T + r.
  by_type   kva and bypass with fill_by_type: the projector is paid only on the late layer types filled with
            'proj' (proj_types_of); identity, oracle and kvfold fills pay no projector. Oracle fills read the true
            layer inputs: a diagnostic, not a real arm (diagnostic()), so its speedup means nothing.
  repair    deferred repair after the first answer (kva/multiturn.py): the late layers exactly over the
            first R cached rows, attention over the rows before each (repair()); paid between turns.
            repair-q: R rows picked by the next question, out of order, costed per row as bypass's exact rows
            (repair_selected()), plus the selector's scores.
  rank r    a low-rank projector applied factored, 2 x hidden x r MACs per prefix token (if below hidden^2).
  context K the context projector (kva.kva Arm.context) reads (K + 1) x hidden inputs: (K + 1) x hidden^2 MACs per
            prefix token per late layer it fills; the lower layers' mixer outputs it reads are computed anyway.
  refresh R layer R runs in full on the prefix rows (layer_exact over [0, P) instead of the fill), and layers > R
            use a plain projector of split R + 1 (hidden^2, also with context: the context projector fills S..R).
            The second projector replaces the first past R, so it adds no MACs (nor weights: a deployment keeps the
            first one's maps S..R only); the extra cost is layer_exact(R, 0, P) - layer_fill(R, P). With exact rows
            (bypass(refresh=R)) layer R runs in full on all P rows, theirs included, and the other P - r rows
            still pay the projector there.
  K/V splice  exact_part (kva.kva, KV1) reads the true late inputs: a diagnostic, as oracle fills.
  rescale   knorm / kdir / vnorm (kva.kva, KN1 / VN1) read the true late inputs: diagnostics. ktail / vtail are
            deployable; not counted here: per late attention layer the projector on the T - k tail rows before the
            scored window and their K (or V) projection, as tail_bias_from's projector pass.
"""
import json
from pathlib import Path


class Dims:
    def __init__(self, path):
        raw = json.loads((Path(path) / "config.json").read_text())
        c = raw.get("text_config", raw)
        self.L = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.I = c["intermediate_size"]
        self.V = c["vocab_size"]
        self.hq = c["num_attention_heads"]
        self.hkv = c["num_key_value_heads"]
        self.d = c.get("head_dim") or self.H // self.hq
        lt = c.get("layer_types")
        self.types = ["gdn" if t == "linear_attention" else "attn" for t in lt] if lt else ["attn"] * self.L
        self.gated = bool(c.get("attn_output_gate"))
        if "linear_num_key_heads" in c:
            self.nk, self.nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
            self.dk, self.dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
            self.kc = c["linear_conv_kernel_dim"]

    # per-token MACs
    def mlp(self):
        return 3 * self.H * self.I

    def attn_linear(self):
        q = self.H * self.hq * self.d * (2 if self.gated else 1)
        kv = 2 * self.H * self.hkv * self.d
        return q + kv + self.hq * self.d * self.H

    def attn_fill(self):
        return 2 * self.H * self.hkv * self.d

    def gdn_inproj(self):
        conv = 2 * self.nk * self.dk + self.nv * self.dv
        return self.H * conv + 2 * self.H * self.nv + conv * self.kc

    def gdn_linear(self):
        return self.gdn_inproj() + self.H * self.nv * self.dv + self.nv * self.dv * self.H   # + z, out_proj

    def gdn_rec(self, read=True):
        return (4 if read else 3) * self.nv * self.dk * self.dv

    def attn_ctx(self, first, last):
        """QK^T + PV MACs for tokens at positions first..last-1 (each attends to position+1 keys)."""
        n = last - first
        keys = n * (first + last + 1) / 2
        return 2 * self.hq * self.d * keys

    def layer_exact(self, i, first, last):
        n = last - first
        if self.types[i] == "attn":
            return n * (self.attn_linear() + self.mlp()) + self.attn_ctx(first, last)
        return n * (self.gdn_linear() + self.gdn_rec() + self.mlp())

    def layer_fill(self, i, n):
        if self.types[i] == "attn":
            return n * self.attn_fill()
        return n * (self.gdn_inproj() + self.gdn_rec(read=False))


def proj_macs(dm, i, proj_types=None, rank=None):
    """Projector MACs per prefix token at late layer i: 0 if its type is not in proj_types (None: every type)."""
    if proj_types is not None and dm.types[i] not in proj_types:
        return 0
    return dm.H * dm.H if rank is None else min(dm.H * dm.H, 2 * dm.H * rank)


def proj_types_of(fill_by_type):
    """The late layer types ('attn', 'gdn') that a fill_by_type runs the projector on."""
    return {"attn" if t == "attention" else t for t, f in fill_by_type.items() if f == "proj"}


def diagnostic(fill, fill_by_type=None, exact_part=None, rescale=None):
    """True for arms that read the true late-layer inputs (oracle, a K/V splice, an oracle rescale): not deployable,
    kept off any frontier."""
    return (exact_part is not None or rescale in ("knorm", "kdir", "vnorm")
            or "oracle" in (fill_by_type.values() if fill == "by_type" else (fill,)))


def exact(dm, n):
    return sum(dm.layer_exact(i, 0, n) for i in range(dm.L)) + dm.H * dm.V


def kva(dm, n, split, tail, projector=True, kvfold=False, proj_types=None, rank=None, context=0, refresh=None):
    p = max(0, n - tail)
    c = sum(dm.layer_exact(i, 0, n) for i in range(split))
    for i in range(split, dm.L):
        c += (dm.layer_exact(i, 0, p) if i == refresh else dm.layer_fill(i, p)) + dm.layer_exact(i, p, n)
        if projector and not (kvfold and dm.types[i] == "attn"):
            wide = 1 + context if refresh is None or i <= refresh else 1
            c += p * wide * proj_macs(dm, i, proj_types, rank)
    return c + dm.H * dm.V


def skip(dm, n, split, tail):
    """nofill (fill none): layers < S on every token; layers >= S on the exact tail alone (the prefix rows never reach
    them, so the tail attends to the tail only)."""
    return sum(dm.layer_exact(i, 0, n) for i in range(split)) + sum(dm.layer_exact(i, 0, tail) for i in range(split, dm.L)) \
        + dm.H * dm.V


def tail_budget(dm, n, split, tail):
    """The fewest extra exact tail rows E for which nofill with tail + E costs at least ridgefill (kva, projector) at
    (split, tail): the T1.5 control that spends the fill's modelled budget on a longer tail (skip+tailbudget, b = E/n
    declared in the FINAL record). Bisection over E in [0, n - tail] (the cost rises with E)."""
    target, lo, hi = kva(dm, n, split, tail), 0, n - tail
    while lo < hi:
        mid = (lo + hi) // 2
        if skip(dm, n, split, tail + mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def swiftkv(dm, n, split=16):
    c = sum(dm.layer_exact(i, 0, n) for i in range(split))
    c += (dm.L - split) * n * (2 * dm.H * dm.hkv * dm.d)
    c += sum(dm.layer_exact(i, n - 1, n) for i in range(split, dm.L))
    return c + dm.H * dm.V


def qattn_select(dm, n, split, window=64, layers=(3, 7, 11, 15)):
    """qattn's scores (kva/select.py WINDOW, LAYERS): queries and QK^T of the last `window` rows at the
    exact lower attention layers."""
    q = dm.H * dm.hq * dm.d * (2 if dm.gated else 1)
    used = [i for i in layers if i < split and dm.types[i] == "attn"]
    return len(used) * (window * q + dm.attn_ctx(n - window, n) / 2)


def bypass(dm, n, split, tail, rows, select=None, gdn="efficient", pos=0.5, proj_types=None, rank=None, refresh=None):
    """kva with `rows` exact prefix rows (capped at P) at mean position pos x P, plus the selector's cost
    (select: 'qattn' or None); proj_types, rank: as kva; refresh R: layer R in full on all P rows (FQ3's stack)."""
    p = max(0, n - tail)
    r = max(0, min(int(rows), p))
    ref = gdn == "reference"
    if gdn not in ("efficient", "reference"):
        raise ValueError(gdn)
    c = sum(dm.layer_exact(i, 0, n) for i in range(split))
    for i in range(split, dm.L):
        c += dm.layer_exact(i, p, n) + (p if ref else p - r) * proj_macs(dm, i, proj_types, rank)
        if i == refresh:
            c += dm.layer_exact(i, 0, p)
            continue
        c += dm.layer_fill(i, p - r)
        if dm.types[i] == "attn":
            c += r * (dm.attn_linear() + dm.mlp() + 2 * dm.hq * dm.d * pos * (p + 1))
        else:
            c += r * (dm.gdn_linear() + dm.gdn_rec() + dm.mlp())
        if ref and dm.types[i] == "gdn":
            c += (p - r) * dm.nv * dm.dk * dm.dv
    if select == "qattn":
        c += qattn_select(dm, n, split)
    return c + dm.H * dm.V


def bypass_arm(dm, n, split, tail, kind, rows=None, budget=None, gdn="efficient", pos=0.5, proj_types=None, rank=None,
               refresh=None):
    """MACs of a bypass arm of select kind `kind` (kva/select.py) with `rows` exact rows, or
    round(budget * n) rows (qattn's rule); tail_match spends them on the exact tail."""
    if rows is None:
        rows = round(budget * n)
    if kind == "tail_match":
        return kva(dm, n, split, tail + rows, proj_types=proj_types, rank=rank, refresh=refresh)
    if kind not in ("digits", "qattn", "random"):
        raise ValueError(kind)
    return bypass(dm, n, split, tail, rows, select=kind, gdn=gdn, pos=pos, proj_types=proj_types, rank=rank,
                  refresh=refresh)


def repair(dm, split, rows):
    """Deferred repair (kva/multiturn.py): layers >= split run exactly over rows [0, rows), each attending to
    the rows before it, as chunked prefill of those rows would."""
    return sum(dm.layer_exact(i, 0, rows) for i in range(split, dm.L))


def repair_selected(dm, split, rows, prefix, pos=0.5):
    """repair-q (kva/multiturn.py): `rows` prefix rows at mean position pos x prefix recomputed through layers
    >= split, each attending to the cached rows before it; GDN rows charged as in bypass() (efficient engine)."""
    c = 0
    for i in range(split, dm.L):
        if dm.types[i] == "attn":
            c += rows * (dm.attn_linear() + dm.mlp() + 2 * dm.hq * dm.d * pos * (prefix + 1))
        else:
            c += rows * (dm.gdn_linear() + dm.gdn_rec() + dm.mlp())
    return c


def bypass_speedup(dm, n, split, tail, kind, rows=None, budget=None, gdn="efficient", pos=0.5, proj_types=None, rank=None):
    return exact(dm, n) / bypass_arm(dm, n, split, tail, kind, rows, budget, gdn, pos, proj_types, rank)


def speedup(dm, n, fill, split=None, tail=0, fill_by_type=None, rank=None, context=0, refresh=None):
    base = exact(dm, n)
    if fill == "exact":
        return 1.0
    if fill == "swiftkv_native":
        return base / swiftkv(dm, n, split)
    if fill == "kvfold":
        return base / kva(dm, n, split, tail, kvfold=True)
    if fill == "by_type":
        return base / kva(dm, n, split, tail, proj_types=proj_types_of(fill_by_type), rank=rank)
    return base / kva(dm, n, split, tail, projector=fill == "proj", rank=rank, context=context, refresh=refresh)
