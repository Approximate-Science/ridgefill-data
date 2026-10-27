"""Approximate prefill (KVA) and its baselines over the layer-wise model API.

Layer-major form of the engine's chunk-major computation (equivalent by causality):
  1. layers < split run exactly on all N prompt rows;
  2. for each layer >= split: the prefix rows [0, P) only fill the layer's cache (from a predicted
     input, or SwiftKV's direct K/V); then the exact tail rows [P, N) run the full layer and
     attend to / continue from those caches.
P = N - tail (no block rounding). Paper arm names (OUTLINE T1) map onto these fills: nofill = 'none' (the skip arm:
late layers see only the exact tail; equals POP's "w/o Indep. KV" ablation, checks/T0.1), identity = 'identity' (POP's
fill at our split and tail), ridgefill = 'proj', pop-native = 'identity' at pop_split(L) with tail_from
'continuation' (POP's boundary: only the scored rows run the full model), swiftkv-native = 'swiftkv_native'. Fills:
  proj      ridge projector: x_i = W_i b + c_i  (b = the layer-split input of the row)
  identity  no projector: x_i = b for every late layer (training-free SingleInputKV ablation)
  oracle    the true layer inputs (all rows: must reproduce exact prefill; per layer type: a diagnostic)
  by_type   fill_by_type = {'attention': f, 'gdn': g}: fill f on attention layers, g on GDN layers
  swiftkv   SwiftKV's distilled K/V for prefix rows; tail and decode run the base model exactly
  swiftkv_native  SwiftKV as released: every row's late K/V from SwiftKV, queries through
            q_proj_swiftkv (prefill and decode)
  kvfold    K/V fold (scripts/fit_kvfold.py): attention layers' pre-RoPE K/V straight from the unit-RMS
            boundary state, [k | v] = (b / rms(b)) W_i^T + c_i; attention layers only (Qwen3.5: set
            fill_by_type for the GDN layers); no exact rows, no f1
Options: st = {layer: dS} terminal-state correction added to each late GDN layer's recurrent
state after its prefix fill; f1 = position 0 uses its exact late-layer inputs (exploratory).
  tail_bias_from = L  (BV3, exploratory) every late layer i >= L filled 'proj' adds the tail's mean projector
                residual m_i = mean_tail(x_i - P_i(h_S)) to the prefix rows' filled input (BV1: that mean transfers
                to the prefix). Layer-major order makes it one pass: the tail's layer-i input is the carried tail
                state when layer i's prefix fill is built (it saw the corrected fills of layers < i), so the cost
                is one projector matmul on the T tail rows per corrected layer. Exact rows and f1's row 0 keep
                their exact inputs; no context, refresh or fill reuse.
  tail_bias_skip_last = k  the mean covers tail rows [P, N - k) only: the grade scores the last 512 prompt
                tokens, which must not reach the prefix fill they are predicted from (0 <= k < T).
Fill quality (protocol v1.1 draft 6c, exploratory; scripts/fill_screen.py screens them):
  context = K   the projector reads [b, c_{S-1}, .., c_{S-K}] (context_input): c_j is lower layer j's mixer
                output for the row (Model.layer's mix: attention or GDN block output, before the MLP), kept by
                run_lower(mix=...) and passed to finish(mix=...). The row's own state b holds only what layer S
                has mixed in; c adds what the last layers gathered from context (scripts/fit_context.py).
  refresh = R   fill layers S..R from the projector, but run layer R in full on the prefix rows' filled
                inputs (attention over the filled cache of layer R, or the GDN recurrence, plus the MLP): its
                output h' carries freshly mixed context. Layers > R are filled from refresh_projector (a plain
                projector of split R + 1, fitted on exact data) applied to h'. Fill 'proj' only. With exact rows
                (FQ3's stack) the rows enter layer R with their carried inputs, so their h' is their exact-path
                output (what layer_rows gives them) and they carry on from it.
K/V splice (KV1, a diagnostic: it reads the true late inputs, as oracle fills do):
  exact_part = 'k' | 'v'  on late attention layers the prefix rows' K (or V) come from their true layer inputs
                (oracle, kept by oracle_tap from the exact pass) and the other half from the fill, so the NLL cost
                of a filled K and a filled V can be told apart. GDN layers and the tail run as usual. No exact
                rows, refresh or fill reuse.
K or V rescale (KN1 / VN1, exploratory): on late attention layers the prefix rows' filled K (or V) is rescaled, the
other half and GDN layers filled as usual. Norms per row and KV head in fp32; RoPE is a rotation, so pre- and
post-RoPE norms are the same.
  rescale = 'knorm' | 'vnorm'  (oracle) each row's K (V) rescaled to the norm of its true K (V), filled direction kept
          = 'kdir'             (oracle) the true K direction with the filled K norm
          = 'ktail' | 'vtail'  (deployable) times one scale per KV head: the mean over tail rows [P, N - k) of
                ||true K|| / ||filled K|| (V for vtail), the true K from the tail's own layer-i input (it runs exactly)
                and the filled K from the projector applied to its h_S rows (plus the tail bias where tail_bias_from
                corrects the layer, so the scale measures the fill the prefix gets); k = rescale_skip_last keeps the
                scored last rows out (0 <= k < T)
  Fill 'proj' only; no f1, exact_part, exact rows, refresh or fill reuse; ktail / vtail: no context.
Exact rows (bypass, finish(exact_rows=R)): prefix rows R are carried through every late layer by
the real layer computation over the cache as filled (projected rows plus exact rows; on GDN layers
the recurrent state at the row's position); their late inputs replace the predicted ones, so their
K/V and GDN contributions are exact-path values. Rows outside R are filled as before. With fill
'none' (skip+select, FastKV-style) the other prefix rows have no late K/V: the rows R run each late
layer on their own (attending to each other only; the GDN conv and state see rows R alone, as if
adjacent), and the tail then attends to rows R and itself.
"""
import json
from dataclasses import dataclass, field

import dataclasses
import torch
import torch.nn.functional as F


ORACLE_RESCALE = ("knorm", "kdir", "vnorm")     # read the true late inputs, as oracle fills
TAIL_RESCALE = ("ktail", "vtail")               # deployable: a per-head scale measured on the exact tail


@dataclass
class Arm:
    name: str
    split: int | None = None          # None: exact
    tail: int = 0
    fill: str = "exact"
    projector: object = None
    st: dict | None = None
    f1: bool = False
    fill_by_type: dict | None = None
    kvfold: object = None
    context: int = 0                  # projector input [b, c_{S-1}, .., c_{S-context}]
    refresh: int | None = None        # layer run in full on the filled prefix rows
    refresh_projector: object = None  # split refresh + 1, applied to that layer's output
    tail_bias_from: int | None = None  # proj fills of layers >= this add the tail's mean projector residual
    tail_bias_skip_last: int = 0       # the last rows of the prompt stay out of that mean
    exact_part: str | None = None      # 'k' | 'v': that half of late attention K/V from the true inputs (KV1)
    rescale: str | None = None         # knorm | kdir | vnorm (oracle) | ktail | vtail: late attention K or V rescaled
    rescale_skip_last: int = 0         # ktail / vtail: the last rows of the prompt stay out of the tail scale
    tail_from: str | None = None       # 'continuation': the tail is the scored rows, set per prompt (pop-native)
    meta: dict = field(default_factory=dict)

    def prefix_len(self, n):
        if self.fill == "exact":
            return 0
        if self.fill == "swiftkv_native":
            return n
        return max(0, n - self.tail)

    def fill_of(self, layer_type):
        """The fill of a late layer of this type ('attn' | 'gdn')."""
        if self.fill_by_type is None:
            return self.fill
        return self.fill_by_type["attention" if layer_type == "attn" else layer_type]

    def uses(self, fill):
        return fill in (self.fill_by_type.values() if self.fill_by_type else (self.fill,))

    def uses_oracle(self):
        return self.uses("oracle") or self.exact_part is not None or self.rescale in ORACLE_RESCALE


MAX_ABS_WEIGHT = 1e4   # every fitted projector is below 100 (scan of data/proj, 2026-10-03)


def check_weights(tensors, where):
    """Refuse a projector whose stored tensors ({'layer.i': [d_out, d_in + 1], bias last}) hold a non-finite
    value or |w| > MAX_ABS_WEIGHT: a broken fit, or a file corrupted after it was written (projector-defect,
    2026-10-03: two 64-byte runs of garbage in the page cache, up to 3.4e38). Raises ValueError naming the layer."""
    for k, t in tensors.items():
        t = t.float()
        bad = ~torch.isfinite(t) | (t.abs() > MAX_ABS_WEIGHT)
        if bad.any():
            r, c = bad.nonzero()[0].tolist()
            raise ValueError(f"{where}: {k} has {int(bad.sum())} non-finite or |w| > {MAX_ABS_WEIGHT:g} weights "
                             f"(first at row {r}, col {c}: {t[r, c].item():.4g}); refusing")


class Projector:
    """Per late layer: x_i = b @ W_i^T + c_i in bf16 (as the engine applies it). The maps may live on
    the CPU: the first call on a device copies them there; release() drops the device copy (sweeps
    keep one projector on the GPU at a time). save() and load() refuse non-finite or huge weights
    (check_weights); load() also checks the file's sha256."""

    def __init__(self, split, weights, biases, meta=None):
        self.split = split
        self.W = weights
        self.b = biases
        self.meta = meta or {}
        self._dev = None

    def __call__(self, i, b):
        if self.W[i].device != b.device:
            if self._dev is None or self._dev[0] != b.device:
                self._dev = (b.device, {k: w.to(b.device) for k, w in self.W.items()},
                             {k: x.to(b.device) for k, x in self.b.items()})
            return F.linear(b, self._dev[1][i], self._dev[2][i])
        return F.linear(b, self.W[i], self.b[i])

    def release(self):
        self._dev = None

    @classmethod
    def load(cls, path, device, dtype=torch.bfloat16, sha256=None):
        """The file is read once, its sha256 checked against `sha256` (the one recorded when it was written;
        default: the disk copy read around the page cache, kva/integrity.py: host RAM flips bits) and the maps
        are built from those same bytes. The verified hash is kept as .sha256."""
        from safetensors.torch import load as from_bytes
        from . import integrity
        data, sha = integrity.read_verified(path, sha256)
        meta = json.loads(data[8:8 + int.from_bytes(data[:8], "little")]).get("__metadata__") or {}
        W, B = {}, {}
        for k, t in from_bytes(data).items():
            i = int(k.split(".")[1])
            check_weights({k: t}, f"Projector.load {path}")
            W[i] = t[:, :-1].to(device, dtype).contiguous()
            B[i] = t[:, -1].to(device, dtype).contiguous()
        p = cls(int(meta["split"]), W, B, meta)
        p.sha256 = sha
        return p

    @staticmethod
    def meta_of(path):
        """The file's metadata (strings, as save() writes them) from its safetensors header alone, without the maps:
        what a provenance check reads (fit_sweep.py's stale-projector refusal, finals/lib/kvabackend.py)."""
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            return json.loads(f.read(n)).get("__metadata__") or {}

    def save(self, path, meta):
        from safetensors.torch import save_file
        tensors = {f"layer.{i}": torch.cat((self.W[i], self.b[i][:, None]), 1).to(torch.bfloat16).cpu().contiguous()
                   for i in sorted(self.W)}
        check_weights(tensors, f"Projector.save {path}")
        save_file(tensors, str(path), metadata={k: str(v) for k, v in dict(meta, split=self.split).items()})


class KVFold(Projector):
    """Per late attention layer: pre-RoPE [k | v] = u @ W_i^T + c_i, u = b / rms(b) (eps as the model's
    input norm; fp32, cast back), W_i [2 kv_heads head_dim, H] with the per-head scales already folded in
    (fit_kvfold.py). Storage and device handling as Projector; metadata n_kv and eps."""

    def __init__(self, split, weights, biases, n_kv, eps, meta=None):
        super().__init__(split, weights, biases, meta)
        self.n_kv, self.eps = n_kv, eps

    def kv(self, i, b):
        """(k, v) [tokens, kv_heads, head_dim] of layer i for boundary rows b."""
        xf = b.float()
        u = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(b.dtype)
        return self(i, u).unflatten(-1, (2, self.n_kv, -1)).unbind(1)

    @classmethod
    def load(cls, path, device, dtype=torch.bfloat16, sha256=None):
        p = Projector.load(path, device, dtype, sha256)
        fold = cls(p.split, p.W, p.b, int(p.meta["n_kv"]), float(p.meta["eps"]), p.meta)
        fold.sha256 = p.sha256
        return fold

    def save(self, path, meta):
        super().save(path, dict(meta, n_kv=self.n_kv, eps=self.eps))


def kv_target(model, i, h):
    """The fold's target for layer inputs h: pre-RoPE [k | v] [tokens, 2 kv_heads head_dim] (Model.reads)."""
    r = model.reads(i, h)
    return torch.cat((r["k"].flatten(1), r["v"].flatten(1)), 1)


def splice_kv(model, i, x, xo, part):
    """Pre-RoPE (k, v) of attention layer i for prefix rows (Model.reads, the fold's kv_target layout): `part`
    ('k' | 'v') from the true inputs xo, the other from the filled inputs x (KV1)."""
    f, e = model.reads(i, x), model.reads(i, xo)
    return (e["k"], f["v"]) if part == "k" else (f["k"], e["v"])


def rescale_kv(model, i, x, mode, xo=None, scale=None):
    """Pre-RoPE (k, v) of attention layer i for prefix rows filled from x with one half rescaled per row and KV head
    (Arm.rescale, KN1 / VN1): knorm / vnorm to the norm of the true inputs xo, kdir xo's direction at the filled norm,
    ktail / vtail times scale [kv_heads, 1] (tail_scale). In fp32, cast back."""
    f = model.reads(i, x)
    part = mode[0]
    a = f[part].float()
    if scale is None:
        t = model.reads(i, xo)[part].float()
        na, nt = a.norm(dim=-1, keepdim=True), t.norm(dim=-1, keepdim=True)
        a = t * (na / nt.clamp_min(1e-30)) if mode == "kdir" else a * (nt / na.clamp_min(1e-30))
    else:
        a = a * scale
    a = a.to(f[part].dtype)
    return (a, f["v"]) if part == "k" else (f["k"], a)


def tail_scale(model, i, h_true, x_fill, part):
    """[kv_heads, 1]: mean over tail rows of ||true|| / ||filled|| of attention layer i's K (part 'k') or V ('v'), from
    the rows' true inputs h_true and their fill x_fill. Both are known at run time (the tail runs exactly), so this
    is the deployable scale of ktail / vtail."""
    t = model.reads(i, h_true)[part].float().norm(dim=-1)
    f = model.reads(i, x_fill)[part].float().norm(dim=-1)
    return (t / f.clamp_min(1e-30)).mean(0)[:, None]


def pop_split(n_layers):
    """POP's split (He et al., arXiv:2602.03295v2 section 3.3: prefill prunes the last floor(L / 3) layers): S = L -
    floor(L / 3), so 22 of 32 (Llama-3.1-8B, Qwen3.5-9B) and 24 of 36 (Qwen3-8B)."""
    return n_layers - n_layers // 3


ENGINE_MIN_TAIL = 512   # his scheduler's MIN_TAIL (b0/reference-tcc-ced/v1_ced.py): the engine never runs a shorter tail


def rounded_tail(n, base, block):
    """Exact-tail length when the prefix boundary is rounded down to a block multiple, as a block-granular scheduler does
    (tail_start = (N - base) // block * block): T = N - that start, so T >= base (up to base + block - 1)."""
    return n - max(0, (n - base) // block * block)


def context_input(b, mix, split, k):
    """The context projector's input rows: [b, c_{split-1}, .., c_{split-k}] (mix {layer: mixer output}, the
    same rows as b)."""
    return torch.cat([b] + [mix[split - j].to(b.device, b.dtype) for j in range(1, k + 1)], -1)


def run_lower(model, ids, splits, hs_device=None, hook=None, mix=None):
    """Exact forward up to max(splits). Returns ({split: h_split [N, H]}, cache with layers <
    max(splits) filled). Arms at split S reuse cache[:S]. hs_device: where to keep the boundary
    states (e.g. 'cpu' when many splits are evaluated; finish() moves them back as needed).
    hook(i, h): each layer's input rows (all N), as forward_exact's hook; mix(i, a): each layer's
    mixer output (Model.layer), for context arms."""
    ids = ids.to(model.device)
    pos = torch.arange(ids.shape[0], device=model.device)
    cache = model.new_cache()
    h = model.embed(ids)
    hs = {}
    for i in range(max(splits)):
        if i in splits:
            hs[i] = h if hs_device is None else h.to(hs_device)
        if hook is not None:
            hook(i, h)
        h = model.layer(i, h, pos, cache, mix=mix)
    hs[max(splits)] = h if hs_device is None else h.to(hs_device)
    return hs, cache


def _late_inputs_row0(model, ids, split):
    """Exact late-layer inputs of position 0 (it depends on itself only)."""
    got = {}
    model.forward_exact(ids[:1], hook=lambda i, h: got.__setitem__(i, h) if i != "final" and i >= split else None)
    return got


def oracle_tap(store, first, rows, device="cpu"):
    """hook(i, h) for run_lower / finish of the exact arm: keeps rows [0, rows) of every layer
    input i >= first in store (the true late-layer inputs that oracle fills read)."""
    def hook(i, h):
        if i != "final" and i >= first:
            store[i] = h[:rows].to(device)
    return hook


def finish(model, ids, h_split, lower_cache, arm, keep, oracle=None, keep_cache=True, fill_store=None,
           fill_reuse=None, exact_rows=None, row_inputs=None, hook=None, mix=None):
    """Layers >= arm.split for one arm. keep: number of final rows whose logits to return
    (rows N-keep .. N-1). keep_cache=False drops each late layer's cache once the layer is done
    (scoring without decode; saves memory). fill_store (dict): receives each late layer's cache
    right after the prefix fill (before +st); fill_reuse: such a dict from an arm with the same
    fill, split and P, used instead of recomputing the fill (+st arms reuse their base arm's fill).
    exact_rows: sorted prefix rows (< P) run exactly through the late layers (empty or None: plain
    fill); row_inputs (dict): receives their late-layer inputs {layer: [len(R), H]}. mix: {layer: mixer
    output [N, H]} of the lower layers split-1 .. split-arm.context (context arms).
    Returns (logits [keep, V] fp32, cache, P)."""
    n = ids.shape[0]
    split = arm.split
    P = arm.prefix_len(n)
    pos = torch.arange(n, device=model.device)
    h_split = h_split.to(model.device)
    cache = [dict(c) for c in lower_cache[:split]] + [dict() for _ in range(model.n_layers - split)]
    if arm.fill == "swiftkv_native":
        if split != model.swift_layers:
            raise ValueError(f"SwiftKV native runs at its own split {model.swift_layers}")
        q0 = n - keep
        h = h_split[q0:]
        for i in range(split, model.n_layers):
            model.swift_fill(i, h_split, pos, cache)
            h = model.swift_layer(i, h, pos[q0:], cache)
            if not keep_cache:
                cache[i] = {}
        return model.logits(h), cache, P
    if keep > n - P:
        raise ValueError(f"keep {keep} rows > exact tail {n - P}")
    b = h_split[:P]
    h = h_split[P:]
    bx = b                                               # the projector's input rows
    if arm.context:
        if mix is None or not arm.uses("proj") or arm.uses("swiftkv") or arm.uses("kvfold"):
            raise ValueError("context arms fill from a projector and need the lower mixer outputs (mix)")
        bx = context_input(b, {j: mix[j][:P] for j in range(split - arm.context, split)}, split, arm.context)
    row0 = _late_inputs_row0(model, ids, split) if (arm.f1 and P > 0) else None
    rows = None
    if exact_rows is not None and len(exact_rows) > 0:
        rows = torch.as_tensor(exact_rows, dtype=torch.long, device=model.device)
        if (fill_reuse is not None or arm.fill == "swiftkv" or arm.uses("kvfold") or int(rows[-1]) >= P
                or bool((rows.diff() <= 0).any())):
            raise ValueError("exact rows: sorted unique rows < P, no fill reuse, not swiftkv or kvfold")
        hr = h_split[rows]
    R = arm.refresh
    if R is not None and (not split <= R < model.n_layers - 1 or arm.fill != "proj" or arm.fill_by_type
                          or fill_reuse is not None or arm.f1 or arm.st
                          or getattr(arm.refresh_projector, "split", None) != R + 1):
        raise ValueError("refresh R: split <= R < L - 1, fill 'proj', a refresh_projector of split R + 1; "
                         "no fill reuse, f1 or st")
    part = arm.exact_part
    if part is not None and (part not in ("k", "v") or oracle is None or rows is not None or R is not None
                             or fill_reuse is not None):
        raise ValueError("exact_part 'k' | 'v' reads the true late inputs (oracle); no exact rows, refresh or "
                         "fill reuse")
    tb = arm.tail_bias_from
    if tb is not None and (arm.context or R is not None or fill_reuse is not None):
        raise ValueError("tail_bias_from: no context or refresh (the tail's projector input is h_S), no fill reuse "
                         "(the reusing arm's own tail states set its correction)")
    skip = arm.tail_bias_skip_last
    if (tb is None and skip) or (tb is not None and not 0 <= skip < n - P):
        raise ValueError(f"tail_bias_skip_last {skip}: needs tail_bias_from and 0 <= it < the tail ({n - P})")
    t_mean = n - P - skip                                # tail rows [P, n - skip) enter the tail mean
    rs, rskip = arm.rescale, arm.rescale_skip_last
    if rs is not None and (rs not in ORACLE_RESCALE + TAIL_RESCALE or arm.fill != "proj" or arm.f1
                           or part is not None or rows is not None or R is not None or fill_reuse is not None
                           or (rs in ORACLE_RESCALE and oracle is None) or (rs in TAIL_RESCALE and arm.context)):
        raise ValueError("rescale knorm | kdir | vnorm (true inputs: oracle) | ktail | vtail (no context): "
                         "fill 'proj'; no f1, exact_part, exact rows, refresh or fill reuse")
    if (rs not in TAIL_RESCALE and rskip) or (rs in TAIL_RESCALE and not 0 <= rskip < n - P):
        raise ValueError(f"rescale_skip_last {rskip}: needs rescale ktail | vtail and 0 <= it < the tail ({n - P})")
    t_scale = n - P - rskip                              # tail rows [P, n - rskip) enter the tail scale
    for i in range(split, model.n_layers):
        if P > 0 and fill_reuse is not None:
            if fill_reuse[i].get("n", P) != P:      # attention caches record their length
                raise ValueError(f"reused fill of layer {i} has {fill_reuse[i]['n']} rows, P = {P}")
            cache[i] = dict(fill_reuse[i])
        elif P > 0:
            fill = arm.fill_of(model.layer_types[i])
            if fill == "swiftkv":
                model.swift_fill(i, b, pos[:P], cache)
            elif fill == "kvfold":
                if model.layer_types[i] != "attn" or row0 is not None:
                    raise ValueError("kvfold fills attention layers only (fill_by_type for GDN layers), without f1")
                model.fill_kv(i, *arm.kvfold.kv(i, b), pos[:P], cache)
            elif fill == "none":                         # plain layer skip: prefix rows never reach layer i
                if row0 is not None:
                    raise ValueError("fill 'none' takes no f1")
                if rows is not None:                     # FastKV-style: only the exact rows reach layer i
                    if row_inputs is not None:
                        row_inputs[i] = hr
                    hr = model.layer(i, hr, pos[rows], cache)
            else:
                bias = None
                if fill == "proj":
                    x = arm.projector(i, bx) if R is None or i <= R else arm.refresh_projector(i, bx)
                    if tb is not None and i >= tb:       # h: the tail's layer-i input, before layer i runs
                        r = h[:t_mean].float() - arm.projector(i, h_split[P:n - skip]).float()
                        bias = r.mean(0).to(x.dtype)
                        x = x + bias
                elif fill == "identity":
                    x = b
                elif fill == "oracle":
                    x = oracle[i][:P].to(b.device)
                else:
                    raise ValueError(fill)
                if row0 is not None:
                    x = torch.cat((row0[i], x[1:]), 0)
                if rows is not None:
                    if row_inputs is not None:
                        row_inputs[i] = hr
                    x = x.index_copy(0, rows, hr.to(x.dtype))
                if i == R:                               # refresh: the full layer on the filled prefix rows
                    bx = model.layer(i, x, pos[:P], cache)
                    if rows is not None:
                        hr = bx[rows]
                elif rows is not None:
                    hr = model.layer_rows(i, x, pos[:P], rows, cache)
                elif part is not None and model.layer_types[i] == "attn":
                    model.fill_kv(i, *splice_kv(model, i, x, oracle[i][:P].to(x.device), part), pos[:P], cache)
                elif rs is not None and model.layer_types[i] == "attn":
                    if rs in TAIL_RESCALE:
                        xt = arm.projector(i, h_split[P:n - rskip])
                        kv = rescale_kv(model, i, x, rs, scale=tail_scale(
                            model, i, h[:t_scale], xt if bias is None else xt + bias, rs[0]))
                    else:
                        kv = rescale_kv(model, i, x, rs, xo=oracle[i][:P].to(x.device))
                    model.fill_kv(i, *kv, pos[:P], cache)
                else:
                    model.fill(i, x, pos[:P], cache)
        if P > 0 and fill_store is not None:
            fill_store[i] = dict(cache[i])
        st = arm.st.get(i) if (arm.st and P > 0) else None
        if st is not None:
            cache[i]["rec"] = cache[i]["rec"] + st.to(cache[i]["rec"])
        if hook is not None:
            hook(i, h)
        h = model.layer(i, h, pos[P:], cache)
        if not keep_cache:
            cache[i] = {}
    return model.logits(h[-keep:]), cache, P


def score_tokens(logits, ids, keep_targets):
    """Log-probs of the last keep_targets prompt tokens; logits = the last keep_targets + 1 rows."""
    lp = torch.log_softmax(logits[:-1].float(), -1)
    tgt = ids[-keep_targets:].to(lp.device)
    return lp.gather(1, tgt[:, None])[:, 0].cpu()


class NonFiniteLogits(RuntimeError):
    """Raised when logits contain NaN/inf: on 10-02 three 32K prompts on card 03 produced NaN logits in
    every arm (scored as plain misses). A broken forward must stop the run, never become a data point."""


def check_finite(logits, where):
    if not bool(torch.isfinite(logits).all()):
        raise NonFiniteLogits(f"non-finite logits ({where})")
    return logits



def exclude_scored(arm, scored):
    """The arm with its tail statistics kept off the last `scored` rows of the sequence (v1.1.3). On the sets that
    score prompt tokens teacher-forced (longppl: the continuation; grade: the last 512), those rows are the targets:
    a tail-measured rescale (ktail / vtail) or tail bias that averages over them puts the targets into the prefix
    cache that predicts them. The finals cells of protocol 1.1.1 / 1.1.2 ran with skip 0 (a regression from the dev
    configs' 512; reviews/2026-10-07-codex-critique-assessment-fable.md in the paper workspace)."""
    if scored <= 0:
        return arm
    if arm.rescale in ("ktail", "vtail") and arm.rescale_skip_last < scored:
        arm = dataclasses.replace(arm, rescale_skip_last=scored)
    if arm.tail_bias_from is not None and arm.tail_bias_skip_last < scored:
        arm = dataclasses.replace(arm, tail_bias_skip_last=scored)
    return arm


@torch.no_grad()
def greedy(model, ids, cache, first_logits, steps, arm, eos=()):
    """Greedy continuation of up to `steps` tokens, stopping after an EOS token; decode is exact
    (SwiftKV native: its own decode path). Returns (token ids, 'stop' | 'length')."""
    out = []
    tok = int(check_finite(first_logits[-1], f"{arm.name} first token").argmax())
    n = ids.shape[0]
    native = arm.fill == "swiftkv_native"
    for t in range(steps):
        out.append(tok)
        if tok in eos:
            return out, "stop"
        if t == steps - 1:
            break
        pos = torch.tensor([n + t], device=model.device)
        h = model.embed(torch.tensor([tok], device=model.device))
        for i in range(model.n_layers):
            if native and i >= arm.split:
                if i == arm.split:
                    b = h
                model.swift_fill(i, b, pos, cache)
                h = model.swift_layer(i, h, pos, cache)
            else:
                h = model.layer(i, h, pos, cache)
        tok = int(check_finite(model.logits(h)[-1], f"{arm.name} decode step {t + 1}").argmax())
    return out, "length"
