"""Fill impact (TE2): per approximated prefix row j, the first-order change in the scored loss if row j's late
attention K/V were exact instead of filled:

  impact_j = sum over late attention layers of <dL/dK_j, K_j^true - K_j^fill> + <dL/dV_j, V_j^true - V_j^fill>

L = mean NLL of the scored targets, the gradient taken at the filled cache (every late layer filled, as the
kva / noproj arms). Negative impact: making row j exact lowers the loss. K is the cached (post-RoPE) key; RoPE
is a rotation, so the inner product equals the pre-RoPE one.

GDN layers: the prefix enters through the conv and recurrent state at P, held constant here (per-row GDN
attribution would need backprop through the prefix recurrence). So impact covers attention K/V only.

The model's fast path cannot be differentiated: _append_kv writes into buffers, ops.attend would keep
~P x tail fp32 probabilities per layer for backward, and ops.delta_rule's triangular inverse is in place.
This module has its own tail path for the late layers (same math, blocked attention with a recomputing
backward, a GDN chunked delta rule built from out-of-place ops), checkpointed per layer.
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import ops


class Attend(torch.autograd.Function):
    """ops.attend (bottom-right causal, GQA) with a flash-style backward: probabilities are recomputed per
    K/V head group and query block, so memory stays at one block of scores. The forward's bf16 cast of the
    probabilities is treated as identity in the backward."""

    @staticmethod
    def forward(ctx, q, k, v, scale):
        out = ops.attend(q, k, v, scale)
        ctx.save_for_backward(q, k, v, out)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out = ctx.saved_tensors
        scale = ctx.scale
        hq, nq, d = q.shape
        hkv, nk, _ = k.shape
        rep = hq // hkv
        do = grad.view(nq, hq, -1).transpose(0, 1).float()
        dsum = (do * out.view(nq, hq, -1).transpose(0, 1).float()).sum(-1)          # [hq, nq]
        dq = torch.zeros(hq, nq, d, dtype=torch.float32, device=q.device)
        dk = torch.zeros(hkv, nk, d, dtype=torch.float32, device=q.device)
        dv = torch.zeros(hkv, nk, v.shape[-1], dtype=torch.float32, device=q.device)
        rows = max(16, ops.ATTN_BLOCK_SCORES // (rep * nk))
        for g in range(hkv):
            hs = slice(g * rep, (g + 1) * rep)
            for a in range(0, nq, rows):
                b = min(nq, a + rows)
                end = nk - (nq - b)
                kf, vf = k[g, :end].float(), v[g, :end].float()
                qg = q[hs, a:b].float()
                s = torch.matmul(qg, kf.T).mul_(scale)                                # [rep, b - a, end]
                qpos = torch.arange(nk - nq + a, nk - nq + b, device=q.device)
                s.masked_fill_(torch.arange(end, device=q.device)[None, :] > qpos[:, None], float("-inf"))
                p = torch.softmax(s, -1)
                del s
                dog = do[hs, a:b]
                ds = p * (torch.matmul(dog, vf.T) - dsum[hs, a:b, None])
                dq[hs, a:b] = torch.matmul(ds, kf) * scale
                dk[g, :end] += torch.matmul(ds.transpose(1, 2), qg).sum(0) * scale
                dv[g, :end] += torch.matmul(p.transpose(1, 2), dog).sum(0)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


def delta_rule_ad(q, k, v, g, beta, state, chunk=64):
    """ops.delta_rule's chunked gated delta rule (same chunk grid and formulas) with out-of-place ops only, so
    autograd can run through it; (I + A)^-1 by a unit lower triangular solve. Returns out [H, n, dv] fp32."""
    h, n, dk = k.shape
    dv = v.shape[-1]
    pad = (-n) % chunk
    nc = (n + pad) // chunk

    def seg(x):
        x = x.float()
        return F.pad(x, (0, 0, 0, pad) if x.dim() == 3 else (0, pad)) if pad else x
    qs, ks, vs = seg(q).view(h, nc, chunk, dk), seg(k).view(h, nc, chunk, dk), seg(v).view(h, nc, chunk, dv)
    gam = seg(g).view(h, nc, chunk).cumsum(-1)
    bs = seg(beta).view(h, nc, chunk)
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=k.device).tril()
    diff = gam.unsqueeze(-1) - gam.unsqueeze(-2)
    decay = torch.where(tri, diff, torch.full_like(diff, float("-inf"))).exp()
    a = (bs.unsqueeze(-1) * (ks @ ks.transpose(-1, -2)) * decay).tril(-1)
    eye = torch.eye(chunk, dtype=a.dtype, device=a.device).expand_as(a)
    tinv = torch.linalg.solve_triangular(a + eye, eye, upper=False, unitriangular=True)
    u = tinv @ (bs.unsqueeze(-1) * vs)
    w = tinv @ (bs.unsqueeze(-1) * gam.exp().unsqueeze(-1) * ks)
    qk = (qs @ ks.transpose(-1, -2)) * decay
    qd = qs * gam.exp().unsqueeze(-1)
    kd = ks * (gam[..., -1:] - gam).exp().unsqueeze(-1)
    cd = gam[..., -1].exp()
    state = state.float()
    outs = []
    for c in range(nc):
        v_new = u[:, c] - w[:, c] @ state
        outs.append(qd[:, c] @ state + qk[:, c] @ v_new)
        state = state * cd[:, c, None, None] + kd[:, c].transpose(-1, -2) @ v_new
    return torch.cat(outs, 1)[:, :n]


def kv_of(m, i, x, pos):
    """Post-RoPE K, V [kv_heads, n, d] of attention layer i for normed inputs x (as cached)."""
    return m._attn_kv(i, x, pos) if m.arch == "qwen3_5" else m._kv(i, x, pos)     # Llama, Qwen3: _kv


def _attn_layer(m, i, h, pos, kp, vp):
    """Late attention layer i for tail rows h over [prefix K/V kp, vp; the rows' own K/V] (= Model.layer)."""
    w = m.W[i]
    x = m.in_norm(h, w["ln1"])
    if m.arch == "qwen3_5":
        q, gate = m._attn_q(i, x, pos)
    else:
        q, gate = m.queries(i, x, pos), None
    k, v = kv_of(m, i, x, pos)
    o = Attend.apply(q, torch.cat((kp, k), 1), torch.cat((vp, v), 1), m.scale)
    if gate is not None:
        o = o * torch.sigmoid(gate)
    return m.mlp(i, h + F.linear(o, w["o"]))


def _gdn_layer(m, i, h, conv, rec):
    """Late GDN layer i for tail rows h continuing the (constant) conv and recurrent state (= Model.layer)."""
    w = m.W[i]
    x = m.in_norm(h, w["ln1"])
    n = x.shape[0]
    pre, b, a = m._gdn_inputs(i, x)
    z = F.linear(x, w["z"])
    conv_out, _ = ops.causal_conv(pre, conv, w["conv"])
    kd = m.nk * m.dk
    rep = m.nv // m.nk
    q, k, v = conv_out.split([kd, kd, m.nv * m.dv], -1)
    q = q.reshape(n, m.nk, m.dk).float().repeat_interleave(rep, 1).transpose(0, 1)
    k = k.reshape(n, m.nk, m.dk).float().repeat_interleave(rep, 1).transpose(0, 1)
    v = v.reshape(n, m.nv, m.dv).float().transpose(0, 1)
    beta = torch.sigmoid(b).float().T
    g = (-w["A_log"].float().exp() * F.softplus(a.float() + w["dt_bias"].float())).T
    out = delta_rule_ad(ops.l2norm(q) * m.dk ** -0.5, ops.l2norm(k), v, g, beta, rec).to(pre.dtype).transpose(0, 1)
    y = ops.rms_norm_gated(out, z.view(n, m.nv, m.dv), w["norm"], m.eps).reshape(n, -1)
    return m.mlp(i, h + F.linear(y, w["out"]))


def tail_loss(m, split, h0, pos, kv, gdn, rows, tgt, ckpt=True):
    """Mean NLL (fp32) of targets tgt at tail rows `rows` (indices into the tail), running the tail rows
    (layer-split inputs h0, positions pos) through layers >= split over prefix caches kv {i: (K, V)} and
    gdn {i: dict(conv, rec)}. ckpt: recompute each layer in the backward (memory)."""
    h = h0
    for i in range(split, m.n_layers):
        if m.layer_types[i] == "attn":
            fn, args = _attn_layer, (m, i, h, pos, *(t.to(h.device) for t in kv[i]))
        else:
            fn, args = _gdn_layer, (m, i, h, gdn[i]["conv"], gdn[i]["rec"])
        h = checkpoint(fn, *args, use_reentrant=False) if ckpt and torch.is_grad_enabled() else fn(*args)
    return F.cross_entropy(m.logits(h[rows]), tgt)


@torch.no_grad()
def exact_late(m, h_split, split, P, rows, tgt, final=False, hook=None):
    """Exact late layers over all rows from the layer-split state. Returns ({attention layer: (K, V) of rows
    [0, P) on the CPU, post-RoPE as cached}, exact L over prompt rows `rows`), plus the last layer's output rows
    (before the final norm) if final. hook(i, h): each late layer's input rows (all n), as run_lower's."""
    n = h_split.shape[0]
    pos = torch.arange(n, device=m.device)
    cache = m.new_cache()
    true = {}
    h = h_split
    for i in range(split, m.n_layers):
        if hook is not None:
            hook(i, h)
        h = m.layer(i, h, pos, cache)
        if m.layer_types[i] == "attn":
            true[i] = (cache[i]["k"][:, :P].cpu(), cache[i]["v"][:, :P].cpu())
        cache[i] = {}
    loss = float(F.cross_entropy(m.logits(h[rows]), tgt))
    return (true, loss, h) if final else (true, loss)


@torch.no_grad()
def fill_late(m, split, P, fill_x):
    """Prefix fill of every late layer from inputs fill_x(i) [P, H] (projector or identity), as finish()
    does. Returns ({attention layer: (K, V)}, {GDN layer: dict(conv, rec)})."""
    pos = torch.arange(P, device=m.device)
    cache = m.new_cache()
    kv, gdn = {}, {}
    for i in range(split, m.n_layers):
        m.fill(i, fill_x(i), pos, cache)
        if m.layer_types[i] == "attn":
            kv[i] = (cache[i]["k"], cache[i]["v"])
        else:
            gdn[i] = dict(conv=cache[i]["conv"], rec=cache[i]["rec"])
        cache[i] = {}
    return kv, gdn


def impact(m, split, h_tail, P, kv, true_kv, gdn, rows, tgt):
    """Backward through the tail at the filled cache. Per prefix row [P] (fp64, CPU): impact (signed), the
    relative fill error (mean over late attention layers of (|dK|/|K| + |dV|/|V|) / 2, as TE1) and the
    gradient norm over all late attention K/V. Each layer's gradient is reduced as it arrives and freed.
    Returns (dict of tensors, L at the fill)."""
    dev = h_tail.device
    pos = torch.arange(P, P + h_tail.shape[0], device=dev)
    acc = {k: torch.zeros(P, dtype=torch.float64, device=dev) for k in ("impact", "err", "grad2")}
    leaves = {}

    def reduce(i, which):
        def hook(p):
            true = true_kv[i][which].to(dev).float()
            delta = true - p.detach().float()
            g = p.grad.float()
            acc["impact"] += (g * delta).sum((0, 2)).double()
            acc["grad2"] += g.pow(2).sum((0, 2)).double()
            ref = true.pow(2).sum((0, 2)).sqrt().clamp_min(1e-6)
            acc["err"] += 0.5 * (delta.pow(2).sum((0, 2)).sqrt() / ref).double()
            p.grad = None
        return hook
    for i, (k, v) in kv.items():
        leaves[i] = (k.detach().requires_grad_(), v.detach().requires_grad_())
        for which, p in enumerate(leaves[i]):
            p.register_post_accumulate_grad_hook(reduce(i, which))
    with torch.enable_grad():
        loss = tail_loss(m, split, h_tail, pos, leaves, gdn, rows, tgt)
        loss.backward()
    acc["err"] /= max(1, len(kv))
    out = dict(impact=acc["impact"].cpu(), err=acc["err"].cpu(), grad_norm=acc["grad2"].sqrt().cpu())
    return out, float(loss.detach())


@torch.no_grad()
def received_mass(q, k, scale, rows_below=None, stride=1, entropy=False):
    """Attention mass each key row receives from the strictly later query rows (H2O-style causal column sums),
    summed over query heads. q [Hq, n, d] post-RoPE queries of all n rows, k [Hkv, n, d]. rows_below: also
    return the mass from query rows < rows_below only (a cached prefix's own rows). stride > 1: only every
    stride-th query row (rows 0, s, 2s, ...), scaled by stride - an unbiased-in-expectation estimate that cuts
    the cost by ~stride (10-02: the full pass made qfp arms ~10 min per 32K prompt on Llama). entropy: also return
    each query row's own attention entropy (nats, over its causal keys, summed over heads; needs stride 1).
    Returns fp64 CPU [n] (then [n] for rows_below, then [n] for entropy)."""
    assert stride == 1 or not entropy, "entropy needs every query row (stride 1)"
    hq, n, d = q.shape
    hkv = k.shape[0]
    rep = hq // hkv
    tot = torch.zeros(n, dtype=torch.float64, device=q.device)
    low = torch.zeros(n, dtype=torch.float64, device=q.device)
    ent = torch.zeros(n, dtype=torch.float64, device=q.device)
    rows = max(16, ops.ATTN_BLOCK_SCORES // (rep * n)) * stride
    for g in range(hkv):
        kf = k[g].float()
        for a in range(0, n, rows):
            b = min(n, a + rows)
            qpos = torch.arange(a, b, stride, device=q.device)
            s = torch.matmul(q[g * rep:(g + 1) * rep, qpos].float(), kf[:b].T).mul_(scale)
            keys = torch.arange(b, device=q.device)
            s.masked_fill_(keys[None, :] > qpos[:, None], float("-inf"))
            p = torch.softmax(s, -1)
            if entropy:
                ent[a:b] += torch.special.entr(p).sum((0, 2)).double()
            p.masked_fill_(keys[None, :] == qpos[:, None], 0.0)                      # a row's own mass
            per_q = p.sum(0) * stride                                                 # [rows kept, b]
            tot[:b] += per_q.sum(0).double()
            if rows_below is not None and a < rows_below:
                low[:b] += per_q[: int((qpos < rows_below).sum())].sum(0).double()
    out = (tot.cpu(),) + ((low.cpu(),) if rows_below is not None else ()) + ((ent.cpu(),) if entropy else ())
    return out if len(out) > 1 else out[0]


@torch.no_grad()
def row_stats(q, k, scale, local=64):
    """Shape of each query row's own lookup, averaged over query heads: attention entropy (nats), mass on row 0 (the
    sink) and mass on the `local` most recent keys (self included). A cheap per-token feature for the fill error (BV2).
    q [Hq, n, d] post-RoPE queries, k [Hkv, n, d]. Returns three fp64 CPU [n]."""
    hq, n, d = q.shape
    rep = hq // k.shape[0]
    keys = torch.arange(n, device=q.device)
    ent, sink, near = (torch.zeros(n, dtype=torch.float64, device=q.device) for _ in range(3))
    rows = max(16, ops.ATTN_BLOCK_SCORES // (rep * n))
    for g in range(k.shape[0]):
        kf = k[g].float()
        for a in range(0, n, rows):
            b = min(n, a + rows)
            qpos = keys[a:b]
            s = torch.matmul(q[g * rep:(g + 1) * rep, a:b].float(), kf[:b].T).mul_(scale)
            s.masked_fill_(keys[None, :b] > qpos[:, None], float("-inf"))
            p = torch.softmax(s, -1)
            ent[a:b] -= (p * p.clamp_min(1e-30).log()).sum(-1).sum(0).double()
            sink[a:b] += p[..., 0].sum(0).double()
            recent = (qpos[:, None] - keys[None, :b] < local) & (keys[None, :b] <= qpos[:, None])
            near[a:b] += (p * recent).sum(-1).sum(0).double()
    return (ent / hq).cpu(), (sink / hq).cpu(), (near / hq).cpu()


def leverage_map(stats_path, lam):
    """(mean, L) with ||(x - mean) @ L||^2 = the ridge hat value of a split state x under the projector's own
    calibration scatter (cxx + lam * scale * I, as kva.ridge.RidgeFit fits it): how far a row sits outside the data the
    projector was fitted on, the textbook predictor of a regression's error. CPU fp64."""
    from kva.ridge import RidgeFit
    fit = RidgeFit(torch.load(stats_path, map_location="cpu", weights_only=False, mmap=True))
    return fit.mean_x, fit.evecs / (fit.evals + lam * fit.scale).sqrt()[None, :]


# ---- per-row features of [n, H] states, shared by scripts/bias_probe.py (BV2) and scripts/fill_impact.py (BIGDATA)
ROWS = 4096   # as scripts/bias_probe.py blocks its residual sums


def by_rows(fn, n, chunk=None):
    """torch.cat of fn(rows slice) over row chunks (default ROWS, read at call time), as fp64 CPU. Keeps per-row
    features of [n, H] states at chunk x H temporaries: BV2's full-size fp32/fp64 [n, H] temporaries ran out of memory
    at 32K tokens on Llama (23 GiB card). bias_probe.py's row_sq / block_sum do the same for its residual sums."""
    chunk = chunk or ROWS
    return torch.cat([fn(slice(a, min(n, a + chunk))).double().cpu() for a in range(0, n, chunk)])


def row_norm(x):
    """||x|| per row (TE1's state_norm). fp64 CPU [n]."""
    return by_rows(lambda s: x[s].float().norm(dim=-1), x.shape[0])


def velocity(x, prev):
    """||x - prev|| / ||prev|| per row (TE1's velocity: how far the last layers moved each row). fp64 CPU [n]."""
    def rows(s):
        p = prev[s].float()
        return (x[s].float() - p).norm(dim=-1) / p.norm(dim=-1).clamp_min(1e-6)
    return by_rows(rows, x.shape[0])


def log_leverage(x, mean, L):
    """log ridge hat value of each row of x, with leverage_map's (mean, L) on x's device (fp32). fp64 CPU [n]."""
    return by_rows(lambda s: (((x[s].float() - mean) @ L) ** 2).sum(-1), x.shape[0]).clamp_min(1e-30).log()


def log_max_act(x):
    """log max |x| over each row's hidden dims: massive-activation rows (sink-like tokens). fp64 CPU [n]."""
    return by_rows(lambda s: x[s].float().abs().amax(-1), x.shape[0]).clamp_min(1e-30).log()


@torch.no_grad()
def gdn_gates(m, i, h):
    """Per row at GDN layer i, from its input rows h: the write strength sigmoid(beta) and the log decay
    -exp(A_log) softplus(a + dt_bias), each a mean over value heads. How hard a token overwrites the recurrent
    state may say how much the late layers lean on it. Two fp64 CPU [n]."""
    w = m.W[i]

    def rows(s):
        _, b, a = m._gdn_inputs(i, m.in_norm(h[s], w["ln1"]))
        decay = -w["A_log"].float().exp() * F.softplus(a.float() + w["dt_bias"].float())
        return torch.stack((torch.sigmoid(b.float()).mean(-1), decay.mean(-1)), -1)
    g = by_rows(rows, h.shape[0])
    return g[:, 0], g[:, 1]


def repetition(ids, cap):
    """Per token: log distance to the previous occurrence of the same id (capped at cap, which first occurrences
    get), a first-occurrence flag, and whether (previous id, id) occurred earlier: copy-like rows may be the ones the
    projector gets wrong. Three fp64 lists."""
    last, seen = {}, set()
    dist, first, bigram = [], [], []
    for j, t in enumerate(ids):
        d = j - last[t] if t in last else cap
        dist.append(math.log(max(1, min(d, cap))))
        first.append(0.0 if t in last else 1.0)
        pair = (ids[j - 1], t) if j else None
        bigram.append(1.0 if pair in seen else 0.0)
        if pair is not None:
            seen.add(pair)
        last[t] = j
    return dist, first, bigram
