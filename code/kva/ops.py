"""Numerical building blocks: RMS norms, RoPE, causal attention over a cache, and the Gated
DeltaNet (GDN) causal conv and chunked gated delta rule with an explicit carried state.

Layouts: activations are [tokens, features]; per-head tensors are [heads, tokens, dim]. The GDN
recurrent state is [value_heads, key_dim, value_dim] in fp32, the conv state the last
(kernel - 1) pre-conv rows [kernel - 1, channels].
"""
import math

import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    """Llama RMSNorm: normalise in fp32, cast back, then scale (HF order)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def rms_norm_zc(x, weight, eps):
    """Qwen3.5 zero-centred RMSNorm: (1 + w) applied in fp32, then cast."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).to(x.dtype)


def rms_norm_gated(x, gate, weight, eps):
    """GDN output norm: per-head RMSNorm, then x silu(gate) in fp32 (HF Qwen3_5RMSNormGated)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = weight * xf.to(x.dtype)
    return (out * F.silu(gate.float())).to(x.dtype)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------------------- RoPE

def inv_freq_default(theta, rot_dim):
    return 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float) / rot_dim))


def inv_freq_llama3(theta, dim, factor, low_freq_factor, high_freq_factor, original_max):
    """Llama 3.1 frequency rescaling (as transformers' _compute_llama3_parameters)."""
    inv = inv_freq_default(theta, dim)
    low_wavelen = original_max / low_freq_factor
    high_wavelen = original_max / high_freq_factor
    wavelen = 2 * math.pi / inv
    scaled = torch.where(wavelen > low_wavelen, inv / factor, inv)
    smooth = (original_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(medium, smoothed, scaled)


def rope_cos_sin(inv_freq, positions, dtype):
    """cos/sin [tokens, rot_dim] computed in fp32 and cast (HF order)."""
    freqs = positions.float()[:, None] * inv_freq.to(positions.device)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    """x [heads, tokens, head_dim]; rotates the first cos.shape[-1] dims (partial RoPE)."""
    rot = cos.shape[-1]
    if rot == x.shape[-1]:
        return x * cos + _rotate_half(x) * sin
    xr, xp = x[..., :rot], x[..., rot:]
    return torch.cat((xr * cos + _rotate_half(xr) * sin, xp), dim=-1)


# ----------------------------------------------------------------------------------- attention

ATTN_BLOCK_SCORES = 1 << 26   # max fp32 scores per query block (256 MiB; ~0.6 GiB peak with softmax)


def query_block(hq, nk):
    """Queries per block in attend / attend_at: ATTN_BLOCK_SCORES fp32 scores over nk keys, at least 16."""
    return max(16, ATTN_BLOCK_SCORES // (hq * nk))


def _attend_block(q, kf_t, v, scale):
    """Exact attention for queries that are the last q.shape[1] positions of the keys (bottom-right
    causal). kf_t = K^T in fp32 [Hkv, d, nk]. GQA by grouping query heads per K/V head (no K/V
    expansion); scores and softmax in fp32, probabilities cast to V's dtype for the second matmul."""
    hq, nq, d = q.shape
    hkv, _, nk = kf_t.shape
    rep = hq // hkv
    qg = q.reshape(hkv, rep * nq, d).float()
    s = torch.matmul(qg, kf_t).mul_(scale)                                         # [hkv, rep*nq, nk]
    if nq > 1:
        diag = s.view(hkv, rep, nq, nk)[..., nk - nq:]
        diag.masked_fill_(torch.ones(nq, nq, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    p = torch.softmax(s, -1).to(v.dtype)
    del s
    return torch.matmul(p, v).view(hkv * rep, nq, -1)


def attend(q, k, v, scale):
    """Causal attention of the LAST q.shape[1] positions of a sequence over all k.shape[1]
    positions (queries are bottom-right aligned with the keys). q [Hq, nq, d], k/v [Hkv, nk, d].
    Exact query-blocked matmul attention (the fused SDPA kernels fail on gfx1201 with this torch
    build): each block is again the last rows of a shorter key range. Returns [nq, Hq * d]."""
    hq, nq, d = q.shape
    hkv, nk, _ = k.shape
    kf_t = k.float().transpose(1, 2)
    rows = query_block(hq, nk)
    if nq <= rows:
        out = _attend_block(q, kf_t, v, scale)
    else:
        parts = []
        for a in range(0, nq, rows):
            b = min(nq, a + rows)
            end = nk - (nq - b)
            parts.append(_attend_block(q[:, a:b], kf_t[:, :, :end], v[:, :end], scale))
        out = torch.cat(parts, 1)
    return out.transpose(0, 1).reshape(nq, -1)


def attend_at(q, k, v, scale, qpos):
    """Causal attention of queries at arbitrary sorted positions qpos [nq] over keys at positions
    0 .. k.shape[1] - 1 (query t sees keys j <= qpos[t]). Same arithmetic as _attend_block, blocked
    over queries; each block reads keys up to its last query. Returns [nq, Hq * d].
    The scores are freed before the probabilities' cast, the probabilities before the next block: memory only (the
    old order kept both fp32 tensors, the bf16 copy and the previous block's copy alive at once, ~0.75 GiB at 32K)."""
    hq, nq, d = q.shape
    hkv, nk, _ = k.shape
    rep = hq // hkv
    kf_t = k.float().transpose(1, 2)
    rows = query_block(hq, nk)
    parts = []
    for a in range(0, nq, rows):
        b = min(nq, a + rows)
        end = int(qpos[b - 1]) + 1
        qg = q[:, a:b].reshape(hkv, rep * (b - a), d).float()
        s = torch.matmul(qg, kf_t[:, :, :end]).mul_(scale)
        mask = torch.arange(end, device=q.device)[None, :] > qpos[a:b, None]
        s.view(hkv, rep, b - a, end).masked_fill_(mask, float("-inf"))
        p = torch.softmax(s, -1)
        del s
        p = p.to(v.dtype)
        parts.append(torch.matmul(p, v[:, :end]).view(hq, b - a, -1))
        del p
    return torch.cat(parts, 1).transpose(0, 1).reshape(nq, -1)


def attend_rows(q, k, v, scale, qpos, max_rows):
    """attend_at, for more than max_rows queries in runs of whole query blocks (about max_rows queries each) written
    into one output. The runs make the blocks one call makes, so the result is bitwise the same; the block outputs
    and their concatenation shrink from all queries to one run. A run still casts all of K to fp32: its last
    block reads (nearly) every key, so a per-block cast would not lower the peak."""
    hq, nq, _ = q.shape
    if nq <= max_rows:
        return attend_at(q, k, v, scale, qpos)
    blk = query_block(hq, k.shape[1])
    step = max(blk, max_rows // blk * blk)
    out = v.new_empty(nq, hq * v.shape[-1])
    for a in range(0, nq, step):
        out[a:a + step] = attend_at(q[:, a:a + step], k, v, scale, qpos[a:a + step])
    return out


# ------------------------------------------------------------------------------ GDN primitives

def causal_conv(x, state, weight):
    """Depthwise causal conv + SiLU over time, continuing from `state`.
    x [n, C] (pre-conv), state [K-1, C] or None (zeros), weight [C, K].
    Returns (out [n, C] in x.dtype, new_state [K-1, C] in x.dtype)."""
    kk = weight.shape[1]
    if state is None:
        state = x.new_zeros(kk - 1, x.shape[1])
    xx = torch.cat((state.to(x.dtype), x), 0)
    n = x.shape[0]
    w = weight.float()
    acc = torch.zeros(n, x.shape[1], dtype=torch.float32, device=x.device)
    for j in range(kk):
        acc += xx[j:j + n].float() * w[:, j]
    return F.silu(acc).to(x.dtype), xx[-(kk - 1):].clone()


def _unit_lower_inverse(a):
    """(I + A)^-1 for strictly lower-triangular A [..., C, C], by forward substitution."""
    c = a.shape[-1]
    t = -a.clone()
    for i in range(1, c):
        row = t[..., i, :i].clone()
        t[..., i, :i] = row + (row.unsqueeze(-1) * t[..., :i, :i]).sum(-2)
    return t + torch.eye(c, dtype=a.dtype, device=a.device)


def delta_rule(q, k, v, g, beta, state=None, chunk=64, segment=4096, checkpoint_every=None,
               offset=0, need_out=True):
    """Chunked gated delta rule (fp32), continuing from `state`.

    q, k [H, n, dk] (already l2-normed; q already scaled), v [H, n, dv], g/beta [H, n].
    Per step: S <- exp(g) S; S <- S + k (beta (v - S^T k))^T; o = S^T q.
    The chunk grid starts at this call's first token. `checkpoint_every`: also return
    {offset + t: S_t} at every t (tokens consumed in this call) that is a multiple of it and a
    chunk end. Returns (out [H, n, dv] or None, final state [H, dk, dv], checkpoints dict)."""
    h, n, dk = k.shape
    dv = v.shape[-1]
    if state is None:
        state = torch.zeros(h, dk, dv, dtype=torch.float32, device=k.device)
    state = state.float()
    out = torch.empty(h, n, dv, dtype=torch.float32, device=k.device) if need_out else None
    checkpoints = {}
    if checkpoint_every:
        assert checkpoint_every % chunk == 0 and segment % chunk == 0
    tri_incl = None
    for s0 in range(0, n, segment):
        s1 = min(n, s0 + segment)
        m = s1 - s0
        pad = (-m) % chunk
        nc = (m + pad) // chunk

        def seg(x, fill=0.0):
            x = x[:, s0:s1].float()
            if pad:
                x = F.pad(x, (0, 0, 0, pad) if x.dim() == 3 else (0, pad), value=fill)
            return x

        qs, ks, vs = seg(q), seg(k), seg(v)
        gs, bs = seg(g), seg(beta)
        qs = qs.view(h, nc, chunk, dk)
        ks = ks.view(h, nc, chunk, dk)
        vs = vs.view(h, nc, chunk, dv)
        gam = gs.view(h, nc, chunk).cumsum(-1)                      # cumulative log decay
        bs = bs.view(h, nc, chunk)
        if tri_incl is None or tri_incl.device != k.device:
            tri_incl = torch.ones(chunk, chunk, dtype=torch.bool, device=k.device).tril()
        diff = gam.unsqueeze(-1) - gam.unsqueeze(-2)                # gamma_i - gamma_j
        decay = torch.where(tri_incl, diff, torch.full_like(diff, float("-inf"))).exp()
        kk = ks @ ks.transpose(-1, -2)
        a = (bs.unsqueeze(-1) * kk * decay).tril(-1)                # strictly lower
        tinv = _unit_lower_inverse(a)
        u = tinv @ (bs.unsqueeze(-1) * vs)
        w = tinv @ (bs.unsqueeze(-1) * gam.exp().unsqueeze(-1) * ks)
        if need_out:
            qk = (qs @ ks.transpose(-1, -2)) * decay
            qd = qs * gam.exp().unsqueeze(-1)
        kd = ks * (gam[..., -1:] - gam).exp().unsqueeze(-1)
        cd = gam[..., -1].exp()
        for c in range(nc):
            v_new = u[:, c] - w[:, c] @ state
            if need_out:
                o = qd[:, c] @ state + qk[:, c] @ v_new
                lo = s0 + c * chunk
                hi = min(s1, lo + chunk)
                out[:, lo:hi] = o[:, :hi - lo]
            state = state * cd[:, c, None, None] + kd[:, c].transpose(-1, -2) @ v_new
            t = s0 + (c + 1) * chunk
            if checkpoint_every and t <= n and t % checkpoint_every == 0:
                checkpoints[offset + t] = state.clone()
    return out, state, checkpoints


def delta_rule_recurrent(q, k, v, g, beta, state=None):
    """Token-by-token reference for delta_rule (tests and single-token decode)."""
    h, n, dk = k.shape
    dv = v.shape[-1]
    s = torch.zeros(h, dk, dv, dtype=torch.float32, device=k.device) if state is None else state.float().clone()
    out = torch.empty(h, n, dv, dtype=torch.float32, device=k.device)
    for t in range(n):
        s = s * g[:, t].float().exp()[:, None, None]
        kt, vt = k[:, t].float(), v[:, t].float()
        delta = (vt - (s * kt[:, :, None]).sum(1)) * beta[:, t].float()[:, None]
        s = s + kt[:, :, None] * delta[:, None, :]
        out[:, t] = (s * q[:, t].float()[:, :, None]).sum(1)
    return out, s
