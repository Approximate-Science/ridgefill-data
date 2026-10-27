import math

import torch

from kva import ops


def _gdn_inputs(h=3, n=200, dk=16, dv=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = ops.l2norm(torch.randn(h, n, dk, generator=g)) * dk ** -0.5
    k = ops.l2norm(torch.randn(h, n, dk, generator=g))
    v = torch.randn(h, n, dv, generator=g)
    gate = -torch.rand(h, n, generator=g) * 0.5
    beta = torch.rand(h, n, generator=g)
    s0 = torch.randn(h, dk, dv, generator=g) * 0.1
    return q, k, v, gate, beta, s0


def test_delta_rule_matches_recurrent():
    q, k, v, g, b, s0 = _gdn_inputs()
    ref, sref = ops.delta_rule_recurrent(q, k, v, g, b, s0)
    out, s, _ = ops.delta_rule(q, k, v, g, b, s0, chunk=16, segment=64)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s, sref, atol=1e-5, rtol=1e-5)


def test_delta_rule_split_and_checkpoints():
    q, k, v, g, b, s0 = _gdn_inputs(n=192)
    _, s_all, ck = ops.delta_rule(q, k, v, g, b, s0, chunk=16, segment=64, checkpoint_every=32, need_out=False)
    for t in (32, 96, 192):
        _, s_t = ops.delta_rule_recurrent(q[:, :t], k[:, :t], v[:, :t], g[:, :t], b[:, :t], s0)
        torch.testing.assert_close(ck[t], s_t, atol=1e-5, rtol=1e-5)
    # two calls, the second continuing from the first's state (grid restarts at the split)
    o1, s1, _ = ops.delta_rule(q[:, :77], k[:, :77], v[:, :77], g[:, :77], b[:, :77], s0, chunk=16)
    o2, s2, _ = ops.delta_rule(q[:, 77:], k[:, 77:], v[:, 77:], g[:, 77:], b[:, 77:], s1, chunk=16)
    ref, sref = ops.delta_rule_recurrent(q, k, v, g, b, s0)
    torch.testing.assert_close(torch.cat((o1, o2), 1), ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s2, sref, atol=1e-5, rtol=1e-5)


def test_causal_conv_continuation():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(50, 12, generator=g)
    w = torch.randn(12, 4, generator=g)
    full, st = ops.causal_conv(x, None, w)
    a, sa = ops.causal_conv(x[:20], None, w)
    b, sb = ops.causal_conv(x[20:], sa, w)
    torch.testing.assert_close(torch.cat((a, b)), full)
    torch.testing.assert_close(sb, st)
    ref = torch.nn.functional.conv1d(x.T[None], w[:, None], padding=3, groups=12)[0, :, :50].T
    torch.testing.assert_close(full, torch.nn.functional.silu(ref), atol=1e-6, rtol=1e-5)


def test_attend_bottom_right_causal():
    g = torch.Generator().manual_seed(0)
    nk, nq, d = 40, 7, 8
    q = torch.randn(4, nq, d, generator=g)
    k = torch.randn(2, nk, d, generator=g)
    v = torch.randn(2, nk, d, generator=g)
    out = ops.attend(q, k, v, d ** -0.5)
    kr, vr = k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)
    s = q @ kr.transpose(1, 2) * d ** -0.5
    qpos = torch.arange(nk - nq, nk)[:, None]
    s = s.masked_fill(torch.arange(nk)[None, :] > qpos, -math.inf)
    ref = (s.softmax(-1) @ vr).transpose(0, 1).reshape(nq, -1)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    old = ops.ATTN_BLOCK_SCORES
    try:
        ops.ATTN_BLOCK_SCORES = 4 * 40 * 3          # forces query blocks
        torch.testing.assert_close(ops.attend(q, k, v, d ** -0.5), ref, atol=1e-5, rtol=1e-5)
        full = ops.attend(q.repeat(1, 6, 1)[:, :40], k, v, d ** -0.5)   # nq == nk, blocked
        ops.ATTN_BLOCK_SCORES = old
        torch.testing.assert_close(full, ops.attend(q.repeat(1, 6, 1)[:, :40], k, v, d ** -0.5), atol=1e-5, rtol=1e-5)
    finally:
        ops.ATTN_BLOCK_SCORES = old


def test_niah_scoring():
    from kva.data import niah_correct
    assert niah_correct(" 1234567.", "1234567")
    assert niah_correct("<think>9999999</think> The number is 1234567.", "1234567")
    assert not niah_correct("12345678", "1234567")
    assert not niah_correct("<think>unfinished 1234567", "1234567")
    assert not niah_correct("7654321 then 1234567", "1234567")
