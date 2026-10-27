"""The own forward pass against transformers' reference models (fp32, CPU)."""
import torch

from kva import model as M


def _ids(n, seed=0):
    return torch.randint(0, 300, (n,), generator=torch.Generator().manual_seed(seed))


def _check(path, hf, n=150):
    m = M.load(path, dtype=torch.float32)
    ids = _ids(n)
    keep = torch.arange(n)
    mine, _ = m.forward_exact(ids, keep=keep)
    with torch.no_grad():
        ref = hf(ids[None]).logits[0].float()
    torch.testing.assert_close(mine, ref, atol=2e-4, rtol=2e-4)


def test_llama_matches_hf(tiny_llama):
    _check(*tiny_llama)


def test_qwen35_matches_hf(tiny_qwen):
    _check(*tiny_qwen, n=200)


def test_qwen35_cached_decode_matches_full(tiny_qwen):
    """Prefill then token-by-token decode through the caches equals one full forward."""
    path, _ = tiny_qwen
    m = M.load(path, dtype=torch.float32)
    ids = _ids(140, 3)
    full, _ = m.forward_exact(ids, keep=torch.arange(140))
    _, cache = m.forward_exact(ids[:130])
    rows = []
    for t in range(130, 140):
        h = m.embed(ids[t:t + 1])
        pos = torch.tensor([t])
        for i in range(m.n_layers):
            h = m.layer(i, h, pos, cache)
        rows.append(m.logits(h))
    torch.testing.assert_close(torch.cat(rows), full[130:], atol=1e-4, rtol=1e-4)


def test_qwen35_gdn_row_segments_match(tiny_qwen):
    path, _ = tiny_qwen
    m = M.load(path, dtype=torch.float32)
    ids = _ids(300, 5)
    ref, c1 = m.forward_exact(ids, keep=torch.arange(300), gdn_checkpoint_every=64)
    old = M.GDN_ROWS
    try:
        M.GDN_ROWS = 128
        seg, c2 = m.forward_exact(ids, keep=torch.arange(300), gdn_checkpoint_every=64)
    finally:
        M.GDN_ROWS = old
    torch.testing.assert_close(seg, ref, atol=1e-5, rtol=1e-5)
    assert sorted(c1[0]["checkpoints"]) == sorted(c2[0]["checkpoints"]) == [64, 128, 192, 256]
    torch.testing.assert_close(c2[0]["checkpoints"][256], c1[0]["checkpoints"][256], atol=1e-5, rtol=1e-5)
