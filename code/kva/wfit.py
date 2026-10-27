"""WF1: a sensitivity-weighted projector fit (paper/GAINS.md item 17; the BIGDATA blind analyses).

The plain ridge fit weights every calibration row equally. But the loss from the fill follows how much later text
reads a row (sensitivity), the fill error is nearly flat across rows, and the fill is worst on the rows read least.
WF1 weights each calibration row by qfp, the best question-free proxy of being read: the attention the row receives
from later rows in the attention layers below the split, per later row (kva.select.per_later_token, the selector
STACK1 runs; here the whole calibration document is the prefix).

Weight shapes, per document over rows >= 1, then scaled to mean 1 over the document:
  qfp   qfp itself
  sqrt  qfp ** 0.5 (tempers qfp's heavy tail)
  rank  1 / max(1 - u, RANK_FLOOR), u = the row's rank fraction by qfp in its document (ties share their mean rank):
        a row in the top fraction f gets 1 / f, at most 1 / RANK_FLOOR. Bounded, so a few rows cannot hold the fit.
Row 0 gets weight 1 (the document mean): its qfp is the attention sink's (BIGDATA, S12: ~2600x the mean on Llama,
~40x on Qwen3.5), and Llama's BOS is sensitive but filled almost exactly (the blind analyses: proxies must exclude it).
"""
import torch

from kva import impact as IM
from kva import select as SEL

SHAPES = ("qfp", "sqrt", "rank")
RANK_FLOOR = 0.1


def rank_fraction(x):
    """Mean rank of each value among x (0-based; ties share their mean rank), divided by len(x)."""
    _, inv, cnt = torch.unique(x, return_inverse=True, return_counts=True)
    return (cnt.cumsum(0) - (cnt + 1) / 2).double()[inv] / x.numel()


def row_weights(qfp, shape):
    """Fit weights [n] (fp64) of one document's rows from their qfp [n] (see the module docstring)."""
    q = qfp.double().cpu()
    if shape == "qfp":
        s = q[1:]
    elif shape == "sqrt":
        s = q[1:].clamp_min(0).sqrt()
    elif shape == "rank":
        s = 1 / (1 - rank_fraction(q[1:])).clamp_min(RANK_FLOOR)
    else:
        raise ValueError(f"weight shape {shape!r} not in {SHAPES}")
    w = torch.ones_like(q)
    if s.numel() and s.mean() > 0:
        w[1:] = s / s.mean()
    return w


@torch.no_grad()
def doc_pass(m, ids, split, stride, late):
    """One exact forward of a document (all rows are prefix). By layer `split` its qfp [n] (fp64, CPU) is complete:
    attention received from strictly later rows at every attention layer below the split, summed over heads and
    layers, per later row; stride s: every s-th query row, scaled by s (kva.impact.received_mass, STACK1's
    qfp_stride). late(i, h, qfp) then sees every late layer's input rows h (i >= split)."""
    n = ids.shape[0]
    pos = torch.arange(n, device=m.device)
    recv = torch.zeros(n, dtype=torch.float64)
    qfp = []

    def hook(i, h):
        if i == "final":
            return
        if i < split and m.layer_types[i] == "attn":
            x = m.in_norm(h, m.W[i]["ln1"])
            recv.add_(IM.received_mass(m.queries(i, x, pos), IM.kv_of(m, i, x, pos)[0], m.scale, stride=stride))
        if i >= split:
            if not qfp:
                qfp.append(SEL.per_later_token(recv, n))
            late(i, h, qfp[0])
    m.forward_exact(ids.to(m.device), hook=hook)
