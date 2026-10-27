"""Decode-side agreement with the exact model (paper T1.7; scripts/evaluate.py set 'decode').

Per prompt the exact model and every arm prefill the same prompt (the arm through kva.kva.finish, so its late layers
hold the filled prefix cache for the whole generation); decode is exact for every arm (SwiftKV native: its own decode
path, as kva.kva.greedy). Fixed length: `steps` tokens, EOS ignored (min_tokens-style, so every prompt contributes
`steps` positions; the exact run records where its first EOS fell so an analysis can cut there).
  exact        greedy: tokens y_0..y_{s-1} and the log-probs at every step (step t: given the prompt and y_<t).
  arm, forced  the arm's cache teacher-forced on y (step t: given the prompt and the EXACT tokens y_<t): per-step
               KL(exact || arm) in nats over the full vocabulary, and the share of steps whose arm argmax is y_t.
  arm, free    greedy from the arm's own cache: tokens z; first_divergence = the first t with z_t != y_t (None if
               none), frac_identical = the share of positions t with z_t == y_t.
The forced pass runs on a fork of the arm's prefill cache (dict copies share the K/V buffers; rows the fork appends are
overwritten by the free pass afterwards, which only reads rows below the prompt length), so both start from the same
filled cache; the free pass then uses the original.
"""
import torch
import torch.nn.functional as F

from kva.kva import check_finite


@torch.no_grad()
def step(model, tok, pos, cache, arm):
    """Logits [V] (fp32) after feeding token tok at position pos to every layer (exact; SwiftKV native: its own
    late layers, as kva.kva.greedy)."""
    p = torch.tensor([pos], device=model.device)
    h = model.embed(torch.tensor([tok], device=model.device))
    for i in range(model.n_layers):
        if arm.fill == "swiftkv_native" and i >= arm.split:
            if i == arm.split:
                b = h
            model.swift_fill(i, b, p, cache)
            h = model.swift_layer(i, h, p, cache)
        else:
            h = model.layer(i, h, p, cache)
    return model.logits(h)[-1]


@torch.no_grad()
def decode(model, n, cache, first_logits, steps, arm, feed=None, logprobs=False):
    """`steps` decode steps from the prompt's last-row logits (prompt length n). feed None: greedy (each argmax is
    fed back); feed = token list: teacher-forced (feed[t] is fed after step t). Returns (argmax per step [steps],
    log-probs [steps, V] fp32 on the CPU if logprobs else None). EOS does not stop it."""
    if feed is not None and len(feed) < steps:
        raise ValueError(f"teacher forcing needs {steps} tokens, got {len(feed)}")
    tops, lps = [], []
    logits = first_logits[-1]
    for t in range(steps):
        check_finite(logits, f"{arm.name} decode step {t}")
        tops.append(int(logits.argmax()))
        if logprobs:
            lps.append(torch.log_softmax(logits.float(), -1).cpu())
        if t == steps - 1:
            break
        logits = step(model, tops[t] if feed is None else int(feed[t]), n + t, cache, arm)
    return tops, (torch.stack(lps) if logprobs else None)


def kl_steps(lp_exact, lp_arm):
    """Per-step KL(exact || arm) [steps] in nats from log-probs [steps, V]: sum_v p_exact (log p_exact - log p_arm)."""
    return F.kl_div(lp_arm, lp_exact, reduction="none", log_target=True).sum(-1)


def agreement(exact_tokens, arm_tokens):
    """(first_divergence: the first position whose tokens differ, None if none; frac_identical: share of equal
    positions). Both lists have one token per step."""
    if len(exact_tokens) != len(arm_tokens) or not exact_tokens:
        raise ValueError(f"{len(exact_tokens)} exact vs {len(arm_tokens)} arm tokens")
    same = [a == b for a, b in zip(exact_tokens, arm_tokens)]
    return next((t for t, s in enumerate(same) if not s), None), sum(same) / len(same)


def compare(model, n, cache, first_logits, steps, arm, y_exact, lp_exact):
    """The arm's forced and free passes from its prefill cache (module doc) against the exact run (y_exact,
    lp_exact). Returns the record fields."""
    fork = [dict(c) for c in cache]
    forced, lp_arm = decode(model, n, fork, first_logits, steps, arm, feed=y_exact, logprobs=True)
    del fork
    kl = kl_steps(lp_exact, lp_arm)
    del lp_arm
    free, _ = decode(model, n, cache, first_logits, steps, arm)
    first, frac = agreement(y_exact, free)
    return dict(gen=free, first_divergence=first, frac_identical=round(frac, 6),
                forced_top1_agree=round(sum(a == b for a, b in zip(forced, y_exact)) / steps, 6),
                kl_mean=round(float(kl.mean()), 8), kl=[round(float(x), 8) for x in kl])
