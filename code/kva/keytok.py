"""Token-level NLL readouts over a scored window (paper T1.3): novel tokens and LongPPL-style key tokens.

Novel tokens: kva.data.novel_mask (its docstring is the definition: a token-id 8-gram copy rule; the grade runs it on
the prompt without its first id, the long-prompt NLL on the whole prompt). The mask is stored per prompt as "novel".

Key tokens (after LongPPL: Fang et al., "What is Wrong with Perplexity for Long-context Language Modeling?", 2024):
the tokens whose loss under the EXACT model rises by more than `threshold` nats when the prefix is removed. With
lp_full = the exact model's log-probs of the scored tokens given the whole prompt (the exact arm's run) and lp_tail =
the exact model's log-probs of the same tokens given only the last T prompt tokens (exact_tail_only), token j is a key
token iff lp_full[j] - lp_tail[j] > threshold. LongPPL's own selection also keeps only tokens whose long-context
log-prob is above a floor (its LCL term); that term is not applied here. THRESHOLD is LongPPL's alpha; the final
record declares the value used.

The tail-only context is [BOS] + ids[-T:]: the BOS (the prompt's own first id) only when the prompt starts with the
tokenizer's BOS (Llama), so the model keeps its attention sink; Qwen has none. T is the config's keytoken_tail
(evaluate.py records it with the log-probs as rec["tail_only"]); it must leave a prefix to remove (T < len(ids) minus
the BOS) and hold every scored token plus the row that predicts the first (T >= scored + 1).

Readouts (readout()): per arm, the mean over a mask of (-lp_arm) - (-lp_exact) in nats (arm minus exact) and the
mask's size, for all scored tokens, the novel tokens and the key tokens. An empty mask gives delta None, never 0.
"""
import torch

from kva.kva import score_tokens

THRESHOLD = 2.0          # LongPPL's alpha (nats); the paper's final record declares the value it uses


class KeyTokenReferenceMissing(ValueError):
    """A key-token readout was asked of a record without the exact tail-only log-probs (rec["tail_only"]): the run's
    config had no keytoken_tail. Never scored as 'no key tokens'."""


def check_tail(n, scored, tail, bos):
    """Refuse a tail-only context that removes nothing or cannot score every token (see the module doc)."""
    if not scored + 1 <= tail < n - len(bos):
        raise ValueError(f"keytoken_tail {tail}: needs scored + 1 = {scored + 1} <= T < {n - len(bos)} "
                         f"(the prompt's {n} ids minus its BOS)")


@torch.no_grad()
def exact_tail_only(model, ids, tail, bos, scored):
    """The exact model's log-probs [scored] of the last `scored` ids given [bos] + ids[-tail:] (bos: [] or the
    prompt's own first id)."""
    ids = list(ids)
    check_tail(len(ids), scored, tail, bos)
    ctx = torch.tensor(list(bos) + ids[-tail:])
    n = ctx.shape[0]
    logits, _ = model.forward_exact(ctx, keep=torch.arange(n - scored - 1, n))
    return score_tokens(logits, ctx, scored)


def key_mask(lp_full, lp_tail, threshold=THRESHOLD):
    """True where the exact NLL rises by more than threshold when the prefix is removed: lp_full - lp_tail > it."""
    if len(lp_full) != len(lp_tail):
        raise ValueError(f"{len(lp_full)} full-context log-probs vs {len(lp_tail)} tail-only")
    return [f - t > threshold for f, t in zip(lp_full, lp_tail)]


def nll_delta(lp_arm, lp_exact, mask=None):
    """(mean over the masked tokens of (-lp_arm) - (-lp_exact), number of masked tokens); (None, 0) if none."""
    xs = [e - a for i, (a, e) in enumerate(zip(lp_arm, lp_exact)) if mask is None or mask[i]]
    return (sum(xs) / len(xs) if xs else None), len(xs)


def readout(rec, threshold=THRESHOLD):
    """{arm: {all, novel, key: {delta, n}}} for every non-exact arm of a grade or longppl record."""
    ref = rec.get("tail_only")
    if ref is None:
        raise KeyTokenReferenceMissing(f"{rec.get('name')}: no tail_only log-probs (run with keytoken_tail)")
    ex = rec["runs"]["exact"]["logprobs"]
    masks = dict(all=None, novel=rec["novel"], key=key_mask(ex, ref["logprobs"], threshold))
    out = {}
    for name, run in rec["runs"].items():
        if name == "exact":
            continue
        out[name] = {k: dict(zip(("delta", "n"), nll_delta(run["logprobs"], ex, mask))) for k, mask in masks.items()}
    return out
