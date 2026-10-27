"""Predictability scores for the pred / blend selectors (Dylan, R16/R18a): a token the model cannot predict from
its context (a number, a name, an ID) can only be copied later, so it should stay exact. The score needs no
question, so it also works for question-blind prefill, cached documents and follow-up turns.

Predictor (scripts/fit_surprisal.py; r15/surprisal_ridge.py's iiib_b16emb on the reference models): a ridge from
f(t) = [h_S(t-1) ; embed(ids[t])] to log1p of the exact model's surprisal of token t, -log p(ids[t] | ids[<t]).
h_S is the pre-norm residual input of layer S, taken at t-1 because that row is the state that must predict t;
embed is the input embedding row of t. The ridge runs on standardised columns (as r15: the residual has a few
huge dimensions that would otherwise take the whole penalty budget) and is folded back to the raw features:
s_hat(t) = f(t) . w + c. File: BULK/pred/<model>-s<S>.safetensors, tensors w [2H] and c [1] (fp32), metadata
split, hidden, features, target and the fit's provenance.
Free baselines (the freq / frag selectors): a unigram log-frequency table per model (scripts/token_freq.py),
BULK/pred/<model>-freq.safetensors, tensor logfreq [V] fp32 = log((count + 1) / (tokens + V)).
"""
import torch

FEATURES = "[h_S(t-1) ; embed(ids[t])]"
TARGET = "log1p(exact surprisal of t)"


class Predictor:
    def __init__(self, split, w, c, meta=None):
        self.split, self.w, self.c = int(split), w.float(), float(c)
        self.meta = meta or {}

    @torch.no_grad()
    def score(self, h, e, block=8192):
        """Predicted log1p surprisal [m] (fp64, CPU) of tokens whose h_S(t-1) rows are h and embedding rows e."""
        H = h.shape[1]
        w = self.w.to(h.device)
        out = torch.empty(h.shape[0], dtype=torch.float64)
        for r in range(0, h.shape[0], block):
            s = h[r:r + block].float() @ w[:H] + e[r:r + block].to(h.device).float() @ w[H:]
            out[r:r + block] = s.double().cpu()
        return out + self.c

    @classmethod
    def load(cls, path):
        from safetensors import safe_open
        with safe_open(str(path), "pt", device="cpu") as f:
            meta = f.metadata() or {}
            w, c = f.get_tensor("w"), f.get_tensor("c")
        return cls(int(meta["split"]), w, float(c[0]), meta)

    def save(self, path, meta):
        from safetensors.torch import save_file
        save_file({"w": self.w.float().contiguous(), "c": torch.tensor([self.c], dtype=torch.float32)}, str(path),
                  metadata={k: str(v) for k, v in dict(meta, split=self.split, hidden=self.w.numel() // 2,
                                                       features=FEATURES, target=TARGET).items()})


def log_freq(counts):
    """Add-one smoothed log unigram frequency [V] of a count vector: an id never seen gets the lowest value."""
    c = counts.double() + 1
    return (c / c.sum()).log().float()


def save_freq(path, logfreq, meta):
    from safetensors.torch import save_file
    save_file({"logfreq": logfreq.float().contiguous()}, str(path), metadata={k: str(v) for k, v in meta.items()})


def load_freq(path):
    from safetensors import safe_open
    with safe_open(str(path), "pt", device="cpu") as f:
        return f.get_tensor("logfreq")


def prefix_scores(pred, h_split, emb, P):
    """Predicted surprisal [P] of every prefix row j in [0, P), from h_split[j - 1] and emb[j]: only rows < P are
    read, and h_S of row j depends on ids[:j + 1] only, so the score never sees the tail or the question. Row 0
    has no context to be predicted from: -inf (ranked last)."""
    out = torch.full((P,), float("-inf"), dtype=torch.float64)
    if P > 1:
        out[1:] = pred.score(h_split[:P - 1], emb[1:P])
    return out


def standardise(state):
    """The sums of x / sd (sd = each column's standard deviation over the rows) and sd. A RidgeStats.state()."""
    n, sx = state["n"], state["sx"].double()
    sd = ((state["sxx"].diagonal().double() - sx * sx / n) / n).clamp_min(1e-12).sqrt()
    return dict(state, cx=state["cx"].double() / sd, sx=sx / sd, sxx=state["sxx"].double() / torch.outer(sd, sd),
                sxy={i: v.double() / sd[:, None] for i, v in state["sxy"].items()}), sd


def solve(state, alphas, layer=0):
    """{alpha: (w [d] fp64, c)}: ridge on standardised columns with penalty alpha * n (RidgeFit's lam times the
    mean diagonal of the centred scatter, which is n once standardised; r15's alpha), folded back to raw x."""
    from kva.ridge import RidgeFit
    st, sd = standardise(state)
    out = {}
    for a, (w, b) in RidgeFit(st).layer(layer, alphas).items():
        out[a] = (w[0].double() / sd, float(b[0]))
    return out


def ranks(x):
    """1-based ranks, ties sharing their mean rank."""
    x = x.double()
    order = torch.argsort(x, stable=True)
    _, inv, counts = torch.unique_consecutive(x[order], return_inverse=True, return_counts=True)
    mean_rank = counts.cumsum(0).double() - (counts.double() - 1) / 2
    r = torch.empty_like(x)
    r[order] = mean_rank[inv]
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-300))


def topk_recall(pred, true, frac):
    """Share of the true top-frac rows found by the predicted top-frac rows (r15's recall_top{2,5}pct)."""
    k = max(1, round(frac * true.numel()))
    return len(set(pred.topk(k).indices.tolist()) & set(true.topk(k).indices.tolist())) / k
