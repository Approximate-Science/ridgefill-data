"""Closed-form ridge projectors from sufficient statistics.

Accumulate (on the device): n, shifted sums Sx, Sy_i and scatters Sxx, Sxy_i with fixed shifts
cx, cy_i (the first batch's means) for numerical stability; fp32 per-batch GEMMs added into
fp64 accumulators. Fit (CPU, fp64): centred scatters C = S - s s^T / n,
    W(lam) = (Cxx + lam * mean(diag Cxx) I)^-1 Cxy        (one eigendecomposition, every lam)
bias unpenalised: y = (x - mean_x) W + mean_y.
residual(): the statistics of the target y - x (ridge towards the identity map, not towards 0).
RidgeFit.reduced_rank(): the best rank-r map under the same ridge objective.
RidgeFit.sq(): per output, the training rows' sums of squares of the fitted values and of the targets.
Row weights (WF1): begin(x, w=w) makes every sum a weighted one (sum of w, w x, w x x^T, w y, w x y^T), so n holds
the sum of w and RidgeFit's centring uses weighted means; the fit minimises sum_j w_j |y_j - x_j W - b|^2 + ridge.
w = None runs the unweighted code; w = 1 gives the same sums bit for bit. The accumulators may live on another device
than the rows (fp32 products on the rows' device, added into fp64 on the accumulators' device, as fit_sweep.py).
"""
import torch


class RidgeStats:
    def __init__(self, split, layers, dim, device, out_dim=None):
        """out_dim: the targets' width (default dim: layer inputs; 1 for a scalar target)."""
        self.split, self.layers, self.dim, self.device = split, list(layers), dim, device
        out_dim = dim if out_dim is None else out_dim
        f64 = dict(dtype=torch.float64, device=device)
        self.n = 0
        self.cx = None
        self.cy = {}
        self.sx = torch.zeros(dim, **f64)
        self.sxx = torch.zeros(dim, dim, **f64)
        self.sy = {i: torch.zeros(out_dim, **f64) for i in self.layers}
        self.sxy = {i: torch.zeros(dim, out_dim, **f64) for i in self.layers}

    @torch.no_grad()
    def begin(self, x, block=8192, w=None):
        """Boundary rows x [n, d] of one document; follow with add_layer for every layer. w [n]: row weights (None:
        every row counts once); n then grows by their sum."""
        if self.cx is None:
            self.cx = x[:block].float().mean(0)
        self._x = x
        self._w = None if w is None else w.to(x.device, torch.float32)[:, None]
        self._block = block
        self.n += x.shape[0] if w is None else w.double().sum().item()
        for r in range(0, x.shape[0], block):
            xc = x[r:r + block].float() - self.cx
            xw = self._weigh(xc, r)
            self.sx += xw.sum(0).double().to(self.sx.device)
            self.sxx += (xw.T @ xc).to(self.sxx.device)

    def _weigh(self, t, r):
        """w * t for rows t of the current document starting at row r (t itself when unweighted)."""
        return t if self._w is None else t * self._w[r:r + t.shape[0]]

    @torch.no_grad()
    def add_layer(self, i, y):
        """The same rows' true inputs of layer i."""
        if i not in self.cy:
            self.cy[i] = y[:self._block].float().mean(0)
        for r in range(0, y.shape[0], self._block):
            xw = self._weigh(self._x[r:r + self._block].float() - self.cx, r)
            yc = y[r:r + self._block].float() - self.cy[i]
            self.sy[i] += self._weigh(yc, r).sum(0).double().to(self.sy[i].device)
            self.sxy[i] += (xw.T @ yc).to(self.sxy[i].device)

    def add(self, x, ys, block=8192, w=None):
        """x [n, d] boundary rows; ys {layer: [n, d]} the same rows' true layer inputs; w [n] row weights or None."""
        self.begin(x, block, w)
        for i in self.layers:
            self.add_layer(i, ys[i])

    def state(self):
        cpu = lambda t: t.cpu()  # noqa: E731
        return dict(split=self.split, layers=self.layers, dim=self.dim, n=self.n, cx=cpu(self.cx).double(),
                    cy={i: cpu(v).double() for i, v in self.cy.items()}, sx=cpu(self.sx), sxx=cpu(self.sxx),
                    sy={i: cpu(v) for i, v in self.sy.items()}, sxy={i: cpu(v) for i, v in self.sxy.items()})


class RidgeFit:
    """Eigendecomposition of the centred input scatter, shared by every layer and lambda."""

    def __init__(self, state):
        n = state["n"]
        self.n = n
        sx = state["sx"]
        mx_c = sx / n
        self.mean_x = state["cx"] + mx_c
        cxx = state["sxx"] - torch.outer(sx, sx) / n
        self.scale = cxx.diagonal().mean().item()
        self.evals, self.evecs = torch.linalg.eigh(cxx)
        self.state = state

    def _cross(self, i):
        st = self.state
        sx, sy = st["sx"], st["sy"][i]
        cxy = st["sxy"][i] - torch.outer(sx, sy) / self.n
        return self.evecs.T @ cxy, st["cy"][i] + sy / self.n

    def layer(self, i, lams):
        """{lam: (weight [d_out, d_in] fp32, bias [d_out] fp32)} for layer i (F.linear layout)."""
        a, mean_y = self._cross(i)
        out = {}
        for lam in lams:
            w = self.evecs @ (a / (self.evals + lam * self.scale)[:, None])     # [d_in, d_out]
            bias = mean_y - self.mean_x @ w
            out[lam] = (w.T.float().contiguous(), bias.float())
        return out

    def sq(self, i, weight):
        """Per output over the training rows: (sum of squares of the fitted values (x - mean_x) w + mean_y,
        sum of squares of the targets y), fp64. weight [d_out, d_in] as layer() returns; needs state['sq'][i],
        the per-output sum of (y - cy)^2."""
        st = self.state
        cy, sy = st["cy"][i], st["sy"][i]
        z = self.evecs.T @ weight.double().T                                   # [d_in, d_out]
        fitted = (z * z * self.evals[:, None]).sum(0) + self.n * (cy + sy / self.n) ** 2
        return fitted, st["sq"][i] + 2 * cy * sy + self.n * cy * cy

    def reduced_rank(self, i, lam, ranks):
        """Reduced-rank ridge (Mukherjee & Zhu 2011): the rank-r minimiser of |Y - XW|^2 + lam s |W|^2 is
        W(lam) V_r V_r^T, V_r the top r eigenvectors of W^T (Cxx + lam s I) W = a^T diag(1 / (e + lam s)) a
        (the fitted values of the ridge-augmented regression). Returns ({r: (weight, bias)} as layer(),
        the cumulative share of that fitted energy kept by the top k directions, k = 1..d)."""
        a, mean_y = self._cross(i)
        d = self.evals + lam * self.scale
        w = self.evecs @ (a / d[:, None])
        ev, v = torch.linalg.eigh(a.T @ (a / d[:, None]))
        ev, v = ev.flip(0).clamp(min=0), v.flip(1)
        out = {}
        for r in ranks:
            wr = (w @ v[:, :r]) @ v[:, :r].T
            out[r] = (wr.T.float().contiguous(), (mean_y - self.mean_x @ wr).float())
        return out, ev.cumsum(0) / ev.sum().clamp(min=1e-300)     # 0 where the target is 0 (i = S, residual)


def residual(state):
    """Statistics of the target y - x: shifts cy - cx, shifted sums sy - sx and sxy - sxx (exact)."""
    sx, sxx = state["sx"].double(), state["sxx"].double()
    return dict(state, sx=sx, sxx=sxx, cy={i: state["cy"][i] - state["cx"] for i in state["layers"]},
                sy={i: state["sy"][i].double() - sx for i in state["layers"]},
                sxy={i: state["sxy"][i].double() - sxx for i in state["layers"]})


def combine(a, b, w):
    """Sufficient statistics of a's rows plus w x b's rows (b re-expressed about a's shifts).
    Both are RidgeStats.state() dicts over the same split/layers; n becomes n_a + w n_b."""
    d = b["cx"] - a["cx"]
    out = dict(a, n=a["n"] + w * b["n"])
    sxb = b["sx"] + b["n"] * d
    out["sx"] = a["sx"] + w * sxb
    out["sxx"] = a["sxx"] + w * (b["sxx"] + torch.outer(b["sx"], d) + torch.outer(d, b["sx"]) + b["n"] * torch.outer(d, d))
    out["sy"], out["sxy"] = {}, {}
    for i in a["layers"]:
        e = b["cy"][i] - a["cy"][i]
        out["sy"][i] = a["sy"][i] + w * (b["sy"][i] + b["n"] * e)
        out["sxy"][i] = a["sxy"][i] + w * (b["sxy"][i] + torch.outer(b["sx"], e) + torch.outer(d, b["sy"][i])
                                          + b["n"] * torch.outer(d, e))
    return out
