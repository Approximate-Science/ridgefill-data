"""v1.1.3: on sets that score prompt tokens teacher-forced (longppl, grade), no adaptive decision may read the scored
rows. Before 1.1.3 the qattn window ended at n (inside the scored continuation) and ktail / vtail averaged their scale
over the scored rows, so changing only future targets changed earlier logits (Codex and Fable reviews, 2026-10-07).

The invariance test: hold the input through the first scored position fixed, change only later target tokens, and
require identical logits at that position, identical selected rows and identical correction settings. The positive
control runs the pre-1.1.3 settings and must show the leak, so the test can fail."""
import torch

from kva import model as M
from kva import select as SEL
from kva.kva import Arm, Projector, exclude_scored, finish, run_lower

N, SPLIT, TAIL, SCORED = 200, 4, 100, 39
P = N - TAIL
KEEP = SCORED + 1                            # logits of rows N-KEEP .. N-1 score the last SCORED tokens


def _projector(m, split, seed=11):
    g = torch.Generator().manual_seed(seed)
    d = m.hidden
    W = {i: torch.eye(d) + 0.1 * torch.randn(d, d, generator=g) for i in range(split, m.n_layers)}
    B = {i: 0.1 * torch.randn(d, generator=g) for i in range(split, m.n_layers)}
    return Projector(split, W, B)


def _ids():
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 300, (N,), generator=g)
    b = a.clone()
    b[N - SCORED + 1:] = torch.randint(0, 300, (SCORED - 1,), generator=g)   # change targets 2.. only
    assert torch.equal(a[:N - SCORED + 1], b[:N - SCORED + 1])
    return a, b


@torch.no_grad()
def _first_scored_logits(m, ids, arm, known):
    """The evaluator's composition: windows captured at the known end, rows via SEL.choose, tail statistics
    through exclude_scored (known = N - scored; known = N and scored = 0 reproduce the pre-1.1.3 path)."""
    scored = N - known
    arm = exclude_scored(arm, scored)
    wins = {}

    def hook(i, h):
        for end in {known, P}:
            wins.setdefault(end, {})[i] = h[max(0, end - SEL.WINDOW):end].clone()
    hs, lower = run_lower(m, ids, {SPLIT}, hook=hook)
    rows = None
    if arm.meta.get("select"):
        def mass_of(split, layers, end):
            return SEL.window_mass(m, wins[end], lower, SEL.qattn_layers(m, split, layers), end)
        pick = SEL.choose([arm], N, ids, "probe", mass_of=mass_of, known=None if known == N else known)[arm.name]
        rows = pick["rows"]
    logits, _, _ = finish(m, ids, hs[SPLIT], lower, arm, KEEP, exact_rows=rows)
    return logits[0], rows, arm


def _arms(m):
    proj = _projector(m, SPLIT)
    base = dict(split=SPLIT, tail=TAIL, fill="proj", projector=proj)
    return {"ridgefill": Arm("rf", **base),
            "ktail": Arm("kt", rescale="ktail", **base),
            "vtail": Arm("vt", rescale="vtail", **base),
            "rows": Arm("rows", meta=dict(select=dict(kind="qattn", budget=0.1)), **base)}


def test_exclude_scored_moves_tail_statistics_off_the_targets():
    a = Arm("kt", split=4, tail=100, fill="proj", rescale="ktail")
    assert exclude_scored(a, 0) is a
    assert exclude_scored(a, 39).rescale_skip_last == 39
    assert exclude_scored(Arm("kt", split=4, tail=100, rescale="ktail", rescale_skip_last=512), 39).rescale_skip_last == 512
    plain = Arm("rf", split=4, tail=100, fill="proj")
    assert exclude_scored(plain, 39) == plain


def test_choose_window_ends_where_the_scored_rows_begin():
    seen = []

    def mass_of(split, layers, end):
        seen.append(end)
        return torch.arange(N, dtype=torch.float64)
    q = Arm("q", split=4, tail=100, fill="identity", meta=dict(select=dict(kind="qattn", budget=0.1)))
    qp = Arm("qp", split=4, tail=100, fill="identity", meta=dict(select=dict(kind="qattn", budget=0.1, window="prefix")))
    out = SEL.choose([q, qp], N, torch.zeros(N, dtype=torch.long), "t", mass_of=mass_of, known=N - SCORED)
    assert seen == [N - SCORED, P] and out["q"]["window_end"] == N - SCORED and out["qp"]["window_end"] == P
    seen.clear()
    assert SEL.choose([q], N, torch.zeros(N, dtype=torch.long), "t", mass_of=mass_of)["q"]["window_end"] == N
    for bad in (0, N + 1):
        try:
            SEL.choose([q], N, torch.zeros(N, dtype=torch.long), "t", mass_of=mass_of, known=bad)
        except ValueError:
            continue
        raise AssertionError(f"known={bad} was accepted")


def test_changing_only_future_targets_never_changes_earlier_logits(tiny_llama):
    path, _ = tiny_llama
    m = M.load(str(path), dtype=torch.float32)
    ids_a, ids_b = _ids()
    for name, arm in _arms(m).items():
        la, ra, arm_a = _first_scored_logits(m, ids_a, arm, N - SCORED)
        lb, rb, arm_b = _first_scored_logits(m, ids_b, arm, N - SCORED)
        assert torch.equal(la, lb), f"{name}: earlier logits depend on future targets"
        assert ra == rb and arm_a == arm_b, f"{name}: selection or correction depends on future targets"


def test_the_pre_113_settings_leak(tiny_llama):
    """Positive control: the 1.1.1 / 1.1.2 finals settings (window to n, skip 0) do change earlier logits."""
    path, _ = tiny_llama
    m = M.load(str(path), dtype=torch.float32)
    ids_a, ids_b = _ids()
    arms = _arms(m)
    leaked = {}
    for name in ("ktail", "vtail", "rows"):
        la, ra, _ = _first_scored_logits(m, ids_a, arms[name], N)
        lb, rb, _ = _first_scored_logits(m, ids_b, arms[name], N)
        leaked[name] = (not torch.equal(la, lb)) or ra != rb
    assert all(leaked.values()), leaked
