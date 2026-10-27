#!/usr/bin/env python3
"""Split sweep: ONE capture pass, ridge projectors for every split in SPLITS.

1. capture  exact forward over the calibration documents (data.calib_docs('train'), all positions).
            For every split S and every later layer i: shifted sums sx_S, sxx_S, sy_Si, sxy_Si
            (shifts = the first document's means), per-document fp32 GEMMs on the GPU added into
            fp64 accumulators in host RAM (not saved: ~15 GB per model).
            --unitrms  also the sums of the unit-RMS targets y / rms(y) (rms with the model's norm eps), same x
                       (triage #2: the late layer RMS-normalises its input, so only the prediction's direction is
                       used, while the raw fit is dominated by high-norm rows).
            --kvfold   also the K/V fold sums for scripts/fit_kvfold.py (triage #6): x = the unit-RMS boundary
                       state b / rms(b), targets the pre-RoPE K | V of every late attention layer (kva.kv_target),
                       plus the targets' per-feature sums of squares sq (the fold's per-head scale).
2. per S    CPU fp64 ridge for the lambda grid (as fit_projector.py), held-out read-cosine lambda
            selection on data.calib_docs('held') (the same rule as fit_projector.py, per S), save
            BULK/proj/<model>-sweep-s<S>.safetensors and WORK/results/fit-sweep-<model>-s<S>.json
            (per-lambda and per-layer read cosines, incl. the no-projector map). With either flag the raw fit
            is the control of the same capture: -sweep-s<S>-rawctl, and the unit-RMS fit -sweep-s<S>-unitrms
            (same grid and selection rule; reports fit-sweep-<model>-s<S>-{rawctl,unitrms}.json).
3. stats    the sufficient statistics of every split go to STATS_DIR/<model>-s<S>.pt (refits at any lambda and
            continual training without recapture; coordinator 2026-10-01): fp64 if the home filesystem keeps
            >= 15 GB free afterwards, else fp32 (noted in the file), else only S in {8, 16, 24}. With either
            flag: every split's sums in fp64 (raw keys as above plus 'unitrms' / 'kvfold' dicts) go to
            paths.SUMS_RMSFOLD/<model>-s<S>.pt, and the job stops before the capture unless they fit with
            15 GB to spare.
Provenance (protocol 1.1, 2026-10-04): every projector's metadata records rows (the calibration rows of the capture),
docs (the calibration document count; also in the fit report), calib_exclude_sha256 (datasets/calibration-exclude.json, T0.2), code_commit and code_dirty. A projector file that
already exists counts as done only if its rows equal the current calibration list's; one fitted on another list (a
pre-exclusion projector: more rows) is refused by name (StaleProjector) before the capture, never skipped as done and
never overwritten: move it aside first (jobs/paper-refit-clean.jobs says how).
Usage: fit_sweep.py MODEL_DIR S1,S2,... [--unitrms] [--kvfold]
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from kva import gpu, paths  # noqa: E402

LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0)
STATS_DIR = paths.SUMS
KEEP_FREE = 15 * 10**9
SUBSET = (8, 16, 24)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stale_projectors(paths, rows):
    """[(path, its recorded rows)] of the existing projector files whose metadata 'rows' is not this calibration list's
    row count (fitted on another list), read from the safetensors header only."""
    from kva.kva import Projector
    return [(p, Projector.meta_of(p).get("rows")) for p in paths
            if Path(p).exists() and Projector.meta_of(p).get("rows") != str(rows)]


def accumulators(splits, L, H, unitrms=False, kv=None):
    """Host fp64 sums for the capture. kv {late attention layer: K|V width}: also the fold sums."""
    f64 = dict(dtype=torch.float64)
    pairs = [(S, i) for S in splits for i in range(S, L)]
    acc = dict(n=0, cx={}, cy={}, sx={S: torch.zeros(H, **f64) for S in splits},
               sxx={S: torch.zeros(H, H, **f64) for S in splits},
               sy={p: torch.zeros(H, **f64) for p in pairs}, sxy={p: torch.zeros(H, H, **f64) for p in pairs})
    if unitrms:
        acc["unitrms"] = dict(cy={}, sy={p: torch.zeros(H, **f64) for p in pairs},
                              sxy={p: torch.zeros(H, H, **f64) for p in pairs})
    if kv:
        kp = [(S, i) for S in splits for i in sorted(kv) if i >= S]
        acc["kvfold"] = dict(cx={}, cy={}, sx={S: torch.zeros(H, **f64) for S in splits},
                             sxx={S: torch.zeros(H, H, **f64) for S in splits},
                             sy={p: torch.zeros(kv[p[1]], **f64) for p in kp},
                             sxy={p: torch.zeros(H, kv[p[1]], **f64) for p in kp},
                             sq={p: torch.zeros(kv[p[1]], **f64) for p in kp})
    return acc


def unit_rms(h, eps):
    hf = h.float()
    return hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)


def capture_hook(acc, xs, eps=None, kv_target=None):
    """hook(i, h) for Model.forward_exact adding one document's rows to acc. xs keeps the document's centred
    boundary rows (the caller clears it per document and adds the row count to acc['n']). eps: the norm eps
    of the unit-RMS rows (extras only); kv_target(i, h): the fold targets of layer i (--kvfold)."""
    pending = list(acc["sx"])
    ur, kf = acc.get("unitrms"), acc.get("kvfold")

    def hook(i, h):
        if i == "final":
            return
        if i in acc["sx"]:
            if i not in acc["cx"]:
                acc["cx"][i] = h.float().mean(0)
            xc = h.float() - acc["cx"][i]
            xs[i] = xc
            acc["sx"][i] += xc.sum(0).double().cpu()
            acc["sxx"][i] += (xc.T @ xc).cpu()
        for S in pending:
            if S <= i:
                p = (S, i)
                if p not in acc["cy"]:
                    acc["cy"][p] = h.float().mean(0)
                yc = h.float() - acc["cy"][p]
                acc["sy"][p] += yc.sum(0).double().cpu()
                acc["sxy"][p] += (xs[S].T @ yc).cpu()
        if ur is not None and i >= pending[0]:
            y = unit_rms(h, eps)
            for S in pending:
                if S <= i:
                    p = (S, i)
                    if p not in ur["cy"]:
                        ur["cy"][p] = y.mean(0)
                    yc = y - ur["cy"][p]
                    ur["sy"][p] += yc.sum(0).double().cpu()
                    ur["sxy"][p] += (xs[S].T @ yc).cpu()
        if kf is None:
            return
        if i in kf["sx"]:
            u = unit_rms(h, eps)
            if i not in kf["cx"]:
                kf["cx"][i] = u.mean(0)
            uc = u - kf["cx"][i]
            xs["u", i] = uc
            kf["sx"][i] += uc.sum(0).double().cpu()
            kf["sxx"][i] += (uc.T @ uc).cpu()
        todo = [S for S in pending if (S, i) in kf["sy"]]
        if todo:
            t = kv_target(i, h).float()
            for S in todo:
                p = (S, i)
                if p not in kf["cy"]:
                    kf["cy"][p] = t.mean(0)
                tc = t - kf["cy"][p]
                kf["sy"][p] += tc.sum(0).double().cpu()
                kf["sxy"][p] += (xs["u", S].T @ tc).cpu()
                kf["sq"][p] += (tc * tc).sum(0).double().cpu()

    return hook


def split_state(acc, S, L, H):
    """RidgeStats.state()-style sums of split S (tensors shared with acc); extras as 'unitrms' (cy, sy, sxy over the
    same x) and 'kvfold' (a full state over the late attention layers, plus sq) entries."""
    late = list(range(S, L))
    cpu = lambda t: t.double().cpu()  # noqa: E731
    state = dict(split=S, layers=late, dim=H, n=acc["n"], cx=cpu(acc["cx"][S]),
                 cy={i: cpu(acc["cy"][(S, i)]) for i in late}, sx=acc["sx"][S], sxx=acc["sxx"][S],
                 sy={i: acc["sy"][(S, i)] for i in late}, sxy={i: acc["sxy"][(S, i)] for i in late})
    if "unitrms" in acc:
        u = acc["unitrms"]
        state["unitrms"] = dict(cy={i: cpu(u["cy"][(S, i)]) for i in late}, sy={i: u["sy"][(S, i)] for i in late},
                                sxy={i: u["sxy"][(S, i)] for i in late})
    if "kvfold" in acc:
        k = acc["kvfold"]
        kl = [i for i in late if (S, i) in k["sy"]]
        state["kvfold"] = dict(split=S, layers=kl, dim=H, n=acc["n"], cx=cpu(k["cx"][S]),
                               cy={i: cpu(k["cy"][(S, i)]) for i in kl}, sx=k["sx"][S], sxx=k["sxx"][S],
                               sy={i: k["sy"][(S, i)] for i in kl}, sxy={i: k["sxy"][(S, i)] for i in kl},
                               sq={i: k["sq"][(S, i)] for i in kl})
    return state


def drop_split(acc, S, L):
    """Free split S's accumulators (sx stays: small)."""
    for d in [acc] + [acc[k] for k in ("unitrms", "kvfold") if k in acc]:
        for k in ("sxx", "sy", "sxy", "sq"):
            if k not in d:
                continue
            for key in [key for key in d[k] if key == S or (isinstance(key, tuple) and key[0] == S)]:
                del d[k][key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("splits")
    ap.add_argument("--unitrms", action="store_true")
    ap.add_argument("--kvfold", action="store_true")
    args = ap.parse_args()
    device, gpu_info = gpu.setup()
    import torch.nn.functional as F
    from kva import data, prov
    from kva import model as M
    from kva.kva import Projector, kv_target
    from kva.ridge import RidgeFit
    torch.set_num_threads(4)   # host limit while other CPU workers run

    model_dir = Path(args.model_dir)
    splits = sorted(int(s) for s in args.splits.split(","))
    extra = args.unitrms or args.kvfold
    kinds = ("-rawctl",) + (("-unitrms",) if args.unitrms else ()) if extra else ("",)
    stats_dir = paths.SUMS_RMSFOLD if extra else STATS_DIR
    name = model_dir.name
    tok = data.tokenizer(model_dir)
    m = M.load(model_dir, device=device, load_lm_head=False, extras=False)
    L, H = m.n_layers, m.hidden
    proj_path = lambda S, kind: prov.BULK / "proj" / f"{name}-sweep-s{S}{kind}.safetensors"  # noqa: E731
    docs = data.calib_ids(tok, "train")
    calib_rows = sum(len(d["ids"]) for d in docs)
    stale = stale_projectors([proj_path(S, k) for S in splits for k in kinds], calib_rows)
    if stale:
        raise SystemExit("StaleProjector: fitted on another calibration list (rows recorded vs this list's "
                         f"{calib_rows}): " + ", ".join(f"{p} ({r})" for p, r in stale) +
                         ". A refit neither skips nor overwrites them: move them aside first")
    pending = [S for S in splits if not all(proj_path(S, k).exists() for k in kinds)]
    log(f"splits {splits}; to fit {pending}" + (f"; extras {kinds[1:]}{' + kvfold sums' if args.kvfold else ''}"
                                                if extra else ""))
    if not pending:
        return
    kv = {i: 2 * m.n_kv * m.head_dim for i in range(min(pending), L) if m.layer_types[i] == "attn"} if args.kvfold else None
    acc = accumulators(pending, L, H, args.unitrms, kv)
    nbytes = sum(8 * t.numel() for d in [acc] + [acc[k] for k in ("unitrms", "kvfold") if k in acc]
                 for k in ("sx", "sxx", "sy", "sxy", "sq") if k in d for t in d[k].values())
    log(f"host accumulators: {nbytes / 2**30:.1f} GiB fp64 for {len(acc['sxy'])} pairs")

    import shutil
    stats_dir.mkdir(parents=True, exist_ok=True)
    (prov.BULK / "proj").mkdir(parents=True, exist_ok=True)
    size = {S: (L - S + 1) * H * H * 8 for S in pending}
    if extra:
        for S in pending:
            size[S] += (L - S) * H * H * 8 if args.unitrms else 0
            size[S] += (H * H + sum(H * w for i, w in (kv or {}).items() if i >= S)) * 8
        free = shutil.disk_usage(stats_dir).free
        if free - sum(size.values()) < KEEP_FREE:
            raise SystemExit(f"refusing: sums need {sum(size.values()) / 1e9:.1f} GB, {free / 1e9:.1f} GB free in "
                             f"{stats_dir} (keep {KEEP_FREE / 1e9:.0f} GB)")

    # 1. capture
    xs = {}
    hook = capture_hook(acc, xs, m.eps, lambda i, h: kv_target(m, i, h))
    t0 = time.time()
    with torch.no_grad():
        for j, d in enumerate(docs):
            xs.clear()
            ids = torch.tensor(d["ids"])
            m.forward_exact(ids, hook=hook)
            acc["n"] += ids.shape[0]
            if j % 10 == 0:
                log(f"capture {j + 1}/{len(docs)} rows {acc['n']} ({time.time() - t0:.0f}s) peak {gpu.peak_gib(device):.1f} GiB")
    xs.clear()
    log(f"capture done: {acc['n']} rows in {time.time() - t0:.0f}s")

    # stats policy (decided once, from the free space now)
    free = shutil.disk_usage(stats_dir).free
    if extra:
        policy = dict(dtype="float64", splits=pending)
    elif free - sum(size.values()) >= KEEP_FREE:
        policy = dict(dtype="float64", splits=pending)
    elif free - sum(size.values()) // 2 >= KEEP_FREE:
        policy = dict(dtype="float32", splits=pending)
    else:
        sub = [S for S in pending if S in SUBSET]
        dt = "float64" if free - sum(size[S] for S in sub) >= KEEP_FREE else "float32"
        policy = dict(dtype=dt, splits=sub if free - sum(size[S] for S in sub) // (1 if dt == "float64" else 2) >= KEEP_FREE else [])
    policy.update(free_before=free, note="fp32 sums lose ~1e-7 relative precision" if policy["dtype"] == "float32" else "")
    log(f"stats policy: {policy}")

    # 2. per split: fit, select lambda on held docs, save
    held = data.calib_ids(tok, "held")
    for S in sorted(pending, reverse=True):
        t0 = time.time()
        late = list(range(S, L))
        state = split_state(acc, S, L, H)
        if S in policy["splits"]:
            dt = getattr(torch, policy["dtype"])
            out = dict(state, dtype=policy["dtype"], note=policy["note"], docs=[d["name"] for d in docs], model=name,
                       model_commit=prov.model_commit(model_dir), code_commit=prov.COMMIT_AT_START)
            if extra:
                out.update(inputs=str(data.INPUTS), eps=m.eps)
            for k in ("sx", "sxx"):
                out[k] = state[k].to(dt)
            out["sy"] = {i: v.to(dt) for i, v in state["sy"].items()}
            out["sxy"] = {i: v.to(dt) for i, v in state["sxy"].items()}
            spath = stats_dir / f"{name}-s{S}.pt"
            torch.save(out, spath)
            del out
            log(f"S={S}: stats saved {spath} ({spath.stat().st_size / 2**30:.2f} GiB, {policy['dtype']})")
        targets = {"": state, "-rawctl": state, "-unitrms": dict(state, **state.get("unitrms", {}))}
        maps = {}
        for kind in kinds:
            fit = RidgeFit(targets[kind])
            maps[kind] = {lam: {} for lam in LAMBDAS}
            for i in late:
                for lam, (w, b) in fit.layer(i, LAMBDAS).items():
                    maps[kind][lam][i] = (w.to(torch.bfloat16), b.to(torch.bfloat16))
            del fit
        del state, targets
        drop_split(acc, S, L)                            # free this split's accumulators
        gc.collect()
        log(f"S={S}: fitted {len(late)} layers x {len(LAMBDAS)} lambdas x {len(kinds)} targets in {time.time() - t0:.0f}s")
        cand = [(k, lam) for k in kinds for lam in LAMBDAS] + [(None, "identity")]
        sums = {c: {i: 0.0 for i in late} for c in cand}
        count = 0
        with torch.no_grad():
            for d in held:
                got = {}
                m.forward_exact(torch.tensor(d["ids"]), hook=lambda i, h: got.__setitem__(i, h)
                                if i != "final" and i >= S else None)
                count += 1
                for i in late:
                    true = m.reads(i, got[i])
                    for k, c in cand:
                        x = got[S] if k is None else F.linear(got[S], maps[k][c][i][0].to(device), maps[k][c][i][1].to(device))
                        pred = m.reads(i, x)
                        sums[k, c][i] += sum(F.cosine_similarity(pred[r].float(), true[r].float(), dim=-1).mean().item()
                                             for r in true) / len(true)
                del got
        for kind in kinds:
            per_layer = {str(c): {i: v / count for i, v in s.items()} for (k, c), s in sums.items() if k in (kind, None)}
            mean = {c: sum(v.values()) / len(v) for c, v in per_layer.items()}
            best = max(LAMBDAS, key=lambda lam: mean[str(lam)])
            path = proj_path(S, kind)
            W = {i: maps[kind][best][i][0] for i in late}
            B = {i: maps[kind][best][i][1] for i in late}
            meta = dict(model=name, model_commit=prov.model_commit(model_dir), lam=best, rows=acc["n"], taps=S, sweep=True,
                        note="Built with Llama" if "llama" in name else "",
                        calib_exclude_sha256=prov.sha256(data.CALIB_EXCLUDE_FILE), code_commit=prov.COMMIT_AT_START,
                        code_dirty=prov.DIRTY_AT_START, docs=len(docs))
            if extra:
                meta.update(target="y / rms(y)" if kind == "-unitrms" else "y", sums=str(stats_dir / f"{name}-s{S}.pt"))
            Projector(S, W, B).save(path, meta)
            prov.write_json(prov.WORK / "results" / f"fit-sweep-{name}-s{S}{kind}.json",
                            dict(provenance=prov.provenance(model_dir, dict(split=S, lambdas=LAMBDAS, sweep_splits=splits,
                                                                            held=[d["name"] for d in held],
                                                                            **(dict(target=meta["target"], inputs=str(data.INPUTS))
                                                                               if extra else {})), gpu_info),
                                 rows=acc["n"], docs=len(docs), lambda_star=best, mean_read_cos=mean,
                                 per_layer_read_cos=per_layer,
                                 projector=str(path), projector_sha256=prov.sha256(path)))
            log(f"S={S}{kind}: lambda* {best} read cos {mean[str(best)]:.4f} (identity {mean['identity']:.4f}) -> {path}")
        del maps
        gc.collect()


if __name__ == "__main__":
    main()
