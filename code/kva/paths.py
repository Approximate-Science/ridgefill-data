"""Where things live. Everything is relative to this repo except DATA: big local files that are not
committed (model links, projectors, sums, captures, B0/27B server data, run logs and safety files).
DATA defaults to <repo>/data (a local directory or symlink, never committed); override with KVA_DATA.
Configs write data paths as "$KVA_DATA/..."; expand() resolves them."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("KVA_DATA", REPO / "data"))
os.environ.setdefault("KVA_DATA", str(DATA))
RESULTS = REPO / "results"
DATASETS = REPO / "datasets"
MODELS = DATA / "models"
RUNS = DATA / "runs"
SUMS = Path(os.environ.get("KVA_STATS_DIR", DATA / "sums"))   # fit_sweep.py's ridge sums
SUMS_RMSFOLD = SUMS.with_name(SUMS.name + "-rmsfold")          # fit_sweep.py --unitrms/--kvfold sums (kept apart)
SUMS_WFIT = SUMS.with_name(SUMS.name + "-wfit")                # fit_weighted.py's weighted sums (kept apart)


def expand(obj):
    """Expand $KVA_DATA (and other env vars) in every string of a loaded config."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    return obj
