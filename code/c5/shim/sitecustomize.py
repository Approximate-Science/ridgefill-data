"""FIX-5 (2026-10-08). tcclaviger/vllm:29.08.10's PLE offload worker (vllm/v1/ple_offload/worker.py:593) logs
`self._nvme_store.reuse`, an attribute that build removed from PleNvmeStore (reuse is now decided per table:
`_reused`, `is_reused`). Every boot with --ple-nvme-offload therefore dies with AttributeError before the model loads
(the C5 smoke boot, 2026-10-08 16:52). Nothing else in the image reads `.reuse` (grep of vllm/v1/ple_offload). This
adds it back for that log line only, as a read-only property (True when any table was served from disk); no engine
file is changed and no numeric path is touched. Mounted on PYTHONPATH by c5/run_c5.sh; it chains to the image's own
sitecustomize (the apport hook) so nothing else changes."""
import importlib.abc
import os
import sys

_TARGET = "vllm.v1.ple_offload.nvme_table"


def _patch(module):
    cls = getattr(module, "PleNvmeStore", None)
    if cls is not None and not hasattr(cls, "reuse"):
        cls.reuse = property(lambda self: bool(getattr(self, "_reused", None)))
        os.environ["C5_FIX5_APPLIED"] = "1"


class _Fix5(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch(module)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Fix5())

_orig = "/usr/lib/python3.14/sitecustomize.py"
if os.path.isfile(_orig):
    with open(_orig) as f:
        exec(compile(f.read(), _orig, "exec"), {"__name__": "sitecustomize", "__file__": _orig})
