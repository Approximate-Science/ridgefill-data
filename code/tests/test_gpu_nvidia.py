"""kva/gpu.py on an NVIDIA host (KVA_GPU_VENDOR=nvidia; the rented finals boxes), CPU only with a fake torch module:
the device is identified by name (KVA_GPU_NAME), a ROCm or CPU-only torch build, a device count other than 1 and a
name mismatch are refused by name before any allocation; the deterministic flags are set (cuBLAS workspace, cuDNN
deterministic and no benchmark, TF32 off, deterministic algorithms); the AMD path stays the default and an unknown
vendor is refused."""
import types

import pytest

from kva import gpu


def fake_torch(name="NVIDIA A100-SXM4-40GB", count=1, cuda="12.4", hip=None, total=40 * 2**30):
    calls = {}
    props = types.SimpleNamespace(name=name, major=8, minor=0, pci_domain_id=0, pci_bus_id=0x1b, total_memory=total)
    cuda_mod = types.SimpleNamespace(
        device_count=lambda: count, get_device_properties=lambda i: props,
        set_per_process_memory_fraction=lambda f, i: calls.__setitem__("fraction", (f, i)))
    backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False, benchmark=True, allow_tf32=True),
                                     cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)))
    t = types.SimpleNamespace(version=types.SimpleNamespace(cuda=cuda, hip=hip), cuda=cuda_mod, backends=backends,
                              device=lambda s: ("device", s),
                              use_deterministic_algorithms=lambda on, warn_only: calls.__setitem__("det", (on, warn_only)))
    return t, calls


def test_a_named_a100_is_set_up_deterministically():
    t, calls = fake_torch()
    env = {"KVA_GPU_NAME": "A100"}
    device, info = gpu.nvidia_setup(t, env)
    assert device == ("device", "cuda:0")
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert t.backends.cudnn.deterministic is True and t.backends.cudnn.benchmark is False
    assert t.backends.cudnn.allow_tf32 is False and t.backends.cuda.matmul.allow_tf32 is False
    assert calls["det"] == (True, True) and calls["fraction"] == (1.0, 0)
    assert info["name"] == "NVIDIA A100-SXM4-40GB" and info["arch"] == "sm_80" and info["vendor"] == "nvidia"
    assert info["runtime"] == "cuda 12.4" and info["deterministic"] is True and info["pci"].split(":") == ["0000", "1b", "00.0"]


def test_the_vram_cap_and_the_deterministic_opt_out_are_honoured():
    t, calls = fake_torch()
    env = {"KVA_GPU_NAME": "a100", "KVA_VRAM_CAP_GIB": "30", "KVA_DETERMINISTIC": "0"}
    _, info = gpu.nvidia_setup(t, env)
    assert calls["fraction"] == (0.75, 0) and info["cap_gib"] == 30.0
    assert "det" not in calls and info["deterministic"] is False


@pytest.mark.parametrize("torch_kw,env,match", [
    (dict(hip="6.4", cuda=None), {"KVA_GPU_NAME": "A100"}, "ROCm build"),
    (dict(cuda=None), {"KVA_GPU_NAME": "A100"}, "no CUDA"),
    (dict(), {}, "KVA_GPU_NAME is not set"),
    (dict(count=2), {"KVA_GPU_NAME": "A100"}, "2 visible CUDA devices"),
    (dict(count=0), {"KVA_GPU_NAME": "A100"}, "0 visible CUDA devices"),
    (dict(name="NVIDIA H100 80GB HBM3"), {"KVA_GPU_NAME": "A100"}, "not a 'A100'"),
    (dict(), {"KVA_GPU_NAME": "A100", "CUBLAS_WORKSPACE_CONFIG": ":0:0"}, "does not make cuBLAS deterministic"),
])
def test_refusals_by_name_before_any_allocation(torch_kw, env, match):
    t, calls = fake_torch(**torch_kw)
    with pytest.raises(SystemExit, match=match):
        gpu.nvidia_setup(t, dict(env))
    assert "fraction" not in calls and "det" not in calls


def test_setup_dispatches_on_the_vendor(monkeypatch):
    monkeypatch.delenv("KVA_CPU_DRYRUN", raising=False)
    monkeypatch.setenv("KVA_GPU_VENDOR", "intel")
    with pytest.raises(SystemExit, match="unknown KVA_GPU_VENDOR 'intel'"):
        gpu.setup()
    seen = {}
    monkeypatch.setenv("KVA_GPU_VENDOR", "nvidia")
    monkeypatch.setattr(gpu, "nvidia_setup", lambda torch, env: seen.setdefault("nvidia", torch.__name__))
    assert gpu.setup() == "torch"
    # the default stays the AMD path: with no vendor it checks the card selection env (and refuses a bad card id)
    monkeypatch.delenv("KVA_GPU_VENDOR")
    monkeypatch.setenv("KVA_CARD", "zz")
    with pytest.raises(SystemExit, match="KVA_CARD 'zz' is not a PCI bus number"):
        gpu.setup()


def test_the_amd_card_comes_from_the_environment():
    with pytest.raises(SystemExit, match="KVA_CARD '' is not a PCI bus number"):
        gpu.amd_card({})
    with pytest.raises(SystemExit, match="ROCR_VISIBLE_DEVICES '' must name exactly one device"):
        gpu.amd_card({"KVA_CARD": "13"})
    with pytest.raises(SystemExit, match="ROCR_VISIBLE_DEVICES 'GPU-a,GPU-b' must name exactly one device"):
        gpu.amd_card({"KVA_CARD": "13", "ROCR_VISIBLE_DEVICES": "GPU-a,GPU-b"})
    assert gpu.amd_card({"KVA_CARD": "13", "ROCR_VISIBLE_DEVICES": "GPU-a"}) == 0x13
