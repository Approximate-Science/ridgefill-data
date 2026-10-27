"""GPU selection guard: one card per process, asserted before any allocation.

AMD (the default, KVA_GPU_VENDOR=amd): KVA_CARD names the card as its PCI bus number in hex (e.g. 13 for the card on
bus 13), and the launcher exports ROCR_VISIBLE_DEVICES=<that card's KFD unique id (rocminfo or sysfs unique_id)> and
HIP_VISIBLE_DEVICES=0. setup() refuses by name unless KVA_CARD is set, ROCR_VISIBLE_DEVICES names exactly one device,
exactly one device is visible and it sits on the KVA_CARD bus. KVA_VRAM_CAP_GIB caps the caching allocator if set
(default: no cap below the card's memory). No card facts live in this file: they come from the machine's environment.

NVIDIA hosts (the rented finals boxes): KVA_GPU_VENDOR=nvidia selects nvidia_setup() instead. There is no card bus to
check, so the device is identified by NAME: KVA_GPU_NAME (required, e.g. "A100") must
be a substring of the one visible device's name. It refuses a ROCm torch build, a CPU-only build, any device count
but 1 (the container exposes one card per cell) and a name mismatch, all by name, before any allocation. Determinism:
CUBLAS_WORKSPACE_CONFIG=:4096:8 (set if unset, before the first cuBLAS handle; cuBLAS is only deterministic with
it), cuDNN deterministic and benchmark off, TF32 off for matmul and cuDNN (the reference runs bf16/fp32 as written,
never TF32), torch.use_deterministic_algorithms as on AMD (KVA_DETERMINISTIC, default 1, warn_only as on AMD).
KVA_VRAM_CAP_GIB caps the caching allocator if set (default: no cap below the card's memory).
"""
import os
import re

VENDORS = ("amd", "nvidia")
CUBLAS_WORKSPACE = ":4096:8"


def amd_card(env):
    """The KVA_CARD bus number, after checking the card selection env (module doc). Refuses by name."""
    card = env.get("KVA_CARD", "")
    if not re.fullmatch(r"[0-9a-fA-F]{1,2}", card):
        raise SystemExit(f"refusing: KVA_CARD {card!r} is not a PCI bus number in hex: name the card this process "
                         "must run on (e.g. 13 for the card on bus 13)")
    visible = env.get("ROCR_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise SystemExit(f"refusing: ROCR_VISIBLE_DEVICES {visible!r} must name exactly one device: the KVA_CARD "
                         "card's KFD unique id (rocminfo or sysfs unique_id)")
    return int(card, 16)


def setup():
    if os.environ.get("KVA_CPU_DRYRUN") == "1":            # scripts/dryrun.sh: tiny models on the CPU
        import torch
        return torch.device("cpu"), dict(cpu_dryrun=True)
    vendor = os.environ.get("KVA_GPU_VENDOR", "amd")
    if vendor not in VENDORS:
        raise SystemExit(f"refusing: unknown KVA_GPU_VENDOR {vendor!r} (one of {VENDORS})")
    if vendor == "nvidia":
        import torch
        return nvidia_setup(torch, os.environ)
    bus = amd_card(os.environ)
    import torch
    n = torch.cuda.device_count()
    if n != 1:
        raise SystemExit(f"refusing: {n} visible devices, expected exactly 1")
    p = torch.cuda.get_device_properties(0)
    if (p.pci_domain_id, p.pci_bus_id) != (0, bus):
        raise SystemExit(f"refusing: device 0 is on domain {p.pci_domain_id:04x} bus {p.pci_bus_id:02x}, not on bus "
                         f"{bus:02x} (KVA_CARD)")
    cap = float(os.environ.get("KVA_VRAM_CAP_GIB", 0)) * 2**30 or p.total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / p.total_memory), 0)
    # default on: without it Qwen3.5 (GDN fp32 GEMMs) is not run-to-run reproducible (0/12 identical; with it 12/12,
    # ~10% slower). KVA_DETERMINISTIC=0 opts out.
    deterministic = os.environ.get("KVA_DETERMINISTIC", "1") == "1"
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.device("cuda:0"), dict(deterministic=deterministic, name=p.name, arch=p.gcnArchName,
                                        pci=f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:00.0",
                                        total_gib=round(p.total_memory / 2**30, 2), cap_gib=cap / 2**30)


def nvidia_setup(torch, env):
    """One named NVIDIA card per process (module doc). torch is the torch module (a fake in the CPU tests); env the
    process environment, which gets CUBLAS_WORKSPACE_CONFIG if it is unset. Returns (device, gpu_info)."""
    if torch.version.hip:
        raise SystemExit(f"refusing: KVA_GPU_VENDOR=nvidia but this torch is a ROCm build (hip {torch.version.hip})")
    if not torch.version.cuda:
        raise SystemExit("refusing: KVA_GPU_VENDOR=nvidia but this torch has no CUDA (a CPU-only build)")
    want = env.get("KVA_GPU_NAME", "").strip()
    if not want:
        raise SystemExit("refusing: KVA_GPU_NAME is not set: name the rented card (a substring of its device name, "
                         "e.g. A100), so a run never lands on another card type unnoticed")
    n = torch.cuda.device_count()
    if n != 1:
        raise SystemExit(f"refusing: {n} visible CUDA devices, expected exactly 1 (one card per cell: "
                         f"CUDA_VISIBLE_DEVICES)")
    p = torch.cuda.get_device_properties(0)
    if want.lower() not in p.name.lower():
        raise SystemExit(f"refusing: device 0 is {p.name!r}, not a {want!r} (KVA_GPU_NAME)")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    if env["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise SystemExit(f"refusing: CUBLAS_WORKSPACE_CONFIG {env['CUBLAS_WORKSPACE_CONFIG']!r} does not make "
                         "cuBLAS deterministic (:4096:8 or :16:8)")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    deterministic = env.get("KVA_DETERMINISTIC", "1") == "1"
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    cap = float(env.get("KVA_VRAM_CAP_GIB", 0)) * 2**30 or p.total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / p.total_memory), 0)
    return torch.device("cuda:0"), dict(deterministic=deterministic, vendor="nvidia", name=p.name,
                                        arch=f"sm_{p.major}{p.minor}", pci=f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:00.0",
                                        total_gib=round(p.total_memory / 2**30, 2), cap_gib=cap / 2**30,
                                        runtime=f"cuda {torch.version.cuda}",
                                        cublas_workspace=env["CUBLAS_WORKSPACE_CONFIG"])


def peak_gib(device):
    import torch
    return torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0


def used_gib(device):
    import torch
    return torch.cuda.memory_allocated() / 2**30 if device.type == "cuda" else 0.0


def empty_cache(device):
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()
