"""Tiny random checkpoints (HF classes, saved as safetensors) for CPU tests in fp32."""
import json
import os

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def _save(model, path):
    model.save_pretrained(str(path), safe_serialization=True)


@pytest.fixture(scope="session")
def tiny_llama(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=160, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5,
                      rope_theta=500000.0, max_position_embeddings=4096, tie_word_embeddings=False,
                      rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                                        original_max_position_embeddings=64))
    m = LlamaForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.add_(0.1 * torch.randn_like(p))   # non-trivial norm weights
    path = tmp_path_factory.mktemp("llama")
    _save(m, path)
    return path, m


@pytest.fixture(scope="session")
def tiny_swiftkv(tiny_llama, tmp_path_factory):
    """The tiny Llama plus SwiftKV extras for layers >= 3 (random)."""
    from safetensors.torch import load_file, save_file
    src, _ = tiny_llama
    path = tmp_path_factory.mktemp("swiftkv")
    t = load_file(str(src / "model.safetensors"))
    g = torch.Generator().manual_seed(1)
    t["model.norm_swiftkv.weight"] = 1 + 0.1 * torch.randn(64, generator=g)
    for i in range(3, 6):
        p = f"model.layers.{i}.self_attn."
        t[p + "q_proj_swiftkv.weight"] = 0.1 * torch.randn(64, 64, generator=g)
        t[p + "k_proj_swiftkv.weight"] = 0.1 * torch.randn(32, 64, generator=g)
        t[p + "v_proj_swiftkv.weight"] = 0.1 * torch.randn(32, 64, generator=g)
    save_file(t, str(path / "model.safetensors"))
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(swiftkv=True, num_key_value_layers=3, key_value_group_size=1)
    (path / "config.json").write_text(json.dumps(cfg))
    return path


@pytest.fixture(scope="session")
def tiny_qwen(tmp_path_factory):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    types = (["linear_attention"] * 3 + ["full_attention"]) * 2
    cfg = Qwen3_5TextConfig(vocab_size=300, hidden_size=64, intermediate_size=160, num_hidden_layers=8,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32, layer_types=types,
                            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
                            max_position_embeddings=4096, tie_word_embeddings=False,
                            rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25,
                                                 mrope_interleaved=True, mrope_section=[2, 1, 1]))
    m = Qwen3_5ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 1 and "A_log" not in n and "dt_bias" not in n:
                p.add_(0.1 * torch.randn_like(p))
    path = tmp_path_factory.mktemp("qwen")
    _save(m, path)
    return path, m


@pytest.fixture(scope="session")
def tiny_qwen3(tmp_path_factory):
    """Dense Qwen3 (Qwen3-8B's architecture: per-head q/k RMSNorm, default RoPE), 4 layers."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=160, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                      max_position_embeddings=4096, tie_word_embeddings=False,
                      rope_parameters=dict(rope_type="default", rope_theta=1000000.0))
    m = Qwen3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.add_(0.1 * torch.randn_like(p))   # non-trivial norm weights (q_norm / k_norm too)
    path = tmp_path_factory.mktemp("qwen3")
    _save(m, path)
    return path, m
