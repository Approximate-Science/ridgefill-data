#!/usr/bin/env python3
"""Dry-run models: tiny random 32-layer versions of the three checkpoints (real vocab sizes and
tokenizer files, real layer patterns, SwiftKV extras from layer 16) under OUT/<same dir names>.
Usage: make_tiny.py OUT"""
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path(os.environ.get("KVA_DATA", Path(__file__).resolve().parents[1] / "data")) / "models"
TOK = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
       "chat_template.jinja", "generation_config.json")


def copy_tok(src, dst):
    for f in TOK:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)


def main():
    out = Path(sys.argv[1])
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    lc = LlamaConfig(vocab_size=128256, hidden_size=64, intermediate_size=128, num_hidden_layers=32,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5,
                     rope_theta=500000.0, max_position_embeddings=131072, tie_word_embeddings=False,
                     bos_token_id=128000, rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0,
                                                            high_freq_factor=4.0, original_max_position_embeddings=8192))
    base = out / "llama-3.1-8b-instruct"
    LlamaForCausalLM(lc).to(torch.bfloat16).save_pretrained(str(base))
    copy_tok(SRC / "llama-3.1-8b-instruct", base)
    swift = out / "llama-3.1-swiftkv-8b-instruct"
    swift.mkdir(parents=True, exist_ok=True)
    t = load_file(str(base / "model.safetensors"))
    t["model.norm_swiftkv.weight"] = torch.ones(64, dtype=torch.bfloat16)
    for i in range(16, 32):
        p = f"model.layers.{i}.self_attn."
        t[p + "q_proj_swiftkv.weight"] = t[p + "q_proj.weight"].clone()
        t[p + "k_proj_swiftkv.weight"] = (t[p + "k_proj.weight"].float() * 1.1).to(torch.bfloat16)
        t[p + "v_proj_swiftkv.weight"] = t[p + "v_proj.weight"].clone()
    save_file(t, str(swift / "model.safetensors"))
    cfg = json.loads((base / "config.json").read_text())
    cfg.update(swiftkv=True, num_key_value_layers=16, key_value_group_size=1)
    (swift / "config.json").write_text(json.dumps(cfg))
    copy_tok(SRC / "llama-3.1-swiftkv-8b-instruct", swift)
    types = (["linear_attention"] * 3 + ["full_attention"]) * 8
    qc = Qwen3_5TextConfig(vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=32,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32, layer_types=types,
                           linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                           linear_value_head_dim=16, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
                           max_position_embeddings=262144, tie_word_embeddings=False,
                           rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25,
                                                mrope_interleaved=True, mrope_section=[2, 1, 1]))
    q = out / "qwen3.5-9b"
    Qwen3_5ForCausalLM(qc).to(torch.bfloat16).save_pretrained(str(q))
    copy_tok(SRC / "qwen3.5-9b", q)
    print("tiny models in", out)


if __name__ == "__main__":
    main()
