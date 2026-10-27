"""Own forward pass for Llama-3.1 (incl. the SwiftKV extras), dense Qwen3 (Qwen3-8B: Llama's layer plus per-head
q/k RMSNorm) and the Qwen3.5 text model (Gated DeltaNet + gated attention), layer by layer over explicit caches.
load() picks the class from config.json's model_type and refuses any other architecture (UnsupportedArchitecture).

The per-layer API is what KVA needs:
  layer(i, h, pos, cache)       exact layer on rows h (positions pos), appending to cache[i]
  fill(i, x, pos, cache)        cache-only fill from (predicted) layer inputs x: attention K/V,
                                or the GDN conv + recurrent state; no attention output, no MLP
  fill_kv(i, k, v, pos, cache)  attention cache-only fill from given pre-RoPE K/V (the K/V fold)
  layer_rows(i, x, pos, rows, cache)  fill, plus the exact layer output of the rows x[rows]
  queries(i, x, pos)            an attention layer's post-RoPE queries (the qattn selector)
  reads(i, x)                   what layer i reads from its input (for projector scoring)
  layer(..., mix=f) / forward_exact(..., mix=f)  also call f(i, a) with layer i's mixer output a: what the
                                attention block (Llama, Qwen3.5 attention layers) or the GDN block (Qwen3.5 GDN
                                layers) adds to the residual, after o_proj / out_proj, before the MLP (the
                                context projector's features, scripts/fit_context.py)
Weights are read from the checkpoint's safetensors straight onto the target device; load() verifies them.
"""
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import integrity, ops

MLP_ROWS = 4096   # row block for MLPs (peak-memory bound, exact either way)
KV_RESERVE = 64   # spare K/V rows per layer buffer (decode appends in place)
GDN_ROWS = 8192   # row segment of the GDN conv + recurrence (multiple of the 64-token chunk)
ROW_BYTES = 1 << 25   # layer_rows: more exact rows than one [rows, hidden] activation of this size go in runs (4096 on 8B bf16)


class UnsupportedArchitecture(ValueError):
    """config.json describes a model (or a feature of one: attention bias, sliding window, a RoPE type) this forward
    pass does not implement. Raised at load, never approximated."""


def _rows(fn, x):
    if x.shape[0] <= MLP_ROWS:
        return fn(x)
    return torch.cat([fn(x[i:i + MLP_ROWS]) for i in range(0, x.shape[0], MLP_ROWS)], 0)


class Checkpoint:
    """Lazy tensor access over a (possibly sharded) safetensors checkpoint."""

    def __init__(self, path):
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.map = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.path / "model.safetensors", "pt") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        self.files = {}

    def __contains__(self, key):
        return key in self.map

    def get(self, key, device, dtype=None):
        name = self.map[key]
        handle = (name, str(device))
        if handle not in self.files:
            self.files[handle] = safe_open(self.path / name, "pt", device=str(device))
        t = self.files[handle].get_tensor(key)
        return t.to(dtype) if dtype is not None and t.is_floating_point() else t


class Model:
    """Shared machinery; subclasses define the per-layer math."""

    arch = None

    def __init__(self, path, device="cpu", dtype=torch.bfloat16, load_lm_head=True, extras=True):
        self.path = Path(path)
        raw = json.loads((self.path / "config.json").read_text())
        self.config = raw.get("text_config", raw)
        self.raw_config = raw
        self.device = torch.device(device)
        self.dtype = dtype
        self.ckpt = Checkpoint(path)
        self.n_layers = self.config["num_hidden_layers"]
        self.max_positions = self.config.get("max_position_embeddings")   # evaluate.py refuses prompts past it
        self.hidden = self.config["hidden_size"]
        self.eps = self.config["rms_norm_eps"]
        self._setup()
        self.W = [self._load_layer(i) for i in range(self.n_layers)]
        p = self.prefix
        # the embedding table stays on the CPU (a row lookup; saves 1-2 GB of VRAM)
        self.embed_w = self.ckpt.get(f"{p}embed_tokens.weight", "cpu", self.dtype)
        self.norm_w = self._get(f"{p}norm.weight")
        self.lm_head = self._get("lm_head.weight") if load_lm_head else None
        if extras:
            self._load_extras()

    def _get(self, key):
        return self.ckpt.get(key, self.device, self.dtype)

    def _get_raw(self, key):
        return self.ckpt.get(key, self.device)

    # ---- common
    def new_cache(self):
        return [dict() for _ in range(self.n_layers)]

    def embed(self, ids):
        return F.embedding(ids.cpu(), self.embed_w).to(self.device)

    def logits(self, h):
        return F.linear(self.final_norm(h), self.lm_head).float()

    def mlp(self, i, h):
        w = self.W[i]

        def run(x):
            x = self.post_norm(x, w["ln2"])
            return F.linear(F.silu(F.linear(x, w["gate"])) * F.linear(x, w["up"]), w["down"])

        return h + _rows(run, h)

    def row_block(self):
        """layer_rows' threshold and run length: the rows of one [rows, hidden] activation of ROW_BYTES."""
        return max(1, ROW_BYTES // (self.hidden * self.dtype.itemsize))

    def mlp_rows(self, i, h):
        """mlp() for layer_rows. Over row_block() rows: in place, in runs of whole MLP_ROWS blocks, which are the
        blocks mlp() makes on all rows (bitwise the same), without its full-height block outputs, concatenation
        and sum."""
        n = h.shape[0]
        if n <= self.row_block():
            return self.mlp(i, h)
        step = max(MLP_ROWS, self.row_block() // MLP_ROWS * MLP_ROWS)
        for a in range(0, n, step):
            h[a:a + step] = self.mlp(i, h[a:a + step])
        return h

    def _append_kv(self, cache, k, v):
        """Append rows to the layer's K/V. Storage is a buffer with spare capacity written in place;
        cache['n'] is this cache's length and cache['k'] / cache['v'] views of its rows. Caches
        copied per arm (dict copies) share the buffer: rows below every copy's length are never
        rewritten, and arms run one after another, so each arm's appends only overwrite rows that
        no other live cache reads."""
        n = cache.get("n", 0)
        m = k.shape[1]
        buf = cache.get("kbuf")
        if buf is None or n + m > buf.shape[1]:
            cap = n + m + KV_RESERVE
            kb = k.new_empty(k.shape[0], cap, k.shape[2])
            vb = v.new_empty(v.shape[0], cap, v.shape[2])
            if n:
                kb[:, :n] = cache["kbuf"][:, :n]
                vb[:, :n] = cache["vbuf"][:, :n]
            cache["kbuf"], cache["vbuf"] = kb, vb
        cache["kbuf"][:, n:n + m] = k
        cache["vbuf"][:, n:n + m] = v
        cache["n"] = n + m
        cache["k"], cache["v"] = cache["kbuf"][:, :n + m], cache["vbuf"][:, :n + m]

    def fill_kv(self, i, k, v, pos, cache):
        """Cache-only fill of attention layer i from given pre-RoPE K and V [tokens, kv_heads, head_dim]
        (the reads() layout): RoPE on K, then the same append as fill()."""
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, k.dtype)
        self._append_kv(cache[i], ops.apply_rope(k.transpose(0, 1), cos, sin), v.transpose(0, 1))

    def forward_exact(self, ids, hook=None, cache=None, keep=None, gdn_checkpoint_every=None, mix=None):
        """Exact forward over all rows. hook(i, h) sees each layer's input (i = layer index)
        and hook('final', h) the final-norm input; mix(i, a) each layer's mixer output (layer()).
        keep: row indices whose logits to return."""
        pos = torch.arange(ids.shape[0], device=self.device)
        cache = self.new_cache() if cache is None else cache
        h = self.embed(ids)
        for i in range(self.n_layers):
            if hook is not None:
                hook(i, h)
            h = self.layer(i, h, pos, cache, gdn_checkpoint_every=gdn_checkpoint_every, mix=mix)
        if hook is not None:
            hook("final", h)
        out = self.logits(h[keep]) if keep is not None else None
        return out, cache


class Llama(Model):
    arch = "llama"

    def _setup(self):
        c = self.config
        self.prefix = "model."
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c.get("head_dim") or self.hidden // self.n_heads
        self.scale = self.head_dim ** -0.5
        rs = c.get("rope_parameters") or c.get("rope_scaling") or {}   # transformers 5 / 4 layouts
        theta = rs.get("rope_theta", c.get("rope_theta"))
        if rs.get("rope_type") == "llama3":
            self.inv_freq = ops.inv_freq_llama3(theta, self.head_dim, rs["factor"], rs["low_freq_factor"],
                                                rs["high_freq_factor"], rs["original_max_position_embeddings"])
        else:
            self.inv_freq = ops.inv_freq_default(theta, self.head_dim)
        self.layer_types = ["attn"] * c["num_hidden_layers"]
        self.swift_layers = c.get("num_key_value_layers") if c.get("swiftkv") else None

    def _load_layer(self, i):
        p = f"model.layers.{i}."
        g = self._get
        return dict(ln1=g(p + "input_layernorm.weight"), ln2=g(p + "post_attention_layernorm.weight"),
                    q=g(p + "self_attn.q_proj.weight"), k=g(p + "self_attn.k_proj.weight"),
                    v=g(p + "self_attn.v_proj.weight"), o=g(p + "self_attn.o_proj.weight"),
                    gate=g(p + "mlp.gate_proj.weight"), up=g(p + "mlp.up_proj.weight"),
                    down=g(p + "mlp.down_proj.weight"))

    def _load_extras(self):
        self.swift = None
        if self.swift_layers is None:
            return
        self.swift = dict(norm=self._get("model.norm_swiftkv.weight"))
        for i in range(self.swift_layers, self.n_layers):
            p = f"model.layers.{i}.self_attn."
            self.swift[i] = dict(q=self._get(p + "q_proj_swiftkv.weight"), k=self._get(p + "k_proj_swiftkv.weight"),
                                 v=self._get(p + "v_proj_swiftkv.weight"))

    def in_norm(self, x, w):
        return ops.rms_norm(x, w, self.eps)

    post_norm = in_norm

    def final_norm(self, h):
        return ops.rms_norm(h, self.norm_w, self.eps)

    def _heads(self, x, n_heads):
        return x.view(x.shape[0], n_heads, self.head_dim).transpose(0, 1)

    def _kv(self, i, x, pos):
        w = self.W[i]
        k = self._heads(F.linear(x, w["k"]), self.n_kv)
        v = self._heads(F.linear(x, w["v"]), self.n_kv)
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, x.dtype)
        return ops.apply_rope(k, cos, sin), v

    def queries(self, i, x, pos):
        """Post-RoPE queries [Hq, n, d] of normed layer inputs x."""
        q = self._heads(F.linear(x, self.W[i]["q"]), self.n_heads)
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, x.dtype)
        return ops.apply_rope(q, cos, sin)

    def layer(self, i, h, pos, cache, mix=None, **_):
        w = self.W[i]
        x = self.in_norm(h, w["ln1"])
        q = self.queries(i, x, pos)
        k, v = self._kv(i, x, pos)
        self._append_kv(cache[i], k, v)
        o = ops.attend(q, cache[i]["k"], cache[i]["v"], self.scale)
        a = F.linear(o, w["o"])
        if mix is not None:
            mix(i, a)
        return self.mlp(i, h + a)

    def layer_rows(self, i, h, pos, rows, cache):
        """Fill cache[i] from the inputs h of rows at positions pos (an empty cache, pos = 0..n-1),
        as fill(); return the exact layer output of h[rows] over that cache (= layer(...)[rows]).
        Over row_block() rows, attention and MLP take the rows in runs of whole blocks (attend_rows, mlp_rows):
        bitwise the same, with the rows' temporaries bounded (VB1 keeps 8-16K exact rows at 32K). The all-row
        input, K and V are dropped once the cache holds them."""
        w = self.W[i]
        x = self.in_norm(h, w["ln1"])
        k, v = self._kv(i, x, pos)
        self._append_kv(cache[i], k, v)
        del k, v
        q = self.queries(i, x[rows], pos[rows])
        del x
        o = ops.attend_rows(q, cache[i]["k"], cache[i]["v"], self.scale, pos[rows], self.row_block())
        del q
        hr = h[rows] + F.linear(o, w["o"])
        del o
        return self.mlp_rows(i, hr)

    def fill(self, i, x, pos, cache, **_):
        k, v = self._kv(i, self.in_norm(x, self.W[i]["ln1"]), pos)
        self._append_kv(cache[i], k, v)

    def reads(self, i, x):
        """Pre-RoPE K and V per KV head: [tokens, kv_heads, head_dim] each."""
        w = self.W[i]
        x = self.in_norm(x, w["ln1"])
        n = x.shape[0]
        return dict(k=F.linear(x, w["k"]).view(n, self.n_kv, -1), v=F.linear(x, w["v"]).view(n, self.n_kv, -1))

    # ---- SwiftKV (Snowflake): late-layer K/V from norm_swiftkv(boundary) via *_proj_swiftkv
    def swift_fill(self, i, b, pos, cache):
        s = self.swift
        xs = ops.rms_norm(b, s["norm"], self.eps)
        k = self._heads(F.linear(xs, s[i]["k"]), self.n_kv)
        v = self._heads(F.linear(xs, s[i]["v"]), self.n_kv)
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, b.dtype)
        self._append_kv(cache[i], ops.apply_rope(k, cos, sin), v)

    def swift_layer(self, i, h, pos, cache):
        """SwiftKV's late layer for query rows h whose own K/V are already in the cache:
        queries from q_proj_swiftkv; attention, o_proj and MLP as the base layer."""
        w = self.W[i]
        x = self.in_norm(h, w["ln1"])
        q = self._heads(F.linear(x, self.swift[i]["q"]), self.n_heads)
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, x.dtype)
        q = ops.apply_rope(q, cos, sin)
        o = ops.attend(q, cache[i]["k"], cache[i]["v"], self.scale)
        h = h + F.linear(o, w["o"])
        return self.mlp(i, h)


class Qwen3(Llama):
    """Dense Qwen3 (Qwen3ForCausalLM, e.g. Qwen3-8B: 36 layers, GQA 32/8, head_dim 128): Llama's layer with a per-head
    RMSNorm on queries and keys (q_norm / k_norm, plain weight, Llama's RMSNorm) after q_proj / k_proj and before RoPE;
    default RoPE (theta from config.json), no attention bias, no GDN, no SwiftKV extras. Every K the model writes or
    reads goes through _kv / reads, so fills, layer_rows and the projector scoring all see the normed keys.
    Long context: the released config has no RoPE scaling (max_position_embeddings 40960); a scaled RoPE (Qwen's YaRN
    recipe for > 32K) is refused at load, not approximated."""
    arch = "qwen3"

    def _setup(self):
        c = self.config
        rs = c.get("rope_parameters") or c.get("rope_scaling") or {}
        problems = [f"{k} {c[k]!r}" for k in ("attention_bias", "use_sliding_window") if c.get(k)]
        if rs.get("rope_type", "default") != "default":
            problems.append(f"rope_type {rs.get('rope_type')!r} (only default RoPE is implemented)")
        if problems:
            raise UnsupportedArchitecture(f"{self.path}: Qwen3 with {', '.join(problems)}")
        super()._setup()
        self.swift_layers = None

    def _load_layer(self, i):
        w = super()._load_layer(i)
        a = f"model.layers.{i}.self_attn."
        w.update(q_norm=self._get(a + "q_norm.weight"), k_norm=self._get(a + "k_norm.weight"))
        return w

    def _load_extras(self):
        self.swift = None

    def _kv(self, i, x, pos):
        w = self.W[i]
        k = ops.rms_norm(self._heads(F.linear(x, w["k"]), self.n_kv), w["k_norm"], self.eps)
        v = self._heads(F.linear(x, w["v"]), self.n_kv)
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, x.dtype)
        return ops.apply_rope(k, cos, sin), v

    def queries(self, i, x, pos):
        w = self.W[i]
        q = ops.rms_norm(self._heads(F.linear(x, w["q"]), self.n_heads), w["q_norm"], self.eps)
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, x.dtype)
        return ops.apply_rope(q, cos, sin)

    def reads(self, i, x):
        """k-normed pre-RoPE K and V per KV head: [tokens, kv_heads, head_dim] each."""
        w = self.W[i]
        x = self.in_norm(x, w["ln1"])
        n = x.shape[0]
        return dict(k=ops.rms_norm(F.linear(x, w["k"]).view(n, self.n_kv, -1), w["k_norm"], self.eps),
                    v=F.linear(x, w["v"]).view(n, self.n_kv, -1))


class Qwen35(Model):
    arch = "qwen3_5"

    def _setup(self):
        c = self.config
        self.prefix = "model.language_model." if "model.language_model.embed_tokens.weight" in self.ckpt else "model."
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.scale = self.head_dim ** -0.5
        rp = c.get("rope_parameters") or {}
        theta = rp.get("rope_theta", c.get("rope_theta"))
        rot = int(self.head_dim * rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
        # text-only positions: the interleaved M-RoPE sections all carry the same position, which
        # is plain RoPE on the first `rot` dims
        self.inv_freq = ops.inv_freq_default(theta, rot)
        self.layer_types = ["gdn" if t == "linear_attention" else "attn" for t in c["layer_types"]]
        self.nk = c["linear_num_key_heads"]
        self.nv = c["linear_num_value_heads"]
        self.dk = c["linear_key_head_dim"]
        self.dv = c["linear_value_head_dim"]
        self.conv_k = c["linear_conv_kernel_dim"]
        self.swift_layers = None

    def _load_layer(self, i):
        p = f"{self.prefix}layers.{i}."
        g = self._get
        w = dict(ln1=g(p + "input_layernorm.weight"), ln2=g(p + "post_attention_layernorm.weight"),
                 gate=g(p + "mlp.gate_proj.weight"), up=g(p + "mlp.up_proj.weight"), down=g(p + "mlp.down_proj.weight"))
        if self.layer_types[i] == "attn":
            a = p + "self_attn."
            w.update(q=g(a + "q_proj.weight"), k=g(a + "k_proj.weight"), v=g(a + "v_proj.weight"),
                     o=g(a + "o_proj.weight"), q_norm=g(a + "q_norm.weight"), k_norm=g(a + "k_norm.weight"))
        else:
            a = p + "linear_attn."
            w.update(qkv=g(a + "in_proj_qkv.weight"), z=g(a + "in_proj_z.weight"), b=g(a + "in_proj_b.weight"),
                     a=g(a + "in_proj_a.weight"), conv=g(a + "conv1d.weight").squeeze(1),
                     A_log=self._get_raw(a + "A_log"), dt_bias=self._get_raw(a + "dt_bias"),
                     norm=g(a + "norm.weight"), out=g(a + "out_proj.weight"))
        return w

    def _load_extras(self):
        pass

    def in_norm(self, x, w):
        return ops.rms_norm_zc(x, w, self.eps)

    post_norm = in_norm

    def final_norm(self, h):
        return ops.rms_norm_zc(h, self.norm_w, self.eps)

    # ---- gated attention
    def _attn_kv(self, i, x, pos):
        w = self.W[i]
        n = x.shape[0]
        k = ops.rms_norm_zc(F.linear(x, w["k"]).view(n, self.n_kv, self.head_dim), w["k_norm"], self.eps)
        v = F.linear(x, w["v"]).view(n, self.n_kv, self.head_dim)
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, x.dtype)
        return ops.apply_rope(k.transpose(0, 1), cos, sin), v.transpose(0, 1)

    def _attn_q(self, i, x, pos):
        """Post-RoPE queries [Hq, n, d] and the output gate [n, Hq * d] of normed inputs x."""
        w = self.W[i]
        n = x.shape[0]
        qg = F.linear(x, w["q"]).view(n, self.n_heads, 2 * self.head_dim)
        q, gate = qg[..., :self.head_dim], qg[..., self.head_dim:].reshape(n, -1)
        q = ops.rms_norm_zc(q, w["q_norm"], self.eps).transpose(0, 1)
        cos, sin = ops.rope_cos_sin(self.inv_freq, pos, x.dtype)
        return ops.apply_rope(q, cos, sin), gate

    def queries(self, i, x, pos):
        return self._attn_q(i, x, pos)[0]

    def _attn_layer(self, i, h, pos, cache):
        """The attention block's output (layer() adds it to h)."""
        w = self.W[i]
        x = self.in_norm(h, w["ln1"])
        q, gate = self._attn_q(i, x, pos)
        k, v = self._attn_kv(i, x, pos)
        self._append_kv(cache[i], k, v)
        o = ops.attend(q, cache[i]["k"], cache[i]["v"], self.scale)
        o = o * torch.sigmoid(gate)
        return F.linear(o, w["o"])

    # ---- Gated DeltaNet
    def _gdn_inputs(self, i, x):
        w = self.W[i]
        return F.linear(x, w["qkv"]), F.linear(x, w["b"]), F.linear(x, w["a"])

    def _gdn_core(self, i, pre, b, a, cache, need_out, checkpoint_every=None, offset=0):
        """Conv + delta rule from pre-conv qkv rows, continuing cache[i]'s states. Rows go in
        GDN_ROWS segments (a multiple of the chunk size, so the chunk grid is unchanged) to bound
        the fp32 working set. Returns (out [Hv, n, dv] in pre.dtype or None, checkpoints)."""
        w = self.W[i]
        c = cache[i]
        n = pre.shape[0]
        kd = self.nk * self.dk
        rep = self.nv // self.nk
        outs, ck = [], {}
        for r0 in range(0, n, GDN_ROWS):
            r1 = min(n, r0 + GDN_ROWS)
            m = r1 - r0
            conv_out, c["conv"] = ops.causal_conv(pre[r0:r1], c.get("conv"), w["conv"])
            q, k, v = conv_out.split([kd, kd, self.nv * self.dv], -1)
            q = q.reshape(m, self.nk, self.dk).float().repeat_interleave(rep, 1).transpose(0, 1)
            k = k.reshape(m, self.nk, self.dk).float().repeat_interleave(rep, 1).transpose(0, 1)
            v = v.reshape(m, self.nv, self.dv).float().transpose(0, 1)
            q = ops.l2norm(q) * self.dk ** -0.5
            k = ops.l2norm(k)
            beta = torch.sigmoid(b[r0:r1]).float().T
            g = (-w["A_log"].float().exp() * F.softplus(a[r0:r1].float() + w["dt_bias"].float())).T
            if m == 1 and not checkpoint_every:
                out, state = ops.delta_rule_recurrent(q, k, v, g, beta, c.get("rec"))
            else:
                out, state, part = ops.delta_rule(q, k, v, g, beta, c.get("rec"), checkpoint_every=checkpoint_every,
                                                  offset=offset + r0, need_out=need_out)
                ck.update(part)
            c["rec"] = state
            if need_out:
                outs.append(out.to(pre.dtype))
        c["len"] = c.get("len", 0) + n
        return (torch.cat(outs, 1) if need_out else None), ck

    def _gdn_layer(self, i, h, pos, cache, checkpoint_every=None):
        """The GDN block's output (layer() adds it to h)."""
        w = self.W[i]
        x = self.in_norm(h, w["ln1"])
        n = x.shape[0]
        pre, b, a = self._gdn_inputs(i, x)
        z = F.linear(x, w["z"])
        offset = cache[i].get("len", 0)
        out, ck = self._gdn_core(i, pre, b, a, cache, True, checkpoint_every, offset)
        if ck:
            cache[i].setdefault("checkpoints", {}).update(ck)
        out = out.transpose(0, 1)                                              # [n, Hv, dv]
        y = ops.rms_norm_gated(out, z.view(n, self.nv, self.dv), w["norm"], self.eps).reshape(n, -1)
        return F.linear(y, w["out"])

    def layer(self, i, h, pos, cache, gdn_checkpoint_every=None, mix=None):
        if self.layer_types[i] == "attn":
            a = self._attn_layer(i, h, pos, cache)
        else:
            a = self._gdn_layer(i, h, pos, cache, gdn_checkpoint_every)
        if mix is not None:
            mix(i, a)
        return self.mlp(i, h + a)

    def fill(self, i, x, pos, cache, st=None, checkpoint_every=None):
        """Cache-only fill from (predicted) layer inputs. GDN: conv + recurrent state over the
        rows, then + st (the terminal-state correction for this layer) if given."""
        x = self.in_norm(x, self.W[i]["ln1"])
        if self.layer_types[i] == "attn":
            k, v = self._attn_kv(i, x, pos)
            self._append_kv(cache[i], k, v)
            return {}
        pre, b, a = self._gdn_inputs(i, x)
        offset = cache[i].get("len", 0)
        _, ck = self._gdn_core(i, pre, b, a, cache, False, checkpoint_every, offset)
        if st is not None:
            cache[i]["rec"] = cache[i]["rec"] + st.to(cache[i]["rec"])
        return ck

    def layer_rows(self, i, h, pos, rows, cache):
        """Fill cache[i] from the inputs h of rows at positions pos (an empty cache, pos = 0..n-1),
        as fill(); return the exact layer output of h[rows] over that cache (= layer(...)[rows]).
        GDN: the delta rule runs with outputs over all rows (rows' outputs need the state at their
        position); the gated norm, out-projection and MLP run on h[rows] only. Row runs and dropped
        all-row temporaries as Llama.layer_rows."""
        w = self.W[i]
        x = self.in_norm(h, w["ln1"])
        xr = x[rows]
        if self.layer_types[i] == "attn":
            k, v = self._attn_kv(i, x, pos)
            self._append_kv(cache[i], k, v)
            del x, k, v
            q, gate = self._attn_q(i, xr, pos[rows])
            del xr
            o = ops.attend_rows(q, cache[i]["k"], cache[i]["v"], self.scale, pos[rows], self.row_block())
            del q
            o = o * torch.sigmoid(gate)
            del gate
            hr = h[rows] + F.linear(o, w["o"])
            del o
        else:
            pre, b, a = self._gdn_inputs(i, x)
            del x
            out, _ = self._gdn_core(i, pre, b, a, cache, True, offset=cache[i].get("len", 0))
            del pre, b, a
            out = out[:, rows].transpose(0, 1)
            n = xr.shape[0]
            z = F.linear(xr, w["z"])
            y = ops.rms_norm_gated(out, z.view(n, self.nv, self.dv), w["norm"], self.eps).reshape(n, -1)
            hr = h[rows] + F.linear(y, w["out"])
        return self.mlp_rows(i, hr)

    def reads(self, i, x):
        """Attention: k-normed pre-RoPE K and V per KV head. GDN: the in-projection outputs that
        feed the state (pre-conv q/k/v per head, the beta and decay pre-activations)."""
        w = self.W[i]
        x = self.in_norm(x, w["ln1"])
        n = x.shape[0]
        if self.layer_types[i] == "attn":
            k = ops.rms_norm_zc(F.linear(x, w["k"]).view(n, self.n_kv, self.head_dim), w["k_norm"], self.eps)
            return dict(k=k, v=F.linear(x, w["v"]).view(n, self.n_kv, self.head_dim))
        pre, b, a = self._gdn_inputs(i, x)
        kd = self.nk * self.dk
        q, k, v = pre.split([kd, kd, self.nv * self.dv], -1)
        return dict(q=q.reshape(n, self.nk, self.dk), k=k.reshape(n, self.nk, self.dk),
                    v=v.reshape(n, self.nv, self.dv), ba=torch.cat((b, a), -1)[:, None, :])


def model_class(path):
    """The class for config.json's model_type: llama / llama_swiftkv -> Llama, qwen3 -> Qwen3, qwen3_5* -> Qwen35;
    anything else raises UnsupportedArchitecture (no silent fallback to Llama's layer)."""
    raw = json.loads((Path(path) / "config.json").read_text())
    mt = str(raw.get("text_config", raw).get("model_type", raw.get("model_type")))
    if mt in ("llama", "llama_swiftkv"):
        return Llama
    if mt == "qwen3":
        return Qwen3
    if mt.startswith("qwen3_5"):
        return Qwen35
    raise UnsupportedArchitecture(f"{path}: model_type {mt!r} has no forward pass here (llama, llama_swiftkv, qwen3, "
                                  f"qwen3_5)")


def load(path, device="cpu", dtype=torch.bfloat16, **kw):
    """The model at path, its files checked after loading against <path>/SHA256SUMS (kva/integrity.py; the hash
    covers the page-cache pages the weights were read from: host RAM flips bits, 2026-10-03). Files corrupted in
    RAM but intact on disk are dropped from the page cache and the model is loaded once more."""
    cls = model_class(path)
    m = cls(path, device=device, dtype=dtype, **kw)
    bad = integrity.verify_model(path)
    if not bad:
        return m
    m.ckpt.files.clear()                 # unmap the files: mapped pages cannot be dropped
    del m
    gc.collect()
    for name in bad:
        integrity.drop_cache(Path(path) / name)
    m = cls(path, device=device, dtype=dtype, **kw)
    bad = integrity.verify_model(path)
    if bad:
        raise integrity.IntegrityError(f"{path}: {', '.join(bad)} still read corrupted after dropping their cached "
                                       f"pages and reloading (another process may keep the bad pages mapped)")
    return m
