"""Build/verify the identity adapter with the pinned fork's gguf-py (CPU only).

Run with PYTHONPATH=PATH_TO_FORK/gguf-py. Attention requires F16 identity
weights: even exact Q8_0 weights re-quantize h_S on CPU and change its values.
F16 identity is exact on E3's F16 saved h_S, including the native matmul.
"""
import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, dequantize
from common import HERE, atomic, sha


def metadata(reader):
    return {k: (v.types, v.contents()) for k, v in reader.fields.items() if not k.startswith('GGUF.')}


def verify(source, target):
    original, identity = GGUFReader(source), GGUFReader(target)
    assert metadata(original) == metadata(identity), 'Metadata changed'
    fields = metadata(original)
    split = fields['adapter.ridgefill.split'][1]
    n = fields['adapter.ridgefill.n_embd'][1]
    kinds = fields['adapter.ridgefill.model_layer_types'][1].split()
    tensors = {t.name: t for t in identity.tensors}
    expected, native, stored = {'ridgefill.rowsel.score'}, [], []
    for layer in range(split, len(kinds)):
        base = f'blk.{layer}.ridgefill.'
        if kinds[layer] == 'linear_attention':
            native.append(layer)
            expected.add(base + 'st')
            assert np.all(tensors[base + 'st'].data == 0), base
        else:
            stored.append(layer)
            expected.update((base + 'weight', base + 'bias'))
            t = tensors[base + 'weight']
            assert t.tensor_type == GGMLQuantizationType.F16
            # Decode the actual file, not the generator's input arrays.
            decoded = dequantize(t.data, t.tensor_type)
            assert np.array_equal(decoded, np.eye(n, dtype=np.float32)), base
            assert np.all(tensors[base + 'bias'].data == 0), base
    assert set(tensors) == expected
    scores = next(t for t in original.tensors if t.name == 'ridgefill.rowsel.score')
    assert scores.data.tobytes() == tensors[scores.name].data.tobytes()
    return dict(source_sha256=sha(source), identity_sha256=sha(target), bytes=Path(target).stat().st_size,
                metadata_identical=True, row_scores_byte_identical=True, native_identity_layers=native,
                exact_F16_identity_layers=stored, zero_state_corrections=36,
                learned_fill_maps=0, encoding='F16 diagonal=1, other entries=0; F32 bias=0',
                required_hidden_type='f16 (same as E3)',
                rejected_Q8_0='Even with exactly decoded identity weights, native CPU matmul quantizes activations; max error 0.00390625 in operator check.',
                vtail='unchanged dynamic calibration; no learned vtail tensor', alpha=0,
                loader_log_limitation='Only GDN omissions log identity; loader counts the 12 required exact identity attention tensors as maps.')


def build(source, target):
    reader = GGUFReader(source)
    fields = metadata(reader)
    writer = GGUFWriter(target, fields['general.architecture'][1])
    for key, (types, value) in fields.items():
        if key == 'general.architecture':
            continue  # GGUFWriter already emitted this exact STRING value.
        writer.add_key_value(key, value, types[0], types[1] if len(types) > 1 else None)
    kinds = fields['adapter.ridgefill.model_layer_types'][1].split()
    n = fields['adapter.ridgefill.n_embd'][1]
    weights = np.eye(n, dtype=np.float16)
    for t in reader.tensors:
        if t.name == 'ridgefill.rowsel.score':
            writer.add_tensor(t.name, t.data)
            continue
        layer = int(t.name.split('.')[1])
        suffix = t.name.rsplit('.', 1)[1]
        if suffix == 'st':
            # st is an additive recurrent-state correction, so its neutral value is 0.
            writer.add_tensor(t.name, np.zeros_like(t.data))
        elif kinds[layer] == 'linear_attention':
            continue  # native x=h_S, no matrix multiplication
        elif suffix == 'weight':
            assert t.tensor_type == GGMLQuantizationType.Q8_0
            writer.add_tensor(t.name, weights)
        elif suffix == 'bias':
            writer.add_tensor(t.name, np.zeros_like(t.data))
        else:
            raise ValueError(t.name)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE/'models/ridgefill-projector-qwen3.8-27b-i8-S16.gguf')
    p.add_argument('--output', type=Path, default=HERE/'models/identity-projector-qwen3.8-27b-i8-S16.gguf')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if not args.verify_only:
        build(args.source, args.output)
    result = verify(args.source, args.output)
    atomic(HERE/'validation/identity-projector.json', result)
    print(result)


if __name__ == '__main__':
    main()
