"""Repeat the pinned KL 9K document check on CPU, with off/E3/identity."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from common import HERE, atomic, load, rows, sha
from kld_run import SCORE_E3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    args = p.parse_args()
    folder = HERE/'validation/identity-cpu'
    folder.mkdir(parents=True, exist_ok=True)
    source = args.fork/'out/kl-score/document.json'
    document = folder/'document.json'
    document.write_bytes(source.read_bytes())
    assert len(load(document)['ids']) == 9000
    binary = args.fork/'build-cpu/bin/llama-ridgefill-score'
    e = dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH=str(binary.parent))
    for key in list(e):
        if key.startswith(('LLAMA_RIDGEFILL', 'RIDGEFILL_AUDIT')):
            del e[key]
    reference = folder/'reference.logprobs'
    results = {}
    commands = {}
    for arm in ('off', 'e3', 'identity'):
        command = [str(binary), '--model', str(args.model), '--input', str(document), '--scored', '64',
                   '--threads', '6', '--gpu-layers', '0', '--flash-attn', 'off', '--batch', '2048', '--ubatch', '512',
                   '--cache-type', 'f16', '--h-type', 'f16', '--output', str(folder/(arm+'.jsonl'))]
        if arm == 'off':
            command += ['--mode', 'off', '--save-logprobs', str(reference)]
        else:
            projector = HERE/'models'/('identity-projector-qwen3.8-27b-i8-S16.gguf' if arm == 'identity' else
                                       'ridgefill-projector-qwen3.8-27b-i8-S16.gguf')
            command += ['--projector', str(projector), *SCORE_E3, '--ridgefill-stream-projector', 'off',
                        '--kl-against', str(reference)]
        commands[arm] = command
        atomic(folder/'commands.json', commands)
        started = time.monotonic()
        with (folder/(arm+'.log')).open('w') as log:
            subprocess.run(command, env=e, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[arm] = rows(folder/(arm+'.jsonl'))[0]
        print(arm, 'NLL', results[arm]['nll'], 'wall_s', round(time.monotonic()-started, 2), flush=True)
    reference.unlink()
    a, b = results['e3'], results['identity']
    ra, rb = set(a['exact_rows']), set(b['exact_rows'])
    logs = (folder/'identity.log').read_text()
    native = list(map(int, re.findall(r'no map for GDN layer (\d+)', logs)))
    assert native == load(HERE/'validation/identity-projector.json')['native_identity_layers']
    assert '12 maps of type' in logs
    assert b['nll'] != a['nll'] and b['nll'] != results['off']['nll']
    assert a['target_positions'] == b['target_positions'] == list(range(8936, 9000))
    assert a['selector_excludes_targets'] and b['selector_excludes_targets']
    report = dict(document_sha256=sha(document), binary_sha256=sha(binary),
                  libraries={p.name: sha(p) for p in sorted(binary.parent.glob('*.so*'))},
                  nll={k: v['nll'] for k,v in results.items()},
                  mean_kl={k: results[k]['kl_mean'] for k in ('e3','identity')},
                  exact_rows={k: len(results[k]['exact_rows']) for k in ('e3','identity')},
                  exact_prefix={k: results[k]['ridgefill']['n_exact_prefix'] for k in ('e3','identity')},
                  same_selected_rows=ra == rb, intersection=len(ra & rb), e3_only=len(ra-rb), identity_only=len(rb-ra),
                  native_identity_layers=native, required_identity_attention_maps=12,
                  selection_depends_on_map=True,
                  selection_reason='S1 fills prefix attention KV with the current projector, then ranks probe attention mass over that KV.',
                  validation='NLL and native identity pass; equal selection is not achievable with unchanged S1 engine/protocol.')
    atomic(folder/'result.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
