"""Independent full-precision reference: score the same KL windows with the official BF16 Qwen3.8-27B in Hugging Face
transformers. Input = the exact token-id windows used on the A100 box (the last 512 ids are the targets). Output per
doc: target log-probabilities, top-1 ids and top-20 (ids, log-probs) for the 512 scored positions, plus the mean NLL."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
import transformers


def load_model(model_dir):
    kwargs = dict(dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa')
    for name in ('AutoModelForCausalLM', 'AutoModelForImageTextToText'):
        try:
            return getattr(transformers, name).from_pretrained(model_dir, **kwargs), name
        except (ValueError, KeyError) as error:
            last = error
    raise last


@torch.no_grad()
def score(model, ids, scored):
    inputs = torch.tensor([ids], device='cuda')
    out = model(input_ids=inputs, logits_to_keep=scored + 1, use_cache=False)
    logits = out.logits[0, :-1].float()          # positions n-scored-1 .. n-2 predict the scored targets
    assert logits.shape[0] == scored, logits.shape
    logp = torch.log_softmax(logits, dim=-1)
    targets = inputs[0, -scored:]
    target_lp = logp.gather(1, targets[:, None])[:, 0]
    top = logp.topk(20, dim=-1)
    return dict(target_logprobs=target_lp.tolist(), top1=top.indices[:, 0].tolist(),
                top20_ids=top.indices.tolist(), top20_logprobs=top.values.tolist(),
                nll=float(-target_lp.mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--docs', required=True, type=Path, help='directory of {id}.json windows with an "ids" list')
    p.add_argument('--selection', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--scored', type=int, default=512)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, loader = load_model(args.model_dir)
    model.eval()
    env = dict(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), python=platform.python_version(), loader=loader,
               model_dir=args.model_dir, dtype='bfloat16', attn='sdpa')
    (args.out / 'ENV.json').write_text(json.dumps(env, indent=1))
    selection = json.loads(args.selection.read_text())
    for group, ids in selection.items():
        for doc_id in ids:
            target = args.out / f'{doc_id}.json'
            if target.exists():
                continue
            window = json.loads((args.docs / f'{doc_id}.json').read_text())['ids']
            start = time.monotonic()
            result = score(model, window, args.scored)
            result.update(id=doc_id, group=group, n_tokens=len(window), wall_s=time.monotonic() - start)
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(result))
            os.replace(tmp, target)
            print(f"{group} {doc_id} nll={result['nll']:.4f} {result['wall_s']:.0f}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
