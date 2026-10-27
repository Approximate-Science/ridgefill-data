"""Validate data, pinned models, binary consistency on resume, and server tokenization."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from common import DATA, HERE, atomic, load, parser, post, rows, server, sha, validate_binary
from pack_data import document
from build_manifest import verify_sources


def identity_baseline(out):
    """Bind all reused measurements without editing their original manifest."""
    out=Path(out)
    baseline=load(out/'RUN-MANIFEST.json')
    assert (out/'checks/PASS.json').exists(), 'Existing CUDA gates required'
    paths=[out/'RUN-MANIFEST.json',out/'checks/PASS.json']
    for item in rows(DATA/'ruler.jsonl'):
        paths.extend(out/'ruler'/a/(item['id']+'.json') for a in ('stock','e3'))
    for item in load(DATA/'kld-index.json'):
        paths.append(out/'kld'/str(item['length'])/(item['id']+'.json'))
    paths.extend(out/'timing'/str(n)/a/'result.json' for n in (16384,32768,65536,120000)
                 for a in ('stock','resident','streamed'))
    return baseline,{str(p.relative_to(out)):sha(p) for p in paths}


def verify_pure_sources(arm,manifest):
    archive=HERE/'sources'/('stock-1242be66f.tar.gz' if arm=='stock' else 'e3.tar.gz')
    actual=verify_sources(HERE/'src/pure'/arm,archive,'pure')
    assert actual==manifest['source_files'],f'{arm} source manifest differs from pinned archive'


def validate_hardware(hardware,gpu_count):
    assert gpu_count in (2,4) and len(hardware)==gpu_count and len({x[1] for x in hardware})==1,'Need requested N identical GPUs'
    assert {int(x[0]) for x in hardware}==set(range(gpu_count)), 'GPU indices must be 0..N-1'
    for index,name,memory,uuid in hardware:
        normalized=name.upper().replace('-','').replace(' ','')
        assert ('A100PCIE40GB' in normalized or 'A100SXM440GB' in normalized) and 39000<=int(memory)<=42000,(name,memory)


def bind_identity(signature,out,identity_projector):
    baseline,files=identity_baseline(out)
    for key in ('model','projector','sources'):
        assert signature[key]==baseline[key],f'Identity baseline {key} differs'
    assert sha(DATA/'BASELINE-SHA256SUMS')==baseline['data_manifest'], 'Original data manifest differs'
    subprocess.run(['sha256sum','-c','BASELINE-SHA256SUMS'],cwd=DATA,stdout=subprocess.DEVNULL,check=True)
    assert {tuple(x[1:3]) for x in signature['hardware']}=={tuple(x[1:3]) for x in baseline['hardware']}, 'GPU model/memory differs from baseline'
    signature.update(identity_projector=sha(identity_projector),baseline_files=files,
                     baseline_binaries=baseline['binaries'],
                     rebuild_policy='Pinned pure source hashes; rebuilt artifacts bound here; mandatory exact REPRODUCTION gate',
                     identity_vendor={str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'vendor').rglob('*'))
                                      if p.is_file() and '__pycache__' not in p.parts},
                     baseline_hardware=baseline['hardware'],selection='same S1 algorithm; projector-dependent rows')


def main():
    p=parser(__doc__)
    p.add_argument('--verify-tokenization',action='store_true')
    p.add_argument('--tokenizer-port',type=int,default=18299)
    args=p.parse_args()
    subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=DATA,stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['sha256sum','-c','models.sha256'],cwd=HERE,check=True)
    lock=load(HERE/'sources/LOCK.json')
    for name,digest in lock['archives'].items():
        assert sha(HERE/'sources'/name)==digest
    instrumented={a:HERE/f'build/instrumented/{a}/bin/llama-server' for a in ('stock','e3')}
    builds={}
    for arm,binary in [('stock',args.stock_bin),('e3',args.e3_bin)]:
        builds['pure/'+arm]=validate_binary(binary,'pure',arm)
        if args.identity_only:
            verify_pure_sources(arm,builds['pure/'+arm])
        else:
            builds['instrumented/'+arm]=validate_binary(instrumented[arm],'instrumented',arm)
    validate_binary(args.scorer,'pure','e3')
    exact=HERE/'build/pure/stock/bin/exact-score'
    validate_binary(exact,'pure','stock')
    binary_paths=[args.stock_bin,args.e3_bin,args.scorer,exact]
    if not args.identity_only:
        binary_paths+=list(instrumented.values())
    for path in list(binary_paths):
        binary_paths+=list(Path(path).parent.glob('*.so*'))
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,uuid','--format=csv,noheader,nounits'],text=True)
    hardware=[list(map(str.strip,line.split(','))) for line in query.splitlines()]
    validate_hardware(hardware,args.gpus)
    hardware.sort(key=lambda x:int(x[0]))
    signature=dict(binaries={str(Path(b).relative_to(HERE)):sha(b) for b in sorted(set(binary_paths))},
                   model=sha(args.model),projector=sha(args.projector),data_manifest=sha(DATA/'SHA256SUMS'),sources=lock,hardware=hardware,gpu_count=args.gpus,
                   harness={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},builds=builds)
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.out/('IDENTITY-MANIFEST.json' if args.identity_only else 'RUN-MANIFEST.json')
    if args.identity_only:
        bind_identity(signature,args.out,args.identity_projector)
    if manifest.exists():
        assert load(manifest)==signature,'Binary/model/data changed: resume requires a new results directory'
    else:
        atomic(manifest,signature)
    if args.verify_tokenization:
        args.port=args.tokenizer_port
        with server(args,'stock',args.out/'tokenizer-preflight'/str(time.time_ns()),ctx=33792):
            for row in rows(DATA/'tokenizer-probes.jsonl'):
                got=post(args.port,'/tokenize',dict(content=row['text'],add_special=False,parse_special=True))['tokens']
                assert got==row['ids'],row['name']
        atomic(args.out/('IDENTITY-TOKENIZER-PASS.json' if args.identity_only else 'TOKENIZER-PASS.json'),
               dict(data_manifest=signature['data_manifest']))
    print('PREFLIGHT PASS')

if __name__=='__main__':
    main()
