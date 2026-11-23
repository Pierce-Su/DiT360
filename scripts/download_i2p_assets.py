"""Download only pinned inference components into configured personal HF cache.
User/PM has verified legal gate access; this never reads/displays credential values.
Records SHA256 for every cached file, compares Hub LFS digest when available.
"""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import time
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path('/mnt/NAS/pierce/.workstation/S3063/dit360-t_db92abb3')
ROOT.mkdir(exist_ok=True)
ASSETS = [
 ('black-forest-labs/FLUX.1-dev', '45d95c497444bbaebe766e1528792fc9158c0c3a',
  ['model_index.json','scheduler/*','text_encoder/*','text_encoder_2/*',
   'tokenizer/*','tokenizer_2/*','transformer/*','vae/*']),
 ('Insta360-Research/DiT360-Panorama-Image-Generation',
  '88b5915693607d3377e1b5385d7abbe6c671d6ce', ['adapter_config.json','adapter_model.safetensors'])]
start = time.monotonic()
rows = []
try:
    for repo, revision, patterns in ASSETS:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise RuntimeError('model revision mismatch')
        files = [f for f in info.siblings if any(fnmatch.fnmatchcase(f.rfilename,p) for p in patterns)
                 and not f.rfilename.endswith(('.bin','.msgpack','.h5','.onnx'))]
        print(json.dumps(dict(repo=repo,revision=revision,files=len(files),bytes=sum(f.size or 0 for f in files))),flush=True)
        for f in files:
            path = hf_hub_download(repo,f.rfilename,revision=revision)
            with open(path,'rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
            expected = f.lfs.sha256 if f.lfs else None
            if expected and digest != expected:
                raise RuntimeError('Hub LFS SHA256 mismatch: '+f.rfilename)
            row = dict(repo=repo,revision=revision,file=f.rfilename,path=path,
                       bytes=Path(path).stat().st_size,sha256=digest,hub_lfs_sha256=expected)
            rows.append(row)
            with (ROOT/'model-files-progress.json').open('w') as stream:
                json.dump(rows,stream,indent=2)
            print('VERIFIED '+f.rfilename+' '+digest,flush=True)
    with (ROOT/'model-files.json').open('x') as stream:
        json.dump(dict(status='VERIFIED',seconds=time.monotonic()-start,files=rows),stream,indent=2)
    (ROOT/'DOWNLOAD_DONE').touch(exist_ok=False)
except BaseException as e:
    (ROOT/'DOWNLOAD_FAILED').write_text(type(e).__name__+'\n')
    raise
