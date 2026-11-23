"""Exactly one 0106 run. Refuses foreign GPU jobs unless explicitly coordinated.

This flag is NOT a reservation. Never kill, stop or signal any foreign process.
Bulk logs/telemetry live outside Vault. Run output is never reused.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path('/mnt/NAS/E3DQA_project/tools/DiT360')
WORK=Path('/mnt/NAS/pierce/.workstation/S3063/dit360-t_db92abb3')
RUN=Path('/mnt/NAS/E3DQA_project/assets/panoramas/dit360/first-sample-t_db92abb3')

def query(fields,kind='gpu'):
    result=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True)
    return [line.strip().split(', ') for line in result.splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinated-with-foreign-jobs',action='store_true')
    args=parser.parse_args()
    if RUN.exists(): raise SystemExit('Refusing existing output directory: '+str(RUN))
    if not (WORK/'DOWNLOAD_DONE').exists(): raise SystemExit('Pinned verified asset download not finished')
    jobs=query('pid,used_memory','compute-apps')
    foreign=[]
    for pid,mem in jobs:
        if Path('/proc/'+pid).stat().st_uid != os.getuid(): foreign.append((pid,mem))
    if foreign and not args.coordinated_with_foreign_jobs:
        raise SystemExit('WAITING_FOR_GPU_COORDINATION foreign jobs='+json.dumps(foreign))
    # Explicit minimum headroom; does not certify other jobs future needs.
    free=int(query('memory.free')[0][0])
    if free < 70000: raise SystemExit('Insufficient prelaunch free VRAM (<70000MiB)')
    launch=WORK/('launch-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    launch.mkdir(exist_ok=False)
    cmd=[sys.executable,str(REPO/'i2p.py'),'--image',
        '/mnt/NAS/E3DQA_project/curated_set/photorealistic/0106.jpg','--caption',
        'In the Cozy Living Room, a plush sofa sits invitingly beside a well-arranged coffee table adorned with flowers and books.',
        '--expected-sha256','a1b93c357a49bf25a42aaf592ebb2d327136bde845b38d2e6a8022ba98212c89',
        '--hfov-deg','80','--seed','42','--output-dir',str(RUN),'--generate','--gpu-approved',
        '--model-files-manifest',str(WORK/'model-files.json')]
    (launch/'launch.json').write_text(json.dumps(dict(command=cmd,prelaunch_jobs=jobs,
        foreign_coordination=args.coordinated_with_foreign_jobs,free_vram_mib=free),indent=2))
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    start=time.monotonic(); samples=[]
    with (launch/'inference.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env)
        with (launch/'vram.jsonl').open('x') as telemetry:
            while proc.poll() is None:
                try:
                    apps=query('pid,used_memory','compute-apps')
                    sample=dict(seconds=time.monotonic()-start,total_used_mib=int(query('memory.used')[0][0]),
                        own_used_mib=next((int(m) for pid,m in apps if int(pid)==proc.pid),0))
                    samples.append(sample)
                    telemetry.write(json.dumps(sample)+'\n'); telemetry.flush()
                except (subprocess.SubprocessError,ValueError): pass
                time.sleep(1)
        code=proc.wait()
    summary=dict(exit_code=code,wall_seconds=time.monotonic()-start,pid=proc.pid,
        sample_interval_seconds=1,sampled_own_peak_mib=max((s['own_used_mib'] for s in samples),default=None),
        sampled_whole_device_peak_mib=max((s['total_used_mib'] for s in samples),default=None),
        note='Sampled nvidia-smi, may miss transient peaks. Not PyTorch allocator.',run=str(RUN),launch=str(launch))
    (launch/'execution.json').write_text(json.dumps(summary,indent=2))
    (launch/('DONE' if code==0 else 'FAILED')).touch(exist_ok=False)
    if RUN.exists():
        for name in ['inference.log','execution.json','launch.json']:
            with (RUN/name).open('xb') as dest, (launch/name).open('rb') as src:
                shutil.copyfileobj(src,dest)
    print(json.dumps(summary),flush=True)
    raise SystemExit(code)


if __name__=='__main__': main()
