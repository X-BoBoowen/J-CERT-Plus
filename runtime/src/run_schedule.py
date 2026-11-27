"""Run a frozen ordered job list, one worker process at a time, preserving all failures."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]


def save(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--batch-seconds',type=float,default=1800)
    p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--rss-mib',type=float,default=2048)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    a=p.parse_args()
    if min(a.batch_seconds,a.timeout,a.rss_mib)<=0:
        raise ValueError('limits must be positive')
    jobs=json.loads(a.schedule.read_text())
    if not isinstance(jobs,list):
        raise ValueError('expected ordered JSON list of worker argument dictionaries')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    if any(a.out_dir.glob('*.json')):
        raise FileExistsError('results exist; preserve this schedule as a distinct batch')
    lock=(ROOT/'logs/.measurement.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    results=[]
    batch_start=time.monotonic()
    for pos,job in enumerate(jobs):
        name=f'{pos:04d}__{job["graph"]}_{job["weight_mode"]}_{job["stream"]}_{job["seed"]}_{job["method"]}'
        path=a.out_dir/(name+'.json')
        entry=dict(position=pos,job=job,result=path.name,status='batch_cap_not_started')
        if time.monotonic()-batch_start<a.batch_seconds:
            cmd=[sys.executable,'-m','src.worker','--cpu',str(a.cpu),'--output',str(path.resolve()),
                 '--schedule-position',str(pos)]
            for k,v in job.items():
                if k in ('schedule_position','cpu','output'):
                    raise ValueError('job shadows scheduler-owned argument '+k)
                if isinstance(v,bool):
                    if v:
                        cmd.append('--'+k.replace('_','-'))
                else:
                    cmd.extend(('--'+k.replace('_','-'),str(v)))
            with (a.out_dir/(name+'.log')).open('x') as log:
                start=time.monotonic()
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                monitor=psutil.Process(proc.pid)
                peak=0
                stop=None
                while proc.poll() is None:
                    try:
                        peak=max(peak,monitor.memory_info().rss)
                    except psutil.NoSuchProcess:
                        break
                    now=time.monotonic()
                    if peak>a.rss_mib*1024**2:
                        stop='rss_cap'
                    elif now-start>=a.timeout:
                        stop='timeout'
                    elif now-batch_start>=a.batch_seconds:
                        stop='batch_cap'
                    if stop:
                        proc.kill()
                        break
                    time.sleep(.025)
                code=proc.wait()
                wall=time.monotonic()-start
            partial=json.loads(path.read_text()) if path.exists() else {}
            entry.update(status=stop or partial.get('status','worker_exit_without_checkpoint'),
                returncode=code,process_wall_seconds=wall,external_sampled_peak_rss_bytes=peak,
                rss_poll_seconds=.025,completed_requests=len(partial.get('requests',[])))
            if stop or code!=0 or not path.exists():
                partial.update(status=entry['status'],harness=entry,
                    failure_reason=stop or partial.get('failure_reason','worker exit before first checkpoint'))
                save(path,partial)
        results.append(entry)
        save(a.out_dir/'SCHEDULE_RESULTS.json',dict(jobs=results,batch_wall_seconds=time.monotonic()-batch_start,
             timeout_seconds=a.timeout,rss_mib=a.rss_mib,cpu=a.cpu,batch_cap_seconds=a.batch_seconds))
        print(pos,entry['status'],name,flush=True)


if __name__=='__main__':
    main()
