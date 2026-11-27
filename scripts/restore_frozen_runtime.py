#!/usr/bin/env python3
"""Restore the exact Git identity plus the necessary V2 native/source/input overlay.
No benchmarks or numerical replay are started. Requires Git and Python 3.10+.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse, json, subprocess, sys, tempfile
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'artifacts/frozen'
COMMITS={'V2':'4c2c7b7e84c7c03ee56d45453fea6950c94817bd','V3':'c6b752630bd8dbe1f9fd296a7ed02f249f834575','V4':'9d09794b6a6de8a4c97f615abf3f3cae1f458b75'}
def run(*args):subprocess.run(list(map(str,args)),check=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--version',choices=COMMITS,default='V4');a=p.parse_args();dest=a.destination.resolve()
    if dest.exists() or dest==ROOT or ROOT in dest.parents:raise ValueError('Use a new directory outside this repository')
    inv=json.loads((ROOT/'artifacts/ARCHIVE_INVENTORY.json').read_text())['archives']
    for row in inv:
        q=ROOT/row['path']
        if sha256(q.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('archive changed: '+row['path'])
    with tempfile.TemporaryDirectory(prefix='jcert-restore-') as tmp:
        temp=Path(tmp)
        with ZipFile(A/'V2_Code.zip') as z:
            for n in ('reproduction/RESEARCH_HISTORY_V2.bundle','reproduction/overlay_packages.py'):
                (temp/Path(n).name).write_bytes(z.read(n))
        run('git','clone',temp/'RESEARCH_HISTORY_V2.bundle',dest);run('git','-C',dest,'checkout','--detach',COMMITS['V2'])
        run(sys.executable,temp/'overlay_packages.py',dest,A/'V2_Code.zip',A/'V2_Inputs.zip')
        for generation,branch in [('V3','research/v3-value-diagnostics-20261126'),('V4','research/v4-measurement-20261126')]:
            if a.version=='V2' or (generation=='V4' and a.version=='V3'):break
            name=f'{generation}_Runtime_Delta.bundle';path=temp/name
            with ZipFile(A/f'{generation}_Code.zip') as z:path.write_bytes(z.read('runtime_delta/'+name))
            run('git','-C',dest,'fetch',path,branch);run('git','-C',dest,'checkout','--detach',COMMITS[generation])
    actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    if actual!=COMMITS[a.version]:raise ValueError('commit mismatch')
    if a.version=='V4':
        pins=json.loads((ROOT/'runtime/FROZEN_SOURCE_SHA256.json').read_text())['source_files_sha256']
        for name,h in pins.items():
            if sha256((dest/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
    print('Restored',a.version,actual,'at',dest,'; no experiments executed.')
if __name__=='__main__':main()
