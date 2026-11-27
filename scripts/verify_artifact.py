#!/usr/bin/env python3
"""Verify original archives, current runtime pins, and reproduce selected record analyses."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from hashlib import sha256
import argparse, json, stat, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def need(cond,msg):
    if not cond:raise ValueError(msg)
def safe(name):
    p=PurePosixPath(name)
    need(bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in p.parts[0],name)
    return p
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args();out=a.output_dir.resolve()
    need(not out.exists(),'choose a new output directory');need(ROOT not in out.parents and out!=ROOT,'keep outputs outside the repository');out.mkdir(parents=True)
    inventory=json.loads((ROOT/'artifacts/ARCHIVE_INVENTORY.json').read_text())['archives'];counts=[]
    for row in inventory:
        path=ROOT/safe(row['path']);need(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],row['path']);
        with ZipFile(path) as z:
            names=z.namelist();need(len(names)==row['members']==len(set(names)),row['path']+'/roster')
            for i in z.infolist():safe(i.filename);need(not stat.S_ISLNK(i.external_attr>>16),'symlink: '+i.filename)
            need(z.testzip() is None,row['path']+'/CRC')
            manifest_names=[n for n in names if n.startswith('MANIFEST_') and n.endswith('.json') and '/' not in n]
            need(len(manifest_names)==1,row['path']+'/manifest')
            manifest=json.loads(z.read(manifest_names[0]));expected=manifest['members'];need(set(names)=={e['path'] for e in expected}|{manifest_names[0]},row['path']+'/manifest members')
            for e in expected:
                raw=z.read(e['path']);need(len(raw)==e['bytes'] and sha256(raw).hexdigest()==e['sha256'],e['path'])
            counts.append(dict(path=row['path'],members=len(names),payload_files=len(expected)))
    pins=json.loads((ROOT/'runtime/FROZEN_SOURCE_SHA256.json').read_text())
    for name,h in pins['source_files_sha256'].items():need(digest(ROOT/'runtime'/safe(name))==h,'runtime/'+name)
    inputs=json.loads((ROOT/'runtime/INPUT_MANIFEST.json').read_text())['members']
    for row in inputs:
        f=ROOT/'runtime'/safe(row['path']);need(f.stat().st_size==row['bytes'] and digest(f)==row['sha256'],'input/'+row['path'])
    legacy=[]
    for folder in ('legacy/v6','legacy/bounded_scale'):
        rows=json.loads((ROOT/folder/'SOURCE_MANIFEST.json').read_text())['members']
        for row in rows:
            f=ROOT/folder/safe(row['path']);need(f.stat().st_size==row['bytes'] and digest(f)==row['sha256'],str(f))
        legacy.append(dict(folder=folder,members=len(rows)))
    summary=dict(status='passed',archives=counts,scientific_commit=pins['commit'],source_pins=len(pins['source_files_sha256']),input_members=len(inputs),legacy_members=legacy,archive_only_analyses=not a.validate_only,new_production_benchmarks=0,new_numerical_replays=0)
    if not a.validate_only:
        commands=[['verification/audit_g_mechanism.py','--workspace',str(ROOT),'--output',str(out/'g_mechanism.json')],['verification/audit_direct_g_provenance.py','--workspace',str(ROOT),'--output',str(out/'g_direct_provenance.json')],['scripts/recompute_headlines.py','--output-dir',str(out/'timing')]]
        for cmd in commands:
            r=subprocess.run([sys.executable,'-I','-B',str(ROOT/cmd[0]),*cmd[1:]],cwd=ROOT,capture_output=True,text=True,timeout=180)
            (out/(Path(cmd[0]).stem+'.log')).write_text(r.stdout+'\n'+r.stderr);need(r.returncode==0,cmd[0]+' failed; inspect log')
        mechanism=json.loads((out/'g_mechanism.json').read_text())['mechanism'];expected=json.loads((ROOT/'verification/expected_g_mechanism.json').read_text());need(mechanism==expected,'all G mechanism rows must match archive reference')
        direct=json.loads((out/'g_direct_provenance.json').read_text());need(direct['operations']['distinct_states']==30 and direct['operations']['archived_interval_comparisons']==90,'direct provenance scope')
        summary.update(g_mechanism_rows=mechanism['all_unique']['recorded_decision_states'],complete_g_states=mechanism['complete_q4_unique']['recorded_decision_states'],G_associated_states=30,G_interval_observations=90)
    (out/'verification.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
