"""Independent validation of committed confirmation fixtures and sampled timed outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
from scipy import sparse
from .checker import CatalogReplay, worker_request, opinion_hash, need, ROOT

PROTOCOL=ROOT/"protocol/A1_CONFIRMATION_PROTOCOL.json"
MANIFEST=ROOT/"protocol/A1_CONFIRMATION_INPUTS.json"
CHECK=ROOT/"reports/A1_CONFIRMATION_INPUT_CHECK.json"
SEED=196613
COUNT=16


def checksum(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def raw_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def reference_stream(unit,name,seed,count):
    """Independent of common.opinions and holdout_stream; explicit per-row signed average."""
    n=unit.shape[0]
    generator=np.random.Generator(np.random.PCG64(seed))
    rows=[]
    for q in range(count):
        if name=="slow" and q>0:
            result=rows[-1].copy()
            k=(n+9)//10
            chosen=generator.choice(n,k,replace=False)
            draw=2.*generator.random(k)-1.
            result[chosen]=np.rint(.9*result[chosen]+(.1*65536)*draw).astype(np.int64)
        else:
            u=2.*generator.random(n)-1.
            if name=="signed-neighbor":
                smoothed=np.empty(n,dtype=np.float64)
                for i in range(n):
                    l,r=unit.indptr[i:i+2]
                    # Traversal order is the canonical sorted CSR row. Sign products
                    # are exactly representable. No weights from the service graph.
                    total=0.
                    for t in range(l,r):
                        total+=float(unit.data[t])*float(u[unit.indices[t]])
                    mean=total/(r-l) if r>l else 0.
                    smoothed[i]=.5*(float(u[i])+mean)
                u=smoothed
            elif name not in ("slow","independent"):
                raise ValueError("unregistered confirmation family")
            result=np.rint(65536*u).astype(np.int64)
        rows.append(result)
    return np.stack(rows)


def fixture_rows(entry):
    graph,stream=entry["graph"],entry["stream"]
    path=ROOT/f"data/confirmation_196613/{graph}_{stream}.npz"
    need((ROOT/entry["file"]).resolve()==path.resolve() and checksum(path)==entry["npz_sha256"],"fixture file identity")
    with np.load(path,allow_pickle=False) as z:
        need(z.files==["opinions"],"fixture archive members")
        matrix=z["opinions"]
    need(matrix.dtype==np.dtype("<i8") and matrix.flags.c_contiguous and
         list(matrix.shape)==entry["shape"] and matrix.shape[0]==COUNT,"fixture matrix layout")
    need(raw_hash(matrix)==entry["matrix_sha256"] and all(opinion_hash(x)==h for x,h in zip(matrix,entry["opinion_hashes"],strict=True)),
         "fixture matrix or opinion digest")
    need(np.all((matrix>=-65536)&(matrix<=65536)),"quantized opinion out of bounds")
    return matrix


def input_check():
    start=perf_counter()
    manifest=json.loads(MANIFEST.read_text())
    need(manifest["seed"]==SEED and manifest["count"]==COUNT and manifest["protocol_sha256"]==checksum(PROTOCOL),"manifest identity")
    for name,digest in manifest["source_files_sha256"].items():
        need(checksum(ROOT/f"src/{name}.py")==digest,"input source identity")
    entries=manifest["fixtures"]
    grid={(g,f) for g in ("Alpha","OTC") for f in ("slow","independent","signed-neighbor")}
    need(len(entries)==6 and {(e["graph"],e["stream"]) for e in entries}==grid,"fixture grid")
    rows=[]
    for e in entries:
        g,f=e["graph"],e["stream"]
        need(e["fixture_id"]==f"{g}/{f}/{SEED}/Q16","fixture ID")
        unit_path=ROOT/f"data/projected/{g}_unit.npz"
        weighted_path=ROOT/f"data/projected/{g}_weighted.npz"
        meta=json.loads((ROOT/f"data/projected/{g}_metadata.json").read_text())
        need(checksum(unit_path)==e["unit_graph_sha256"]==meta["unit_sha256"] and
             checksum(weighted_path)==e["graph_sha256"]==meta["weighted_sha256"],"graph provenance")
        unit=sparse.load_npz(unit_path)
        weighted=sparse.load_npz(weighted_path)
        need(unit.format=="csr" and unit.has_canonical_format and (unit!=unit.T).nnz==0 and
             not unit.diagonal().any() and np.all(abs(unit.data)==1) and
             (weighted.sign()!=unit).nnz==0,"paired sign support")
        matrix=fixture_rows(e)
        need(matrix.shape[1]==unit.shape[0],"node universe")
        regenerated=reference_stream(unit,f,SEED,COUNT)
        need(np.array_equal(regenerated,matrix),"independent generated rows differ")
        rows.append(dict(fixture_id=e["fixture_id"],file=e["file"],npz_sha256=e["npz_sha256"],
                         matrix_sha256=e["matrix_sha256"],shape=list(matrix.shape),
                         min_code=int(matrix.min()),max_code=int(matrix.max()),
                         independently_matched_rows=COUNT))
    return dict(verified=True,fixture_count=6,opinion_rows=96,seed=SEED,
                protocol_sha256=checksum(PROTOCOL),input_manifest_sha256=checksum(MANIFEST),
                checker_source_sha256=checksum(__file__),fixtures=rows,
                checker_seconds=perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def output_replay(path,positions):
    start=perf_counter()
    check=json.loads(CHECK.read_text())
    man=json.loads(MANIFEST.read_text())
    need(check["verified"] and check["input_manifest_sha256"]==checksum(MANIFEST) and
         check["protocol_sha256"]==checksum(PROTOCOL),"input preflight binding")
    tr=json.loads(path.read_text())
    g,stream=tr["graph"],tr["stream"]
    entries=[e for e in man["fixtures"] if (e["graph"],e["stream"])==(g,stream)]
    need(len(entries)==1 and tr["seed"]==SEED and tr["requests_planned"]==COUNT and tr["budget"]==15 and tr["delta"]==0,"trace contract")
    e=entries[0]
    need(tr["fixture_id"]==e["fixture_id"] and tr["fixture_npz_sha256"]==e["npz_sha256"] and
         tr["fixture_matrix_sha256"]==e["matrix_sha256"] and tr["input_manifest_sha256"]==checksum(MANIFEST)
         and tr["protocol_sha256"]==checksum(PROTOCOL),"measured input hash mismatch")
    matrix=fixture_rows(e)
    graph_path=ROOT/f"data/projected/{g}_weighted.npz"
    graph_hash=checksum(graph_path)
    need(graph_hash==e["graph_sha256"]==tr["graph_hash"] and tr["weight_mode"]=="weighted" and tr["method"]=="D","measured graph identity")
    need(tr["repeat_id"]==0 and (tr["arm"]=="adaptive" and positions==[8,16] or
         g=="Alpha" and stream=="slow" and tr["arm"]=="retain-always" and positions==[16]),"preselected verification positions")
    rep=CatalogReplay(sparse.load_npz(graph_path))
    reqs=tr["requests"]
    need(len(reqs)<=COUNT and all(req["query_position"]==k and req["opinion_hash"]==e["opinion_hashes"][k-1]
                                 for k,req in enumerate(reqs,1)),"request-level opinion identity")
    outputs=[]
    for k in positions:
        if k>len(reqs):
            outputs.append(dict(position=k,status="not_available",worker_status=tr["status"]))
            continue
        req=reqs[k-1]
        need(req["budget"]==15 and req["delta"]==0 and req["method"]=="D" and req["arm"]==tr["arm"],"request method binding")
        proof=worker_request(rep,req,matrix[k-1],graph_hash,kind="weighted")
        outputs.append(dict(position=k,status="verified",opinion_hash=req["opinion_hash"],action=req["discovery"]["action"],
                            anchor_request=req["discovery"]["anchor_request"],**proof))
    return dict(trace=str(path.relative_to(ROOT) if path.is_absolute() else path),trace_sha256=checksum(path),source_commit=tr["source_commit"],
                fixture_id=e["fixture_id"],arm=tr["arm"],repeat_id=0,outputs=outputs,
                decision_checker_sha256=checksum(ROOT/"src/checker.py"),input_check_sha256=checksum(CHECK),
                checker_seconds=perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=["check-inputs","replay"])
    cli.add_argument("--trace",type=Path)
    cli.add_argument("--position",type=int,action="append")
    cli.add_argument("--output",type=Path)
    args=cli.parse_args()
    if args.action=="check-inputs":
        result=input_check()
        out=CHECK
    else:
        if not args.trace or not args.position or not args.output: cli.error("replay requires trace, position and output")
        result=output_replay(args.trace,args.position)
        out=args.output
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
