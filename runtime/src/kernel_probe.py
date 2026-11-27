"""Bounded P/L/Q parity probe; `prepare` builds the common primitives, `suite` times serially."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from time import perf_counter
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/"baseline_r15/experimental_records/legacy_v6/tools"
BUILD=ROOT/"raw/parity/build"
PROTO=ROOT/"protocol/KERNEL_PROBE.json"
DEB="libsuitesparse-dev_7.10.1+dfsg-1_amd64.deb"
DEB_URL="https://deb.debian.org/debian/pool/main/s/suitesparse/"+DEB
DEB_SHA="877e6c63aafbe9290b89fd6e586a5c10ea75b1a642ee66a561dac6e220f78c83"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial")
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False,default=lambda x:x.item())+"\n")
    tmp.replace(path)


def prepare():
    BUILD.mkdir(parents=True,exist_ok=True)
    args=["cc","-O3","-fno-fast-math","-ffp-contract=off","-fPIC","-shared"]
    cmds=[]
    generic=BUILD/"paired_pcg.so"
    direct=BUILD/"libjcert_cholmod.so"
    def compile_cmd(extra):
        cmd=[*args,*map(str,extra)]
        cmds.append(cmd)
        subprocess.run(cmd,check=True)
    compile_cmd([TOOLS/"paired_pcg.c","-o",generic,"-lm"])
    status={"fused":"ready","paired_pcg_sha256":digest(generic)}
    try:
        # Pin the header package to this host's CHOLMOD runtime. Only unpack locally.
        lib=Path("/usr/lib/x86_64-linux-gnu/libcholmod.so.5.3.1")
        if not lib.is_file(): raise OSError("pinned CHOLMOD 5.3.1 runtime missing")
        deb=BUILD/DEB
        if not deb.exists():
            with urllib.request.urlopen(DEB_URL,timeout=30) as r:
                payload=r.read(2_000_000)
            if hashlib.sha256(payload).hexdigest()!=DEB_SHA:
                raise ValueError("SuiteSparse header checksum")
            deb.write_bytes(payload)
        if digest(deb)!=DEB_SHA:
            raise ValueError("SuiteSparse header checksum")
        headers=BUILD/"headers"
        subprocess.run(["dpkg-deb","-x",str(deb),str(headers)],check=True)
        compile_cmd(["-DJCERT_UPDATE_READY","-I"+str(headers/"usr/include/suitesparse"),
                     TOOLS/"cholmod_bridge.c","-o",direct,lib,"-lm"])
        status.update(cholmod="ready",bridge_sha256=digest(direct),runtime_sha256=digest(lib))
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        status.update(cholmod="blocked",reason=str(exc))
    status.update(commands=cmds,header_source=DEB_URL,header_sha256=DEB_SHA,
                  compiler_version=subprocess.check_output(["cc","--version"],text=True).splitlines()[0])
    save(BUILD/"build.json",status)
    return status


def source_check(graph):
    plans=[]
    data=TOOLS.parent/"data"
    for fname in (f"persistent_batch_{graph}_2500.csv",f"linview_persistent_{graph}_2500.csv"):
        with (data/fname).open() as f:
            rows=[r for r in csv.DictReader(f) if r["b"]=="40" and r["rep"]=="0"
                  and r["backend"]=="cholmod" and r["mode"] in ("planner","linview")]
        if len(rows)!=1: raise ValueError("historical reference row ambiguity")
        plans.append(json.loads(rows[0]["sequence"]))
    if plans[0]!=plans[1]: raise ValueError("published reference plans differ")
    return plans[0]


def fixture_identity(A,s,edits):
    import numpy as np
    n=len(s)
    if (s.shape!=(n,) or A.shape!=(n,n) or not np.isfinite(s).all() or
        np.max(abs(s))>1 or np.any(s*2**30!=np.rint(s*2**30)) or
        not np.isfinite(A.data).all() or (A!=A.T).nnz or not np.all(A.data==np.rint(A.data))):
        raise ValueError("registered operator/opinion range or dyadic grid")
    if len(edits)!=2500 or len({(e.i,e.j) for e in edits})!=2500:
        raise ValueError("catalog size or duplicate")
    for e in edits:
        if not 0<=e.i<e.j<n or e.w!=1 or float(A[e.i,e.j])!=(1. if e.flip else 0.):
            raise ValueError("catalog weight, endpoint or existing edge")
    diag=A.diagonal()
    if not np.all(2*diag-np.asarray(abs(A).sum(axis=1)).ravel()==1):
        raise ValueError("signed Laplacian dominance margin")
    h=hashlib.sha256()
    for arr in (A.indptr,A.indices,A.data,s): h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps([[int(e.i),int(e.j),float(e.w),bool(e.flip)] for e in edits],separators=(",",":")).encode())
    return dict(input_hash=h.hexdigest(),opinion_min=float(s.min()),opinion_max=float(s.max()),opinion_denominator=2**30)


def check_fixture(graph):
    """Independent energy enclosures for all 40 historical tolerance-optimal steps."""
    from fractions import Fraction
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import cg
    sys.path.insert(0,str(TOOLS))
    from planned_matrix import matrix_inputs
    start=perf_counter()
    A,s,edits=matrix_inputs(graph,2500,persistent=True)
    identity=fixture_identity(A,s,edits)
    A=A.astype(np.int64)
    plan=source_check(graph)
    if len(plan)!=40 or len(set(plan))!=40: raise ValueError("registered budget/sequence")
    D=2**40
    rhs=np.rint(s*D).astype(np.int64)
    # Use the exact binary64 tolerance provided to the released methods.
    delta=Fraction(1e-9)*D**2
    allowed=delta.numerator//delta.denominator
    def child(M,e):
        i,j=int(e.i),int(e.j)
        vals=(0,-2,-2,0) if e.flip else (1,-1,-1,1)
        return (M+sparse.coo_matrix((vals,([i,i,j,j],[i,j,i,j])),shape=M.shape)).tocsr()
    def enclosure(M):
        z,status=cg(M,s,M=sparse.diags(1./M.diagonal()),rtol=1e-13,atol=0.)
        if status or not np.isfinite(z).all() or np.max(abs(z))>1.00001:
            raise ArithmeticError("uncertified trial response")
        q=np.rint(D*z).astype(np.int64)
        if int(abs(M).sum(axis=1).max())*int(abs(q).max())+D>=2**62:
            raise OverflowError("integer variational residual range")
        r=rhs-M@q
        lo=sum(int(a)*int(b) for a,b in zip(q,rhs+r))
        rad=sum(int(v)**2 for v in r)
        return q,r,lo,lo+rad
    chosen=set()
    rows=[]
    q,r,lo,hi=enclosure(A)
    for epoch,k in enumerate(plan,1):
        M=child(A,edits[k])
        cq,cr,cl,ch=enclosure(M)
        lower,upper=lo-ch,hi-cl
        if lower<=0: raise ValueError(f"nonpositive or unresolved step {epoch}")
        diag=A.diagonal()
        rivals=0
        for j,e in enumerate(edits):
            if j==k or j in chosen: continue
            u,v=int(e.i),int(e.j)
            a,b=int(q[u]),int(q[v])
            if e.flip: key,va,vb,shift=-4*a*b,2*abs(b),2*abs(a),1
            else: key,va,vb,shift=(a-b)**2,abs(a-b),abs(a-b),2
            pa=max(0,va-abs(int(r[u])))
            pb=max(0,vb-abs(int(r[v])))
            cap=key+(hi-lo)-pa**2//(int(diag[u])+shift)-pb**2//(int(diag[v])+shift)
            if cap<=lower+allowed: continue
            rivals+=1
            _,_,rl,_=enclosure(child(A,e))
            if hi-rl>lower+allowed:
                raise ValueError(f"rival {j} unresolved at step {epoch}")
        rows.append(dict(epoch=epoch,winner=k,low=lower,high=upper,
                         remaining=len(edits)-len(chosen),refined_rivals=rivals))
        chosen.add(k)
        A,q,r,lo,hi=M,cq,cr,cl,ch
    return dict(status="passed",graph=graph,source_sha256=digest(__file__),protocol_sha256=digest(PROTO),
                gain_denominator=D**2,delta_binary64_exact=str(Fraction(1e-9)),epochs=rows,
                offline_seconds=perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**identity)


def worker(graph,cell):
    os.environ["JCERT_CHOLMOD_LIB"]=str(BUILD/"libjcert_cholmod.so")
    sys.path.insert(0,str(TOOLS))
    import numpy as np
    from threadpoolctl import threadpool_info
    from paired_pcg import load_paired_pcg, solve_columns
    from planned_matrix import matrix_inputs,trial as planner_trial
    import linview_delta
    from linview_delta import GlobalDelta,trial as linview_trial
    from unittest.mock import patch
    from full_quad import trial as quad_trial

    backend,method=cell.split("-")
    start=perf_counter()
    A,s,edits=matrix_inputs(graph,2500,persistent=True)
    fixture_s=perf_counter()-start
    expected=source_check(graph)
    identity=fixture_identity(A,s,edits)
    check=json.loads((BUILD/f"check_{graph}.json").read_text())
    if (check.get("status")!="passed" or check["input_hash"]!=identity["input_hash"] or
        check["source_sha256"]!=digest(__file__) or check["protocol_sha256"]!=digest(PROTO)):
        raise ValueError("independent fixture/sequence certificate missing or stale")
    row=dict(graph=graph,cell=cell,fixture_seconds=fixture_s,**identity,
             expected_reference=expected,threadpools=threadpool_info(),n=A.shape[0],M=len(edits),
             delta=1e-9,b=40,source_sha256=digest(__file__),protocol_sha256=digest(PROTO),
             affinity=sorted(os.sched_getaffinity(0)))
    native=None
    if backend in ("fused","cholmod"):
        build=json.loads((BUILD/"build.json").read_text())
        if build.get(backend)!="ready":
            raise ValueError(f"{backend} unavailable: {build}")
        if backend=="fused":
            if digest(BUILD/"paired_pcg.so")!=build["paired_pcg_sha256"]: raise ValueError("fused binary hash")
            native=load_paired_pcg(BUILD/"paired_pcg.so")
        elif digest(BUILD/"libjcert_cholmod.so")!=build["bridge_sha256"]:
            raise ValueError("CHOLMOD bridge hash")
        row["build"]=build

    class FusedL(GlobalDelta):
        """Only the numerical base RHS solver changes. GlobalDelta owns all policy decisions."""
        def solve(self,rhs):
            yy,count=solve_columns(native,self.A,self.invdiag,rhs[:,None],2e-12,0.)
            self.ledger["cg_iterations"]+=int(count.sum())
            x=yy[:,0]
            return x,self.residual_radius(rhs,x)

        def base_action(self,e):
            rank=1+int(self.flip[e])
            rhs=np.zeros((len(self.z),rank))
            rhs[self.i[e]]=1.
            rhs[self.j[e]]=[-1.,1.] if rank==2 else [-1.]
            y,count=solve_columns(native,self.A,self.invdiag,rhs,2e-12,0.)
            self.ledger["base_action_rhs"]+=rank
            self.ledger["cg_iterations"]+=int(count.sum())
            return rhs,y

    start=perf_counter()
    if method=="Q":
        result=quad_trial(A,edits,s,delta=1e-9)[-1]
    elif method=="P":
        result=planner_trial(A,edits,s,"cholmod" if backend=="cholmod" else "cg",
                   "planner",None,0.,fast=True,persistent=True,batch_transport=True,native_pcg=native)[-1]
    elif backend!="fused":
        result=linview_trial(A,edits,s,backend="cholmod" if backend=="cholmod" else "cg")[-1]
    else:
        # Same trial loop, b=20 checkpoint, memory/counter accounting and cleanup.
        with patch.object(linview_delta,"GlobalDelta",FusedL):
            result=linview_trial(A,edits,s,backend="cg")[-1]
    algorithm_s=perf_counter()-start
    seq=json.loads(result.pop("sequence"))
    counts=json.loads(result.pop("counters"))
    row.update(algorithm_seconds=algorithm_s,setup_plus_algorithm_seconds=fixture_s+algorithm_s,
               plan=seq,counters=counts,native_record=result,matched=seq==expected,
               status="completed" if seq==expected else "different_trajectory",peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return row


def suite(out,cpu):
    protocol=json.loads(PROTO.read_text())
    cap=protocol["suite_cap_seconds"]
    start=perf_counter()
    count=0
    runs=[]
    env=os.environ.copy()
    env.update({name:"1" for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    env["PYTHONDONTWRITEBYTECODE"]="1"
    for rep in range(protocol["repetitions"]):
        cells=[(g,c) for g in protocol["graphs"] for c in protocol["methods"]]
        random.Random(23117+rep).shuffle(cells)
        for graph,cell in cells:
            count+=1
            path=out/f"{count:02d}_{graph}_{cell}_r{rep}.json"
            if path.exists(): raise FileExistsError(path)
            before=perf_counter()
            left=cap-(before-start)
            cmd=[sys.executable,"-m","src.kernel_probe","run","--graph",graph,"--cell",cell,"--cpu",str(cpu),"--out",str(path)]
            status="suite_cap" if left<=0 else None
            if status is None:
                import psutil
                with (out/f"{count:02d}_stderr.txt").open("w") as err:
                    child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=err)
                    proc=psutil.Process(child.pid)
                    limit=min(left,protocol["worker_cap_seconds"])
                    while child.poll() is None:
                        try: rss=proc.memory_info().rss
                        except psutil.NoSuchProcess: break
                        if rss>protocol["rss_cap_bytes"] or perf_counter()-before>limit:
                            status="memory_cap" if rss>protocol["rss_cap_bytes"] else "timeout"
                            child.kill()
                            break
                        time.sleep(.05)
                    code=child.wait()
                    if status is None:
                        status=json.loads(path.read_text())["status"] if path.exists() else f"exit_{code}"
            ended=perf_counter()
            if not path.exists():
                save(path,dict(graph=graph,cell=cell,status=status,rep=rep))
            runs.append(dict(rep=rep,graph=graph,cell=cell,file=path.name,status=status,
                             wall_seconds=ended-before))
            save(out/"schedule.json",dict(protocol_sha256=digest(PROTO),elapsed_seconds=ended-start,runs=runs))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["prepare","check","run","suite"])
    p.add_argument("--graph",choices=["Alpha","OTC"])
    p.add_argument("--cell")
    p.add_argument("--cpu",type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument("--out",type=Path,default=ROOT/"raw/parity/measurements")
    a=p.parse_args()
    if a.action=="prepare":
        print(json.dumps(prepare(),indent=2))
    elif a.action=="check":
        os.sched_setaffinity(0,{a.cpu})
        ok=True
        for graph in ([a.graph] if a.graph else ["Alpha","OTC"]):
            try: row=check_fixture(graph)
            except Exception as exc: row=dict(status="failed",reason=repr(exc),graph=graph)
            save(BUILD/f"check_{graph}.json",row)
            ok=ok and row["status"]=="passed"
        if not ok: sys.exit(1)
    elif a.action=="suite":
        a.out.mkdir(parents=True,exist_ok=True)
        suite(a.out,a.cpu)
    else:
        os.sched_setaffinity(0,{a.cpu})
        try: row=worker(a.graph,a.cell)
        except Exception as exc: row=dict(graph=a.graph,cell=a.cell,status="failed",reason=repr(exc))
        save(a.out,row)
        if row["status"]!="completed": sys.exit(1)


if __name__=="__main__":
    main()
