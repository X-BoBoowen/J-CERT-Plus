"""Optional two-class global root-key index for the signed promotion control."""
import argparse
import copy
import ctypes as ct
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from long_promotion import DATA, SCREEN, OPINION_DEN, search
from signed_promotion import build_screen, catalog, run
from universe_index import build_incidence
from workloads import NAMES, load_graph, opinions


def compile_index(directory):
    source = Path(__file__).with_suffix(".c")
    path = Path(directory) / ("index.dll" if os.name=="nt" else "index.so")
    cc = shutil.which("gcc") or shutil.which("cc")
    subprocess.run([cc,"-O3","-fPIC","-shared",str(source),"-o",str(path)],check=True,
                   creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    lib = ct.CDLL(str(path))
    ptr=ct.c_void_p
    lib.index_pass.argtypes = [ct.c_int32,ptr,ptr,ptr,ptr,ct.c_int64,ct.c_int64,
                               ct.c_int64,ct.c_int32,ptr,ptr]
    lib.index_pass.restype = None
    lib.gather.argtypes = [ct.c_int32,ptr,ptr,ptr,ptr,ptr,ptr,ct.c_uint8,ptr,ptr]
    lib.gather.restype = ct.c_int32
    lib.dyadic_plan.argtypes = ([ct.c_int32,ptr,ptr]+[ct.c_int64]*6
                                +[ct.c_int32,ptr,ct.c_int64]+[ptr]*4)
    lib.dyadic_plan.restype = ct.c_int32
    lib.gather_at.argtypes = ([ct.c_int32,ptr,ct.c_int64]+[ptr]*5
                              +[ct.c_uint8,ptr,ptr])
    lib.gather_at.restype = ct.c_int32
    return lib


class ClassIndex:
    def __init__(self, lib, pool, x, r, scale, *, fixed_tau=None):
        self.lib, self.scale = lib, scale
        self.root_input = (tuple(pool),x,r)
        if fixed_tau is not None and not 1 <= fixed_tau <= SCREEN:
            raise ValueError("Fixed drift threshold must lie in [1, SCREEN]")
        self.fixed_tau = fixed_tau
        self.n, self.m = len(x), len(pool[0])
        if self.m>=2**31 or not 1<=scale<=16 or SCREEN%scale:
            raise ValueError("Unsupported catalog length or operator scale")
        for arr, kind in zip((*pool,x),(np.int32,np.int32,np.bool_,np.int64)):
            if arr.dtype!=kind or not arr.flags.c_contiguous:
                raise TypeError("Expected contiguous typed catalog and root coordinates")
        self.root_norm = int(np.abs(x).max())+int(r)
        if self.root_norm>2*SCREEN:
            raise OverflowError("Root exceeds the bounded state contract")
        # Each class uses the unweighted root cap. Place it in an outward bin
        # by retaining the entire bin containing a query threshold.
        self.offset = 4*self.root_norm**2
        self.width = 1<<max(0,(2*self.offset).bit_length()-15)
        self.bins = 2*self.offset//self.width+1
        args = [self.m,*(a.ctypes.data for a in (*pool,x)),r,self.offset,self.width,self.bins]
        counts = np.zeros(2*self.bins,dtype=np.int32)
        lib.index_pass(*args,counts.ctypes.data,None)
        self.offsets = np.zeros(len(counts)+1,dtype=np.int32)
        np.cumsum(counts,out=self.offsets[1:])
        assert int(self.offsets[-1])==self.m
        self.order = np.empty(self.m,dtype=np.int32)
        cursors = self.offsets[:-1].copy()
        lib.index_pass(*args,cursors.ctypes.data,self.order.ctypes.data)
        self.inc = build_incidence(*pool[:2],self.n)
        self.inc.left = np.ascontiguousarray(self.inc.left,dtype=np.int64)
        self.inc.right = np.ascontiguousarray(self.inc.right,dtype=np.int64)
        self.degree = np.diff(self.inc.left)+np.diff(self.inc.right)
        self.stamps = None
        self.epoch = 0
        self.nbytes = self.m + sum(a.nbytes for a in (self.offsets,self.order,self.degree,
                          self.inc.left,self.inc.right,self.inc.ids))

    def fork_cursor(self, fixed_tau=None):
        """Share immutable bin/posting arrays and isolate the live query's stamps."""
        if fixed_tau is not None and not 1 <= fixed_tau <= SCREEN:
            raise ValueError("Fixed drift threshold must lie in [1, SCREEN]")
        cursor = copy.copy(self)
        cursor.fixed_tau = fixed_tau
        cursor.stamps = np.zeros(self.m,dtype=np.uint8)
        cursor.epoch = 0
        return cursor

    def query(self,drift,target,excluded):
        if self.stamps is None:
            self.stamps = np.zeros(self.m,dtype=np.uint8)
        max_drift = max(1,int(drift.max()))
        choices = (np.array([self.fixed_tau], dtype=np.int64) if self.fixed_tau is not None
                   else np.array([1<<p for p in range((max_drift-1).bit_length()+1)],dtype=np.int64))
        # True endpoints of all states have |z| <= 1/scale. On nonmoving
        # vertices both squared-difference and -4xy variation are bounded by
        # 8*tau/scale and by 8*||z0||inf*tau + 4*tau^2, in state quanta.
        shift = np.minimum(8*(SCREEN//self.scale)*choices,
                           8*self.root_norm*choices+4*choices**2)
        costs = np.array([self.degree[drift>v].sum() for v in choices],dtype=np.int64)
        ranges = []
        for c,w in enumerate((1,self.scale)):
            # U0*w+w*shift >= target. ceil(target/w) is exact at either sign.
            cut = (int(target)+w-1)//w-shift
            bins = np.clip((cut+self.offset)//self.width,0,self.bins)
            start = self.offsets[c*self.bins+bins]
            end = self.offsets[(c+1)*self.bins]
            ranges.append((start,end))
            costs += end-start
        k = int(np.argmin(costs))
        tau = int(choices[k])
        vertices = np.flatnonzero(drift>tau).astype(np.int32)
        bounds = np.array([v for a,b in ranges for v in (a[k],b)],dtype=np.int32)
        self.epoch += 1
        if self.epoch==256:
            self.stamps.fill(0)
            self.epoch=1
        self.stamps[list(excluded)] = self.epoch
        ids = np.empty(min(self.m,int(costs[k])),dtype=np.int32)
        count = self.lib.gather(len(vertices),vertices.ctypes.data,self.inc.left.ctypes.data,
            self.inc.right.ctypes.data,self.inc.ids.ctypes.data,self.order.ctypes.data,
            bounds.ctypes.data,self.epoch,self.stamps.ctypes.data,ids.ctypes.data)
        ids = ids[:count]
        ids.sort()
        return ids,choices,costs,tau


class FusedClassIndex(ClassIndex):
    """Dyadic cost histogram and posting gather in native loops.

    fork_cursor allocates private work arrays. Returned IDs and trace arrays
    belong to the caller, including across subsequent queries and forks.
    """
    def _new_scratch(self):
        self._choices = np.empty(32,dtype=np.int64)
        self._costs = np.empty(32,dtype=np.int64)
        self._bounds = np.empty(4,dtype=np.int32)
        self._best_index = ct.c_int32()
        self._out = np.empty(0,dtype=np.int32)

    def fork_cursor(self, fixed_tau=None):
        cursor = super().fork_cursor(fixed_tau)
        cursor._new_scratch()
        return cursor

    def query(self,drift,target,excluded):
        if self.stamps is None:
            self.stamps = np.zeros(self.m,dtype=np.uint8)
            self._new_scratch()
        if drift.shape!=(self.n,) or drift.dtype!=np.int64:
            raise ValueError("expected one integer drift per vertex")
        drift = np.ascontiguousarray(drift)
        # This saturation is outside the entire attainable root-cap and
        # drift-shift range, and keeps every signed bin subtraction in int64.
        target = max(-(1<<60),min(1<<60,int(target)))
        size = self.lib.dyadic_plan(self.n,drift.ctypes.data,self.degree.ctypes.data,
            target,SCREEN,self.scale,self.root_norm,self.offset,self.width,self.bins,
            self.offsets.ctypes.data,self.fixed_tau or 0,self._choices.ctypes.data,
            self._costs.ctypes.data,self._bounds.ctypes.data,ct.byref(self._best_index))
        if size<1:
            raise ValueError("drift outside bounded state contract")
        k = self._best_index.value
        tau = int(self._choices[k])
        capacity = min(self.m,int(self._costs[k]))
        if capacity>len(self._out):
            self._out = np.empty(capacity,dtype=np.int32)
        self.epoch += 1
        if self.epoch==256:
            self.stamps.fill(0)
            self.epoch=1
        self.stamps[np.fromiter(excluded,dtype=np.int32,count=len(excluded))] = self.epoch
        count = self.lib.gather_at(self.n,drift.ctypes.data,tau,self.inc.left.ctypes.data,
            self.inc.right.ctypes.data,self.inc.ids.ctypes.data,self.order.ctypes.data,
            self._bounds.ctypes.data,self.epoch,self.stamps.ctypes.data,self._out.ctypes.data)
        return (np.sort(self._out[:count]), self._choices[:size].copy(),
                self._costs[:size].copy(), tau)


def selftest(lib):
    from itertools import combinations
    rng=np.random.default_rng(3451)
    for scale in (1,16):
        n=19
        i,j=np.array(list(combinations(range(n),2)),dtype=np.int32).T.copy()
        f=rng.integers(2,size=len(i)).astype(bool)
        x=rng.integers(-SCREEN//scale+512,SCREEN//scale-512,size=n,dtype=np.int64)
        indexes=[ClassIndex(lib,(i,j,f),x,1,scale,fixed_tau=v) for v in (None,256)]
        private = indexes[0].fork_cursor(fixed_tau=256)
        assert private.order is indexes[0].order and private.stamps is not indexes[0].stamps
        indexes.append(private)
        fused = FusedClassIndex(lib,(i,j,f),x,1,scale)
        fast = [fused.fork_cursor(fixed_tau=v.fixed_tau) for v in indexes]
        for trial in range(28):
            root=x+rng.integers(-1,2,size=n)
            state=root+rng.integers(-256,257,size=n)
            drift=np.abs(state-x)+1
            gain=np.where(f,-4*scale*state[i]*state[j],(state[i]-state[j])**2)
            target=int(np.sort(gain)[rng.integers(len(i))])
            excluded={0,1,3}
            for index,optimized in zip(indexes,fast):
                ids,choices,costs,tau=index.query(drift,target,excluded)
                got=optimized.query(drift,target,excluded)
                assert all(np.array_equal(a,b) for a,b in zip(got,(ids,choices,costs,tau)))
                assert set(np.flatnonzero(gain>=target))-excluded<=set(ids)
                assert not excluded.intersection(ids) and len(np.unique(ids))==len(ids)
                if index.fixed_tau is not None:
                    assert choices.tolist()==[tau]==[index.fixed_tau]


    # Exercise empty and tied bins, dyadic boundaries, stamp rollover and
    # persistent returned arrays on independently forked cursors.
    for fixed_tau in (None,3):
        left=indexes[0].fork_cursor(fixed_tau)
        right=fused.fork_cursor(fixed_tau)
        for t in range(260):
            d=np.resize(np.array([0,1,2,3,4,5,SCREEN,4*SCREEN],np.int64),n)
            if t%5==0:
                d.fill(0)
            target=(0,1,-1,1<<55,-(1<<55))[t%5]
            excluded=set(range(t%(len(i)+1)))
            want=left.query(d,target,excluded)
            got=right.query(d,target,excluded)
            assert all(np.array_equal(a,b) for a,b in zip(got,want))
            if t==0:
                first=got
                saved=[np.copy(a) for a in first]
        assert all(np.array_equal(a,b) for a,b in zip(first,saved))


def trial(name,lib,native):
    W=load_graph(name)
    pool=catalog(W,name)
    s=np.rint(OPINION_DEN*opinions(W,"S1",17)).astype(np.int64)
    paths=[]
    for version in ("legacy","indexed","full"):
        mode="full" if version=="full" else "promoted"
        factory=(lambda *args:ClassIndex(lib,*args)) if version=="indexed" else None
        rows,lo,hi,elapsed=search(W,s,pool,mode,adaptive=mode=="promoted",operator_scale=16,
            dyadic_tau=True,trace=True,broad_pass=native,index_factory=factory)
        first=rows[0]
        ids=[r["winner_id"] for r in rows]
        paths.append(ids)
        print(name,version,"total",round(elapsed,4),"online",
            round(elapsed-first["setup_seconds"]-first["common_seconds"],4),"visits",
            sum(r["visited"] for r in rows),"index_bytes",first["index_bytes"],flush=True)
        if version=="legacy":
            legacy=rows
        else:
            assert ids==paths[0] and all(max(a["low"],b["low"])<=min(a["high"],b["high"])
                for a,b in zip(legacy,rows))


def indexed_file(name,control,suffix):
    if control:
        raise ValueError("Indexed main trials use signed weighted additions")
    return DATA / f"indexed_{name}_{suffix}.csv"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph",nargs="*",choices=NAMES,default=NAMES)
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--reps",type=int,default=10)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp, threadpool_limits(1):
        lib=compile_index(tmp)
        full=build_screen(tmp)
        try:
            selftest(lib)
            for name in args.graph:
                if args.benchmark:
                    run(name,args.reps,False,full,
                        index_factory=lambda *args:ClassIndex(lib,*args),output_file=indexed_file)
                else:
                    trial(name,lib,full)
        finally:
            if os.name=="nt":
                from _ctypes import FreeLibrary
                FreeLibrary(lib._handle)
                FreeLibrary(full.library._handle)


