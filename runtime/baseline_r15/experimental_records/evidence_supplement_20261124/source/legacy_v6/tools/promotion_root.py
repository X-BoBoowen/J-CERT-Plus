"""Bind one privately prepared promotion root to its exact query inputs."""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


class RankedSeeds(set):
    """Keep the source of the initial ranking while permitting legacy hot-set updates."""
    def __init__(self,ids,pool,x,r,diag,hot_size,scale):
        super().__init__(ids)
        self.root_input = tuple(pool),x,r,diag,hot_size,scale
        self.initial = frozenset(self)


def _signature(a):
    return a.shape,a.strides,a.dtype


def _freeze(a):
    # CSR components may be NumPy views of an owning allocation.
    owner = a
    while isinstance(owner,np.ndarray):
        owner.flags.writeable = False
        owner = owner.base


@dataclass(frozen=True,eq=False)
class BoundRoot(Mapping):
    fields: Mapping
    graph: object
    opinion: object
    pool: tuple
    policy: tuple
    arrays: tuple
    members: tuple
    metadata: tuple
    fixed_tau: object

    def __getitem__(self,key):
        return self.fields[key]

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def validate(self,W,s,pool,mode,*,hot_size,operator_scale,proof_unit,
                 adaptive,dyadic_tau,index_factory):
        if (W is not self.graph or s is not self.opinion or len(pool)!=3
                or any(x is not y for x,y in zip(pool,self.pool))):
            raise ValueError("prepared root belongs to different graph, opinion or ordered catalog")
        policy = (mode,hot_size,operator_scale,proof_unit,adaptive,
                  dyadic_tau if adaptive else None,index_factory)
        if policy!=self.policy:
            raise ValueError("prepared root policy, index factory, scale or precision changed")
        index = self.fields["index"]
        if index is not None and index.fixed_tau is not None:
            raise ValueError("prepared root base index threshold policy changed")
        for owner,name,expected in self.members:
            if getattr(owner,name) is not expected:
                raise ValueError(f"prepared root member replaced: {name}")
        for owner,name,expected in self.metadata:
            if getattr(owner,name)!=expected:
                raise ValueError(f"prepared root metadata changed: {name}")
        for a,expected in self.arrays:
            if a.flags.writeable or _signature(a)!=expected:
                raise ValueError("prepared array became writable or its layout changed")


def bind_root(W,s,pool,root,method,*,operator_scale,proof_unit,hot_size,
              index_factory=None,fixed_tau=None):
    """Seal the root after setup, before exposing it to independent cursors.

    The builder transfers ownership of its arrays; mutable aliases must not be
    exported during construction. Epoch and visitation stamps are index scratch.
    """
    from long_promotion import Iterative, OPINION_DEN, SCREEN, UpdatedFactor, certified_state
    from workloads import system
    if method not in ("static","full","adaptive","fixed") or hot_size<0:
        raise ValueError("invalid root method or hot seed count")
    indexed = method in ("adaptive","fixed")
    index = root["index"]
    if (indexed!=(index is not None) or indexed!=(index_factory is not None)
            or (method=="fixed")!=(fixed_tau is not None)):
        raise ValueError("root method and index disagree")
    if (len(pool)!=3 or any(a.ndim!=1 or len(a)!=len(pool[0]) or a.dtype!=kind
            or not a.flags.c_contiguous for a,kind in zip(pool,(np.int32,np.int32,np.bool_)))):
        raise ValueError("invalid ordered catalog layout")
    A,fac = root["A"],root["fac"]
    if getattr(fac,"root_operator",None) is not A:
        raise ValueError("root solver was prepared for another operator")
    n = W.shape[0]
    if (W.shape!=(n,n) or W.format!="csr" or A.shape!=W.shape or A.format!="csr"
            or A.dtype!=np.int64 or s.shape!=(n,) or s.dtype!=np.int64
            or np.any(s < -OPINION_DEN) or np.any(s > OPINION_DEN)):
        raise ValueError("invalid graph, operator or bounded opinion")
    if (not isinstance(proof_unit,int) or proof_unit<SCREEN or proof_unit&(proof_unit-1)
            or operator_scale<1 or (A!=operator_scale*system(W).astype("int64")).nnz):
        raise ValueError("root proof grid or graph operator mismatch")
    fields = dict(root)
    fields["hot"] = frozenset(root["hot"])
    fields["state"] = tuple(root["state"])
    fields["opinion_float"] = s/OPINION_DEN
    z,x,r,xc,rc = fields["state"]
    if (len(fields["hot"])!=min(hot_size,len(pool[0]))
            or any(not 0<=e<len(pool[0]) for e in fields["hot"])
            or any(a.shape!=(n,) for a in (z,x,xc)) or r<0 or rc<0):
        raise ValueError("invalid root seed or state")
    if (z.dtype!=np.float64 or not np.isfinite(z).all() or np.abs(z).max()>1.5
            or x.dtype!=np.int64 or xc.dtype!=np.int64):
        raise ValueError("invalid certified state layout or range")
    _,x_check,r_check,xc_check,rc_check = certified_state(A,fac,s,proof_unit,z)
    if (not np.array_equal(x,x_check) or not np.array_equal(xc,xc_check)
            or (r,rc)!=(r_check,rc_check)):
        raise ValueError("root state certificate does not match opinion or proof grid")
    seeds = root["hot"]
    if not isinstance(seeds,RankedSeeds) or seeds!=seeds.initial:
        raise ValueError("root needs unedited seeds from the certified ranking")
    seed_pool,seed_x,seed_r,seed_diag,seed_count,seed_scale = seeds.root_input
    if (any(l is not v for l,v in zip(pool,seed_pool)) or seed_x is not xc
            or (seed_r,seed_count,seed_scale)!=(rc,hot_size,operator_scale)
            or not np.array_equal(seed_diag,A.diagonal())):
        raise ValueError("root seed ranking belongs to different inputs")
    arrays = [s,*pool,z,x,xc,fields["opinion_float"]]
    members,meta = [],[]
    def pin(owner,name):
        value = getattr(owner,name)
        members.append((owner,name,value))
        if isinstance(value,np.ndarray):
            arrays.append(value)
    def pin_csr(matrix):
        for name in ("data","indices","indptr"):
            pin(matrix,name)
        meta.append((matrix,"shape",matrix.shape))
    pin_csr(W)
    pin_csr(A)
    pin(fac,"root_operator")
    if isinstance(fac,Iterative):
        for name in ("A","inverse_diag","M"):
            pin(fac,name)
        pin_csr(fac.A)
    elif isinstance(fac,UpdatedFactor) and not fac.steps:
        pin(fac,"base")
    else:
        raise ValueError("expected an unedited root factor")
    if index is not None:
        if (not isinstance(index,index_factory) or (index.n,index.m,index.scale)
                !=(n,len(pool[0]),operator_scale)):
            raise ValueError("root index and graph/catalog dimensions or scale disagree")
        origin,root_x,root_r = index.root_input
        if (any(l is not v for l,v in zip(pool,origin)) or root_x is not xc or root_r!=rc):
            raise ValueError("root index was constructed on another catalog or state")
        for name in ("lib","inc","order","offsets","degree"):
            pin(index,name)
        for name in ("left","right","ids"):
            pin(index.inc,name)
        for name in ("n","m","scale","bins","width","offset","root_norm"):
            meta.append((index,name,getattr(index,name)))
    for a in arrays:
        _freeze(a)
    mode = "promoted" if indexed else method
    policy = (mode,hot_size,operator_scale,proof_unit,indexed,True if indexed else None,index_factory)
    return BoundRoot(MappingProxyType(fields),W,s,tuple(pool),policy,
                     tuple((a,_signature(a)) for a in arrays),tuple(members),tuple(meta),fixed_tau)


def selftest():
    from long_promotion import UNIT, UpdatedFactor, certified_state, search, seed_view
    from universe_index import build_universe
    from workloads import load_graph,system
    W = load_graph("Alpha",n=16)
    pool = build_universe(W)
    s = np.arange(W.shape[0],dtype=np.int64)
    A = 16*system(W).astype("int64")
    fac = UpdatedFactor(A)
    state = certified_state(A,fac,s)
    hot,_ = seed_view(pool,state[3],state[4],A.diagonal(),4,False,16)
    payload = dict(A=A,fac=fac,state=state,hot=hot,index=None,index_bytes=0,
                   common_seconds=0.,rank_seconds=0.,index_seconds=0.,setup_seconds=0.)
    def reject(fn):
        try:
            fn()
        except (ValueError,TypeError):
            return
        raise AssertionError("accepted incompatible root or writable input")
    bind_options = dict(operator_scale=16,proof_unit=UNIT,hot_size=4)
    reject(lambda: bind_root(W,s+1,pool,payload,"full",**bind_options))
    reject(lambda: bind_root(W,s,pool,payload,"full",**(bind_options|{"proof_unit":UNIT*2})))
    reject(lambda: bind_root(W,s,pool,payload,"full",**(bind_options|{"operator_scale":4})))
    reject(lambda: bind_root(W,s,(pool[1],pool[0],pool[2]),payload,"full",**bind_options))
    wrong_A = A.copy()
    wrong_A.data[1] += 1
    reject(lambda: bind_root(W,s,pool,payload|{"A":wrong_A},"full",**bind_options))
    root = bind_root(W,s,pool,payload,"full",**bind_options)
    payload["A"] = None
    hot.clear()
    assert root["A"] is A and len(root["hot"])==4
    options = dict(**bind_options,prepared_root=root,budget=2)
    control = search(W,s,pool,"full",**options)[0]
    assert control
    reject(lambda: search(W,s.copy(),pool,"full",**options))
    reject(lambda: search(W.copy(),s,pool,"full",**options))
    reject(lambda: search(W,s,(pool[1],pool[0],pool[2]),"full",**options))
    for override in ({"operator_scale":4},{"proof_unit":UNIT*2},{"hot_size":3}):
        reject(lambda: search(W,s,pool,"full",**(options|override)))
    reject(lambda: search(W,s,pool,"static",**options))
    for a in (s,*pool,A.data,W.indices,state[0],state[1],state[3]):
        reject(lambda: a.__setitem__(0,0))
    W.indices = W.indices.copy()
    reject(lambda: search(W,s,pool,"full",**options))


if __name__=="__main__":
    from promotion_root import selftest
    selftest()
