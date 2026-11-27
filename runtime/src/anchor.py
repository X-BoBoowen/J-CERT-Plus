"""Cross-opinion tangent anchor. No Gram, winner or response survives a query."""
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
import math
from types import MappingProxyType

from .implicit_catalog import pair_cap, up

def upward_fraction(q):
    v=float(q)
    return up(v) if Fraction(v)<q else v

@dataclass(frozen=True,order=True)
class Edit:
    i:int
    j:int
    kind:str
    omega:float

@dataclass(frozen=True)
class Graph:
    n:int
    edges:tuple
    universe:str='numbered'
    lookup:object=field(init=False,compare=False,repr=False)
    def __post_init__(self):
        if type(self.n) is not int or self.n<1:
            raise ValueError("empty vertex universe")
        edges=tuple(tuple(e) for e in self.edges)
        lookup={}
        for i,j,sign,w in edges:
            if not 0<=i<j<self.n or sign not in (-1,1) or not math.isfinite(w) or w<=0 or (i,j) in lookup:
                raise ValueError("invalid or duplicate signed edge")
            lookup[i,j]=sign,w
        object.__setattr__(self,'edges',edges)
        object.__setattr__(self,'lookup',MappingProxyType(lookup))

    def validate(self,e):
        if not 0<=e.i<e.j<self.n or not math.isfinite(e.omega) or e.omega<=0:
            raise ValueError("invalid edit endpoints or effective weight")
        edge=self.lookup.get((e.i,e.j))
        if not ((e.kind=='insert' and edge is None) or (e.kind=='flip' and edge==(-1,e.omega))):
            raise ValueError("candidate update is illegal in current graph")

@dataclass(frozen=True,init=False,eq=False)
class State:
    graph:Graph
    opinions:tuple
    x:tuple
    radius:float

    @classmethod
    def certify(cls,graph,opinions,centers):
        opinions=tuple(opinions); centers=tuple(centers)
        if len(opinions)!=graph.n or len(centers)!=graph.n or not all(math.isfinite(v) and abs(v)<=1 for v in opinions) or not all(math.isfinite(v) for v in centers):
            raise ValueError("invalid signed-FJ state or opinions")
        # With identity anchoring the signed operator's inverse infinity norm <= 1.
        x=tuple(map(Fraction,centers))
        residual=[v-Fraction(s) for v,s in zip(x,opinions)]
        for i,j,sign,w in graph.edges:
            term=Fraction(w)*(x[i]-sign*x[j])
            residual[i]+=term
            residual[j]-=sign*term
        r=upward_fraction(max(map(abs,residual)))
        state=object.__new__(cls)
        for key,v in [('graph',graph),('opinions',opinions),('x',centers),('radius',r)]:
            object.__setattr__(state,key,v)
        return state

@dataclass(frozen=True)
class GainUpper:
    upper:float

@dataclass(frozen=True)
class TangentCap:
    reference:State
    edit:Edit
    upper:float

def tangent(state,e):
    state.graph.validate(e)
    x,y=state.x[e.i],state.x[e.j]
    r=state.radius
    if e.kind=='insert':
        cap=pair_cap(x,y,r,e.omega)
    else:
        # Use exact corner products on the bounded box for signed flips.
        x,y,r=map(Fraction,(x,y,r))
        cap=upward_fraction(max(-4*Fraction(e.omega)*a*b for a in (x-r,x+r) for b in (y-r,y+r)))
    return TangentCap(state,e,cap)

@dataclass(frozen=True)
class Plan:
    anchor:object
    state:State
    drift:tuple
    hot:tuple
    starts:tuple
    postings:int
    prefix:int
    base_prefix:int
    cold_upper:float

@dataclass(frozen=True)
class Prices:
    rebuild:float
    posting:float
    inspect:float
    response:float
    response_rate:float=1.

def choose_plan(policy,p,count,prices):
    if policy=='retain-always': return 'retain'
    if policy=='rebuild-always': return 'rebuild'
    if policy!='adaptive': raise ValueError("unknown anchor policy")
    visits=min(count,p.postings+p.prefix)
    extra=max(0,visits-p.base_prefix)
    retain_cost=p.postings*prices.posting + extra*(prices.inspect+prices.response_rate*prices.response)
    return 'retain' if retain_cost<=count*prices.rebuild else 'rebuild'

class Anchor:
    def __init__(self,state,edits):
        self.state=state
        self.edits=tuple(edits)
        self.caps={e:tangent(state,e) for e in self.edits}
        if len({(e.i,e.j) for e in self.edits})!=len(self.edits):
            raise ValueError("a pair can admit only one edit")
        postings=[[] for _ in state.x]
        groups=defaultdict(list)
        for e,cap in self.caps.items():
            groups[e.omega].append((cap.upper,e))
            postings[e.i].append(e); postings[e.j].append(e)
        self.postings=tuple(tuple(row) for row in postings)
        self.groups={w:tuple(sorted(rows)) for w,rows in groups.items()}

    @classmethod
    def from_caps(cls,state,edits,caps):
        a=cls(state,edits)
        if set(caps)!=set(a.edits): raise ValueError("missing or unexpected reference cap")
        for e,cap in caps.items():
            if not isinstance(cap,TangentCap): raise TypeError("a gain upper is not a tangent cap")
            if cap.reference is not state or cap.edit!=e or not cap.upper>=a.caps[e].upper:
                raise ValueError("unverified reference tangent cap")
        # Canonical verified caps are at least as tight as admitted inputs.
        return a

    def drift(self,cur):
        if cur is self.state:
            return (0.,)*len(cur.x)  # Identical certified mathematical state.
        if self.state.graph.n!=cur.graph.n or self.state.graph.universe!=cur.graph.universe:
            raise ValueError("different endpoint coordinate systems")
        return tuple(upward_fraction(abs(Fraction(x)-Fraction(y))+Fraction(cur.radius)+Fraction(self.state.radius))
                     for x,y in zip(cur.x,self.state.x))

    def bounds(self,cur,edits):
        d=self.drift(cur)
        vals={}
        for e in edits:
            cur.graph.validate(e)
            if e in self.caps:
                vals[e]=self.local(e,d)
            else:
                vals[e]=tangent(cur,e).upper
        return vals

    def local(self,e,d):
        shift=up(up(4*e.omega)*up(d[e.i]+d[e.j]))
        return up(self.caps[e].upper+shift)

    def plan(self,cur,threshold,tau):
        """Fixed catalog. A graph change triggers complete current legality validation.
        For filter/extension/weight changes construct a new catalog or use bounds()."""
        if not math.isfinite(tau) or tau<0 or math.isnan(threshold):
            raise ValueError("invalid frontier threshold")
        if cur.graph is not self.state.graph:
            for e in self.edits: cur.graph.validate(e)
        d=self.drift(cur)
        hot=tuple(i for i,v in enumerate(d) if v>tau)
        postings=sum(len(self.postings[i]) for i in hot)
        cold=-math.inf; starts=[]; prefix=base=0
        for w,rows in self.groups.items():
            shift=up(up(8*w)*tau)
            def boundary(s):
                # Keep equality even for exact decisions and hidden earlier IDs.
                l,h=0,len(rows)
                while l<h:
                    k=(l+h)//2
                    if up(rows[k][0]+s)<threshold: l=k+1
                    else: h=k
                return l
            start=boundary(shift)
            starts.append((w,start)); prefix+=len(rows)-start
            base+=len(rows)-boundary(0.)
            if start: cold=max(cold,up(rows[start-1][0]+shift))
        return Plan(self,cur,d,hot,tuple(starts),postings,prefix,base,cold)

    def frontier(self,p):
        if p.anchor is not self: raise ValueError("plan belongs to another anchor")
        ids={e for i in p.hot for e in self.postings[i]}
        for w,start in p.starts:
            ids.update(e for _,e in self.groups[w][start:])
        return ids,p.cold_upper
