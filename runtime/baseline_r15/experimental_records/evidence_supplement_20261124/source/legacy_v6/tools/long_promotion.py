"""Full-graph unit-sign promotion, fixed initial two-hop / negative-edge catalog."""
import argparse
import copy
import csv
import gc
import math
import sys
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu
from threadpoolctl import threadpool_limits

from promotion import (OPINION_DEN, Q, VerifiedGramViews, exchange_tie, resolve_comparison)
from workloads import NAMES, load_graph, opinions, system

DATA = Path(__file__).resolve().parents[1] / "data"
UNIT = 1 << 40
SCREEN = 1 << 20
BLOCK = 131072
TAU = SCREEN // 4096


class Iterative:
    def __init__(self, A, count_iterations=False, paired_pcg=None):
        self.root_operator = A
        self.A = A.astype(float)
        inv = 1. / A.diagonal()
        self.inverse_diag = inv
        self.paired_pcg = paired_pcg
        self.M = LinearOperator(A.shape, matvec=lambda x: inv*x)
        self.iterations = 0
        self.callback = self.count_iteration if count_iterations else None

    def count_iteration(self, _):
        self.iterations += 1

    def solve(self, rhs, x0=None):
        if rhs.ndim == 2:
            if self.paired_pcg is not None and rhs.shape[1]==2 and x0 is None:
                from paired_pcg import solve_pair
                x,count = solve_pair(self.paired_pcg,self.A,self.inverse_diag,rhs)
                self.iterations += count
                return x
            return np.column_stack([self.solve(col) for col in rhs.T])
        x, info = cg(self.A, rhs, M=self.M, x0=x0, rtol=2e-13, atol=1e-15, callback=self.callback)
        if info:
            raise ArithmeticError(f"CG did not converge ({info})")
        return x


class Views(VerifiedGramViews):
    def __init__(self, pool, scale, operator_scale=1):
        super().__init__(pool, scale)
        self.retain_winner = False
        self.scored_response = self.retained = None
        self.profiling = False
        self.response_s = 0.
        self.response_rhs = 0
        self.flip_column_scale = math.isqrt(operator_scale)
        if self.flip_column_scale**2 != operator_scale:
            raise ValueError("A squared integer operator scale is required")

    def project(self, e, columns, r):
        g = super().project(e, columns, r)
        k = self.flip_column_scale if self.pool[2][e] else 1
        if k == 1:
            return g
        return [[(k*lo, k*hi) for lo, hi in row] for row in g]

    def response(self, e, A, fac):
        if self.retained is not None and self.retained[0]==e:
            result = self.retained[1]
            self.latest = result[:2]
            return result
        if self.profiling:
            started = perf_counter()
        result = super().response(e,A,fac)
        if self.flip_column_scale!=1:
            # The unscaled graph's dimension bound above need not dominate
            # the diagonal of an integer-scaled operator. Certify the exact
            # int64 residual product before using the returned enclosure.
            row_norm=2*int(A.diagonal().max())
            if row_norm>2*A.shape[0]+1:
                if row_norm*int(np.abs(result[2]).max())+self.scale>=2**63:
                    raise OverflowError("weighted response residual exceeds int64")
        if self.pool[2][e] and self.flip_column_scale != 1:
            result = tuple(self.flip_column_scale * v for v in result)
        self.latest = result[:2]
        if self.profiling:
            self.response_s += perf_counter()-started
            self.response_rhs += result[0].shape[1]
        if self.retain_winner:
            self.scored_response = e,result
        return result

    def advance(self, e, t, A, fac, z):
        result = super().advance(e,t,A,fac,z)
        if isinstance(fac,UpdatedFactor):
            U,Y = self.latest
            signs = [1.,-1.] if U.shape[1]==2 else [1.]
            fac.steps.append((Y,np.linalg.inv(np.diag(signs)+U.T@Y)))
        self.latest = None
        self.scored_response = self.retained = None
        # Outward dyadic reboxing prevents rational denominator growth.
        for key,(version,g) in self.entries.items():
            boxed = [[(Q(math.floor(lo*self.scale),self.scale),Q(math.ceil(hi*self.scale),self.scale))
                      for lo,hi in row] for row in g]
            self.entries[key] = version,boxed
        return result


class UpdatedFactor:
    def __init__(self,A):
        self.root_operator = A
        self.base = splu(A.astype(float).tocsc())
        self.steps = []

    def solve(self,rhs):
        result = self.base.solve(rhs)
        for Y,inverse in self.steps:
            result -= Y@(inverse@(Y.T@rhs))
        return result


def store_bytes(value):
    seen = set()
    def count(v):
        if id(v) in seen:
            return 0
        seen.add(id(v))
        total = sys.getsizeof(v)
        if isinstance(v,dict):
            total += sum(count(k)+count(w) for k,w in v.items())
        elif isinstance(v,(list,tuple,set)):
            total += sum(map(count,v))
        elif isinstance(v,Q):
            total += count(v.numerator)+count(v.denominator)
        return total
    return count(value)


def cap(pool, ids, x, r, diag, tighten=True, operator_scale=1, *, wide=False):
    i,j,f = (arr[ids] for arr in pool)
    u,v = x[i],x[j]
    if wide:
        u,v = u.astype(object),v.astype(object)
    addition = (abs(u-v)+2*r)**2
    flip = -4*u*v+4*r*(abs(u)+abs(v)+r)
    if tighten:
        a,b = diag[i],diag[j]
        if wide:
            a,b = a.astype(object),b.astype(object)
        prod = a*b
        den = prod+a+b
        # ceil(g*ab/(ab+a+b)) = g-floor(g*(a+b)/(ab+a+b)).
        if not wide and int(addition.max(initial=0))*int((a+b).max(initial=0)) >= 2**63:
            raise OverflowError("addition screen exceeds int64")
        addition -= addition*(a+b)//den
    return np.where(f,operator_scale*flip,addition)


def certified_state(A,fac,s_int,proof_unit=UNIT,z=None,*,residual_limit=None,certified_rows=None):
    if z is None:
        z = fac.solve(s_int/OPINION_DEN)
    x = np.rint(z*proof_unit).astype("int64")
    diag = A.diagonal()
    if (2*int(diag.max())+1)*int(abs(x).max())+proof_unit >= 2**63:
        if residual_limit is not None:
            return None
        raise OverflowError("state residual exceeds the int64 bound")
    abs_residual = abs(A@x-s_int*(proof_unit//OPINION_DEN))
    r = int(abs_residual.max())
    if residual_limit is not None and r>residual_limit:
        return None
    xc = np.rint(z*SCREEN).astype("int64")
    divisor = proof_unit//SCREEN
    rc = (int(abs(x-xc*divisor).max())+r+divisor-1)//divisor
    assert max(abs(xc))+rc <= 2*SCREEN
    if certified_rows is not None:
        certified_rows[:] = [(A,s_int,x,proof_unit,diag,abs_residual)]
    return z,x,r,xc,rc


def certify_child(A,fac,s_int,z,proof_unit,certified_rows=None):
    """Box the rounded Woodbury state using the child operator's integer residual."""
    # One screen quantum, allowing the worst-case rounding of an accurate solve.
    limit = max(proof_unit//SCREEN, int(A.diagonal().max())+1)
    if np.isfinite(z).all() and np.max(np.abs(z))<=1.5:
        result = certified_state(A,fac,s_int,proof_unit,z,residual_limit=limit,certified_rows=certified_rows)
        if result is not None:
            return result,"woodbury"
    if isinstance(fac,Iterative) and np.isfinite(z).all():
        corrected = fac.solve(s_int/OPINION_DEN,x0=z)
    else:
        corrected = fac.solve(s_int/OPINION_DEN)
    return certified_state(A,fac,s_int,proof_unit,corrected,certified_rows=certified_rows),"corrected"


def seed_view(pool,xc,rc,diag,hot_size,rank_cold,operator_scale=1):
    m = len(pool[0])
    seeds,sorted_chunks = [],[]
    for a in range(0,m,BLOCK):
        ids = np.arange(a,min(a+BLOCK,m),dtype="int32")
        caps = cap(pool,ids,xc,rc,diag,False,operator_scale)
        k = min(hot_size,len(ids))
        if k:
            cutoff = np.partition(caps,len(caps)-k)[-k]
            top = np.r_[np.flatnonzero(caps>cutoff),np.flatnonzero(caps==cutoff)[:k]]
            seeds.extend((int(caps[k]),int(ids[k])) for k in top)
        if rank_cold:
            order = np.argsort(-caps,kind="stable")
            sorted_chunks.append((ids[order],-caps[order]))
    from promotion_root import RankedSeeds
    hot = RankedSeeds((k for _,k in sorted(seeds,key=lambda p:(-p[0],p[1]))[:hot_size]),
                      pool,xc,rc,diag,hot_size,operator_scale)
    return hot,sorted_chunks


def dyadic_costs(degree,drift,sorted_chunks,target):
    # tau_0 = 256/SCREEN. Above the first power covering max(drift) the
    # posting cost stays zero while prefixes can only grow.
    max_drift=max(1,int(drift.max()))
    choices=np.asarray([1<<k for k in range((max_drift-1).bit_length()+1)],
                       dtype=np.int64)
    costs=np.asarray([degree[drift>v].sum() for v in choices],dtype=np.int64)
    thresholds=8*SCREEN*choices-target
    for _,keys in sorted_chunks:
        costs+=np.searchsorted(keys,thresholds,side="right")
    return choices,costs


def resolve_weighted(A,s,pool,evaluated,winner,best,proof_unit,operator_scale):
    overlapping=[e for e,hi in evaluated.items() if e!=winner and hi>=best]
    if not overlapping or (winner is None and all(hi<=0 for hi in evaluated.values())):
        return winner,best
    if winner is not None and all(exchange_tie(A,s,pool,winner,e) for e in overlapping):
        return min([winner,*overlapping]),best
    if A.shape[0]>64:
        raise ArithmeticError("Unresolved weighted gain overlap on a large graph.")
    # Small unresolved contests use direct rational child objectives with the
    # same scaled integer operator as the interval path.
    from sympy import Matrix, Rational
    parent=Matrix(A.toarray())
    rhs=Matrix([Rational(float(value)) for value in s])
    parent_energy=(rhs.T*parent.inv()*rhs)[0]
    selected,value=None,0
    for e in sorted(overlapping+([winner] if winner is not None else [])):
        i,j,flip=(v[e].item() for v in pool)
        child=parent.copy()
        if flip:
            child[i,j]-=2*operator_scale
            child[j,i]-=2*operator_scale
        else:
            child[i,i]+=1
            child[j,j]+=1
            child[i,j]-=1
            child[j,i]-=1
        gain=parent_energy-(rhs.T*child.inv()*rhs)[0]
        if gain>value:
            selected,value=e,gain
    if selected is not None:
        evaluated[selected]=math.ceil(value*proof_unit**2)
    return selected,math.floor(value*proof_unit**2)


def search(W, s_int, pool, mode, budget=15, hot_size=64, adaptive=False, *,
           operator_scale=1, dyadic_tau=False, trace=False, broad_pass=None, proof_unit=UNIT,
           index_factory=None, woodbury_state=False, retain_winner=False, prepared_root=None,
           paired_pcg=None, directional=False, transport=True, matched_screen=False, fine_screen=False, strict_screen=False):
    """Search fixed lexical edits with unit sign repairs and 1/operator_scale additions."""
    from universe_index import build_incidence
    start = perf_counter()
    if prepared_root is not None:
        from promotion_root import BoundRoot
        if not isinstance(prepared_root,BoundRoot):
            raise ValueError("prepared search needs a root sealed by bind_root")
        prepared_root.validate(W,s_int,pool,mode,hot_size=hot_size,operator_scale=operator_scale,
                               proof_unit=proof_unit,adaptive=adaptive,dyadic_tau=dyadic_tau,
                               index_factory=index_factory)
    A = (operator_scale*system(W).astype("int64") if prepared_root is None
         else prepared_root["A"])
    n,m = A.shape[0],len(pool[0])
    if prepared_root is None and (s_int.shape!=(n,) or not np.issubdtype(s_int.dtype,np.integer)
            or np.any(s_int < -OPINION_DEN) or np.any(s_int > OPINION_DEN)):
        raise ValueError("Expected one bounded dyadic opinion per vertex")
    if (mode not in ("static","full","promoted") or hot_size<0 or budget<0 or
            (adaptive and mode!="promoted")):
        raise ValueError("Invalid search mode, hot size, or budget")
    if strict_screen and (not hasattr(broad_pass,"library") or fine_screen
                          or (mode=="promoted" and index_factory is None)):
        raise ValueError("Strict screening requires the endpoint library and indexed promotion")
    if index_factory is not None and not (mode=="promoted" and adaptive and dyadic_tau):
        raise ValueError("A custom index requires adaptive dyadic promotion")
    s = s_int/OPINION_DEN if prepared_root is None else prepared_root["opinion_float"]
    chosen, hot, rows = set(),set(),[]
    views = Views(pool,proof_unit,operator_scale)
    if directional:
        from promotion_directional import directional_block
        views.directional_block = directional_block
    views.retain_winner = retain_winner
    child_state = None
    certified_rows = [] if strict_screen else None
    profile_state = woodbury_state or retain_winner
    views.profiling = profile_state
    index = None
    sorted_chunks = []
    ratio = (proof_unit//SCREEN)**2
    total_lo = total_hi = 0
    index_bytes = setup_seconds = 0
    array_bytes = sum(v.nbytes for v in pool)
    if prepared_root is None:
        fac = UpdatedFactor(A) if n<=10000 else None
    else:
        fac = copy.copy(prepared_root["fac"])
        if isinstance(fac,UpdatedFactor):
            fac.steps = []
        elif isinstance(fac,Iterative):
            fac.iterations = 0
            fac.paired_pcg = paired_pcg
            fac.callback = fac.count_iteration if profile_state else None
    for t in range(budget):
        diag = certified_rows[0][4] if certified_rows else A.diagonal()
        state_start = perf_counter()
        if t==0 and prepared_root is not None:
            z,x,r,xc,rc = prepared_root["state"]
            state_s = prepared_root.get("state_s",0.)
            state_iterations = prepared_root.get("state_iterations",0)
            state_source = "shared"
        elif child_state is None:
            if n>10000:
                fac = Iterative(A,profile_state,paired_pcg)
            z,x,r,xc,rc = certified_state(A,fac,s_int,proof_unit,certified_rows=certified_rows)
            state_s = perf_counter()-state_start
            state_iterations = getattr(fac,"iterations",0)
            state_source = "solve"
        else:
            z,x,r,xc,rc = child_state
        if t==0 and prepared_root is not None:
            x0,r0 = xc,rc
            hot = set(prepared_root["hot"])
            index = prepared_root["index"]
            if index is not None:
                index = index.fork_cursor(fixed_tau=prepared_root.fixed_tau)
            index_bytes = prepared_root["index_bytes"]
            common_seconds = prepared_root["common_seconds"]
            rank_seconds = prepared_root["rank_seconds"]
            index_seconds = prepared_root["index_seconds"]
            setup_seconds = prepared_root["setup_seconds"]
            if mode=="promoted" and index_factory is None:
                raise ValueError("Prepared promotion requires a custom dyadic index")
        elif t == 0:
            common_seconds = perf_counter()-start
            setup = perf_counter()
            x0,r0 = xc.copy(),rc
            hot,sorted_chunks = seed_view(pool,xc,rc,diag,hot_size,
                                          mode=="promoted" and index_factory is None,operator_scale)
            rank_seconds = perf_counter()-setup
            index_start = perf_counter()
            if index_factory is not None:
                index = index_factory(pool,xc,rc,operator_scale)
                index_bytes = index.nbytes
            elif mode=="promoted":
                index = build_incidence(pool[0],pool[1],n)
                index_bytes = sum(a.nbytes for pair in sorted_chunks for a in pair)
                index_bytes += sum(a.nbytes for a in vars(index).values())
            index_seconds = perf_counter()-index_start if index is not None else 0.
            setup_seconds = perf_counter()-setup
        reset_s = perf_counter()-start if t==0 and prepared_root is not None else 0.
        score_s = cold_query_s = cold_screen_s = 0.
        response_start = views.response_s
        rhs_start = views.response_rhs
        iteration_start = getattr(fac,"iterations",0)
        evaluated = {}
        admissions = {}
        best,winner = 0,None
        visited = promoted = 0
        def refine(ids, upper, cold, conversion=ratio):
            nonlocal best,winner,promoted,score_s
            started = perf_counter()
            for k in np.argsort(-upper,kind="stable"):
                if int(upper[k])*conversion < best:
                    break
                e = int(ids[k])
                if e in chosen or e in evaluated:
                    continue
                views.scored_response = None
                lo,hi = views.score(e,t,A,fac,x,r)
                evaluated[e] = hi
                if cold and mode=="promoted":
                    hot.add(e)
                    promoted += 1
                    if trace:
                        admissions[e] = (lo*operator_scale,hi*operator_scale)
                if lo>best or (winner is not None and lo==best and e<winner):
                    best,winner = lo,e
                    views.retained = views.scored_response
                views.scored_response = None
            score_s += perf_counter()-started
        seed = np.array(sorted(hot-chosen),dtype="int32")
        seed_started = perf_counter()
        if strict_screen:
            from promotion_strict import StrictScreen
            strict = StrictScreen(A,s_int,x,proof_unit,operator_scale,pool,library=broad_pass.library,
                                  certified_rows=certified_rows[0] if certified_rows else None)
            strict_setup_s = perf_counter()-seed_started
            seed_ids,seed_upper = strict.screen(seed,best)
        else:
            seed_ids = seed
            seed_upper = (cap(pool,seed,x,r,diag,operator_scale=operator_scale,wide=True)
                          if fine_screen else cap(pool,seed,xc,rc,diag,operator_scale=operator_scale))
        seed_screen_s = perf_counter()-seed_started
        refine(seed_ids,seed_upper,False,1 if fine_screen or strict_screen else ratio)
        if strict_screen and mode!="static":
            query_start = perf_counter()
            excluded = hot|chosen
            if mode=="promoted":
                drift = abs(xc-x0)+rc+r0
                ids,choices,costs,tau = index.query(drift,(best+ratio-1)//ratio,excluded)
                cold_query_s = perf_counter()-query_start
                screen_start = perf_counter()
                survivors,upper = strict.screen(ids,best)
            else:
                excluded_ids = np.array(sorted(excluded),dtype=np.int32)
                cold_query_s = perf_counter()-query_start
                screen_start = perf_counter()
                live_ids,live_upper = [],[]
                for start_id in range(0,m,BLOCK):
                    block_ids,block_upper = strict.scan(start_id,min(start_id+BLOCK,m),best,excluded_ids)
                    if len(block_ids):
                        live_ids.append(block_ids)
                        live_upper.append(block_upper)
                survivors = np.concatenate(live_ids) if live_ids else np.empty(0,dtype=np.int32)
                upper = np.concatenate(live_upper) if live_upper else np.empty(0,dtype=object)
            visited = strict.rows-len(seed)
            cold_screen_s = perf_counter()-screen_start
            refine(survivors,upper,True,1)
        elif mode!="static":
            query_start = perf_counter()
            drift = abs(xc-x0)+rc+r0
            if index_factory is not None:
                ids,choices,costs,tau = index.query(drift,(best+ratio-1)//ratio,hot|chosen)
                def blocks(selected_ids=ids):
                    for start_id in range(0,len(selected_ids),BLOCK):
                        yield selected_ids[start_id:start_id+BLOCK]
            elif mode=="promoted":
                tau = TAU
                if adaptive:
                    degree = np.diff(index.left)+np.diff(index.right)
                    target = (best+ratio-1)//ratio
                    def cost(tau):
                        return int(degree[drift>tau].sum())+sum(
                            int(np.searchsorted(keys,8*SCREEN*tau-target,side="right"))
                            for _,keys in sorted_chunks)
                    if dyadic_tau:
                        choices,costs=dyadic_costs(degree,drift,sorted_chunks,target)
                        tau=int(choices[np.argmin(costs)])
                    else:
                        tau = min((TAU//4,TAU,4*TAU,16*TAU,64*TAU),key=cost)
                moving = np.flatnonzero(drift>tau)
                incident = index.incident(moving)
                # Merge per-block sorted prefixes with the incident posting union.
                def blocks():
                    for b,(order,negative_caps) in enumerate(sorted_chunks):
                        limit = np.searchsorted(negative_caps,
                            8*SCREEN*tau-(best+ratio-1)//ratio,side="right")
                        edges = incident[np.searchsorted(incident,b*BLOCK):
                                         np.searchsorted(incident,(b+1)*BLOCK)]
                        yield np.union1d(edges,order[:limit])
            else:
                def blocks():
                    for a in range(0,m,BLOCK):
                        if broad_pass is None:
                            yield np.arange(a,min(a+BLOCK,m),dtype="int32")
                        else:
                            yield broad_pass(pool, xc, rc, operator_scale,
                                             (best+ratio-1)//ratio, a, min(a+BLOCK,m))
            pending_ids,pending_upper = [],[]
            screen_start = perf_counter()
            cold_query_s = screen_start-query_start
            for ids in blocks():
                if mode=="promoted" and index_factory is None and len(ids):
                    ids = ids[~np.isin(ids,list(hot|chosen))]
                if not len(ids):
                    continue
                visited += len(ids)
                if mode!="full" or broad_pass is None:
                    upper = cap(pool,ids,xc,rc,diag,False,operator_scale)
                    live = upper >= (best+ratio-1)//ratio
                    ids = ids[live]
                if not len(ids):
                    continue
                upper = cap(pool,ids,xc,rc,diag,operator_scale=operator_scale)
                if mode=="promoted" and not matched_screen:
                    u,v = pool[0][ids],pool[1][ids]
                    upper = np.minimum(upper,cap(pool,ids,x0,r0,diag,False,operator_scale)
                                       +4*SCREEN*(drift[u]+drift[v]))
                mask = upper >= (best+ratio-1)//ratio
                if fine_screen:
                    ids = ids[mask]
                    upper = cap(pool,ids,x,r,diag,operator_scale=operator_scale,wide=True)
                    mask = upper >= best
                if mask.any():
                    pending_ids.append(ids[mask])
                    pending_upper.append(upper[mask])
            cold_screen_s = perf_counter()-screen_start
            if pending_ids:
                refine(np.concatenate(pending_ids),np.concatenate(pending_upper),True,
                       1 if fine_screen else ratio)
        if operator_scale==1:
            winner,best=resolve_comparison(A,s,pool,evaluated,winner,best,proof_unit)
        else:
            winner,best=resolve_weighted(A,s,pool,evaluated,winner,best,proof_unit,operator_scale)
        if winner is None:
            break
        assert best>0
        total_lo += best
        total_hi += evaluated[winner]
        u,v,f = (arr[winner].item() for arr in pool)
        chosen.add(winner)
        materialized = None if matched_screen else store_bytes((views.entries,hot))
        retained = views.retained is not None and views.retained[0]==winner
        scored_response_s = views.response_s-response_start
        scored_rhs = views.response_rhs-rhs_start
        before_commit = getattr(fac,"iterations",0)
        score_iterations = before_commit-iteration_start
        response_start = views.response_s
        advance_start = perf_counter()
        if not transport:
            views.entries.clear()
        next_state = views.advance(winner,t,A,fac,z)
        winner_correction_s = perf_counter()-advance_start
        commit_response_s = views.response_s-response_start
        hot.discard(winner)
        rows.append(dict(epoch=t+1,i=u,j=v,flip=int(f),low=best*operator_scale,high=evaluated[winner]*operator_scale,
                         visited=visited if mode!="full" else m-t,
                         promoted=promoted,view=len(hot),scores=len(evaluated),
                         fresh=views.fresh,hits=views.hits,transfers=views.transfers,
                         winner_blocks=views.winner_blocks,graph_bytes=sum(a.nbytes for
                         a in (W.data,W.indices,W.indptr,A.data,A.indices,A.indptr)),
                         catalog_bytes=array_bytes,index_bytes=index_bytes,
                         view_bytes=materialized,setup_seconds=setup_seconds))
        if trace:
            import json
            rows[-1].update(winner_id=winner,admissions=json.dumps(admissions),
                            common_seconds=common_seconds,rank_seconds=rank_seconds,
                            index_seconds=index_seconds)
        if profile_state:
            rows[-1].update(state_s=state_s,state_residual=r,state_iterations=state_iterations,
                            reset_s=reset_s,cold_query_s=cold_query_s,cold_screen_s=cold_screen_s,
                            score_s=score_s,scored_response_s=scored_response_s,
                            scored_rhs=scored_rhs,score_iterations=score_iterations,
                            commit_response_s=commit_response_s,
                            commit_transfer_s=winner_correction_s-commit_response_s,
                            commit_rhs=0 if retained else 1+int(f),
                            commit_iterations=getattr(fac,"iterations",0)-before_commit,
                            state_source=state_source,
                            winner_response_source="retained" if retained else "recomputed",
                            winner_correction_s=winner_correction_s)
        if matched_screen:
            rows[-1].update(seed_visits=len(seed),catalog_visits=m if mode=="full" else visited,
                            seed_screen_s=seed_screen_s,transport_enabled=int(transport))
        if strict_screen:
            rows[-1].update(strict_rows=strict.rows,strict_setup_s=strict_setup_s,catalog_visits=visited,
                            endpoint_lower_max=strict.lower_max)
            assert strict.rows==len(seed)+visited
            if mode=="full":
                assert strict.rows==m-t
        if adaptive:
            rows[-1].update(tau_units=tau,tau_denominator=SCREEN)
            if dyadic_tau and trace:
                rows[-1].update(tau_grid_max=int(choices[-1]),tau_work=int(costs.min()))
        w = (0,-2*operator_scale,-2*operator_scale,0) if f else (1,-1,-1,1)
        A = A + sparse.coo_matrix((w,([u,u,v,v],[u,v,u,v])),shape=A.shape).tocsr()
        if woodbury_state:
            state_start = perf_counter()
            if n>10000:
                fac = Iterative(A,True,paired_pcg)
            child_state,state_source = certify_child(A,fac,s_int,next_state,proof_unit,certified_rows)
            state_iterations = getattr(fac,"iterations",0)
            state_s = perf_counter()-state_start
            rows[-1].update(child_state_s=state_s,child_state_residual=child_state[2],
                            child_state_source=state_source,child_state_iterations=state_iterations)
    return rows,total_lo*operator_scale,total_hi*operator_scale,perf_counter()-start


def write_rows(file,rows):
    with file.open("a",newline="") as stream:
        out = csv.DictWriter(stream,fieldnames=rows[0].keys())
        if stream.tell()==0:
            out.writeheader()
        out.writerows(rows)


def comparison_rows(name):
    from promotion import complete_pool
    from universe_index import build_universe
    with (DATA/f"long_promotion_{name}_rounds.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["repetition"]=="0"]
    paths = [[r for r in rows if r["method"]==m] for m in ("static","promoted","full")]
    assert all(len(path)==15 for path in paths)
    def action(r):
        return tuple(int(r[k]) for k in ("i","j","flip"))
    selected = {action(r) for r in paths[0]}
    if any(action(r) not in selected for r in paths[2]):
        # Every static selection belongs to the original hot set. Rebuild that
        # set whenever equality with a selected static action does not settle membership.
        W = load_graph(name.split("_")[0])
        pool = complete_pool(W) if name.endswith("_all") else build_universe(W)
        A = system(W).astype("int64")
        fac = UpdatedFactor(A) if A.shape[0]<=10000 else Iterative(A)
        s = np.rint(OPINION_DEN*opinions(W,"unsigned_control",seed=17)).astype("int64")
        *_,xc,rc = certified_state(A,fac,s)
        seed,_ = seed_view(pool,xc,rc,A.diagonal(),64,False)
        selected = {tuple(int(v[k]) for v in pool) for k in seed}
        assert all(action(r) in selected for r in paths[0])
    result = []
    promotions = misses = static_lo = static_hi = full_lo = full_hi = 0
    for t,(st,dy,fu) in enumerate(zip(*paths),1):
        assert action(dy)==action(fu)
        dlo,dhi = int(dy["low"]),int(dy["high"])
        flo,fhi = int(fu["low"]),int(fu["high"])
        slo,shi = int(st["low"]),int(st["high"])
        assert dlo>0 and max(dlo,flo)<=min(dhi,fhi)
        mismatch = action(st)!=action(fu)
        misses += mismatch
        promotions += int(dy["promoted"])
        full_lo,full_hi = full_lo+flo,full_hi+fhi
        static_lo,static_hi = static_lo+slo,static_hi+shi
        live = int(dy["universe"])-t+1
        result.append(dict(graph=dy["graph"],catalog="all" if name.endswith("_all") else "twohop",
            epoch=t,live_before=live,static_action=action(st),full_action=action(fu),
            promoted_action=action(dy),full_winner_in_initial_hot=int(action(fu) in selected),
            certified_positive=int(dlo>0),different_static_action=int(mismatch),
            static_mismatches=misses,visited=int(dy["visited"]),
            visited_fraction=int(dy["visited"])/live,promoted=int(dy["promoted"]),
            cumulative_promoted=promotions,hot_remaining=int(dy["view"]),
            view_bytes=int(dy["view_bytes"]),remaining_universe=live-1,
            dynamic_gain_low=dlo,dynamic_gain_high=dhi,full_gain_low=flo,full_gain_high=fhi,
            static_gain_low=slo,static_gain_high=shi,
            trajectory_gain_difference_low=flo-shi if misses else 0,
            trajectory_gain_difference_high=fhi-slo if misses else 0,
            cumulative_full_low=full_lo,cumulative_full_high=full_hi,
            cumulative_static_low=static_lo,cumulative_static_high=static_hi,
            cumulative_loss_low=full_lo-static_hi if misses else 0,
            cumulative_loss_high=full_hi-static_lo if misses else 0,denominator=UNIT**2))
    return result


def write_comparisons(names=(*NAMES,"Conflict_all","Bio_all")):
    grouped = {name:comparison_rows(name) for name in names}
    for filename,controls in (("comparison",False),("scope_comparison",True)):
        output = [r for name,rows in grouped.items() if (name in ("Bio","Conflict"))==controls
                  for r in rows]
        if output:
            path = DATA/f"long_promotion_{filename}.csv"
            path.unlink(missing_ok=True)
            write_rows(path,output)
    return grouped


def summarize(names=(*NAMES,"Conflict_all","Bio_all")):
    summaries = []
    comparisons = write_comparisons(names)
    rng = np.random.default_rng(108)
    for name in names:
        with (DATA/f"long_promotion_{name}_runs.csv").open() as f:
            runs = list(csv.DictReader(f))
        with (DATA/f"long_promotion_{name}_rounds.csv").open() as f:
            rounds = list(csv.DictReader(f))
        methods = ("static","promoted","full")
        path = {mode:[x for x in rounds if x["repetition"]=="0" and x["method"]==mode]
                for mode in methods}
        final = comparisons[name][-1]
        misses = final["static_mismatches"]
        full_lo,full_hi = final["cumulative_full_low"],final["cumulative_full_high"]
        static_lo,static_hi = final["cumulative_static_low"],final["cumulative_static_high"]
        for mode in methods:
            series = [r for r in runs if r["method"]==mode]
            assert sorted(int(r["repetition"]) for r in series)==list(range(10))
            times = np.asarray([float(r["seconds"]) for r in series])
            lo,median,hi = np.quantile(times,[.25,.5,.75])
            boot = np.quantile(np.median(rng.choice(times,size=(10000,len(times))),axis=1),
                               [.025,.975])
            full_times = {r["repetition"]:float(r["seconds"]) for r in runs if r["method"]=="full"}
            ratios = [full_times[r["repetition"]]/float(r["seconds"]) for r in series]
            paired_ci = np.quantile(np.median(rng.choice(ratios,size=(10000,len(ratios))),axis=1),
                                    [.025,.975])
            p = path[mode]
            summaries.append(dict(graph=name,n=p[0]["n"],universe=p[0]["universe"],method=mode,
                reps=len(series),median_s=median,q25_s=lo,q75_s=hi,ci_low_s=boot[0],ci_high_s=boot[1],
                paired_full_over_method=np.median(ratios),paired_ci_low=paired_ci[0],
                paired_ci_high=paired_ci[1],
                max_sampled_rss=max(int(r["peak_rss"]) for r in series),
                graph_bytes=max(int(r["graph_bytes"]) for r in p),catalog_bytes=p[0]["catalog_bytes"],
                index_bytes=p[0]["index_bytes"],peak_view_bytes=max(int(r["view_bytes"]) for r in p),
                median_setup_s=np.median([float(r["setup_seconds"]) for r in rounds
                                          if r["epoch"]=="1" and r["method"]==mode]),
                static_mismatches=misses,loss_low=full_lo-static_hi if misses else 0,
                loss_high=full_hi-static_lo if misses else 0,
                gain_low=series[0]["gain_low"],gain_high=series[0]["gain_high"],denominator=UNIT**2,
                promoted=sum(int(r["promoted"]) for r in p),visited=sum(int(r["visited"]) for r in p),
                max_view=max(int(r["view"])+1 for r in p)))
    file = DATA/"long_promotion_summary.csv"
    file.unlink(missing_ok=True)
    write_rows(file,summaries)
    adaptive_records = []
    for name in ("Alpha","OTC"):
        with (DATA/f"long_promotion_{name}_adaptive_runs.csv").open() as f:
            runs = list(csv.DictReader(f))
        with (DATA/f"long_promotion_{name}_adaptive_rounds.csv").open() as f:
            rounds = [r for r in csv.DictReader(f) if r["repetition"]=="0"]
        times = [float(r["seconds"]) for r in runs]
        assert len(times)==10 and len(rounds)==15
        lo,med,hi = np.quantile(times,[.25,.5,.75])
        ci = np.quantile(np.median(rng.choice(times,size=(10000,len(times))),axis=1),
                         [.025,.975])
        adaptive_records.append(dict(graph=name,repetitions=len(times),median_s=med,q25_s=lo,
            q75_s=hi,ci_low_s=ci[0],ci_high_s=ci[1],visited=sum(int(r["visited"]) for r in rounds),
            total_live_epochs=sum(int(r["universe"])-int(r["epoch"])+1 for r in rounds)))
    file = DATA/"long_promotion_adaptive_summary.csv"
    file.unlink(missing_ok=True)
    write_rows(file,adaptive_records)


def experiment(name,reps,catalog,adaptive):
    from universe_index import build_universe
    from promotion import complete_pool
    tag = name+("_all" if catalog=="all" else "")+("_adaptive" if adaptive else "")
    for suffix in ("runs","rounds"):
        (DATA/f"long_promotion_{tag}_{suffix}.csv").unlink(missing_ok=True)
    W = load_graph(name)
    s = np.rint(OPINION_DEN*opinions(W,"unsigned_control",seed=17)).astype("int64")
    if catalog=="all" and W.shape[0]*(W.shape[0]-1)//2-(W>0).nnz//2>=2**31:
        raise OverflowError("The all-pairs universe exceeds the signed 32-bit catalog ID range")
    pool = complete_pool(W) if catalog=="all" else build_universe(W)
    methods = ["promoted"] if adaptive else ["static","promoted","full"]
    for rep in range(reps):
        paths = {}
        for mode in methods[rep%3:]+methods[:rep%3]:
            gc.collect()
            process = psutil.Process()
            peak = [process.memory_info().rss]
            stop = Event()
            def sample():
                while not stop.wait(.015):
                    peak[0] = max(peak[0],process.memory_info().rss)
            monitor = Thread(target=sample)
            monitor.start()
            try:
                rows,lo,hi,elapsed = search(W,s,pool,mode,adaptive=adaptive)
            finally:
                stop.set()
                monitor.join()
            paths[mode] = [(row['i'],row['j'],row['flip']) for row in rows]
            record = dict(graph=name,n=W.shape[0],universe=len(pool[0]),repetition=rep,method=mode)
            write_rows(DATA/f"long_promotion_{tag}_runs.csv",
                       [dict(**record,seconds=elapsed,peak_rss=peak[0],gain_low=lo,
                             gain_high=hi,denominator=UNIT**2)])
            write_rows(DATA/f"long_promotion_{tag}_rounds.csv",[dict(**record,**r) for r in rows])
            print(tag,rep,mode,len(rows),round(elapsed,3),flush=True)
        if not adaptive:
            assert paths["promoted"]==paths["full"]
        else:
            with (DATA/f"long_promotion_{name}_rounds.csv").open() as f:
                ref = [(int(r["i"]),int(r["j"]),int(r["flip"])) for r in csv.DictReader(f)
                       if r["repetition"]=="0" and r["method"]=="full"]
            assert paths["promoted"]==ref


def selftest():
    """Compare the search and its enclosures against an exact rational refactor."""
    from promotion import graph_case, search as rational_search
    from universe_index import build_universe
    for name in ("Alpha","OTC","Bio"):
        W = load_graph(name,n=10)
        A,s,actions = graph_case(name,n=10)
        pool = build_universe(W)
        assert actions==list(zip(*pool))
        opinions_int = np.asarray([int(v*OPINION_DEN) for v in s])
        ids,total,_,_ = rational_search(A,s,actions,"full",3,5)
        for method,adaptive in (("full",False),("promoted",False),("promoted",True)):
            rows,lo,hi,_ = search(W,opinions_int,pool,method,budget=5,hot_size=3,adaptive=adaptive)
            assert [tuple(row[k] for k in ("i","j","flip")) for row in rows]==ids
            assert lo<=total*UNIT**2<=hi
        for method, adaptive in (("full", False), ("promoted", True)):
            rows, lo, hi, _ = search(W, opinions_int, pool, method, budget=5, hot_size=3,
                adaptive=adaptive, matched_screen=True, fine_screen=True,
                transport=method != "full", proof_unit=1 << 48)
            assert [tuple(row[k] for k in ("i","j","flip")) for row in rows] == ids
            assert lo <= total*2**96 <= hi
    # These candidate caps exceed signed int64 on the fine coordinate grid.
    xx = np.array([2**44, -(2**43), 2**42], dtype=np.int64)
    dd = np.array([64, 96, 128], dtype=np.int64)
    pairs = np.array([0,0]),np.array([1,2]),np.array([False,True])
    wide = cap(pairs, np.arange(2), xx, 7, dd, operator_scale=16, wide=True)
    u,v,w = map(int,xx)
    a,b = map(int,dd[:2])
    assert wide[0] == math.ceil(Q((abs(u-v)+14)**2*a*b, a*b+a+b))
    assert wide[1] == 16*(-4*u*w+28*(abs(u)+abs(w)+7))
    assert search(W,np.zeros(W.shape[0],dtype="int64"),pool,"promoted",budget=5)[0]==[]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--graph",choices=NAMES)
    group.add_argument("--summary",action="store_true")
    group.add_argument("--selftest",action="store_true")
    parser.add_argument("--reps",type=int,default=10)
    parser.add_argument("--catalog",choices=("twohop","all"),default="twohop")
    parser.add_argument("--adaptive",action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.selftest:
            selftest()
        elif args.summary:
            summarize()
        else:
            experiment(args.graph,args.reps,args.catalog,args.adaptive)
