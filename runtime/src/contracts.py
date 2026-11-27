"""Finite-catalog decision evidence. IDs are in declared tie order."""
from fractions import Fraction
import math

def cutoff(lower, delta):
    """Greatest binary64 <= exact sum of the two nonnegative binary64 inputs."""
    if not all(math.isfinite(v) and v >= 0 for v in (lower, delta)):
        raise ValueError("nonnegative finite lower bound and tolerance required")
    total = Fraction(lower) + Fraction(delta)
    value = lower + delta
    if math.isinf(value) or Fraction(value) > total:
        return math.nextafter(value, -math.inf)
    return value

def can_skip(upper, lower, delta, *, earlier=False, hidden=False):
    floor = cutoff(lower, delta)
    if delta == 0 and lower > 0 and (earlier or hidden):
        return upper < floor
    return upper <= floor

def cold_key_threshold(lower, delta, drift, scale):
    """For non-strict pruning, the largest integer key k with k/scale+D <= L+delta."""
    cutoff(lower, delta)
    if type(scale) is not int or scale <= 0 or not math.isfinite(drift) or drift < 0:
        raise ValueError("invalid scale or drift upper bound")
    return math.floor((Fraction(lower) + Fraction(delta) - Fraction(drift))*scale)

class DecisionFrame:
    """An external certifier supplies intervals bound to this immutable decision context."""
    def __init__(self, binding, ids, delta):
        cutoff(0.,delta)
        self._binding, self._delta = binding, delta
        self._ids = tuple(ids)
        self._bounds = dict.fromkeys(self._ids, (-math.inf, math.inf))
        if len(self._bounds) != len(self._ids):
            raise ValueError("duplicate candidate identity")
        self._lower = 0.

    @property
    def lower(self):
        return self._lower

    def observe(self, binding, edge, lo, hi):
        if binding != self._binding or edge not in self._bounds:
            raise ValueError("evidence from another decision or unknown candidate")
        old_lo, old_hi = self._bounds[edge]
        if math.isnan(lo) or math.isnan(hi) or lo == math.inf or hi == -math.inf:
            raise ValueError("invalid certificate")
        lo, hi = max(old_lo,lo), min(old_hi,hi)
        if lo > hi:
            raise ValueError("inconsistent same-context intervals")
        self._bounds[edge] = lo, hi
        self._lower = max(self._lower, lo)

    def restart(self, binding, ids, delta):
        return DecisionFrame(binding,ids,delta)

    def decide(self, *, budget_remaining=1, equal=None):
        """equal(a,b) is an independently verified current-version equality oracle."""
        if type(budget_remaining) is not int or budget_remaining < 0:
            raise ValueError("invalid remaining budget")
        if budget_remaining == 0:
            return 'budget_stop',None
        # Equal gains can transfer their lower evidence to an earlier candidate.
        ranks=range(len(self._ids))
        if self._delta>0:
            ranks=sorted(ranks,key=lambda i:-self._bounds[self._ids[i]][0])
        for rank in ranks:
            a=self._ids[rank]
            lo = self._bounds[a][0]
            if equal and self._delta == 0:
                lo = max([lo]+[self._bounds[e][0] for e in self._ids if e!=a and equal(a,e)])
            if lo <= 0:
                continue
            for j,e in enumerate(self._ids):
                if e == a or can_skip(self._bounds[e][1], lo, self._delta, earlier=j<rank):
                    continue
                if self._delta == 0 and j > rank and equal and equal(a,e):
                    continue
                break
            else:
                return 'accepted',a
        if all(hi <= self._delta for _,hi in self._bounds.values()):
            return 'tolerance_stop',None
        return 'unresolved',None

def screen_decision(binding,ids,delta,upper,score,*,visits=None,budget_remaining=1,equal=None):
    """Finite explicit search. upper/score must certify this exact bound context.

    visits may change work order only. Unvisited, omitted or duplicate IDs cannot
    silently change the catalog. Every skipped contender retains its current U.
    """
    ids=tuple(ids)
    frame=DecisionFrame(binding,ids,delta)
    if budget_remaining==0:
        return frame.decide(budget_remaining=0),frame,()
    ranks={e:i for i,e in enumerate(ids)}
    visits=ids if visits is None else tuple(visits)
    if len(visits)!=len(ids) or set(visits)!=set(ids):
        raise ValueError("visits must permute the legal catalog")
    incumbent=None; evaluated=[]
    for e in visits:
        hi=upper(e)
        frame.observe(binding,e,-math.inf,hi)
        earlier=incumbent is not None and ranks[e]<ranks[incumbent]
        if can_skip(hi,frame.lower,delta,earlier=earlier):
            continue
        lo,hi=score(e)
        previous=frame.lower
        frame.observe(binding,e,lo,hi)
        if lo>previous or (lo==previous and lo>0 and earlier):
            incumbent=e
        evaluated.append(e)
    return frame.decide(budget_remaining=budget_remaining,equal=equal),frame,tuple(evaluated)
