import math
from fractions import Fraction as Q
import unittest
from src.contracts import DecisionFrame, cutoff, can_skip, cold_key_threshold

B = ('graph/version', 's', 'catalog/filter')

class Cases(unittest.TestCase):
    def test_zero_stop_requires_full_coverage_and_budget_separate(self):
        f = DecisionFrame(B, (1,2), .25)
        f.observe(B, 1, -math.inf, .125)
        assert f.decide()[0] == 'unresolved'
        f.observe(B, 2, -math.inf, .25)
        assert f.decide() == ('tolerance_stop', None)
        assert f.decide(budget_remaining=0)[0] == 'budget_stop'
        assert DecisionFrame(B, (), 0).decide() == ('tolerance_stop',None)

    def test_threshold_neighbors_scaled_drift_and_monotonicity(self):
        f = DecisionFrame(B, (1,2), .25)
        f.observe(B,1,.5,.8)
        f.observe(B,1,.4,.9)
        assert f.lower == .5
        t=cutoff(f.lower,.25)
        assert t == .75 and can_skip(t,.5,.25)
        assert can_skip(math.nextafter(t,-math.inf),.5,.25)
        assert not can_skip(math.nextafter(t,math.inf),.5,.25)
        assert can_skip(.25,0,.25)
        for a,d,h in [(.1,.2,.3),(1.,2**-53,2**-54),(.125,.5,.875)]:
            ex=Q(a)+Q(d)
            assert Q(cutoff(a,d))<=ex
            k=cold_key_threshold(a,d,h,1<<50)
            assert k <= (ex-Q(h))*(1<<50) < k+1

    def test_tie_order_equality_and_exhaustion(self):
        f=DecisionFrame(B,(10,11,12),0)
        for i in (10,11,12): f.observe(B,i,.5, .6)
        assert f.decide()[0]=='unresolved'
        assert not can_skip(.5,.5,0,earlier=True)
        assert not can_skip(.5,.5,0,hidden=True)
        assert can_skip(.5,.5,0)
        assert f.decide(equal=lambda a,b: a in (10,11,12) and b in (10,11,12))==('accepted',10)
        for i in (10,11,12): f.observe(B,i,.5,.5)
        assert f.decide()==('accepted',10)

    def test_rebinding_revokes_old_exclusions(self):
        f=DecisionFrame(B,(1,2),.4)
        f.observe(B,1,.5,.51); f.observe(B,2,-math.inf,.8)
        assert f.decide()==('accepted',1)
        for b,ids,d in [(B,(2,),.4),(B,(1,2),.1),(('child','s','cat'),(1,2),.4),
                        (('version','other','cat'),(1,2),.4),(B,(1,2,3),.4)]:
            fresh=f.restart(b,ids,d)
            assert fresh.lower==0 and fresh.decide()[0]=='unresolved'
        with self.assertRaises(ValueError): f.observe(('wrong',),1,1,1)
        with self.assertRaises(ValueError): f.observe(B,2,.9,1.)

    def test_fixed_and_free_approximate_choices(self):
        q={0:{1:Q(5,8),2:Q(3,4)},1:{2:Q(1,4)},2:{1:Q(1,8)}}
        def step(node,order):
            f=DecisionFrame((node,), tuple(sorted(q[node])), .25)
            for i in order:
                g=float(q[node][i]); f.observe((node,),i,-math.inf,g)
            for i in order:
                g=float(q[node][i])
                if not can_skip(g,f.lower,.25): f.observe((node,),i,g,g)
            return f.decide()
        for order in ((1,2),(2,1)):
            result,i=step(0,order)
            assert result=='accepted' and q[0][i]>0 and q[0][i]+Q(1,4)>=max(q[0].values())
            assert step(i,tuple(q[i]))[0]=='tolerance_stop'
        assert step(0,(1,2))[1]!=step(0,(2,1))[1]

    def test_real_fj_fixed_and_free_routes(self):
        from src.contracts import screen_decision
        s=[Q(1),Q(0),Q(-1,8)]
        edits=((0,1),(0,2))
        def energy(active):
            a=[[Q(i==j) for j in range(3)]+[s[i]] for i in range(3)]
            for i,j in active:
                a[i][i]+=1; a[j][j]+=1; a[i][j]-=1; a[j][i]-=1
            for j in range(3):
                a[j]=[v/a[j][j] for v in a[j]]
                for i in range(3):
                    if i!=j: a[i]=[v-a[i][j]*w for v,w in zip(a[i],a[j])]
            return sum(si*row[-1] for si,row in zip(s,a))
        # Independently derived score via rank-one formula at the root.
        root_gain={e:(s[e[0]]-s[e[1]])**2/3 for e in edits}
        def enclosing(x):
            v=float(x); q=Q(v)
            return (math.nextafter(v,-math.inf) if q>x else v,
                    math.nextafter(v,math.inf) if q<x else v)
        def compare(active,reverse):
            ids=tuple(e for e in edits if e not in active)
            gains={e:energy(active)-energy((*active,e)) for e in ids}
            if not active: assert gains==root_gain
            result,_,vis=screen_decision((tuple(active),'s',ids,Q(1,8)),ids,.125,
                lambda e:enclosing(gains[e])[1],lambda e:enclosing(gains[e]),visits=ids[::-1] if reverse else ids)
            status,w=result
            if status=='accepted':
                assert gains[w]>0 and gains[w]+Q(1,8)>=max(gains.values())
            else: assert status=='tolerance_stop' and max(gains.values(),default=Q(0))<=Q(1,8)
            return w,vis
        # Fixed trajectory: both work orders certified at the same root and same forced child.
        assert compare([],False)[0]==edits[0] and compare([],True)[0]==edits[1]
        assert compare([edits[0]],False)==compare([edits[0]],True)
        # Free: follow each first choice on its own child; stop and gains may differ.
        free=[]
        for reverse in (False,True):
            a=[]; w,_=compare(a,reverse); a.append(w)
            w,_=compare(a,reverse)
            if w is not None: a.append(w)
            free.append((a,energy([])-energy(a)))
        assert free[0][0][0]!=free[1][0][0]
