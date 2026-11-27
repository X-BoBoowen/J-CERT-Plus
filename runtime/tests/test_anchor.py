from fractions import Fraction as Q
import unittest
from src.anchor import Graph, Edit, State, Anchor, GainUpper, tangent, choose_plan, Prices

class Cases(unittest.TestCase):
    def test_old_gain_rejected_and_both_radii_and_exact_gains(self):
        graph=Graph(2,())
        e=Edit(0,1,'insert',1.)
        old=State.certify(graph,(1.,-1.),(.99,-.98))
        new=State.certify(graph,(-1.,1.),(-.96,.98))
        a=Anchor(old,(e,))
        u=a.bounds(new,(e,))[e]
        assert old.radius>=.02 and new.radius>=.04
        assert u>=4 and Q(tangent(old,e).upper)>=4
        # I, s=(1,-1), weight=1: exact old gain=4/3 while tangent=4.
        with self.assertRaises(TypeError): Anchor.from_caps(old,(e,),{e:GainUpper(4/3)})
        assert Q(4,3)<4

    def test_branch_filter_expand_weight_flip(self):
        e=Edit(0,1,'insert',1.); f=Edit(1,2,'flip',.5)
        g=Graph(3,((1,2,-1,.5),))
        old=State.certify(g,(.5,-.75,.25),(.5,-.5,0.))
        a=Anchor(old,(e,f))
        # Sibling graph with changed edge and opinion: only current legal edits are admitted.
        branch=Graph(3,((0,2,1,.5),(1,2,-1,.5)))
        cur=State.certify(branch,(0.,.5,-.5),(0.,.25,-.25))
        changed=Edit(0,1,'insert',2.)
        vals=a.bounds(cur,(changed,f))
        assert vals[changed]==tangent(cur,changed).upper
        assert set(a.bounds(cur,(f,)))=={f}
        with self.assertRaises(ValueError): a.bounds(cur,(Edit(0,2,'insert',1.),))
        with self.assertRaises(ValueError): a.bounds(cur,(Edit(1,2,'flip',1.),))
        with self.assertRaises(ValueError): State.certify(g,(1.5,0.,0.),(0.,0.,0.))

    def test_frontier_prices_and_cold_coverage(self):
        graph=Graph(4,()); es=tuple(Edit(i,j,'insert',1.) for i in range(4) for j in range(i+1,4))
        s=(1.,-.5,.25,0.); old=State.certify(graph,s,s); a=Anchor(old,es)
        for new_s in [s,(-1.,1.,-1.,1.)]:
            cur=State.certify(graph,new_s,new_s)
            p=a.plan(cur,.5,.125)
            ids,cold=a.frontier(p)
            for e in set(es)-set(ids): assert cold>=tangent(cur,e).upper-1e-12 and cold<=.5
            assert p.postings==sum(len(a.postings[v]) for v in p.hot)
        p=a.plan(State.certify(graph,(-1.,1.,-1.,1.),(-1.,1.,-1.,1.)),.5,.125)
        assert choose_plan('adaptive',p,len(es),Prices(1,10,10,10))=='rebuild'
        assert choose_plan('retain-always',p,len(es),Prices(1,10,10,10))=='retain'

    def test_old_gain_fails_on_an_actual_zero_drift_branch(self):
        e=Edit(0,1,'insert',1.)
        z=(.25,-.25,.25)
        g0=Graph(3,()); g1=Graph(3,((0,2,1,1.),))
        a=Anchor(State.certify(g0,z,z),(e,))
        new=State.certify(g1,z,z)
        # For g1, A^-1[0,0]=2/3; cross response between endpoints is zero.
        old_gain=Q(1,4)/(1+Q(2))
        new_gain=Q(1,4)/(1+Q(5,3))
        assert new.radius==0 and a.drift(new)==(0.,0.,0.)
        assert old_gain==Q(1,12)<new_gain==Q(3,32)<=Q(a.bounds(new,(e,))[e])
        with self.assertRaises(TypeError):
            Anchor.from_caps(a.state,(e,),{e:GainUpper(float(old_gain))})

    def test_rational_gain_reference_independent_of_response_formula(self):
        from src.prototype_bench import MatchingOracle, catalog, direct_gain
        g=Graph(4,((0,1,-1,1.),(2,3,1,1.)))
        old=MatchingOracle(g,(.25,-.5,-.25,1.))
        cur=MatchingOracle(g,(-.25,.5,1.,-.125))
        es=catalog(g)
        a=Anchor(old.state,es)
        bounds=a.bounds(cur.state,es)
        for e in es:
            ref=direct_gain(g,cur.state.opinions,e)
            assert ref==cur.score(e) and Q(bounds[e])>=ref and Q(tangent(cur.state,e).upper)>=ref

    def test_spd_counterexample_exact(self):
        # (A-I) has positive diagonal and positive determinant.
        assert Q(17,23)*Q(57,23)-Q(30,23)**2>0
        p,q=Q(11,10),Q(7,10)
        assert (40*p-30*q)/23==1 and (-30*p+80*q)/23==1 and p>1

    def test_no_prior_winner_can_restart_and_empty_catalog_stops(self):
        from unittest.mock import patch
        from src.prototype_bench import anchor_query, catalog
        g=Graph(4,((0,1,-1,1.),(2,3,1,1.)))
        with patch('src.prototype_bench.clock',return_value=0.):
            for es in (catalog(g),()):
                a=w=None; prices=Prices(0,0,0,0)
                for _ in range(2):
                    a,w,prices,row=anchor_query(g,es,(0.,)*4,a,w,'retain-always',prices)
                    assert w is None and row['gain']=='0'
