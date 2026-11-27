from fractions import Fraction as Q
import math
from itertools import combinations
import unittest
from src.implicit_catalog import ImplicitNonedges, pair_cap

class Cases(unittest.TestCase):
    def test_exhaustive_grid(self):
        for centers,r in [((0.,0.,0.,0.),0.),((0.,.75,-.5,.125),0.),((0.,.75,-.5,.125),.0625)]:
            for blocked in [(),((0,3),(1,2)),tuple(combinations(range(4),2))]:
                with self.subTest(centers=centers,r=r,blocked=blocked):
                    self.check_grid(centers,r,blocked)

    def check_grid(self,centers,r,blocked):
        h=ImplicitNonedges(centers,r,1.5,blocked[:1],blocked[1:])
        remaining=set(combinations(range(4),2))-set(blocked)
        seen=set()
        while remaining:
            assert h.tail>=max(pair_cap(centers[i],centers[j],r,1.5) for i,j in remaining)
            pair,bound=h.pop()
            assert bound==max(pair_cap(centers[i],centers[j],r,1.5) for i,j in remaining)
            assert pair in remaining and pair not in seen
            i,j=pair
            assert Q(bound)>=Q(1.5)*(abs(Q(centers[i])-Q(centers[j]))+2*Q(r))**2
            seen.add(pair); remaining.remove(pair)
            assert len(h.heap)<=len(centers)
        assert h.pop() is None and h.tail==-math.inf
        assert h.checked==len(seen) and h.rejected==len(blocked)

    def test_cutoff_boundaries_and_explicit_flip_union(self):
        h=ImplicitNonedges((1.,1.,-1.),0.,1.,[(0,2)],[])
        p,u=h.pop()
        assert p==(1,2)
        assert max(h.tail,3.)>=3.
        assert not h.covered(math.nextafter(h.tail,-math.inf),exact=False)
        assert h.covered(h.tail,exact=False) and not h.covered(h.tail,exact=True)

    def test_bad_center_radius_or_pair(self):
        for args in [((1.,math.nan),0.,1.,(),()),((0.,),-.1,1.,(),()),((0.,1.),0.,1.,((1,1),),())]:
            with self.assertRaises(ValueError): ImplicitNonedges(*args)

    def test_hidden_earlier_tie_survives_tail(self):
        centers=(1.,-1.,-1.,1.)
        h=ImplicitNonedges(centers,0.,1.)
        u=h.tail; rivals=[]
        while not h.covered(u,exact=True):
            pair,key=h.pop()
            if key==u: rivals.append(pair)
        assert len(rivals)==4 and min(rivals)==(0,1)

    def test_candidate_specific_corrections_can_break_row_order(self):
        # At fixed left endpoint 0, the middle pair has a larger private correction.
        assert pair_cap(0.,.5,1.,1.)>pair_cap(0.,1.,0.,1.)
