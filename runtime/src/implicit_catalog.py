"""A fixed-version implicit all-nonedges tangent frontier, with sparse exclusions."""
from bisect import bisect_left
import heapq
import math

def up(v):
    return math.nextafter(v, math.inf)

def pair_cap(a,b,r,w):
    """Same monotone evaluation order across every row, including equal centers."""
    width = up(up(abs(b-a)) + up(r+r))
    return up(w*up(width*width))

class ImplicitNonedges:
    def __init__(self, centers,r,weight,initial_edges=(),committed=()):
        self.x=tuple(centers)
        if not all(math.isfinite(v) for v in self.x) or not math.isfinite(r) or r<0 or not math.isfinite(weight) or weight<=0:
            raise ValueError("one finite center per vertex, common radius >=0 and weight >0 required")
        self.r,self.weight = r,weight
        self.order=sorted(range(len(self.x)),key=lambda i:(self.x[i],i))
        blocked=[]
        for edges in (initial_edges,committed):
            for i,j in edges:
                if not (0<=i<len(self.x) and 0<=j<len(self.x)) or i==j:
                    raise ValueError("invalid excluded pair")
                blocked.append(tuple(sorted((i,j))))
        self.blocked=tuple(sorted(blocked))
        self.checked=self.rejected=self.popped=0
        n=len(self.order)
        self.heap=[self._entry(i,n-1) for i in range(n-1)]
        heapq.heapify(self.heap)

    def _entry(self,i,j):
        a,b=self.order[i],self.order[j]
        return -pair_cap(self.x[a],self.x[b],self.r,self.weight),i,j

    @property
    def tail(self):
        # May include a blocked row head. Keeping it only weakens the certificate.
        return -self.heap[0][0] if self.heap else -math.inf

    def covered(self,cutoff,*,exact):
        return self.tail < cutoff if exact else self.tail <= cutoff

    def pop(self):
        while self.heap:
            neg,i,j=heapq.heappop(self.heap)
            self.popped+=1
            if j>i+1:
                heapq.heappush(self.heap,self._entry(i,j-1))
            pair=tuple(sorted((self.order[i],self.order[j])))
            k=bisect_left(self.blocked,pair)
            if k<len(self.blocked) and self.blocked[k]==pair:
                self.rejected+=1
                continue
            self.checked+=1
            return pair,-neg
        return None
