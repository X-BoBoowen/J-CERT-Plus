"""Dyadic directional certificates for legal unit-sign repair and addition pivots."""
from fractions import Fraction
from math import isqrt


def _box(interval, unit):
    lo,hi = interval
    if lo>hi:
        raise ArithmeticError("empty symmetric pivot intersection")
    low = lo.numerator*unit//lo.denominator
    high = -((-hi.numerator*unit)//hi.denominator)
    center = (low+high)//2
    return center,high-center


def _nearest(num,den):
    if den<0:
        num,den = -num,-den
    return (2*num+den)//(2*den)


def directional_block(g,rhs_rows,unit):
    """Enclose H (J+G)^-1 H.T entrywise with Theorem 4B.

    Rows of H and entries of G are input intervals. The caller supplies a
    legal graph action, for which ||(J+G)^-1||_2 <= 3: signed Woodbury gives
    J - J U.T A_child^-1 U J and ||U||^2 <= 2*lambda_min(A_child).
    The graph bound remains valid for every true pivot represented by the
    caller, including a candidate whose approximate midpoint is singular.
    """
    d = len(g)
    if d not in (1,2) or any(len(row)!=d for row in (*g,*rhs_rows)) or unit<=0:
        raise ValueError("expected one- or two-column graph pivot and matching directions")
    b,db = [],[]
    for i in range(d):
        mids,radii = [],[]
        for j in range(d):
            lo = max(g[i][j][0],g[j][i][0])
            hi = min(g[i][j][1],g[j][i][1])
            midpoint,radius = _box((lo,hi),unit)
            if i==j:
                midpoint += unit if i==0 else -unit
            mids.append(midpoint)
            radii.append(radius)
        b.append(mids)
        db.append(radii)
    det = b[0][0] if d==1 else b[0][0]*b[1][1]-b[0][1]**2
    directions = []
    for row in rhs_rows:
        h,dh = zip(*(_box(interval,unit) for interval in row))
        if det==0:
            p = [0]*d
        elif d==1:
            p = [_nearest(h[0]*unit,det)]
        else:
            p = [_nearest(unit*(b[1][1]*h[0]-b[0][1]*h[1]),det),
                 _nearest(unit*(b[0][0]*h[1]-b[0][1]*h[0]),det)]
        # p/unit is an arbitrary trial. Bound each component of h - B p
        # in units of unit^-2, then round its Euclidean norm upward.
        bp = [sum(v*w for v,w in zip(br,p)) for br in b]
        dbp = [sum(v*abs(w) for v,w in zip(dr,p)) for dr in db]
        err2 = sum((abs(h[k]*unit-bp[k])+dh[k]*unit+dbp[k])**2 for k in range(d))
        norm = 1+isqrt(err2-1) if err2 else 0
        directions.append((h,dh,p,bp,dbp,norm))
    den = unit**2
    result = [[None]*len(directions) for _ in directions]
    for i,(h,dh,p,bp,dbp,norm) in enumerate(directions):
        for j in range(i+1):
            hj,dhj,pj,_,_,normj = directions[j]
            # The exact center is H_i P_j + P_i H_j - P_i B P_j.
            # Linear input uncertainty and the quadratic residual remainder
            # are bounded separately, using integer numerators throughout.
            center = sum((h[k]*pj[k]+p[k]*hj[k])*unit-bp[k]*pj[k] for k in range(d))
            error = sum((dh[k]*abs(pj[k])+abs(p[k])*dhj[k])*unit
                        +dbp[k]*abs(pj[k]) for k in range(d))
            low = ((center-error)*unit-3*norm*normj)//den
            high = -((-((center+error)*unit+3*norm*normj))//den)
            result[i][j] = result[j][i] = Fraction(low,den),Fraction(high,den)
    return result


def selftest():
    """Compare to exact inverses at every vertex of small uncertain boxes."""
    from itertools import product
    q = Fraction
    for rank in (1,2):
        intervals = [(q(1,5),q(1,4)),(q(-1,9),q(1,10)),(q(1,6),q(1,5))]
        g = [[intervals[0]]] if rank==1 else [
            [intervals[0],intervals[1]],[intervals[1],intervals[2]]]
        # All these real symmetric pivots have inverse norm below three
        # by strict diagonal dominance with row margin greater than 1/3.
        rows = [[(q(-1,7),q(2,9)),(q(1,6),q(1,4))][:rank],
                [(q(1,5),q(1,3)),(q(-1,4),q(-1,7))][:rank]]
        boxed = directional_block(g,rows,1<<16)
        unique = intervals[:1] if rank==1 else intervals
        for values in product(*unique,*rows[0],*rows[1]):
            a = 1+values[0]
            if rank==1:
                inv = [[1/a]]
            else:
                b,c = values[1],values[2]-1
                det = a*c-b*b
                inv = [[c/det,-b/det],[-b/det,a/det]]
            h = values[len(unique):]
            for i,j in product(range(2),repeat=2):
                exact = sum(h[i*rank+k]*inv[k][l]*h[j*rank+l]
                            for k,l in product(range(rank),repeat=2))
                lo,hi = boxed[i][j]
                assert lo<=exact<=hi
        singleton_g = [[(v[0],v[0]) for v in row] for row in g]
        zero = [[(0,0)]*rank]
        assert directional_block(singleton_g,zero,128)==[[(0,0)]]


if __name__=="__main__":
    selftest()
