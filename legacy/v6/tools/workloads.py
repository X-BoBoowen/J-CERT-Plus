"""Portable signed-graph workloads; run with --build to regenerate the six matrices."""
import argparse
import csv
import gzip
import io
from pathlib import Path
import urllib.request

# Fix numeric concurrency for the published fixture-construction timings.
if __name__ == "__main__":
    import sys
    if any(flag in sys.argv for flag in ("--matched-setup", "--clustered-setup", "--clustered-witness", "--persistent-setup", "--persistent-witness")):
        import os
        for thread_option in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[thread_option] = "1"

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import cg

DATA = Path(__file__).resolve().parents[1] / "datasets"
NAMES = ("Alpha", "OTC", "Slashdot", "Conflict", "Bio", "Fin")
SOURCES = dict(Alpha="soc-sign-bitcoinalpha.csv.gz", OTC="soc-sign-bitcoinotc.csv.gz",
               Slashdot="soc-sign-Slashdot090221.txt.gz")


def _observed(name):
    url = "https://snap.stanford.edu/data/" + SOURCES[name]
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = io.BytesIO(gzip.decompress(response.read()))
    rows = np.loadtxt(raw, delimiter=None if name == "Slashdot" else ",",
                      usecols=(0, 1, 2), dtype=int)
    vertices, indices = np.unique(rows[:, :2], return_inverse=True)
    i, j = indices.reshape(-1, 2).T
    W = sparse.coo_matrix((np.sign(rows[:, 2]), (i, j)),
                          shape=(len(vertices), len(vertices))).tocsr()
    W = W + W.T
    W.setdiag(0)
    W.eliminate_zeros()
    W.data = np.sign(W.data).astype(np.int8)
    W.sort_indices()
    return W


def _synthetic(name):
    rng = np.random.default_rng(2819 + NAMES.index(name))
    n = {"Conflict": 768, "Bio": 1536, "Fin": 1024}[name]
    edges = {}
    for j in range(1, n):
        edges[j-1, j] = 1
    if name == "Conflict":
        # Overlapping antagonistic triads in a banded interaction network.
        for i in range(n):
            for j in range(i + 1, min(i + 9, n)):
                edges[i, j] = -1 if rng.random() < 0.52 else 1
    elif name == "Bio":
        # Inhibitory/activating interactions along a sparse cascade.
        for i in range(n - 1):
            edges[i, i+1] = -1 if rng.random() < 0.28 else 1
            if i + 2 < n and rng.random() < 0.18:
                edges[i, i+2] = -1 if rng.random() < 0.35 else 1
    else:
        # Sixteen activity sectors, positive within and antagonistic across.
        for i in range(n):
            for j in range(i+1, n):
                inside = i // 64 == j // 64
                if rng.random() < (0.16 if inside else 0.0012):
                    edges[i, j] = -1 if rng.random() < (0.07 if inside else 0.84) else 1
    i, j = np.array(list(edges)).T
    W = sparse.coo_matrix((np.array(list(edges.values()), dtype=np.int8), (i, j)),
                          shape=(n, n)).tocsr()
    W = W + W.T
    W.sort_indices()
    return W


def load_graph(name, n=None):
    """Full signed CSR or the induced subgraph on the first n BFS discoveries."""
    if name not in NAMES:
        raise ValueError(name)
    W = sparse.load_npz(DATA / (name + ".npz")).tocsr()
    if n is None:
        return W
    if n < 1 or n > W.shape[0]:
        raise ValueError("invalid restricted size")
    degree = np.diff(W.indptr)
    root = int(degree.argmax())
    order = csgraph.breadth_first_order(W, root, directed=False, return_predecessors=False)
    if len(order) < n:
        raise ValueError("restriction exceeds root component size")
    idx = order[:n]
    result = W[idx][:, idx].tocsr()
    result.sort_indices()
    return result


def system(W, alpha=1.0):
    degree = np.asarray(abs(W).sum(axis=1)).ravel()
    return (sparse.diags(1 + alpha * degree) - alpha * W).tocsr()


def opinions(W, model, seed=0):
    """Signed diffusion, neighborhood camps, conflict anchors and controls."""
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    U = abs(W).astype(float)
    degree = np.asarray(U.sum(axis=1)).ravel()
    noise = rng.uniform(-1, 1, n)
    if model == "unsigned_control":
        s = noise.copy()
        for _ in range(3):
            s = 0.5*s + 0.5*(U @ s) / np.maximum(degree, 1)
        return s
    def signed_walk(x):
        return (W @ x) / np.maximum(degree, 1)

    if model == "S1":
        s = noise.copy()
        for _ in range(3):
            s = 0.5 * (s - signed_walk(s))
        return np.clip(s, -1, 1)
    if model == "S2":
        field = noise - signed_walk(np.sign(noise))
        return 0.98 * np.where(field >= 0, 1., -1.) + 0.02 * noise
    if model == "planted":
        camps = np.where(noise - signed_walk(np.sign(noise)) >= 0, 1., -1.)
        target = 0.999 * camps + 0.001 * noise
        rhs = system(W) @ target
        return rhs / np.max(np.abs(rhs))
    if model == "S3":
        negatives = np.asarray((W < 0).sum(axis=1)).ravel()
        rank = np.lexsort((rng.random(n), -negatives))
        anchors = rank[:max(2, int(np.ceil(0.02*n)))]
        h = np.zeros(n)
        h[anchors] = np.where(np.arange(len(anchors)) % 2, -1., 1.)
        field = h - signed_walk(h)
        return 0.98 * np.sign(np.where(field == 0, noise, field)) + 0.02 * noise
    raise ValueError(model)


def candidates(W, m, seed=0, alpha=1.0):
    """Uniform flips and uniform nonedge additions, without replacement."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    upper = sparse.triu(W, k=1, format="coo")
    negative = np.flatnonzero(upper.data < 0)
    nonedges = n*(n-1)//2 - upper.nnz
    if not 0 <= m <= len(negative) + nonedges:
        raise ValueError("requested pool exceeds legal edits")
    k = max(m-nonedges, min(len(negative), m//2))
    edits = [(int(upper.row[t]), int(upper.col[t]), float(-alpha*upper.data[t]), True)
             for t in rng.choice(negative, k, replace=False)]
    additions = set()
    if m-k > nonedges//4:
        pairs = [(i,j) for i in range(n) for j in range(i+1,n) if W[i,j] == 0]
        additions.update(pairs[t] for t in rng.choice(len(pairs), m-k, replace=False))
    else:
        while len(additions) < m-k:
            i, j = sorted(map(int, rng.integers(n, size=2)))
            if i < j and W[i,j] == 0:
                additions.add((i,j))
    edits.extend((i,j,float(alpha),False) for i,j in sorted(additions))
    rng.shuffle(edits)
    return edits


def contention_candidates(W, m, s, seed=17, alpha=1.0):
    """Uniform opposite-camp catalog, balanced by edit type when feasible."""
    rng = np.random.default_rng(seed)
    positive = np.asarray(s) >= 0
    plus, minus = np.flatnonzero(positive), np.flatnonzero(~positive)
    upper = sparse.triu(W, k=1, format="coo")
    crossing = positive[upper.row] != positive[upper.col]
    flips = np.flatnonzero(crossing & (upper.data < 0))
    available = len(plus) * len(minus) - np.count_nonzero(crossing)
    if not 0 <= m <= available + len(flips):
        raise ValueError("requested pool exceeds legal cross-camp edits")
    k = max(m - available, min(len(flips), m // 2))
    edits = [(int(upper.row[t]), int(upper.col[t]), float(-alpha*upper.data[t]), True)
             for t in rng.choice(flips, k, replace=False)]
    additions = set()
    while len(additions) < m-k:
        i, j = sorted((int(rng.choice(plus)), int(rng.choice(minus))))
        if W[i, j] == 0:
            additions.add((i, j))
    edits.extend((i, j, float(alpha), False) for i, j in sorted(additions))
    rng.shuffle(edits)
    return edits

def calibrated_candidates(W, m, s, seed=17, alpha=1.0):
    """Match addition upper bounds to a common conflict scale at the root."""
    edits = contention_candidates(W, m, s, seed, alpha)
    A = system(W, alpha)
    D = A.diagonal()
    z, info = cg(A, s, M=sparse.diags(1/D), rtol=1e-12, atol=0)
    if info:
        raise ArithmeticError("root calibration solve failed")
    i, j, w, flip = map(np.asarray, zip(*edits))
    diff = z[i] - z[j]
    pair = z[i] + z[j]
    x, y = D[i], D[j]
    v, u = w*(diff-pair), w*(-diff-pair)
    upper = w*(diff**2-pair**2)-(y*v*v+2*w*v*u+x*u*u)/(x*y-w*w)
    resistance = 1/x + 1/y
    feasible = diff**2/(resistance+1/(4*alpha))
    target = np.quantile(feasible[~flip], 0.72)
    if np.any(flip):
        target = min(target, max(0.05*target, np.quantile(upper[flip], 0.99)))
    for t in np.flatnonzero(~flip):
        score = min(target, feasible[t])
        if score > 0:
            weight = score / (diff[t]**2-score*resistance[t])
            edits[t] = (int(i[t]), int(j[t]), float(max(1, round(weight * 1024)) / 1024), False)
    return edits

def matched_inputs(W, m, seed=17):
    """Unit edits with 64 topology-selected disjoint, root-calibrated dyads."""
    if m < 64:
        raise ValueError("matched pool requires 64 frontier edits")
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    used = np.zeros(n, dtype=bool)
    edges = sparse.triu(W, k=1, format="coo")
    frontier = []
    for t in rng.permutation(np.flatnonzero(edges.data == -1)):
        i, j = int(edges.row[t]), int(edges.col[t])
        if not (used[i] or used[j]):
            used[[i, j]] = True
            frontier.append((i, j, 1., True))
            if len(frontier) == 32:
                break
    if len(frontier) < 32:
        raise ValueError("insufficient disjoint negative edges")
    centers = np.flatnonzero(np.diff(W.indptr) >= 2)
    for _ in range(100000):
        v = rng.choice(centers)
        neighbors = W.indices[W.indptr[v]:W.indptr[v+1]]
        i, j = sorted(map(int, rng.choice(neighbors, 2, replace=False)))
        if not (used[i] or used[j]) and W[i,j] == 0:
            used[[i,j]] = True
            frontier.append((i, j, 1., False))
            if len(frontier) == 64:
                break
    if len(frontier) < 64:
        raise ValueError("insufficient disjoint two-hop additions")
    A = system(W)
    pre = sparse.diags(1 / A.diagonal())
    x = rng.uniform(-0.005, 0.005, n)
    for i, j, _, flip in frontier:
        U = np.array([[1.,1.],[-1.,1.]]) if flip else np.array([[1.],[-1.]])
        gram = np.empty((U.shape[1], U.shape[1]))
        for k in range(U.shape[1]):
            rhs = np.zeros(n)
            rhs[[i,j]] = U[:,k]
            response, status = cg(A, rhs, M=pre, rtol=1e-11, atol=0)
            if status:
                raise ArithmeticError("calibration CG failed")
            gram[:,k] = U.T @ response[[i,j]]
        gram = (gram + gram.T)/2
        signed_inverse = np.diag([1.,-1.] if flip else [1.])
        h = U.T @ np.array([1.,-1.])
        coeff = h @ np.linalg.solve(signed_inverse + gram, h)
        amplitude = rng.uniform(.999,1.001) / np.sqrt(coeff)
        x[i], x[j] = amplitude, -amplitude
    s = A @ x
    s = np.rint((s / np.max(np.abs(s))) * 2**30) / 2**30
    tail_rng = np.random.default_rng(seed + 1)
    tail_size = m - len(frontier)
    eligible = np.flatnonzero((edges.data < 0) & ~used[edges.row] & ~used[edges.col])
    k = min(len(eligible), tail_size // 2)
    neg_order = tail_rng.permutation(eligible)
    need = tail_size - k
    additions = set()
    # Sampling wedges avoids constructing the full two-hop matrix on Slashdot.
    for _ in range(40 * need):
        if len(additions) == need:
            break
        v = tail_rng.choice(centers)
        nb = W.indices[W.indptr[v]:W.indptr[v+1]]
        i, j = sorted(map(int, tail_rng.choice(nb, 2, replace=False)))
        if not (used[i] or used[j]) and W[i,j] == 0:
            additions.add((i,j))
    if len(additions) < need:
        available = set()
        for v in centers:
            nb = W.indices[W.indptr[v]:W.indptr[v+1]]
            nb = nb[~used[nb]]
            for pos, i in enumerate(nb):
                for j in nb[pos+1:]:
                    if W[i,j] == 0:
                        available.add((int(i),int(j)))
        remaining = sorted(available.difference(additions))
        count = min(len(remaining), need-len(additions))
        additions.update(remaining[t] for t in tail_rng.choice(len(remaining),count,replace=False))
        k = min(len(eligible), tail_size-len(additions))
    # Bio at 2500 exhausts both legal two-hop additions and observed flips.
    vertices = np.flatnonzero(~used)
    while len(additions) + k < tail_size:
        i, j = sorted(map(int, tail_rng.choice(vertices, 2, replace=False)))
        if W[i,j] == 0:
            additions.add((i,j))
    edits = frontier + [(int(edges.row[t]),int(edges.col[t]),1.,True) for t in neg_order[:k]]
    edits.extend((i,j,1.,False) for i,j in sorted(additions))
    rng.shuffle(edits)
    return s, edits

def _motif_field(A, pre, D, a, b, c):
    n = A.shape[0]
    motif = [a, b, c]
    G = np.empty((3, 3))
    for t, v in enumerate(motif):
        rhs = np.zeros(n)
        rhs[v] = 1.
        y, info = cg(A, rhs, M=pre, rtol=1e-11, atol=0)
        if info:
            raise ArithmeticError("motif calibration solve failed")
        G[:, t] = y[motif]
    G = (G + G.T)/2
    inv_delta = np.array([[0., -.5], [-.5, 0.]])
    ab, ac = [0, 1], [0, 2]
    H = np.linalg.inv(inv_delta + G[np.ix_(ab, ab)])
    J = np.linalg.inv(inv_delta + G[np.ix_(ac, ac)])
    coupling = G[np.ix_(ac, ab)]
    J1 = np.linalg.inv(inv_delta + G[np.ix_(ac, ac)] - coupling @ H @ coupling.T)
    # Fixed coefficient grid; optimize a gain margin before catalog generation.
    az, cz = np.meshgrid(np.arange(1, 97)/(32*D[a]), np.arange(1, 97)/64)
    abx = np.stack((az, -np.ones_like(az)), axis=-1)
    acx = np.stack((az, cz), axis=-1)
    updated = acx - abx @ H @ coupling.T
    def quad(X, Q):
        return np.sum((X @ Q)*X, axis=-1)
    g1, g0, g2 = quad(abx, H), quad(acx, J), quad(updated, J1)
    margin = np.minimum.reduce((g1, -g0, g2))
    cell = np.unravel_index(np.argmax(margin), margin.shape)
    if margin[cell] <= 0:
        raise ValueError("no root-negative revival on the fixed motif grid")
    return np.array([az[cell], -1., cz[cell]]), margin[cell], min(g1[cell], g2[cell])


def clustered_inputs(W, m, seed=17):
    """Unit edits on overlapping stars and a three-vertex sign-revival motif."""
    return _cluster_family(W, m, seed, persistent=False)


def persistent_inputs(W, m, seed=17):
    """Four calibrated adjacent-flip pairs and a sustained two-hop/flip frontier."""
    return _cluster_family(W, m, seed, persistent=True)


def _cluster_family(W, m, seed, persistent):
    if persistent and m not in (1000, 2500):
        raise ValueError("persistent catalog requires M=1000 or 2500")
    if not 100 <= m <= 2500:
        raise ValueError("clustered catalog requires 100 <= m <= 2500")
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    A = system(W)
    D = A.diagonal()
    neg = (W < 0).tocsr()
    nd = np.diff(neg.indptr)
    centers = np.flatnonzero(nd >= 2)
    used = np.zeros(n, dtype=bool)
    def edit(i, j, flip):
        return min(i,j), max(i,j), 1., flip
    edits = []
    pre = sparse.diags(1/D)
    x = None
    for a in sorted(centers, key=lambda v: (D[v], nd[v], v)):
        if used[a]:
            continue
        neighbors = sorted((v for v in neg[a].indices if not used[v]),
                           key=lambda v: (D[v], v))
        if len(neighbors) < 2:
            continue
        b, c = neighbors[:2]
        motif = [int(a), int(b), int(c)]
        field, margin, gain = _motif_field(A, pre, D, *motif)
        if x is None:
            x = rng.uniform(-.0005, .0005, n)*np.sqrt(margin)
            common_gain = gain
        x[motif] = field * np.sqrt(common_gain/gain)
        used[motif] = True
        for v in motif:
            used[W[v].indices] = True
        edits.extend((edit(*motif[:2],True), edit(motif[0],motif[2],True)))
        if len(edits) == (8 if persistent else 2):
            break
    if len(edits) != (8 if persistent else 2):
        raise ValueError("insufficient separate revival motifs")
    edges = sparse.triu(neg, k=1, format="coo")
    # Each star couples one negative flip and two positive two-hop insertions.
    vertices = rng.permutation(n)
    vertex_rank = np.empty(n, dtype=int)
    vertex_rank[vertices] = np.arange(n)
    def two_hop(h):
        neighbors = W.indices[W.indptr[h]:W.indptr[h+1]]
        if not len(neighbors):
            return np.empty(0, dtype=int)
        reached = np.unique(np.concatenate([
            W.indices[W.indptr[v]:W.indptr[v+1]] for v in neighbors]))
        return np.setdiff1d(reached, np.append(neighbors, h))
    scale = np.sqrt(common_gain*.15)
    order = rng.permutation(edges.nnz)
    frontier_size = 8 + 3*76 if persistent else 98
    star_order = order
    if persistent:
        maximum = np.maximum(D[edges.row[order]], D[edges.col[order]])
        star_order = order[np.argsort(maximum > 60, kind="stable")]
    for t in star_order:
        i, j = int(edges.row[t]), int(edges.col[t])
        if used[i] or used[j]:
            continue
        if D[j] > D[i]:
            i, j = j, i
        star = (len(edits)-8)//3
        possible = two_hop(i)
        possible = possible[~used[possible]]
        if len(possible) < 2:
            continue
        leaves = possible[np.argsort(vertex_rank[possible])[:2]].tolist()
        used[[i,j,*leaves]] = True
        # Diagonal resistance approximation sets a common initial gain scale.
        x[i] = scale*np.sqrt(1+1/D[i]+1/D[j])/2
        x[j] = -x[i]
        if persistent and star < 36:
            # Give overlapping alternatives positive root gains below the main
            # band and calibrate their endpoint screen above that band.
            G = np.empty((2,2))
            for col,v in enumerate((i,j)):
                rhs = np.zeros(n)
                rhs[v] = 1.
                response,status = cg(A,rhs,M=pre,rtol=1e-11,atol=0)
                if status:
                    raise ArithmeticError("flip calibration solve failed")
                G[:,col] = response[[i,j]]
            delta = np.array([[0.,-2.],[-2.,0.]])
            H = np.linalg.inv(delta/4 + (G+G.T)/2)
            cap = delta - delta @ np.linalg.inv(
                [[D[i],-1],[-1,D[j]]]) @ delta
            Q = H - (.75/1.65)*cap
            slopes = np.roots([Q[1,1],2*Q[0,1],Q[0,0]])
            rays = [np.array([1.,r.real]) for r in slopes if abs(r.imag)<1e-10]
            rays = [v*np.sqrt(.75/(v@H@v)) for v in rays if v@H@v>0]
            v = min(rays,key=np.linalg.norm)
            x[[i,j]] = scale*v
        if persistent and star >= 36:
            # A descending band retains real, overlapping candidates as the
            # stronger alternatives are committed.
            band = np.sqrt(1.25 - .25*(star-36)/39)
            x[i] *= band
            x[j] *= band
        edits.append(edit(i,j,True))
        for k in leaves:
            x[k] = x[i] - scale*np.sqrt(1+1/D[i]+1/D[k])*rng.uniform(.999,1.001)
            if persistent:
                strength = (.65 if star<36 else band**2*(1 if k==leaves[0] else .8))
                x[k] = x[i] - scale*np.sqrt(strength*(1+1/D[i]+1/D[k]))
            edits.append(edit(i,k,False))
        if len(edits) == frontier_size:
            break
    if len(edits) != frontier_size:
        raise ValueError("insufficient frontier stars")
    s = A @ x
    s = np.rint(s / np.max(np.abs(s)) * 2**30) / 2**30
    tail = m-len(edits)
    eligible = order[~used[edges.row[order]] & ~used[edges.col[order]]]
    # Reuse the highest-negative-degree endpoint before moving to the next hub.
    rank = np.maximum(nd[edges.row[eligible]], nd[edges.col[eligible]])
    flips = eligible[np.argsort(-rank, kind="stable")[:tail//2]]
    edits.extend(edit(int(edges.row[t]), int(edges.col[t]), True) for t in flips)
    free = np.flatnonzero(~used)
    hubs = free[np.argsort(-nd[free], kind="stable")]
    additions = set()
    # Fill from the first available hubs, exhausting exact two-hop nonedges.
    for h in hubs:
        possible = two_hop(h)
        possible = possible[~used[possible]]
        for k in possible[np.argsort(vertex_rank[possible])]:
            additions.add(edit(int(h), int(k), False))
            if len(additions) == m-len(edits):
                break
        if len(additions) == m-len(edits):
            break
    if len(additions) < m-len(edits):
        for k in vertices:
            if used[k]:
                continue
            for h in hubs[:16]:
                if h != k and W[h,k] == 0:
                    additions.add(edit(int(h), int(k), False))
                    if len(additions) == m-len(edits):
                        break
            if len(additions) == m-len(edits):
                break
    if len(additions) != m-len(edits):
        raise ValueError("insufficient hub additions")
    edits.extend(sorted(additions))
    rng.shuffle(edits)
    return s, edits


def clustered_witness(W, m):
    """Certify the first two winners and enclose the second gain at both states."""
    from engine import Edit
    from fast_policy_kernel import FastPolicyKernel
    s, catalog = clustered_inputs(W, m)
    A = system(W).astype(np.int64)
    kernel = FastPolicyKernel(A, [Edit(*e) for e in catalog], backend="cg",
                              cached=False, winner_slots=1)
    wins = []
    try:
        for _ in range(2):
            winner, _ = kernel.select(s, delta=1e-9)
            wins.append(int(winner))
            kernel.commit(winner)
    finally:
        kernel.close()

    unit = 2**40
    rhs = np.rint(unit*s).astype(np.int64)
    def objective(B):
        z, status = cg(B, s, M=sparse.diags(1/B.diagonal()), rtol=1e-13, atol=0)
        if status:
            raise ArithmeticError("witness solve failed")
        t = np.rint(unit*z).astype(np.int64)
        r = rhs - B @ t
        lo = sum(int(i)*int(j) for i,j in zip(t, rhs+r))
        return lo, lo + sum(int(v)**2 for v in r)
    def child(B, idx):
        i,j,_,flip = catalog[idx]
        val = [-2,-2] if flip else [1,1,-1,-1]
        rr,cc = ([i,j],[j,i]) if flip else ([i,j,i,j],[i,j,j,i])
        return B + sparse.coo_matrix((val,(rr,cc)), shape=B.shape).tocsr()
    parent = child(A,wins[0])
    intervals = []
    for state in (A, parent):
        lo,hi = objective(state)
        cl,ch = objective(child(state,wins[1]))
        intervals.extend((lo-ch, hi-cl))
    ids = np.array([(i,j) for i,j,_,_ in catalog])
    degree = np.bincount(ids.ravel(), minlength=W.shape[0])
    two_hop_count = 0
    flips = 0
    for i,j,w,flip in catalog:
        assert w == 1 and i < j and W[i,j] == (-1 if flip else 0)
        if flip:
            flips += 1
        else:
            ni = W.indices[W.indptr[i]:W.indptr[i+1]]
            nj = W.indices[W.indptr[j]:W.indptr[j+1]]
            two_hop_count += bool(np.intersect1d(ni,nj,assume_unique=True).size)
    assert len(set(catalog)) == m and two_hop_count >= 64
    assert intervals[1] < 0 < intervals[2]
    return [m, *wins, *ids[wins].ravel(), flips, two_hop_count,
            m-flips-two_hop_count, int(degree.max()),
            int(np.count_nonzero(degree >= 2)), *intervals, unit**2]


def persistent_witness(W, m):
    """Check all adaptively selected gains; integer-enclose every root revival."""
    from engine import Edit
    from fast_policy_kernel import FastPolicyKernel

    s, catalog = persistent_inputs(W, m)
    base = system(W).astype(np.int64)
    kernel = FastPolicyKernel(base, [Edit(*e) for e in catalog], backend="cg",
                              cached=True, winner_slots=1)
    unit = 2**40
    rhs = np.rint(unit*s).astype(np.int64)

    def objective(B):
        z, status = cg(B, s, M=sparse.diags(1/B.diagonal()), rtol=1e-13, atol=0)
        if status:
            raise ArithmeticError("witness solve failed")
        t = np.rint(unit*z).astype(np.int64)
        residual = rhs - B @ t
        lo = sum(int(i)*int(j) for i,j in zip(t, rhs+residual))
        return lo, lo + sum(int(v)**2 for v in residual)

    def edited(B, e):
        i,j,_,flip = catalog[e]
        values = [-2,-2] if flip else [1,1,-1,-1]
        rows,cols = ([i,j],[j,i]) if flip else ([i,j,i,j],[i,j,j,i])
        return B + sparse.coo_matrix((values,(rows,cols)),shape=B.shape).tocsr()

    root_lo,root_hi = objective(base)
    try:
        for step in range(1,41):
            z,rho = kernel.solve(s)
            ids = np.flatnonzero(~kernel.active)
            caps = kernel.upper(ids,z,rho)
            if step == 1:
                root_caps = caps.copy()
            winner,lower = kernel.select_solved(s,z,rho,delta=1e-9)
            if winner is None or lower <= 0:
                raise AssertionError("incomplete forty-commit trace")
            sign_change = root_caps[winner] < 0
            exact = [""]*4
            if sign_change:
                after_lo,after_hi = objective(edited(base,winner))
                current = kernel.A.astype(np.int64)
                parent_lo,parent_hi = objective(current)
                child_lo,child_hi = objective(edited(current,winner))
                exact = [root_lo-after_hi,root_hi-after_lo,parent_lo-child_hi,parent_hi-child_lo]
                if not exact[1] < 0 < exact[2]:
                    raise AssertionError("unconfirmed root-to-current sign change")
            i,j,_,flip = catalog[winner]
            yield (m,step,winner,i,j,int(flip),root_caps[winner],lower,
                   int(sign_change),int(np.count_nonzero(caps>lower+1e-9)),
                   *exact,unit**2)
            kernel.commit(winner)
    finally:
        kernel.close()


def root_flip_signs(W, seed):
    """Classify S2 flip gains with the signed two-sided root screen."""
    s = opinions(W, "S2", seed)
    A = system(W)
    degree = A.diagonal()
    z, status = cg(A, s, M=sparse.diags(1/degree), rtol=1e-12, atol=0)
    if status:
        raise ArithmeticError("root diagnostic CG failed")
    edits = contention_candidates(W, 2500, s, seed)
    i, j = np.array([(i, j) for i, j, _, flip in edits if flip]).T
    x, y = degree[i], degree[j]
    d, p = z[i]-z[j], z[i]+z[j]
    a = d*d-p*p
    v, u = d-p, -d-p
    hi = a - (y*v*v + 2*v*u + x*u*u)/(x*y-1)
    lo = a - v*v - u*u + (v-u)**2/3
    positive, negative = np.count_nonzero(lo>1e-10), np.count_nonzero(hi< -1e-10)
    return positive, negative, len(i)-positive-negative

def profile(W):
    """Compute full-graph signed-triad counts and a diameter lower bound."""
    degree = np.diff(W.indptr)
    order = np.argsort(degree, kind="stable")
    F = sparse.triu(W[order][:, order], k=1, format="csr").astype(np.int64)
    U = abs(F)
    triangles = int((U @ U).multiply(U).sum())
    signed_sum = int((F @ F).multiply(F).sum())
    root = int(degree.argmax())
    def distances(v):
        d = csgraph.shortest_path(abs(W), indices=v, directed=False, unweighted=True)
        return np.where(np.isfinite(d), d, -1)
    diameter_lower = int(distances(distances(root).argmax()).max())
    return (W.shape[0], W.nnz//2, np.count_nonzero(W.data < 0)//2,
            triangles, (triangles-signed_sum)//2, diameter_lower)


def response_profile(W, count=64, seed=48231):
    """Quantiles of grounded positive-dipole response over sampled observed edges."""
    A = system(W)
    preconditioner = sparse.diags(1 / A.diagonal()).tocsr()
    edges = sparse.triu(W, k=1, format="coo")
    draw = np.random.default_rng(seed).choice(edges.nnz, count, replace=False)
    values = []
    errors = []
    for t in draw:
        i, j = edges.row[t], edges.col[t]
        b = np.zeros(W.shape[0])
        b[i], b[j] = 1, -1
        x, status = cg(A, b, M=preconditioner, rtol=1e-10, atol=0)
        if status:
            raise ArithmeticError(f"CG failed with status {status}")
        values.append(x[i]-x[j])
        errors.append(np.sqrt(2)*np.linalg.norm(b-A@x))
    return (int(np.diff(W.indptr).max()), count, seed,
            *np.quantile(values, [0.25, 0.50, 0.75]), max(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", nargs="*", choices=NAMES)
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--response-profile", action="store_true")
    parser.add_argument("--matched-setup", action="store_true")
    parser.add_argument("--clustered-witness", action="store_true")
    parser.add_argument("--persistent-witness", action="store_true")
    parser.add_argument("--persistent-setup", action="store_true")
    parser.add_argument("--clustered-setup", action="store_true")
    parser.add_argument("--root-signs", action="store_true")
    args = parser.parse_args()
    if args.persistent_witness:
        with (DATA / "persistent_witness.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("graph","M","round","winner","i","j","flip","root_gain_upper",
                             "current_gain_lower","root_negative","caps_above_winner_plus_delta",
                             "root_lo","root_hi","current_lo","current_hi","denominator"))
            for name in NAMES:
                W = load_graph(name)
                for m in (1000,2500):
                    for row in persistent_witness(W,m):
                        writer.writerow((name,*row))
                    stream.flush()
    if args.clustered_witness:
        with (DATA / "clustered_witness.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("graph", "M", "winner1", "winner2", "i1", "j1", "i2", "j2",
                             "flips", "two_hop_adds", "longer_adds", "max_endpoint_load",
                             "shared_endpoints", "root_lo", "root_hi",
                             "child_lo", "child_hi", "denominator"))
            for name in NAMES:
                W = load_graph(name)
                for m in (1000, 2500):
                    writer.writerow([name, *clustered_witness(W, m)])
                    stream.flush()
    if args.root_signs:
        with (DATA / "root_signs.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("graph", "seed", "M", "positive", "negative", "unresolved"))
            for name in NAMES:
                W = load_graph(name)
                for seed in (17, 31):
                    writer.writerow((name, seed, 2500, *root_flip_signs(W, seed)))
    for label, fixture, rhs_count, enabled in (("matched", matched_inputs, 96, args.matched_setup),
                                               ("clustered", clustered_inputs, 3, args.clustered_setup),
                                               ("persistent", persistent_inputs, 84, args.persistent_setup)):
        if not enabled:
            continue
        from time import perf_counter
        summary = []
        boot_rng = np.random.default_rng(48231)
        with (DATA / f"{label}_setup_raw.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("graph", "M", "seed", "calibration_rhs", "rep", "seconds"))
            for name in NAMES:
                W = load_graph(name)
                for m in (1000, 2500):
                    samples = []
                    for rep in range(10):
                        start = perf_counter()
                        s, catalog = fixture(W, m)
                        seconds = perf_counter() - start
                        if rep == 0:
                            first_s, first_catalog = s.tobytes(), catalog
                        elif s.tobytes() != first_s or catalog != first_catalog:
                            raise RuntimeError(f"fixture mismatch on {name} M={m} rep={rep}")
                        samples.append(seconds)
                        writer.writerow((name, m, 17, rhs_count, rep, seconds))
                        stream.flush()
                    q1, median, q3 = np.quantile(samples, [.25, .5, .75])
                    bootstrap = np.median(boot_rng.choice(samples, (20000, 10)), axis=1)
                    lo, hi = np.quantile(bootstrap, [.025, .975])
                    summary.append((name, m, 17, rhs_count, median, q1, q3, lo, hi, 10))
        with (DATA / f"{label}_setup.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("graph", "M", "seed", "calibration_rhs", "seconds",
                             "q1_seconds", "q3_seconds", "median_ci_lo", "median_ci_hi", "reps"))
            writer.writerows(summary)
    if args.build is not None:
        DATA.mkdir(exist_ok=True)
        for name in args.build or NAMES:
            W = _observed(name) if name in SOURCES else _synthetic(name)
            sparse.save_npz(DATA / (name + ".npz"), W)
            print(name, W.shape, W.nnz // 2, flush=True)

    if args.profiles:
        for name in NAMES:
            print(name, *profile(load_graph(name)))


    if args.response_profile:
        with (DATA / "grounded_response.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("network", "max_degree", "edge_count_sampled", "seed",
                             "Q1", "median", "Q3", "max_residual_error_bound"))
            for name in NAMES:
                writer.writerow((name, *response_profile(load_graph(name))))
