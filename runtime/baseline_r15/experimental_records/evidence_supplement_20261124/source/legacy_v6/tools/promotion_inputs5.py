"""Frozen topology-first conflict-clamp stress sources on the full signed graphs."""
from itertools import combinations
from time import perf_counter

import numpy as np

from long_promotion import Iterative, OPINION_DEN, UpdatedFactor
from promotion_v4 import motif_sites, query_opinion
from workloads import opinions, system


def _sites(W, degree):
    negative = (W < 0).tocsr()
    centers = np.flatnonzero(np.diff(negative.indptr) >= 2)
    # Search the narrowest center degrees, allowing two extra neighbors.
    # Prefer a positive closing edge, then minimize degree and vertex keys.
    degree_limit = degree[centers].min()+2
    best = None
    for a in sorted(centers, key=lambda a: (degree[a], a)):
        if degree[a] > degree_limit:
            break
        for u, v in combinations(negative[a].indices, 2):
            if W[u, v] < 0:
                continue
            b, c = sorted((u, v), key=lambda v: (degree[v], v))
            key = (W[u, v] == 0, degree[a], degree[c], degree[b], a, b, c)
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError("no negative wedge without a negative closing edge")
    return np.array(best[-3:], dtype=int)


def _targets(G, open_wedge):
    """Maximize a model margin before catalog construction or any timing."""
    G = (G + G.T)/2
    ab, ac = [0, 1], [0, 2]
    inverse_edit = np.array([[0., -.5], [-.5, 0.]])
    H = np.linalg.inv(inverse_edit + G[np.ix_(ab, ab)])
    J = np.linalg.inv(inverse_edit + G[np.ix_(ac, ac)])
    cross = G[np.ix_(ac, ab)]
    child = np.linalg.inv(inverse_edit + G[np.ix_(ac, ac)] - cross @ H @ cross.T)
    alpha, gamma = np.meshgrid(np.arange(1, 97)/64, np.arange(1, 97)/64)
    x = np.stack((alpha, -np.ones_like(alpha), gamma), axis=-1)
    x1 = x - x[..., ab] @ H @ G[:, ab].T

    def score(pair, matrix):
        return ((pair @ matrix)*pair).sum(axis=-1)

    first = score(x[..., ab], H)
    root_target = score(x[..., ac], J)
    second = score(x1[..., ac], child)
    # Include a nominal 1/16 bridge joining the most separated field values
    # (including zero background). Full-catalog winner proofs use actual bounds.
    def bridge(v):
        spread = np.max(np.abs(v), axis=-1)
        if open_wedge:
            spread = np.maximum(spread, np.abs(v[..., 1]-v[..., 2]))
        return spread**2/16

    margin = np.minimum.reduce((first-bridge(x), -root_target, second-bridge(x1)))
    idx = np.unravel_index(margin.argmax(), margin.shape)
    return 2048*x[idx], float(margin[idx])


def clamped_source(W, name, seed=17, metadata=None):
    """Return the quantized source. Caller charges this whole call to root setup."""
    if metadata is None:
        metadata = {}
    if name == "Alpha":
        sites, _ = motif_sites(W, system(W))
        metadata.update(zip(("clamp_center", "clamp_first", "clamp_target"), map(int, sites)))
        return query_opinion(W, "AlphaClamp", seed, metadata)
    start = perf_counter()
    A = system(W).astype("int64")
    sites = _sites(W, A.diagonal())
    signed = np.rint(OPINION_DEN*opinions(W, "S1", seed)).astype("int64")
    solver = UpdatedFactor(A) if W.shape[0] <= 10000 else Iterative(A)
    field = solver.solve(signed.astype(float))
    G = np.empty((3, 3))
    for k, v in enumerate(sites):
        unit = np.zeros(W.shape[0])
        unit[v] = 1
        G[:, k] = solver.solve(unit)[sites]
    target, margin = _targets(G, W[sites[1], sites[2]] == 0)
    c = 3/np.abs(field).max()
    background = c*field
    rhs = c*signed + A[:, sites] @ (target-background[sites])
    scale = min(1, OPINION_DEN/np.abs(rhs).max())
    metadata.update(zip(("clamp_center", "clamp_first", "clamp_target"), map(int, sites)))
    metadata.update(clamp_aux_seconds=perf_counter()-start, clamp_background_scale=c,
                    clamp_background_rms=float(np.sqrt(np.mean(background**2))),
                    clamp_background_max=float(np.abs(background).max()),
                    clamp_source_scale=scale, clamp_model_margin=margin)
    return np.rint(scale*rhs).astype("int64")

