"""Full Bio and Bitcoin-Alpha operators, rank-two greedy flips, and residual audits."""
import argparse
import csv
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.linalg import solveh_banded
from scipy.sparse.linalg import cg, splu
from bounds import prior_constants
from workloads import load_graph, system

ROOT = Path(__file__).resolve().parents[1]
EPS = np.finfo(float).eps
TOLS = (1e-13, 1e-10, 1e-7, 1e-5)
BUDGETS = (15, 20, 30, 50)
J = np.diag([1., -1.])
WEIGHTS = (1, 4, 16, 64)
GRAM_BUDGET = 1e-5
DIAGNOSTIC_DELTA = 1e-6


def fixture(w):
    W = load_graph("Bio").astype(float)
    upper = sparse.triu(W, 1, format="coo")
    rng = np.random.default_rng(1701)
    edges, used = [], set()
    for idx in rng.permutation(np.flatnonzero(upper.data < 0)):
        i, j = int(upper.row[idx]), int(upper.col[idx])
        if i not in used and j not in used:
            edges.append((i, j))
            used.update((i, j))
            if len(edges) == 72:
                break
    i, j = np.array(edges).T
    W[i, j] = W[j, i] = -w
    s = rng.uniform(-1, 1, W.shape[0])
    s[i] = rng.uniform(.35, 1, len(i))
    s[j] = -s[i]
    return system(W), i, j, s


def alpha_fixture(w):
    W = load_graph("Alpha").astype(float)
    neg_degree = np.asarray((W < 0).sum(axis=1)).ravel()
    hubs = np.lexsort((np.arange(len(neg_degree)), -neg_degree))[:8]
    upper = sparse.triu(W, 1, format='coo')
    left_hub = np.isin(upper.row, hubs)
    right_hub = np.isin(upper.col, hubs)
    eligible = np.flatnonzero((upper.data < 0) & (left_hub ^ right_hub))
    rng = np.random.default_rng(1701)
    chosen = rng.choice(eligible, 60, replace=False)
    i, j = upper.row[chosen], upper.col[chosen]
    W[i, j] = W[j, i] = -w
    s = rng.uniform(.25, 1., W.shape[0])
    neighbors = np.where(left_hub[chosen], j, i)
    s[neighbors] *= -1
    return system(W), i, j, s


def sparse_objectives(A, solver, s, X, ref, left, right, w):
    """Reconstruct all edited states, then certify them against actual edits."""
    base = solver.solve(s)
    h = np.sqrt(w)*np.stack((base[left]-base[right], base[left]+base[right]),axis=1)
    coeff = np.linalg.solve(ref+J, h[:,:,None]).squeeze(-1)
    states = base[:,None]-np.einsum('nei,ei->ne',X,coeff)
    residual = s[:,None]-A@states
    idx = np.arange(len(left))
    residual[left,idx] += 2*w*states[right,idx]
    residual[right,idx] += 2*w*states[left,idx]
    mag = abs(s[:,None])+abs(A)@abs(states)
    mag[left,idx] += 2*w*abs(states[right,idx])
    mag[right,idx] += 2*w*abs(states[left,idx])
    rounding = (8+np.diff(A.indptr))[:,None]*EPS
    edited_cap = vec_cap(abs(residual)+rounding*mag, axis=0)*(1+32*EPS)
    base_cap = cap(A,base[:,None],s[:,None])[0]
    gaps = s@base - s@states
    allowance = 4*len(s)*EPS*(abs(s)@abs(base)+abs(s)@abs(states))
    return gaps, (vec_cap(s)*(edited_cap+base_cap)+allowance)*(1+32*EPS)


def vec_cap(X, axis=None):
    """Upper Euclidean norm: scale, n squares and sums, then sqrt."""
    n = X.size if axis is None else X.shape[axis]
    scale = np.max(abs(X), axis=axis, keepdims=True)
    unit = X / np.where(scale == 0, 1., scale)
    magnitude = np.squeeze(scale, axis=axis)
    result = np.sqrt(np.sum(unit*unit, axis=axis))*magnitude
    return np.where(magnitude == 0, 0.,
                    np.nextafter(result*(1+(4*n+16)*EPS), np.inf))


def norm(X):
    """Upper spectral norm for scalar, column and two-by-two blocks."""
    if X.shape[-2:] != (2, 2):
        return vec_cap(X.reshape(X.shape[:-2]+(-1,)), axis=-1)
    # sigma_max = (||(a+d,b-c)||_2 + ||(a-d,b+c)||_2)/2.
    # Scaling bounds the absolute errors of the sums and differences by
    # 4 eps. Each two-term Euclidean reduction and rescaling fit in 32 eps.
    scale = np.max(abs(X), axis=(-2,-1))
    U = X / np.where(scale == 0, 1., scale)[...,None,None]
    a,b,c,d = U[...,0,0],U[...,0,1],U[...,1,0],U[...,1,1]
    plus = np.sqrt((a+d)**2 + (b-c)**2)
    minus = np.sqrt((a-d)**2 + (b+c)**2)
    bound = scale*((plus+minus)*.5+32*EPS)
    return np.where(scale == 0, 0., np.nextafter(bound,np.inf))


def band(A):
    ab = np.zeros((3, A.shape[0]))
    for d in range(3):
        ab[d, :A.shape[0]-d] = A.diagonal(-d)
    return ab


def cap(A, X, B):
    return (vec_cap(B-A@X, axis=0)
            + vec_cap(np.maximum(32, 8+np.diff(A.indptr))[:,None]*EPS*(abs(B)+abs(A)@abs(X)), axis=0))*(1+32*EPS)


def solve(A, B, tol):
    M = sparse.diags(1/A.diagonal())
    X = np.empty_like(B)
    for j in range(B.shape[1]):
        X[:, j], info = cg(A, B[:, j], M=M, rtol=tol, atol=0.)
        if info:
            raise RuntimeError(("CG", info))
    return X, cap(A, X, B)


def inverse(G, E, beta):
    B = J+G
    T = np.linalg.inv(B)
    eta_b = 16*EPS*(1+norm(G))
    eta_t = norm(np.eye(2)-B@T)+32*EPS*(1+norm(abs(B)@abs(T)))
    zeta = beta*(eta_t+(E+eta_b)*norm(T))*(1+32*EPS)
    return T, zeta, max(np.max(eta_b), np.max(eta_t))


def run(w, tol, cohort="bio", directional=False):
    budgets = BUDGETS if cohort == "bio" else (15, 20)
    A, left, right, s = fixture(w) if cohort == "bio" else alpha_fixture(w)
    initial = len(left)
    n, k, beta = len(s), np.sqrt(2*w), 1+2*w
    root = np.sqrt(w)
    ids = np.arange(len(left))
    constants = prior_constants(w, max(budgets), DIAGNOSTIC_DELTA)
    master = np.zeros((n, 2*len(ids)))
    master[left, 2*ids] = master[left, 2*ids+1] = root
    master[right, 2*ids] = -root
    master[right, 2*ids+1] = root
    named = max_residual = max_residual_cap = 0.
    counts, steps, events = [], [], []
    R = W = state_refines = direct_fallbacks = 0

    def record(x):
        nonlocal named
        named = max(named, float(np.max(x)))

    def cg_blocks(B, tol, width):
        nonlocal max_residual, max_residual_cap
        X, caps = solve(A, B, tol)
        residual = np.linalg.norm(B-A@X, axis=0).reshape(-1, width)
        max_residual = max(max_residual, np.max(np.linalg.norm(residual, axis=1)))
        block_caps = vec_cap(caps.reshape(-1, width), axis=1)
        if width > 1:
            block_caps *= 1+8*EPS
        max_residual_cap = max(max_residual_cap, np.max(block_caps))
        return X, caps

    def diagnostics(budget):
        epsilon_star = prior_constants(w, budget, DIAGNOSTIC_DELTA)['epsilon']
        return dict(max_residual=max_residual, max_residual_cap=max_residual_cap,
                    max_named=named, diagnostic_delta=DIAGNOSTIC_DELTA,
                    prior_epsilon_star=epsilon_star, prior_strict=int(named<=epsilon_star))

    def fresh(chosen, precision=None):
        rhs = master[:, (2*chosen[:, None]+[0, 1]).ravel()]
        if precision is None:
            X = (solveh_banded(ab, rhs, lower=True, check_finite=False) if cohort == "bio"
                 else solver.solve(rhs))
            residual = cap(A, X, rhs)
        else:
            X, residual = cg_blocks(rhs, precision, 2)
        X = X.reshape(n, -1, 2)
        row = np.arange(len(chosen))
        l, r = X[left[chosen], row], X[right[chosen], row]
        G = root*np.stack([l-r, l+r], axis=1)
        eta = 32*EPS*(1+2*root*vec_cap(abs(l)+abs(r), axis=1))
        rho = vec_cap(residual.reshape(-1, 2), axis=1)*(1+8*EPS)
        if precision is not None:
            record(eta)
            record(rho)
        return G, (k*rho+eta)*(1+16*EPS), X, rho

    details = {}

    def completed(B, rhs, P, rhs_error, B_error):
        # Column-oriented equations B P = rhs for both scores and blocks.
        pt = P.swapaxes(-1, -2)
        rt = rhs.swapaxes(-1, -2)
        pn = norm(P)*(1+32*EPS)
        sym = (B+B.swapaxes(-1, -2))*.5
        center = rt@P+pt@rhs-pt@sym@P
        rho = (norm(rhs-B@P)+64*EPS*(norm(abs(rhs))+norm(abs(B)@abs(P))))*(1+32*EPS)
        eta = 128*EPS*(norm(abs(rt)@abs(P))+norm(abs(pt)@abs(rhs))
                       +norm(abs(pt)@(abs(B)+abs(B.swapaxes(-1,-2)))@abs(P)))
        rad = (2*rhs_error*pn+B_error*pn**2
               +beta*(rho+rhs_error+B_error*pn)**2+eta)*(1+128*EPS)
        return center, rad, rho

    def scores(z, rho):
        l, r = z[left[ids], 0], z[right[ids], 0]
        h = root*np.stack([l-r, l+r], axis=1)
        eta = 32*EPS*(1+k*(abs(l)+abs(r)))
        eh = k*rho+eta
        hnorm = vec_cap(h, axis=1)
        T, et, named_t = inverse(G, E, beta)
        score = (h[:, None, :]@T@h[:, :, None]).ravel()
        arith = 64*EPS*(1+hnorm**2*norm(abs(T)))
        record(max(np.max(eta), named_t, np.max(arith), max(rho)))
        radius = (beta*(2*hnorm+eh)*eh + hnorm**2*et + arith)*(1+64*EPS)
        if directional:
            B = G+J
            H = h[:,:,None]
            q, dr, local = completed(B,H,T@H,eh,E+16*EPS*(1+norm(G)))
            q = q.ravel()
            use = dr < radius
            details.update(old=radius.copy(), directional=dr, local=local)
            score, radius = np.where(use,q,score), np.minimum(radius,dr)
        return score, radius

    def objective(ab_edit):
        x = solveh_banded(ab_edit, s, lower=True, check_finite=False)
        Ax = ab_edit[0]*x
        mag = abs(ab_edit[0])*abs(x)
        for d in (1, 2):
            off = ab_edit[d, :-d]
            Ax[d:] += off*x[:-d]
            Ax[:-d] += off*x[d:]
            mag[d:] += abs(off*x[:-d])
            mag[:-d] += abs(off*x[d:])
        err = vec_cap(s)*(vec_cap(s-Ax)+32*EPS*vec_cap(abs(s)+mag))
        return s@x, (err+4*n*EPS*(abs(s)@abs(x)))*(1+32*EPS)

    def totals(budget, accepted, stop_upper=''):
        subset = np.array([e["ratio"] for e in events])
        return dict(w=w,b=budget,rtol=tol,beta=beta,C=initial,W=W,R=R,
                    blocks=initial+W+R,state_refines=state_refines,
                    direct_fallbacks=direct_fallbacks,events=len(subset),
                    frac_small=np.mean(subset<.1),frac_large=np.mean(subset>=1),
                    min_margin=min(r['margin'] for r in steps),accepted=accepted,
                    stop_upper=stop_upper, **diagnostics(budget))

    age = np.zeros(len(ids), int)
    coupling = np.zeros(len(ids))
    for epoch in range(1, max(budgets)+1):
        if cohort == "bio":
            ab = band(A)
        else:
            solver = splu(A.tocsc())
        if epoch == 1:
            G, E, _, _ = fresh(ids, tol)
            uniform = E.copy()
        ref, refcap, direct_columns, _ = fresh(ids)
        discrepancy = np.linalg.norm(ref-G, ord=2, axis=(-2,-1))  # diagnostic only
        if np.any(discrepancy > E+refcap):
            raise AssertionError("Gram containment")
        if cohort == "bio":
            base, basecap = objective(ab)
            values, radii = [], []
            for i, j in zip(left[ids], right[ids]):
                edited = ab.copy()
                edited[j-i, i] -= 2*w
                value, radius = objective(edited)
                values.append(base-value)
                radii.append(basecap+radius+8*EPS*(abs(base)+abs(value)))
            truth, rc = np.array(values), np.array(radii)
        else:
            truth, rc = sparse_objectives(A, solver, s, direct_columns, ref, left[ids], right[ids], w)
        winner = int(np.argmax(truth))
        others = np.arange(len(ids)) != winner
        margin = min(truth[winner]-rc[winner],
                     truth[winner]-rc[winner]-max((truth+rc)[others]))
        if max(truth+rc) <= 0:
            z, rho = cg_blocks(s[:, None], tol, 1)
            g, radius = scores(z, rho)
            if max(g+radius) > 0:
                z, rho = cg_blocks(s[:, None], 2e-14, 1)
                state_refines += 1
                g, radius = scores(z, rho)
            unresolved_stop = g+radius > 0
            if np.any(unresolved_stop):
                G[unresolved_stop], E[unresolved_stop], _, _ = fresh(ids[unresolved_stop], 2e-14)
                R += int(sum(unresolved_stop))
                g, radius = scores(z, rho)
            if max(g+radius) > 0:
                direct_fallbacks += 1
            for budget in budgets:
                if budget >= epoch:
                    counts.append(totals(budget, epoch-1, max(truth+rc)))
            break
        assert margin > 0, (w, tol, epoch, margin)
        tref, etref, _ = inverse(ref[winner],refcap[winner],beta)
        z, rho = cg_blocks(s[:, None], tol, 1)
        g, radius = scores(z, rho)
        ratio = radius / margin
        if np.any(abs(g-truth) > radius+rc):
            raise AssertionError("score containment")
        old_named = named
        for p, edge in enumerate(ids):
            prior = (constants['a']-1 + age[p]*constants['ell'])*GRAM_BUDGET
            valid = old_named <= GRAM_BUDGET and 4*constants['a']*beta*GRAM_BUDGET <= 1
            events.append(dict(w=w,rtol=tol,epoch=epoch,edge=edge,age=age[p],
                coupling=coupling[p],error=discrepancy[p],ref_cap=refcap[p],
                max_named=old_named,gram_budget=GRAM_BUDGET,prior_eligible=int(valid),
                posterior=E[p],uniform=uniform[p],prior=prior if valid else '',
                score=g[p],radius=radius[p],margin=margin,ratio=ratio[p],
                true_gain=truth[p],gain_ref_cap=rc[p],winner=int(p==winner),
                competitive=int(g[p]+radius[p]>=max(g-radius))))
            if directional:
                events[-1].update(same_state_old=details['old'][p],
                                  completed=details['directional'][p],rho_p=details['local'][p])
        def unresolved():
            lead = int(np.argmax(g-radius))
            mask = g+radius >= g[lead]-radius[lead]
            mask[lead] = False
            if g[lead]-radius[lead] > 0 and not np.any(mask):
                return lead, mask
            mask[lead] = True
            return -1, mask
        pick, mask = unresolved()
        pre = int(pick < 0)
        fallback = 0
        if pick < 0:
            z, rho = cg_blocks(s[:, None], 2e-14, 1)
            state_refines += 1
            g, radius = scores(z, rho)
            pick, mask = unresolved()
        refreshed = 0
        if pick < 0:
            G[mask], E[mask], _, _ = fresh(ids[mask], 2e-14)
            age[mask] = 0
            uniform[mask] = E[mask]
            refreshed = int(sum(mask))
            R += refreshed
            g, radius = scores(z, rho)
            pick, _ = unresolved()
        if pick < 0:
            pick = winner
            fallback = 1
            direct_fallbacks += 1
        assert pick == winner
        steps.append(dict(w=w,rtol=tol,epoch=epoch,winner=ids[pick],margin=margin,
                          gain=truth[pick],max_ratio=max(ratio),pre_failed=pre,
                          reanchors=refreshed,direct_fallback=fallback,
                          pivot_inverse=float(norm(tref)),pivot_ref_cap=float(etref),
                          left=int(left[ids[pick]]),right=int(right[ids[pick]]), **diagnostics(epoch)))
        if epoch in budgets:
            counts.append(totals(epoch, epoch))
        if epoch == max(budgets):
            break
        keep = others
        chosen = ids[[pick]]
        Gwin, ewin, X, rho = fresh(chosen, tol)
        T, et, nt = inverse(Gwin, ewin, beta)
        record(nt)
        W += 1
        ids = ids[keep]
        X = X[:, 0, :]
        l, r = X[left[ids]], X[right[ids]]
        C = root*np.stack([l-r, l+r], axis=1)
        c = norm(C)
        pad = 32*EPS*(1+2*root*vec_cap(abs(l)+abs(r),axis=1))
        record(pad)
        xi = k*rho[0]+pad
        eta = 128*EPS*(1+norm(G[keep])+norm(abs(C)@abs(T)@abs(C.swapaxes(1,2))))
        record(eta)
        def increment(c):
            return beta*(2*c+xi)*xi + c*c*et + eta
        carried = E[keep]
        E = (carried+increment(c))*(1+64*EPS)
        uniform = (uniform[keep]+increment(2*w+xi))*(1+64*EPS)
        oldG = G[keep]
        G = oldG-C@T@C.swapaxes(1,2)
        if directional:
            B = Gwin+J
            rhs = C.swapaxes(-1,-2)
            Q, er, rp = completed(B,rhs,T@rhs,xi,ewin+16*EPS*(1+norm(Gwin)))
            alternate = oldG-Q
            sub = 32*EPS*(1+norm(oldG)+norm(Q))
            newE = (carried+er+sub)*(1+64*EPS)
            use = newE < E
            G = np.where(use[:,None,None],alternate,G)
            E = np.minimum(E,newE)
            steps[-1].update(max_rho_P=float(max(rp)),directional_blocks=int(sum(use)))
        coupling = c/(2*w)
        age = age[keep]+1
        i, j = left[chosen[0]], right[chosen[0]]
        A[i,j] = A[j,i] = -w
    return counts, steps, events


def write(name, rows):
    with (ROOT/'data'/('flip_stress_'+name+'.csv')).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(events, positive_bins=4):
    output = []
    for name in ('coupling','age','w'):
        vals = np.array([e[name] for e in events])
        if name == 'coupling':
            aged = np.array([e['age'] > 0 for e in events])
            positive = np.flatnonzero(aged & (vals > 0))
            bins = [np.flatnonzero(aged & (vals == 0))]
            bins.extend(np.array_split(positive[np.argsort(vals[positive],kind='stable')],positive_bins))
        elif name == 'age':
            bins = [np.flatnonzero((vals>=lo)&(vals<=hi))
                    for lo,hi in [(0,0),(1,4),(5,14),(15,29),(30,49)]]
        else:
            bins = [np.flatnonzero(vals==w) for w in WEIGHTS]
        for idx in bins:
            if len(idx) == 0:
                continue
            rows = [events[j] for j in idx]
            rat = np.array([r['ratio'] for r in rows])
            result = dict(group=name,lo=min(vals[idx]),hi=max(vals[idx]),n=len(idx),
                          small=np.mean(rat<.1),large=np.mean(rat>=1))
            for key in ('error','posterior','uniform','prior','ratio','ref_cap'):
                valid = [r[key] for r in rows if r[key]!='']
                result[key] = np.median(valid) if valid else ''
                if key=='prior':
                    result['prior_n']=len(valid)
            output.append(result)
    return output


def plot():
    def rows(name):
        with (ROOT/'data'/('flip_stress_'+name+'.csv')).open() as handle:
            return list(csv.DictReader(handle))

    bio = [r for r in rows('bins') if r['group']=='coupling']
    alpha = [r for r in rows('alpha_bins') if r['group']=='coupling']
    events = rows('events')
    age = np.array([int(r['age']) for r in events])
    weight = np.array([int(r['w']) for r in events])
    ratio = np.array([float(r['ratio']) for r in events])
    plt.rcParams.update({'pdf.fonttype':42, 'font.size':8, 'axes.titlesize':8})
    fig, axes = plt.subplots(2,2,figsize=(7.05,3.4),layout='constrained')
    gram, coupling = axes[0]
    for key,fmt,title in [('error','o-','Discrepancy'),('posterior','s--','Posterior'),
                          ('uniform','^:','Uniform'),('prior','d-.','A priori')]:
        gram.semilogy(range(len(bio)),[float(r[key]) for r in bio],fmt,ms=3,label=title)
    gram.set(xticks=range(5),xticklabels=['0','Q1','Q2','Q3','Q4'],
             ylabel='Median magnitude',xlabel='Last coupling bin',
             title='Bio Gram transport')
    gram.legend(fontsize=6,ncol=2,loc='lower right')
    x = np.arange(len(alpha))
    for side,key,title,color in [(-1,'small',r'$r_g/m_t<0.1$','#245d84'),
                                  (1,'large',r'$r_g/m_t\geq1$','#aa5023')]:
        coupling.bar(x+side*.19,[100*float(r[key]) for r in alpha],.38,label=title,color=color)
    coupling.set(xticks=x,xticklabels=['Q1','Q2','Q3','Q4','Q5'],ylim=(0,100),
                 ylabel='Candidate events (%)',xlabel='Last coupling quintile (700–701 events each)',
                 title='Bitcoin-Alpha decision margins')
    coupling.legend(fontsize=6,ncol=2)
    cuts = [(0,0),(1,4),(5,14),(15,29),(30,49)]
    for ax,mask,title in zip(axes[1],(ratio<.1,ratio>=1),
                             (r'Bio events with $r_g/m_t<0.1$ (%)',
                              r'Bio events with $r_g/m_t\geq1$ (%)')):
        cells = np.array([[100*np.mean(mask[(weight==w)&(age>=lo)&(age<=hi)])
                           for lo,hi in cuts] for w in WEIGHTS])
        ax.imshow(cells,cmap='Blues',aspect='auto',vmin=0,vmax=100)
        for i,j in np.ndindex(cells.shape):
            ax.text(j,i,f'{cells[i,j]:.1f}',ha='center',va='center',fontsize=7,
                    color='white' if cells[i,j]>58 else '#102f43')
        ax.set(xticks=range(5),xticklabels=['0','1–4','5–14','15–29','30–49'],
               yticks=range(4),yticklabels=WEIGHTS,xlabel='Age since anchor',ylabel='Flip weight',title=title)
    fig.savefig(ROOT/'figures'/'coupling.pdf')
    plt.close(fig)


def bio_run():
    counts, steps, events = [], [], []
    for w in WEIGHTS:
        for tol in TOLS:
            c, s, e = run(w,tol)
            counts.extend(c)
            steps.extend(s)
            events.extend(e)
            print(w,tol,c[-1],flush=True)
    write('grid',counts)
    write('steps',steps)
    write('events',events)
    bins = summarize(events)
    write('bins',bins)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=('bio','alpha','both','plot'), default='both')
    args = parser.parse_args()
    if args.cohort in ('both', 'bio'):
        bio_run()
    if args.cohort in ('both', 'alpha'):
        counts, steps, events = [], [], []
        for w in WEIGHTS:
            c, s, e = run(w, 1e-7, 'alpha')
            counts.extend(c)
            steps.extend(s)
            events.extend(e)
            print('Alpha', w, c[-1], flush=True)
        write('alpha_grid',counts)
        write('alpha_steps',steps)
        write('alpha_events',events)
        write('alpha_bins',summarize(events, positive_bins=5))
    plot()

