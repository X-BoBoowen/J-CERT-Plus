# Actual V2 numerical certificate and unit audit

## Status and scope

This audits the actual received V2 implementation and paper. The root scheduler ran the bounded production harness once under the authorized 120-second, 2-GiB, single-thread limit: **174 checks passed, zero failed**, and all 17 prescribed physical gains lay in the tested production/generic-kernel intervals. The executable evidence is `numeric_certificate_checks.json`. No production mathematical source was changed, and no historical suite was rerun.

The actual baseline is commit `4c2c7b7e84c7c03ee56d45453fea6950c94817bd` with its separately bound historical overlay. References below are relative to `v3_runtime/` for executable source and `v2_received/paper/paper/` for manuscript source. Line numbers refer to the original V2 snapshots recorded by the harness; later isolated G integration can insert lines in the runtime session/checker without changing the audited numerical kernel. The received originals remain in `v2_received/code/`. The harness records SHA256 of the actual modules it imports, including the legacy overlay modules; its reference gains come exclusively from `v3/code/exact_oracle.py`.

The important conclusion is that **129 and 8256 are inverse norms in different local units**. The weighted adapter explicitly rescales its pivot before invoking the legacy directional primitive. A missing weight factor cannot be inferred from a variable named `beta`, `B`, or `G`. The bounded audit found no invalid returned interval or false decision. This is evidence for the explicitly tested paths; it is not universal verification of historical outputs. The independent checker still has a nonpoint-interval intersection limitation, detailed in Section 4.

Observed results were: three actual small-graph exact fallbacks on the equal path fixture; nine separated/no-overlap resolutions on the near, zero, and empty-graph fixtures; three actual single-transposition resolutions on the extra twin fixture; 15 independently replayed budget-one requests; four valid transported gain intervals; and 16 rejected malformed or out-of-domain cases. The harness's internal wall was 1.098185512 seconds; the scheduler ledger is authoritative for charged process wall time. There were no completion issues in these mathematical fixtures.

## 1. Physical, integer, and proof units

Use `A_p = I + D - W`, `s_p = s_num / 65536`, and `z_p = A_p^-1 s_p`. Write an edit as `omega V J V^T`, where `V` consists of exact endpoint columns: one column `e_i-e_j` for insertion, and two columns `[e_i-e_j, e_i+e_j]` for a flip. These are the manuscript's normalized bases; the paper's unnormalized basis is `U=sqrt(omega)V`.

The implementation uses `S=16`, `Q=2^48`, `A_i=S A_p`, `d=S omega`, and `z_i=A_i^-1 s_p=z_p/S`. Those are exact representation conversions on the admitted half-integer root weights and insertion weight `1/16`. In the table, the quantities with subscript `i` refer to the inverse energy of `A_i`; they are not physical gains until multiplied by 16.

| Quantity | Physical normalized definition | Actual stored/calculated quantity | Conversion |
|---|---|---|---|
| Operator | `A_p` | `A_i=16A_p`, int64 CSR | Divide by 16 |
| Opinion | `s_p` | int64 numerator `s_num`, denominator `2^16` | Divide by `2^16` |
| State | `z_p=A_p^-1 s_p` | trial `z_i`, integer center `x≈Q z_i` | Physical center `16x/Q` |
| Response | `Ybar=A_p^-1 V` | `Y_i=A_i^-1 V=Ybar/16`, integer center `X≈QY_i` | Physical response `16X/Q` |
| Gram | `Gbar=V^T A_p^-1 V` | `G_i=V^T A_i^-1 V=Gbar/16`, rational intervals | Multiply by 16 |
| Projection | `hbar=V^T z_p` | `h_i=V^T z_i=hbar/16`, rational intervals | Multiply by 16 |
| Normalized pivot | `Bbar=J/omega+Gbar` | `B_i=J/d+G_i=Bbar/16` | Multiply by 16 |
| Inverse energy reduction | physical `g` | `h_i^T B_i^-1 h_i=g/16` | Multiply by 16 |
| Raw score integers | physical gain bounds | `[L,U]/Q^2` in integer inverse-energy units | Physical bounds `[16L,16U]/Q^2` |
| Published decision | physical gain bounds | `low=16L`, `high=16U`, `gain_denominator=Q^2` | Divide published numerators only by the denominator |

**Source evidence:** `src/common.py:21–23`; `src/weighted_adapter.py:1–4,54–91,289–299`; `src/a2_session.py:78–85,179–190,265–273`; manuscript `sections/03_responses.tex:5–8,36–37` and `sections/10_appendix.tex:37–38`.

The operator source calls the coefficient `weight` inside `correction`, but that argument is the integer coefficient `d`, not the physical `omega`. The typed argument name alone is therefore not a unit specification. The call site passes `int(self.kernel.d[e])` (`src/weighted_adapter.py:90`), and the coefficient construction is explicit (`:272–277`).

## 2. Residuals, projections, and outward rounding

### 2.1 State and response certificates

For a state center `x/Q`, production recomputes the integer residual

`r_num = A_i x - s_num (Q/65536)`.

Since the signed operator is strictly row diagonally dominant with margin 16, `||A_i^-1||_infinity <= 1/16`. The generic state function retains the conservative raw radius `r=max |r_num|`, so `|z_i-x/Q| <= r/Q` is safe; the sharper `r/(16Q)` is not needed to justify that function's returned box. The endpoint strict screen separately uses `global_r=ceil(max|r_num|/16)` and the row-specific bound

`ceil((|r_num_i| + (A_ii-16) global_r)/A_ii)`.

The response path computes `R_num=A_i X-QV` and returns `ceil(max|R_num|/16)` in center-grid units, directly using the sharper inverse bound. Both paths check sufficient int64 product ranges before trusting integer residuals. The eventual bound concerns the represented dyadic center, so a floating solve or quantization need not itself be exact. A solver tolerance is not used as the certificate.

**Source evidence:** overlay `baseline_r15/experimental_records/evidence_supplement_20261124/source/legacy_v6/tools/long_promotion.py:167–201`; `src/weighted_adapter.py:54–75,124–139`.

Projection sums and differences involve two endpoint rows. The inherited `VerifiedGramViews.project` forms exact rational intervals `(endpoint_center ± 2 radius)/Q`. For a flip it uses both endpoint difference and sum. The same operation supplies state projections, Grams, and cross-response inputs. The one-step transport subtracts interval endpoints in the safe direction and reboxes outward onto the `Q` grid before changing the version number.

**Source evidence:** overlay `.../tools/promotion.py:291–308`; `src/weighted_adapter.py:93–111`.

### 2.2 Why the legacy constant 3 is safe in the weighted adapter

For graph-origin `G_i`, set

`K=J+dG_i`, `c=1+2d/16=1+2omega`.

The signed Woodbury identity gives `||K^-1||_2 <= c` for a legal parent and child. The adapter supplies the legacy routine with a transformed Gram

`G_tilde = cd G_i + (c-1)J`,

so the routine's effective pivot is

`J+G_tilde = cK`.

Its inverse norm is at most 1. The legacy routine uses the conservative bound 3 and returns an interval for `H(cK)^-1 H^T`; the adapter multiplies it by `cd`, yielding exactly the target `H(J/d+G_i)^-1 H^T`. There is no omitted physical weight in this composition. The transformed Gram need not itself be an original graph Gram; the arithmetic proof only needs the certified inverse-norm bound for the represented true pivot, which the adapter establishes.

The legacy routine performs outward rational-to-dyadic boxing, integer trial construction, integer residual norms rounded up with `isqrt`, exact corrected-center arithmetic, and floor/ceiling endpoint divisions. The floating trial is not the certificate for this local operation. Small singular midpoint cases choose a zero trial and remain covered by the residual expression.

**Source evidence:** `src/weighted_adapter.py:24–44`; overlay `.../tools/promotion_directional.py:6–19,22–80`.

### 2.3 The weight-64 example in all relevant units

For the negative two-node edge of magnitude 64,

`A_p=[[65,64],[64,65]]`, `s_p=(1,-1)`, `z_p=(1,-1)`.

The independent physical parent-minus-child objective is `256/129`. The following exact matrices explain the apparent norm discrepancy:

| Object | Exact diagonal | Inverse spectral norm |
|---|---|---:|
| Paper's `B=J+U^T A_p^-1 U` with `U=8V` | `(129, -1/129)` | 129 |
| Paper's `Bbar=B/omega` | `(129/64, -1/8256)` | 8256 |
| Runtime integer-normalized `B_i=Bbar/16` | `(129/1024, -1/132096)` | 132096 |
| Adapter `K=J+dG_i`, `d=1024` | `(129, -1/129)` | 129 |
| Effective legacy pivot `cK`, `c=129` | `(16641, -1)` | 1 |

In these units, `G_i=diag(1/8,1/1032)`, `h_i=(1/8,0)`, the raw integer-operator gain is `16/129`, and multiplying it by 16 gives the physical `256/129`. The paper's unnormalized `h=(16,0)` and solve direction `(16/129,0)` are consistent with the same result.

**Source evidence:** manuscript `sections/05_numerics.tex:59–60` and `sections/03_responses.tex:36–37`; adapter derivation above. The numbers in this table follow algebraically from the displayed physical operator, rather than a floating numerical norm.

**Admission boundary:** `AllCatalog` and `Kernel.validate` reject `|W|>10` and non-half-integer root ratings; the A2 session itself admits 1 through 82,140 vertices. The weight-64 fixture therefore cannot be called an admitted full A2 service test. The harness verifies that rejection, then calls the genuine generic local entrypoint `weighted_adapter.correction` with independently certified Gram/projection boxes. It makes no weight-64 response-acquisition or whole-service claim. The other four prescribed families use actual `EndpointKernel` and actual `A2Session` without any admission monkeypatch.

**Source evidence:** `src/a2_catalog.py:91–95`; `src/weighted_adapter.py:257–266`; `src/a2_session.py:40–46`.

## 3. Equality, strict positivity, versions, and coverage

The production resolver has three relevant paths:

1. If no rival upper overlaps the current positive lower bound, return the ordinary interval decision. A nonpositive catalog can return no winner.
2. If every overlapping rival has a current exact single-transposition witness, intersect valid equal-gain intervals and choose the minimum canonical identifier. Keep a genuine positive lower bound; do not promote a lower endpoint to an upper endpoint.
3. If the structural path is insufficient, graphs of size at most 16 use direct rational full parent/child objectives on the overlap set. Larger graphs return `unresolved`.

**Source evidence:** `src/weighted_adapter.py:319–366`. The exact rational fallback converts `s` from binary64, but `A2Session` constructs it as `s_num/65536` from bounded integer numerators, so this particular conversion preserves the intended dyadic opinion exactly (`src/a2_session.py:78–82`). It would not justify an unrelated arbitrary decimal-opinion API.

The plan's two equal path-end flips have disjoint endpoints. The executable witness rejects a four-vertex symmetric difference because it supports one transposition only. A successful small-path decision can therefore depend on the exact fallback; it must not be attributed to a transposition witness. The harness observes executed resolver lines without replacing the resolver and records fallback usage. It adds one separate three-node twin fixture to exercise the actual successful witness branch.

The witness producer checks the bound operator object, its symmetry/canonical int64 form, exact opinion invariance, current candidate legality, edit kind and coefficient, affected sparse parent rows, and update mapping. Its standalone pair API receives a parent-version string from the caller and cannot validate that string against an external trajectory. In the real request, the session supplies the current version; the independent checker separately reconstructs and checks the version chain. This is an interface trust boundary, not a claim that an arbitrary nonempty version string is independently authenticated by `certify_exchange` alone.

**Source evidence:** `src/exact_ties.py:20–131`; checker `src/a2_checker.py:115–190,360–424`.

The independent checker regenerates all current legal root pairs. A declared equality class does not excuse an omitted stronger rival or an earlier equal canonical candidate. For small graphs it directly solves every relevant child exactly; in its numerical path it verifies or independently discovers required equality relations. It explicitly represents an `unresolved` row as `certified=False` and never certifies its termination.

**Source evidence:** `src/a2_checker.py:296–351,384–400,436–546,579–588`.

The bounded harness attacked actual checker APIs with stale parent/witness versions, wrong edit weight, wrong operator/score scaling, an invalid swap, missing witness coverage, a near tie presented as an exact relation, an omitted stronger rival, an omitted earlier equal rival, and a zero-gain acceptance. All were rejected. The invalid-swap attack failed at exact opinion invariance; this process did not independently exercise every parent-conjugacy and child-conjugacy rejection branch. The valid numerical-checker path did reconstruct and validate a full parent/child witness. Adversarially altered records are labelled as such and never counted as production successes.

## 4. Independent replay does not prove every reported endpoint

There is a precise verification-strength limitation in the actual checker. For an accepted row, it verifies

`max(published_low, independent_low) <= min(published_high, independent_high)`.

When the independent interval is a point from an exact small solve, this is exact containment of the true gain. When the independent interval is not a point, intersection is only a consistency check: a narrower wrong interval can intersect a valid wider interval. The checker records the distinction as `exact containment` versus `independent interval intersection`.

**Source evidence:** `src/a2_checker.py:558–567`; structural source/shared interval consistency uses the same type of intersection at `:205–224`.

This limitation is **not an independently established counterexample to the production gain kernel**. The checker can still independently certify an accepted winner's positivity and full-catalog canonical optimality from its own intervals and structural proofs. It does mean that a claim such as “all published endpoint values are independently proved valid” is stronger than this checker implements for nonpoint independent intervals. In particular, the historical 36 interval-disagreement comparisons cannot be declared fully resolved solely because independently replayed winners agree. A next-step endpoint claim requires exact truth or a narrower independently rigorous enclosure that itself lies inside each published interval, with enough accuracy to discharge the endpoint assertion.

No blanket rejection of all nonpoint interval intersections is proposed: replacing this check with a containment requirement would change the evidence accepted by the checker and may reduce its completion rate without exposing an engine error. The minimal present correction is to retain the existing truthful verification label and use it in reports and manuscript claims.

## 5. Manuscript theorem conditions versus the actual runtime

The finite-horizon theorem requires `delta>0`, complete finite access, bounded weights, legal states, support-preserving bases, and every true residual, assembly error, and complete local arithmetic allowance bounded by a common attainable `epsilon`. Its response count is `N+b`, not total service work. The source explicitly says the theorem does not imply completion of the `delta=0` experiments and that the displayed example requires approximately `5.11e-20`, below generic binary64 roundoff allowances.

**Source evidence:** manuscript `sections/05_numerics.tex:93–115`; full proof in `sections/10_appendix.tex:40–63`. Actual A2 request admission fixes `delta=0` (`src/a2_session.py:68–70`) and calls `WeightedViews`, which transports its stored Grams rather than implementing the general maintenance reanchor policy (`src/weighted_adapter.py:78–111`; manuscript `sections/05_numerics.tex:83–91`).

The legacy `bounds.py` is a scalar reference implementation. It takes certified nonnegative bounds from its caller, evaluates its scalar expressions exactly with `Fraction`, and rounds public caps upward and the sufficient epsilon downward. It does not establish the input caps itself and is not the A2 interval core. Its prior-constant routine is a useful check of the stated theorem's sufficient arithmetic target, not evidence that the runtime attains that target.

**Source evidence:** overlay `.../tools/bounds.py:1–5,10–35,65–92`. The historical weighted stress code uses physical bases containing `sqrt(w)` and `beta=1+2w`, so its `beta=129` for `w=64` belongs to the unnormalized physical pivot, not to `Bbar`. The tested stress weights are `(1,4,16,64)`, whose square roots are exactly representable. **Source:** overlay `.../tools/flip_stress.py:25–27,137–161,201–224`.

The actual scalar check returned `epsilon=5.1139908617745513e-20` for `(w,b,delta)=(1,40,1e-9)`, versus binary64 unit roundoff `2^-53=1.1102230246251565e-16`. This agrees with the paper's stated example and confirms its conditional scope, without establishing or requiring runtime completion at delta zero.

The current paper already makes the main completion qualification. No full-paper rewrite or claim that binary64 invalidates posterior certificates is warranted. The useful manuscript clarification is to spell out the scaling table (or its compact equivalent), retain “conditional arithmetic guarantee,” and continue reporting unresolved states separately. Fixed binary64 trial solves and exact dyadic residual/posterior certification can coexist correctly even when a universal sufficient epsilon is much smaller than ordinary rounding.

## 6. Bounded harness and interpretation

Root-scheduled command from the workspace root:

```bash
python v3/resumed/analysis/production_certificate_checks.py \
  --runtime v3_runtime \
  --oracle v3/code/exact_oracle.py \
  --output v3/resumed/reports/numeric_certificate_checks.json
```

The root scheduler must impose the already authorized 120-second/2-GiB limit and single-thread environment and charge actual process wall time to the 10,800-second ledger. The harness itself additionally uses `threadpool_limits(1)`.

Executed denominators are deliberately separate:

| Evidence | Executed denominator and scope |
|---|---|
| Prescribed physical reference values | 5 mathematical families, 17 gains; existing independent oracle reused |
| Actual A2 root gain intervals | 4 admissible families, 16 gain intervals; all legal candidates, not only winners |
| Generic weight-64 local interval | 1 gain; independently solved parent-system inputs; no A2 admission bypass |
| State/response/Gram/projection/screen checks | All 16 admitted root gains, with exact physical source quantities |
| Transport | 1 actual commit, 4 remaining candidates; direct rational child-system comparison |
| End-to-end small production decisions | 5 families × D/S/H × 1 budget-one request = 15; one family is the extra structural-twin case |
| Checker variants | Actual exact small replay for the 15 requests; one documented `exact_limit=0` replay of a valid witness |
| Rejection checks | 16 expected exception rejections plus producer API negative checks; concrete malformed-evidence risks only |
| Historical real-data numerical replay | 0 in this process |

The actual assertion count was 174 and the status was `passed`. All 15 toy requests completed, with 12 accepted positive canonical edits and three certified zero-gain stops. The near path chose canonical ID 11; the true tie chose ID 1 through the exact fallback. A safe `unresolved` result would have been logged as a completion issue rather than a false answer. Unsupported broader claims remain explicitly unknown. This bounded result permits the root to consider the next correctness gate; it does not replace the separate G access proof or the real Alpha classification.

## 7. Finding classifications and minimal changes

| Finding | Classification | Action |
|---|---|---|
| Norms 129 and 8256 refer to `B` and `Bbar` respectively | Unit distinction; no source-based kernel defect | Preserve explicit unit conversion and test physical gains |
| Adapter changes `K` to `cK` before legacy bound 3 | Correctly motivated normalization in actual source | No mathematical patch proposed |
| Weight64 not admitted by natural-weight A2 | Interface scope | Generic local-kernel test only; retain real admission rejection |
| True path tie needs more than one vertex transposition | Expected structural incompleteness, with a small exact fallback | Record actual fallback; do not credit a witness |
| Unsupported larger overlaps return unresolved | Completion limitation | Classify real Alpha independently; no epsilon/tolerance relaxation |
| Large numerical replay checks interval intersection | Independent endpoint verification limitation | Keep exact wording; do not infer universal endpoint validity |
| Finite theorem excludes delta0 completion | Already qualified theoretical scope | No broad theorem/runtime correctness accusation |

No core patch is proposed: the static derivation and executed bounded harness found no counterexample. The needed claim corrections concern scope and attribution: weight64 is a generic-kernel check, the path equality was completed by the exact fallback, and nonpoint interval intersection is not independent endpoint validity. Any later counterexample must preserve its exact input, physical truth, invalid interval or false decision, and executed path before a minimal coordinated patch is made.
