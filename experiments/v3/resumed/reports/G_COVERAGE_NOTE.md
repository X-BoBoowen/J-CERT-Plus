# G coverage and implementation boundaries

The actual run used commit `c6b752630bd8dbe1f9fd296a7ed02f249f834575`, derived from V2 `4c2c7b7e84c7c03ee56d45453fea6950c94817bd`. G reuses the original numerical kernel, strict endpoint screening, Gram transport, resolver, and stopping contract. The only scheduling differences are processing flips first, deferring warm insertions, and replacing H's row frontier with independent block coverage. G does not call H's insertion frontier as a fallback.

## Block bounds and units

The production state satisfies `A_int=16 A_phys` and `z_int=A_int^-1 s=z_phys/16`. Coarse centers have denominator `SCREEN=2^20`. A coarse upper bound c means that physical gain is at most `16c/SCREEN^2`. The uniform insertion coefficient in the integer operator is d=1.

For a currently legal nonedge `(i,j)`, `A_int[i,j]=0`. Restricting the variational trial vector to the two endpoints gives `R_int >= 1/A_int[ii]+1/A_int[jj]`. With maximum diagonal values D_I and D_J in the respective blocks, `1/D_I+1/D_J` is therefore a valid denominator lower bound for legal nonedges in that block. Let w be the integer width enclosing the difference between the endpoint centers and both radii. Then

`ceil(w^2 D_I D_J / (D_I D_J+D_I+D_J))`

outwardly bounds the coarse gain. Take its min with the global insertion upper bound. This formula certifies insertions only; it never replaces the shared numerical computation for rank-two flips.

Flips do not change absolute weights or operator diagonals; each accepted insertion adds 1 to each of its two integer diagonal entries. G reconstructs the current diagonals from the bound root graph and current chosen set, rather than using stale root diagonals as incorrect denominators.

## Complete pair partition and stopping

Vertices are stably sorted by current center. The initial triangular block contains every unordered pair. A triangular block splits into LL, LR, and RR; a rectangle spanning two intervals splits along its longer side. Each legal pair belongs to exactly one leaf. Singleton same-block leaves are discarded, and diagonal pairs never enter the catalog. At leaves, root edges, committed edits, and already scored warm/cold items are further excluded.

The pending-block heap covers all pairs not yet generated; each expansion replaces a block with disjoint, exhaustive children. The flip tail is sorted by upper bound. Positive-gain equality contenders are excluded only when `upper < ceil(best/RATIO)`; nonpositive zero stopping is handled separately. Unproved equalities still go to the shared resolver: complete source coverage cannot be treated as a complete numerical decision.

All warm insertions are first removed from the warm/hot exclusion sets; existing Gram entries remain maintained in the shared cache. These insertions must be covered again by current block proofs or re-enter shared scoring. Thus, prioritizing flips cannot hide unscored insertions in exclusion sets.

## Verification and performance limits

- Enumeration on irregular small graphs checks pair completeness, absence of duplicates, physical gain bounds for every generated block, exclusion of chosen and processed items, and current diagonals.
- Four consecutive commits by the actual G implementation passed the independent small-graph full-catalog checker. Zero budget and request reset have separate checks.
- The independent full-catalog checker rejected stale parent versions and incorrect weights in actual G outputs, as well as a missing root block injected by a test. The latter is an artificial fault, not an observed production error.
- PairBlocks is an internal object in trusted execution, not a public portable certification API. The independent checker reconstructs the catalog and operator itself and does not trust G's `source_complete`, block list, or upper bounds.
- The current implementation partitions lazily without preallocating an M-entry table; there is no proof that the pending-block heap always remains O(n) for arbitrary inputs. Peak RSS and observed workloads should be reported as measured.
- Deep memory reporting is not mathematical work required for selection or certification. This round retains its actual cost and uses a separate diagnostic to identify its effect on the H/G timing difference; subtracted values are not presented as new timings without observation overhead.
