# Datasets and exact query inputs

The original files in `artifacts/frozen/V2_Inputs.zip` are the authoritative frozen inputs; the same bytes are available under `runtime/` for convenient use. The manifest binds each file by SHA-256.

| Graph | Origin | Main domain |
| --- | --- | --- |
| Bitcoin Alpha | [SNAP Alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) | 3,783 vertices; 7,140,884 logical edits |
| Bitcoin OTC | [SNAP OTC](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) | 5,881 vertices; 17,271,859 logical edits |
| Slashdot | [SNAP signed social network](https://snap.stanford.edu/data/soc-sign-Slashdot090221.html) | Archived unit-signed projection; 82,140 vertices and 3,373,067,797 logical edits. Root numerical preparation stopped before service. |

The exact Alpha/OTC projection metadata and vertex maps are under `runtime/data/projected/`. The projection retains the last timestamped rating in each direction, averages observed reciprocal directions, drops self-loops and zero aggregates, and preserves the sorted vertex universe. The source metadata records the raw source URL, timestamps and hashes. Main experiments preserve projected natural weights; unit-weight versions are separate controls.

Main opinion inputs are generated computational queries, not measurements of user opinions: gradual-change (`slow`), independent uniform (`independent`), and signed-neighbor correlated (`signed-neighbor`). Opinion files store integer numerators on denominator 2^16. The twelve retest streams are in `runtime/data/a2/A2-confirmation-v1/`; the six later streams and one Slashdot input are under `runtime/v2/inputs/`. V4 reuses the six V2 inputs rather than generating new samples. Do not regenerate and substitute different opinions under an archived input identity.

The main domain permits one positive insertion with effective weight 1/16 on every root nonedge, and one magnitude-preserving flip on each negative root edge. Each edit is legal at most once per request. Graph versions and opinions determine current gains.

`legacy/v6/datasets/` contains the original six graph matrices for earlier studies, including controlled generated graphs Conflict, Bio and Fin. Their protocols use separate candidate families and weights. `legacy/bounded_scale/` provides corrected bounded-opinion inputs and records for 32 candidates at four synthetic graph sizes. Graph size is not equivalent to full-catalog size.

Original data sources and attribution remain in the metadata and paper references. Third-party data retain their original redistribution terms.
