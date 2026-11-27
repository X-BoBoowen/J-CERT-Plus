# Verification and archival checks

`reports/` contains the original V2 independent full-domain replay records for two preselected H requests, plus the V3 bounded numerical-coordinate and exact equality diagnostic evidence. The original implementation and checker remain in `runtime/src/` and the immutable archives.

The two Python audit scripts read the seven scientific archives directly. They check source hashes, request and fixture bindings, graph and opinion bytes, selected chains, G mechanism statistics, and exact stored interval inclusion. The G linkage covers the same 30 independently replayed H states and 90 G observations across three repeats.

Run `python scripts/verify_artifact.py --output-dir ../jcert-audit` from the repository root with a fresh output directory. These checks never solve a graph system or re-execute full-catalog replay. They establish properties of the retained evidence, not validity of all possible execution intervals or completion of every exact request. The original V2 confirmation completed five of six streams; the unresolved stream remains in the records.
