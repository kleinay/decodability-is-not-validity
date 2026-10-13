# Response to the critical review (revision 2026-10-12)

Point-by-point mapping from the review to what changed. Numbers quoted here are generated into
`tables/numbers.tex`; consult the PDF for the current values.

| Review point | Change |
|---|---|
| A. Construct validity defined too strongly | Reframed as *discriminant evidence about one named nuisance*; correlated constructs may covary, the test is dependence with the target held fixed. Title now "…Evaluating Nuisance Sensitivity in Language Model Concept Probes". Discussion: low nuisance sensitivity is necessary evidence, not construct validity. |
| B. Edits not validated | Two blind LLM judges (local Qwen3-32B; gpt-oss-120b via Groq, $0.65), random order, symmetric questions. Shared acceptance rule (`src/dv/judging.py`); all CF results use pairs both accept. Placebo edits (synonym swaps; Civil Christian→Protestant), reverse Civil swaps, length-matched subsets, last-token readout ablation. Blinded human-annotation sample: `data/human_audit_sample.csv` (+ key). **Not done: human judgments.** Judge agreement is only moderate (reported). |
| C. LEAK conceals dependence | Replaced by \|LEAK\| = mean of \|AUROC_t − 0.5\|; class-wise AUROCs stored. |
| D. CONFLICT not a validity measure | Stated as a stress test with the s = −N caveat; Conflict-selected scores also lower \|CF_N\|. Four selection criteria compared (max Dec, max Conflict, min \|Leak\| s.t. Dec, Dec − \|Leak\|), standard and +erased pools. |
| E. No equivalence test | Paired-bootstrap TOST (margin 0.01, 90% CI) for near-best extractors and the fixed pair; "tied" removed; CEBaB fixed pair at ρ=0.9 reported as *not* established equivalent. |
| F. Entanglement under-supported | Geometry: cosine of target vs within-class nuisance directions; observational prediction of DoM counterfactual sensitivity (CEBaB vs BiasBios). Synthetic experiment separating overlap from training correlation. Framed as one mechanism, with the error reported. |
| Signed CF = 0 ≠ invariance | \|CF_N\| reported throughout; placebo floors. |
| Denominator sensitivity | SD normalization; rank agreement reported. |
| Civil direction cancellation | Directions reported separately; forward/reverse rank agreement reported. |
| Near-best defined on test | Dev-defined near-best computed (`rev_A_nearbest.csv`); selection experiment uses dev only. |
| Selective correlation reporting | Full table of Spearman correlations with bootstrap CIs, all settings (appendix). Claim reduced to "not a reliable proxy; sign varies". |
| "Negligible" cost | "Modest accuracy cost, not free"; worst-seed costs reported. |
| ρ=0.7 missing | Appendix table. |
| Pairwise counts not independent | Tied-pair percentages removed from main text; PLS k=1 dropped as duplicate of DoM. |
| "Pre-specified pair" | Renamed "fixed-method comparison". |
| Input vs activation interventions | Stated in the introduction. |
| Missing related work | Canby et al. 2025, Bean et al. 2025, Licht et al. 2025 added (metadata verified); Kumar et al. and RAVEL distinctions moved to the introduction. |
| P0 code audit | `scripts/07_audit.py` → `results/AUDIT.md`: found and fixed 3 BiasBios eval texts duplicating training texts; all other checks pass (signs, splits incl. CEBaB families, PLS-1 ≡ DoM, truncation, independent metric re-implementation to <1e-8). CEBaB CIs now resample review families. |

Claims that did **not** survive and were removed: "decodability is often highest for the least valid
extractors"; strong representational-entanglement claim for decorrelated training on all datasets.

## arXiv corrections (2026-10-13, review of V3 — essential items only)

| Review point | Change |
|---|---|
| Judges "from different families" | Now "two distinct judge models, neither identical to a probed model"; Qwen3-32B's shared family with Qwen2.5-7B stated. |
| BiasBios 1,589 vs 800 | Table 1 now separates generated pairs, pairs judged by both (BiasBios: random subset of 800), and acceptance rate among jointly judged pairs. |
| Significance language on dependent configurations | Removed; correlations described as changing sign, intervals labelled descriptive. |
| Selection cost .023 / .049 | Text now reports mean costs (.018 standard; .036 with erased probe) with worst-sample values in parentheses. |
| .34 vs .35 | Mean drops computed from unrounded values (.342); table cells are rounded independently. |
| Which criteria pick the erased probe | Conflict and Dec−\|Leak\| in every sample; leakage-constrained rule in 67%. |
| κ "only moderate" | "Low to moderate"; acceptance described as a conservative filter, not verified correctness (abstract, §3, Limitations, App. A). |
| Synthetic "does not change" | Quantified: CF_N varies by .83–.87 across overlaps, decodability by ≤.03 (.09 for the erased probe). |
| Signed vs absolute at ρ=0.5 | Heading now "the *directional* effect mostly disappears"; example-level CEBaB shifts (median \|CF_N\| vs signed, placebo) stated in the main text. |
| Post-selection equivalence | Equivalence to the test-best reference labelled descriptive; selection comparisons use development data only. |
| Recent related work | Mueller et al. 2026 (ACL), Naowarat et al. 2026 (CoNLL), Li et al. 2026 (ACL), Lysnæs-Larsen et al. 2025 (arXiv) added (metadata verified); "has not been established" replaced by a narrower question. |
| Wording | "scalar proxy for a concept"; edits "intended to change only the nuisance"; materials "will be released". |

Deferred (not essential for arXiv): dev-referenced equivalence as the inferential centerpiece; signed/absolute side-by-side table across regimes; enlarging dense appendix tables; human audit; the NAACL-oriented extensions (multi-nuisance generalization, geometry-based prediction).

## Second reader (2026-10-13) — adopted selectively

Adopted: convergent-evidence claim scoped to CEBaB, the only dataset with target edits (abstract, §4, Discussion); nuisance erasure stated as estimated on the training sample only; effect-size (SD-normalized, Cohen's-d-like) robustness named in Limitations; the ρ definition kept on one line (the reported "missing parenthesis" was a line break, not a typo).
Not adopted: superposition/polysemanticity framing (the overlap is between semantically related aspect sentiments, not evidence of superposition); moving the synthetic experiment to the main text (already summarized there; page limit); judge-selectivity and construct-validity reminders (already in Limitations and the Discussion's last sentence).
Author email changed to ayalk@ariel.ac.il.
