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

## BiasBios LLM edits propagated + related work (2026-10-29)

- The LLM gender edits (gpt-oss-120b hybrid; pronouns, gendered nouns **and first names**), generated/judged/extracted on 10-15, had never reached the results: `results/` and the PDF still used the rule-based pronoun swap (1,589 pairs, 579 accepted). Re-ran `03_fit_population.py bios` (3 readouts) + `run_after_judging.sh`. Now 1,561 pairs, 1,451 accepted by both judges (93%; κ .30→.55). Rule-based pairs kept as kind `nuisance_rule`, unused.
- Effect on paper: BiasBios CF_N slightly higher (e.g. ρ=0.9 equivalent range .08–.41 → .09–.42 Qwen; .09–.29 → .10–.33 Gemma); every qualitative claim re-checked and holds (Conflict lowers CF_N and |CF_N| in every sample; erased probe picked by Conflict in every sample; ±0.08 at ρ=0.5; audit PASS). Text: BiasBios edit description, Table 1 caption, Limitations ("first names" cue removed).
- Related work added (arXiv API metadata): Tiwari et al. 2026 "Decodability is Not Causality" (2609.18080), Muthya Sudheendra & Srivastava 2026 (2609.02438), Shen et al. 2026 "Beyond Decodability" (2605.00607), Noël 2026 (2608.13329); explicit contrast: they ask whether a decodable direction drives behavior, we ask what the score tracks in the input.
- Trimmed ~70 words (redundant Discussion sentence etc.) to keep the main text at 4 pages.
- Human-audit sample regenerated (still unannotated) so its BiasBios rows show the new edits.
- Pre-rebuild snapshot: `backup_pre_bios_llm_20261029/`.

## Civil replaced by Amazon (sentiment x negation, sentiment x first person) (2026-10-29)

- Why: Civil's rule-based identity swaps are mostly rejected by both judges (13% / 29%) because swapping the group changes a claim about it; accepted pairs are a skewed subset. Civil is kept in every computation and reported in a new Appendix C only.
- New settings (`scripts/01j_build_amazon.py`, `01i_amazon_edits.py`): short amazon_polarity reviews (8-40 words), T = negative sentiment; N = negation (natural .64 vs .30) or first person (.66 vs .68, uncorrelated), both defined by word lists so the flip is regex-verified. gpt-oss-120b proposes JSON phrase replacements applied by exact match; judges gpt-oss-120b + Qwen3.8-27B (Groq) with an added strength-of-opinion question; acceptance rule in `src/dv/judging.py` (`nuis_flip_regex`, `strength_same`). Pilot: prompts tuned on seed 13, held-out seed 99 accepted 52% (neg) / 51% (1P) of generated by both judges (pilot CSVs in `data/amz/`; the earlier Rotten Tomatoes pilot is in `data/rt/`).
- Full run: 1,600 test reviews per nuisance -> 1,427 / 1,083 pass automatic checks; both judges accept 58% / 68%; placebos 608 / 567. Activations via gq job 101 (3 readouts). Groq spend: ~$3.0 for the full run (edits ~$1.6, judges $1.39), ~$0.9 for pilots.
- Findings: rho=0.9 near-best CF_N .11-.25 (1P, Qwen) vs .03-.12 (Neg); rho=0.5 ~0 for both. Baseline-adjusted increase .11-.16 (1P) vs .09-.11 (Neg). Negation edits have a constant |CF_N| .07-.09 at every rho (placebo .03) and a negative signed effect at rho=0.5, mostly for removals -> reported as a residual intensity change of the edits, not as nuisance reliance. Conflict selection lowers CF_N where the default is sensitive (CEBaB, BiasBios, Amazon-1P; all samples but one), changes little on Amazon-Neg; with the erased probe CF_N falls everywhere.
- Strict self-review fixes: dropped "first person absorbed more than twice as much" (confounded by negation's negative baseline) for the baseline-adjusted statement; "several-fold" -> "up to several-fold" (CEBaB Qwen is 1.5x); "robust to edit direction" scoped to rho=0.9; main-text macros pool MAIN datasets only (Civil in appendix).
- Audit fixes: the T-label check treated NaN==NaN (CEBaB unlabelled service) as a mismatch; the DoM/PLS-1 check took the first 2,000 train rows (single class in the new builder) and now samples randomly. All checks pass.
- Main text trimmed to 4 pages (Table 2 shows rho=0.9 only; rho=0.5/0.7 and Civil in Appendix B/C). Pre-change snapshot: `backup_pre_amazon_20261029/`.

## Judge validation without new human annotation (2026-11-02)
- `scripts/14_judge_validation.py`:
  - (1) CEBaB human-label check: 529 food/service edit pairs plus 160 ambiance/noise pairs from test+dev families, in `data/cebab_gold/`, judged by Qwen3-32B (local, gq job 102) and gpt-oss-120b. The 89 edits whose intended aspect did not change per the human majority are oversampled.
    - Per question: gpt-oss agrees with the human majority in .92 of pairs, as often as a held-out single worker agrees with the majority of the other four (.90–.91). Qwen3-32B: .77–.84.
    - Both-judge acceptance: precision .94 (vs .83 good in the judged set), rejects .84 of failed edits, keeps .50 of good ones.
  - (2) Planted errors on accepted edits, 50 per type, in `data/<ds>_planted/`. Share passing both judges:
    - flip ≤.08, occupation .00, garble ≤.10, mixed gender .18
    - strength ≤.18 (gpt-oss alone .58–.68)
    - minor fact .52–.88 (tolerated by the rule)
    - clean retest .86–.94
- `scripts/15_filter_robustness.py`: CF recomputed under the filters all / judge 1 / judge 2 / either / both.
  - Spearman vs the paper filter ≥ .87.
  - Conflict selection with the erased probe lowers CF in all 8 settings at ρ 0.9 under every filter.
  - CEBaB magnitudes differ (.41–1.15 on all edits vs .26–.89 on accepted ones).
- Paper: new Appendix A.1 (Tables 7, 8). The main-text edit-validation sentence and Limitations now point to it; the main text is still 4 pages.
- Script changes: alias entries in `02c` (ATTR/NOTE for `cebab_gold`, `bios_planted`) and in `01i` (ds prefix for `*_planted`).
- Groq spend ≈ $0.52.

## LLM readers + arXiv packaging (2026-11-05)
- `scripts/16_author_read_sample.py`: 100 accepted Amazon edits labelled by Claude Opus 5.5 and GPT-6.1 (`data/author_read/labels_*.csv`).
  - Agreement .73, κ .45.
  - Wrong: Amazon-1P .02–.14, Amazon-Neg .12–.22; negation additions .24–.28.
  - Reported in Appendix A.1 and Limitations as LLM readers, not human validation.
- `.gitignore`: texts with LLM edits (bios, amzneg, amzfp, validation sets) are now tracked, since they are not regenerable. Backups and superseded pilots are ignored.
- README updated: Amazon settings, BiasBios LLM edits, Civil appendix-only, pipeline steps 01e–01j and 14–16.
- `paper/arxiv_source.tar.gz`: tex + .bbl + tables + figures. Test-compiled from extraction (12 pages). Audit: all checks pass.
