# Implementation audit (scripts/07_audit.py)

**Overall: ALL CHECKS PASS**


## cebab
- [PASS] uids disjoint across splits
- [PASS] no exact train/test text duplicates (CEBaB: 11 known generic duplicates, excluded by family split)
- [PASS] nuisance: sign == sign(N_b - N_a) for all 210 pairs
- [PASS] nuisance: sources come from test/cfsrc only
- [PASS] placebo: T and N labels identical for source and edit in all 158 pairs
- [PASS] target: sign == sign(T_b - T_a) for all 125 pairs
- [PASS] CEBaB review families disjoint train/dev/test

## bios
- [PASS] uids disjoint across splits
- [PASS] no exact train/test text duplicates (CEBaB: 11 known generic duplicates, excluded by family split)
- [PASS] nuisance: sign == sign(N_b - N_a) for all 1589 pairs
- [PASS] nuisance: sources come from test/cfsrc only
- [PASS] placebo: T and N labels identical for source and edit in all 961 pairs

## civil
- [PASS] uids disjoint across splits
- [PASS] no exact train/test text duplicates (CEBaB: 11 known generic duplicates, excluded by family split)
- [PASS] nuisance: sign == sign(N_b - N_a) for all 762 pairs
- [PASS] nuisance: sources come from test/cfsrc only
- [PASS] nuisance_rev: sign == sign(N_b - N_a) for all 400 pairs
- [PASS] nuisance_rev: sources come from test/cfsrc only
- [PASS] placebo: T and N labels identical for source and edit in all 480 pairs
- [PASS] placebo_group: T and N labels identical for source and edit in all 400 pairs

## PLS(k=1) vs difference of means
- [PASS] cebab: cos(w_DoM, w_PLS1) = 1.000000 (P(T=1)=0.468); same orientation
- [PASS] bios: cos(w_DoM, w_PLS1) = 1.000000 (P(T=1)=0.498); same orientation
- [PASS] civil: cos(w_DoM, w_PLS1) = 1.000000 (P(T=1)=0.528); same orientation
  (PLS-1 weight ∝ Zᵀ(t-t̄) ∝ μ₁-μ₀ exactly when classes are balanced; pls_k1 is therefore dropped from pairwise/correlation analyses as a duplicate of dom)

## Truncation and edit length (Qwen tokenizer)
- cebab: 0.0% of texts exceed 256 tokens (median 25)
  - nuisance: Δtokens mean +0.07, mean |Δ| 3.08, |Δ|≤2 in 67% of pairs; either side truncated: 0.0%
  - placebo: Δtokens mean +0.09, mean |Δ| 0.16, |Δ|≤2 in 100% of pairs; either side truncated: 0.0%
  - target: Δtokens mean +0.50, mean |Δ| 2.26, |Δ|≤2 in 65% of pairs; either side truncated: 0.0%
- bios: 0.0% of texts exceed 256 tokens (median 91)
  - nuisance: Δtokens mean -0.00, mean |Δ| 0.01, |Δ|≤2 in 100% of pairs; either side truncated: 0.0%
  - placebo: Δtokens mean +0.13, mean |Δ| 0.13, |Δ|≤2 in 100% of pairs; either side truncated: 0.0%
- civil: 0.2% of texts exceed 256 tokens (median 73)
  - nuisance: Δtokens mean +0.32, mean |Δ| 0.40, |Δ|≤2 in 98% of pairs; either side truncated: 0.3%
  - nuisance_rev: Δtokens mean -0.03, mean |Δ| 0.06, |Δ|≤2 in 100% of pairs; either side truncated: 0.2%
  - placebo: Δtokens mean +0.08, mean |Δ| 0.08, |Δ|≤2 in 100% of pairs; either side truncated: 0.4%
  - placebo_group: Δtokens mean +0.79, mean |Δ| 0.80, |Δ|≤2 in 96% of pairs; either side truncated: 0.2%

## Independent metric re-implementation vs pipeline (random 40 extractors per dataset/model)
- [PASS] dec: max |independent - pipeline| = 1.63e-14 (float32 score storage)
- [PASS] conflict: max |independent - pipeline| = 2.22e-16 (float32 score storage)
- [PASS] leak: max |independent - pipeline| = 2.22e-16 (float32 score storage)
- [PASS] cf_nuisance: max |independent - pipeline| = 4.93e-09 (float32 score storage)
