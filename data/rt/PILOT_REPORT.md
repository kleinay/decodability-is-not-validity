# Pilot: sentiment (T) x negation (N) on Rotten Tomatoes, as a replacement for Civil (2026-10-29)

Script: `scripts/01h_negation_edits.py` (edits with gpt-oss-120b; Groq judges gpt-oss-120b and qwen3.8-27b, same
judge prompt as `02c`). Total Groq spend for the pilot: about $0.40.

## Data
`cornell-movie-review-data/rotten_tomatoes` (Pang & Lee sentence polarity; 8,530/1,066/1,066). T = negative
sentiment, N = contains a negation word (regex `NEG` in the script).
Natural correlation: P(negation | negative) = .29, P(negation | positive) = .19 (train; dev/test the same).
Real, but modest: ρ=0.7/0.9 are resampled stress regimes, as for Civil. Smallest train cell (positive & negation) = 819.

## Edit method
The LLM proposes phrase replacements as JSON (`"not good" -> "bad"`; `"dull" -> "not interesting"`), applied by
exact match so only the named phrases change; it may skip. Automatic checks: negation status flipped (regex),
≤35% of words changed, length stable.
N is lexical, so it is verified by the regex rather than by the judges: the judges missed real negation flips
(e.g. "doesn't manage" -> "fails" judged "same negation") 7/76 times each. Judges decide: sentiment unchanged,
no major other meaning change, no fluency drop.

## Results (100 test sentences, 25 per T x N cell)
| sample | auto-ok | gpt-oss accepts | qwen accepts | both accept (of generated) |
|---|---|---|---|---|
| seed 13 (prompt development, 3 iterations) | 76 | 91% | 80% | 55% |
| seed 99 (held out, prompt frozen) | 83 | 92% | 80% | **62%** (add 65%, remove 84% of auto-ok) |
Civil for comparison: 13% (forward) / 29% (reverse) accepted by both.
Manual read of 30 accepted held-out pairs: ~26 clean. Residual problems: a movie title edited ("never again"),
garbled multi-negation rewrites, some weakened intensity.

## Known risk to address before adoption
**Intensity confound.** "not + antonym" is often softer than the original ("unwatchable" -> "not watchable",
"strong" -> "not weak"), and removing a negation can sharpen it. A sentiment score that correctly tracks
*intensity* would then move under N edits, which looks like nuisance sensitivity. Mitigations: (1) add a judge
question on strength of opinion and require "same"; (2) report CF_N separately for add vs. remove and per T class
(the softening projects with opposite signs); (3) synonym placebo floor as for the other datasets.

## If adopted: full pipeline
Edit all 1,066 test sentences (~$0.50; expect ~650 accepted pairs vs. Civil's 212), plus placebo edits; local
Qwen3-32B judge + extraction via `gq` (`02c`, `02_extract`); `01_build_datasets` entry for `rt`; fit, cf variants,
tables; paper: replace Civil in the main text, keep Civil as an appendix showing the judge filter rejects
identity swaps that change a claim.
