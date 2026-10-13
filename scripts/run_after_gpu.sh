#!/bin/bash
# Full CPU-side rebuild once GPU job outputs (placebo activations, Qwen3 judge, last-token activations) exist.
set -e
cd /home/ayalk/ISF27/preliminary-projects/Construct-Validity-beyond-Decodability/acl_short
PY=/home/ayalk/.venvs/construct-validity/bin/python
for ds in cebab bios civil; do
  for m in qwen7b gemma9b qwen7b_last; do $PY scripts/03_fit_population.py $ds $m > logs/fit_${ds}_${m}.log 2>&1; done
done
$PY scripts/08_cf_variants.py qwen7b gemma9b qwen7b_last > logs/cfv.log 2>&1
$PY scripts/09_revision_analyses.py > logs/rev_analysis.log 2>&1
$PY scripts/11_figures_rev.py > logs/figs.log 2>&1
$PY scripts/12_tables.py > logs/tables.log 2>&1
$PY scripts/13_pooling_ablation.py > logs/pooling.log 2>&1
$PY scripts/07_audit.py > logs/audit.log 2>&1
cd paper && ~/.local/bin/tectonic -X compile main.tex > ../logs/latex.log 2>&1
echo REBUILD DONE
