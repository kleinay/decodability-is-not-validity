#!/bin/bash
set -e
cd /home/ayalk/ISF27/preliminary-projects/Construct-Validity-beyond-Decodability/acl_short
source /etc/profile.d/shared-model-cache.sh
PY=/home/ayalk/.venvs/construct-validity/bin/python
$PY scripts/02_extract.py qwen7b > logs/rev_extract_qwen7b.log 2>&1          # placebo tails
$PY scripts/02_extract.py gemma9b > logs/rev_extract_gemma9b.log 2>&1
$PY scripts/02_extract.py qwen7b:last > logs/rev_extract_qwen7b_last.log 2>&1 # pooling ablation
$PY scripts/02c_judge_edits.py > logs/judge.log 2>&1
