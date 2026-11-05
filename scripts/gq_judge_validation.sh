#!/bin/bash
# Local judge (Qwen3-32B) on the judge-validation sets of 14_judge_validation.py.
set -e
cd /home/ayalk/ISF27/preliminary-projects/Construct-Validity-beyond-Decodability/acl_short
source /etc/profile.d/shared-model-cache.sh
PY=/home/ayalk/.venvs/construct-validity/bin/python
$PY scripts/02c_judge_edits.py cebab_gold bios_planted > logs/judge_validation_local.log 2>&1
