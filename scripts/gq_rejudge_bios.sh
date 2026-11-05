#!/bin/bash
# Re-judge BiasBios edits with the first-name note (local Qwen3-32B); then rebuild everything downstream of the judges.
set -e
cd /home/ayalk/ISF27/preliminary-projects/Construct-Validity-beyond-Decodability/acl_short
source /etc/profile.d/shared-model-cache.sh
PY=/home/ayalk/.venvs/construct-validity/bin/python
$PY scripts/02c_judge_edits.py bios > logs/judge_bios_v2.log 2>&1
