#!/bin/bash
# GPU side of the Amazon datasets (amzneg, amzfp): activations for all texts incl. appended edits/placebos, 3 readouts.
set -e
cd /home/ayalk/ISF27/preliminary-projects/Construct-Validity-beyond-Decodability/acl_short
source /etc/profile.d/shared-model-cache.sh
PY=/home/ayalk/.venvs/construct-validity/bin/python
$PY scripts/02_extract.py qwen7b amzneg amzfp > logs/amz_extract_qwen7b.log 2>&1
$PY scripts/02_extract.py gemma9b amzneg amzfp > logs/amz_extract_gemma9b.log 2>&1
$PY scripts/02_extract.py qwen7b:last amzneg amzfp > logs/amz_extract_qwen7b_last.log 2>&1
echo AMZ EXTRACT DONE > logs/amz_extract_done.log
