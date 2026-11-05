#!/bin/bash
# GPU side of the BiasBios LLM-edit rebuild: activations for the appended edit texts (3 readouts) + local judge.
set -e
cd /home/ayalk/ISF27/preliminary-projects/Construct-Validity-beyond-Decodability/acl_short
source /etc/profile.d/shared-model-cache.sh
PY=/home/ayalk/.venvs/construct-validity/bin/python
$PY scripts/02_extract.py qwen7b bios > logs/bios_llm_extract_qwen7b.log 2>&1
$PY scripts/02_extract.py gemma9b bios > logs/bios_llm_extract_gemma9b.log 2>&1
$PY scripts/02_extract.py qwen7b:last bios > logs/bios_llm_extract_qwen7b_last.log 2>&1
KINDS=nuisance $PY scripts/02c_judge_edits.py bios > logs/bios_llm_judge_local.log 2>&1
