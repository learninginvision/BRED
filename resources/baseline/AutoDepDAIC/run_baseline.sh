#!/usr/bin/env bash
# DAIC-WOZ baseline in one go: audit the data -> build the feature cache -> run the probe baseline.
#
#   bash resources/baseline/AutoDepDAIC/run_baseline.sh [GPU_ID]
#
# Output:
#   resources/baseline/AutoDepDAIC/code/metrics.json   the probe's floor (where the loop starts from)
#
# The DepAudioNet negative control (code/depaudionet_control.py) scores at chance level under the
# same protocol (macro_f1 .474 / auc .467) and does not need to be run again.
set -euo pipefail

GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-bred}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "###  1/3  Data integrity check"
python "$HERE/audit_data.py"

echo -e "\n###  2/3  Feature cache (cached subjects are skipped; about 50 minutes from scratch)"
python "$HERE/prepare_w2v2.py"

echo -e "\n###  3/3  Probe baseline (generalization protocol; 1-2 minutes with a warm cache)"
python "$HERE/code/experiment.py"

echo -e "\n### Summary"
python - <<'PY'
import json
m = json.load(open("resources/baseline/AutoDepDAIC/code/metrics.json"))
print(f"probe baseline  macro_f1={m['macro_f1']:.4f} ± {m['macro_f1_std']:.4f}"
      f"  auc={m['auc']:.4f} ± {m['auc_std']:.4f}")
print("ref: verified floor               macro_f1 .561 / auc .631")
print("ref: DepAudioNet negative control macro_f1 .474 / auc .467 (chance level)")
print("ref: HAREN-CTC, literature SOTA   macro_f1 .576 / auc .599 (code not released)")
PY
