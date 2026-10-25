#!/usr/bin/env bash
# Usage: bash kaggle/push.sh <kaggle_username> <stage: 1|2>
# Builds a single-file kernel script from src/pipeline.py and pushes it as a private GPU kernel.
set -e
U=$1; S=$2; D=$(dirname "$0")/stage$S
mkdir -p "$D"
{ echo "STAGE='$S'"; cat "$(dirname "$0")/../src/pipeline.py"; } > "$D/stage$S.py"
KS='[]'; [ "$S" != "1" ] && KS="[\"$U/kion-rl-stage1\"]"
cat > "$D/kernel-metadata.json" <<EOF
{"id": "$U/kion-rl-stage$S", "title": "kion-rl-stage$S", "code_file": "stage$S.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": ["$U/kion-dataset"], "kernel_sources": $KS, "competition_sources": []}
EOF
python -m kaggle kernels push -p "$D"
