#!/usr/bin/env bash
# Reproduce every figure (about 90 minutes on 8 cores).
# Options: --quick (low statistics, a few minutes), --workers N (parallel processes).
set -e
cd "$(dirname "$0")/scripts"
for f in fig_rescaling fig_plateau fig_alignment fig_noise_floor fig_phase_diagram fig_plateau_checks fig_relu fig_relu_rescaling; do
  echo "== $f"
  python "$f.py" "$@"
done
