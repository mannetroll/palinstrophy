#!/usr/bin/env bash
set -euo pipefail

#
#   $ nohup stdbuf -oL bash sim_loop.sh > sim_loop.txt 2>&1 &
#

CSV="sim_metadata.csv"
[[ -s "$CSV" ]] || echo "N,Re,K0,CFL,VISC,STEPS" > "$CSV"

for N in 256 512 1024 2048 4096 8192; do
  for K in 5 10 15 20; do
    LOG="output_N${N}_K${K}.log"
    echo "Running N=${N} K=${K} ..."
    # N RE K STEPS CFL backend UPDATE ITERATIONS
    PYTHONUNBUFFERED=1 uv run -- turbulence "$N" 1E15 "$K" 1E5 0.2 auto 10 50000 2>&1 \
      | stdbuf -oL -eL tee -a "$LOG" \
      | awk -v csv="$CSV" '
          $0=="N, Re, K0, CFL, VISC, STEPS" { grab=1; next }
          grab && !done { print >> csv; fflush(csv); done=1; grab=0 }
        '
  done
done