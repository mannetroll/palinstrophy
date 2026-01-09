#!/usr/bin/env bash
set -euo pipefail

# log10(Re) = a*log10(N) + b*log10(K0) + c
Re_from_N_K0 () {
  local N="$1"
  local K0="$2"
  awk -v N="$N" -v K0="$K0" 'BEGIN{
    a=0.725220; b=0.082364; c=2.034392;
    log10Re = a*(log(N)/log(10.0)) + b*(log(K0)/log(10.0)) + c;
    Re = 10.0^log10Re;
    printf "%.6e\n", Re;
  }'
}

#
# nohup stdbuf -oL bash sim_loop.sh > sim_loop.txt 2>&1 &
#
rm -f output_N*
CSV="sim_metadata.csv"
echo "N, K0, Re, CFL, VISC, STEPS, PALIN, SIG, TIME, MINUTES, FPS" > "$CSV"

for N in 256 384 512 768 1024 1536; do
  for K in 2 5 10 15 20 30; do
    LOG="output_N${N}_K${K}.log"

    RE="$(Re_from_N_K0 "$N" "$K")"

    echo "Running N=${N} K=${K} RE=${RE} ..."
    # N K RE STEPS CFL backend UPDATE ITERATIONS
    PYTHONUNBUFFERED=1 uv run -- turbulence "$N" "$K" "$RE" 3E5 0.1 auto 10 50000 2>&1 \
      | stdbuf -oL -eL tee -a "$LOG" \
      | awk -v csv="$CSV" '
          $0=="N, K0, Re, CFL, VISC, STEPS, PALIN, SIG, TIME, MINUTES, FPS" { grab=1; next }
          grab && !done { print >> csv; fflush(csv); done=1; grab=0 }
        '
  done
done
