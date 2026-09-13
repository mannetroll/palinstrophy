# Two-field nonlinear formulation — rejected

Tested on Apple M1 Max, 2026-09-13, against the updated
`tools/sim_512_1001_*.txt` references. The production solver was not edited.

The candidate transforms `u²−v²` and `uv` instead of `u²`, `v²`, and `uv`.
Only product construction and nonlinear spectrum assembly are replaced inside
the experiment process. Initialization, float32 precision, de-aliasing,
integration arithmetic, timestep updates, diagnostics, and FFT normalization
remain as in the current solver. The candidate uses two forward transforms;
the existing three-plane spectral allocation is retained for reconstruction.

## Required numerical comparison

Both runs used seed 1, N=512, Re=10000, K0=10, 1001 steps, CFL=0.25,
UPDATE=100, PAO, and CNAB2. Each backend was compared with its own updated
macOS reference, without adding a tolerance or changing expected values.
Elapsed times, FPS, and environment/header differences are excluded from
the numerical comparison.

The unchanged CPU and MLX solvers each reproduced all 15 numerical records
exactly: the two initial timestep records, 12 iteration records, and final
T/CN/DT record. Thus the candidate differences below are not baseline drift.

| Backend | First changed recorded value | Baseline | Two-field candidate |
|---|---|---:|---:|
| SciPy CPU | CN at iteration 100 | 1.00462324 | 1.00462322 |
| MLX GPU | CN at iteration 100 | 1.00462341 | 1.00462339 |

Each candidate changed 24 of the 60 printed iteration values across
T, DT, CN, CFLM, and TAU_L. All candidate fields remained finite, but neither
candidate preserved the required numerical results.

The full final arrays were also compared with the unchanged solver from
the same backend, using double precision to calculate relative L2 differences:

| Backend | Spectral vorticity `om2` | Nonlinear history `fnm1` | Physical velocity |
|---|---:|---:|---:|
| SciPy CPU | 2.302896e-6 | 8.570174e-5 | 2.524466e-7 |
| MLX GPU | 2.402619e-6 | 8.994580e-5 | 2.675435e-7 |

Combining the squares before the float32 FFT changes rounding relative to
subtracting the two transformed squares. The exact-arithmetic identity does
not ensure identical float32 trajectories.

## Decision and scope

**Reject under the requirement to preserve baseline numerical outputs.**
No production implementation was installed, and no performance benchmark
was pursued after rejection. Elapsed times in the correctness-run logs are
retained as raw evidence, not an accepted speedup measurement.

All four updated reference files and all production Python modules retain
their recorded SHA-256 hashes. Linux/SciPy and Linux/CUDA were not executed
on this Mac; their reference files were preserved without modification.

## Reproduction and evidence

From the palinstrophy repository root, with GPU access for MLX:

```sh
.venv/bin/python tools/two_field_experiment/run_experiment.py baseline cpu /tmp/two-field/baseline-cpu
.venv/bin/python tools/two_field_experiment/run_experiment.py two-field cpu /tmp/two-field/candidate-cpu
.venv/bin/python tools/two_field_experiment/run_experiment.py baseline mlx /tmp/two-field/baseline-mlx
.venv/bin/python tools/two_field_experiment/run_experiment.py two-field mlx /tmp/two-field/candidate-mlx
```

Each command writes a text log, a JSON summary, and final arrays in NPZ format.
The defaults reproduce the reference parameters above. The experimental
overrides support CPU and MLX only and do not persist outside that process.

- [Baseline reproduction](baseline-reproduction.json)
- [All numerical differences and final-field comparisons](comparison.json)
- [Source revision and reference/source hashes](manifest-updated-baselines.json)
- [CPU baseline](baseline-cpu.txt) and [candidate](candidate-cpu.txt)
- [MLX baseline](baseline-mlx.txt) and [candidate](candidate-mlx.txt)

The original final-array captures are at
`/tmp/palinstrophy-two-field-20260913/{baseline,candidate}-{cpu,mlx}.npz`.
Their array hashes are recorded in the corresponding JSON summaries here.
