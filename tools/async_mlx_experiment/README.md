# Bounded asynchronous MLX execution — rejected

Tested on Apple M1 Max with MLX 0.32.0 on 2026-09-13, starting from
revision `b4a16e6`. The candidate preserves numerical results exactly but
reduces measured T_R. No production implementation was installed.

## Scheduling change

The experiment replaces the completed-step blocking evaluation with
`mx.async_eval`, using the same six state-array roots. Python may construct
the next step while the GPU executes the submitted step. Before submitting
again, it waits for the previous submission. Thus only one submitted state
and the next graph being constructed can be outstanding.

The completion roots use `mx.array(a)` descriptor snapshots: retaining the
original Python array objects would let subsequent slice assignments change
which graph a completion check refers to. Explicit `state.sync()` still
evaluates the current state, synchronizes the GPU, and releases the retained
roots. FFTs, floating-point arithmetic, integration, timestep updates, and
rendering expressions are unchanged. The overrides exist only inside the
experiment process. [MLX API documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html).

## Correctness

- The baseline and candidate both reproduce all 15 numerical records in
  `tools/sim_512_1001_gpu_macOS.txt` exactly: seed 1, N=512, Re=10000,
  K0=10, 1001 steps, CFL=0.25, UPDATE=100, PAO, CNAB2.
- Final `om2`, `fnm1`, `uc_full`, `ur_full`, `uc`, and `ur` arrays are
  bitwise identical, and final t/dt/cn/cnm1 values match exactly.
- The candidate CPU run reproduces its macOS reference exactly.
- N=1024 RK3 with rendering every two steps produces identical state arrays,
  scalar values, and pixels at all ten checkpoints through step 20.
- All eight benchmark pairs also finish with identical arrays and scalar
  values after 100 warmup plus 1000 measured steps. Every captured field is
  finite.
- Compilation and the repository's N=128, 200-step, CFL=0.05, KM3, seed 4895
  `CHECK` runs completed on CPU and MLX. CHECK compares the two different
  integrators; it is supplementary to the exact candidate/baseline checks.

All four `sim_512_1001_*.txt` references and every production Python module
were verified byte-for-byte against the starting Git revision. Linux/CUDA
was not executed on this Mac.

## Performance

Four sequential baseline/candidate pairs per workload, reversing order in
alternate pairs, with a fresh process per run. Each run uses 100 warmup
steps and 1000 measured steps at N=1024. T_R is simulated time advanced
divided by elapsed wall time, with GPU completion included at both timing
boundaries. No concurrent benchmark runs were launched.

| Workload | Baseline T_R | Async T_R | Change | Baseline peak MLX | Async peak MLX |
|---|---:|---:|---:|---:|---:|
| CNAB2, frame interval 100 | 0.039942 | 0.034984 | -12.41% | 387.4 MiB | 553.5 MiB |
| LS-IMEX-RK3, frame interval 2 | 0.108826 | 0.106058 | -2.54% | 1042.1 MiB | 1208.2 MiB |

Values are medians across four runs; percentage changes compare those
medians. All paired changes were negative. The fourth RK3 candidate run
was a slower outlier (-8.72%); the other three pairs were -2.49% to -2.60%.
Median per-run p95 frame intervals worsened from 311.61 to 357.07 ms for
CNAB2 and from 19.60 to 19.98 ms for RK3.

These benchmarks exercise `DnsSimulator.step()` and `get_frame_pixels()`.
They include timestep regulation and frame extraction at the stated cadence,
but exclude Qt painting, GUI status diagnostics, and adaptive Reynolds
regulation. The RK3 parameters match the existing MLX correctness fixture:
seed 4895, Re=27380.85174, K0=5, CFL=2, KM3. CNAB2 uses seed 1, Re=10000,
K0=10, CFL=0.25, PAO. Their absolute T_R values are not a comparison of
integrator efficiency, nor a comparison against the native Swift simulator.

The measurements establish that this bounded implementation is slower;
they do not isolate how much of the regression comes from submission
overhead, retained buffers, or GPU scheduling. Reject it under the objective
of improving speed while preserving numerical results.

## Reproduce

Run from the palinstrophy repository root with Metal GPU access:

```sh
.venv/bin/python tools/async_mlx_experiment/run_experiment.py baseline cli /tmp/async-mlx/baseline-mlx
.venv/bin/python tools/async_mlx_experiment/run_experiment.py async cli /tmp/async-mlx/candidate-mlx
.venv/bin/python tools/async_mlx_experiment/run_experiment.py async cli /tmp/async-mlx/candidate-cpu --backend cpu
.venv/bin/python tools/async_mlx_experiment/run_experiment.py baseline frames /tmp/async-mlx/baseline-frames --n 1024 --steps 20 --update 2 --seed 4895 --re 27380.85174 --k0 5 --cfl 2 --spectrum KM3 --method LS_IMEX_RK3
.venv/bin/python tools/async_mlx_experiment/run_experiment.py async frames /tmp/async-mlx/candidate-frames --n 1024 --steps 20 --update 2 --seed 4895 --re 27380.85174 --k0 5 --cfl 2 --spectrum KM3 --method LS_IMEX_RK3
.venv/bin/python tools/async_mlx_experiment/benchmark.py /tmp/async-mlx
.venv/bin/python tools/async_mlx_experiment/summarize.py /tmp/async-mlx
```

The summarizer asserts exact reference records, array hashes, and scalar
values before summarizing timing. It also checks that references and
production Python modules match HEAD; it never rewrites them.

- [Exact correctness and paired performance summary](results/summary.json)
- [Revision and protected-file hashes](results/manifest.json)
- [Raw logs and per-run measurements](results/)

Full correctness arrays are retained in
`/tmp/palinstrophy-async-20260913/*.npz`, with their SHA-256 hashes in the
corresponding JSON files. The repository retains text/JSON evidence only.
