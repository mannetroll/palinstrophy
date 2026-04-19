# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Direct Numerical Simulation (DNS) of 2D homogeneous incompressible turbulence using spectral (Fourier) methods. The solver supports CPU (NumPy/SciPy) and GPU (CuPy/CUDA) backends behind the same API — array ops go through an `xp` alias that resolves to `numpy` or `cupy`.

Python is pinned to exactly 3.13 in `pyproject.toml`. CuPy and Numba are optional; code paths degrade gracefully if they are absent.

## Setup and Commands

```bash
# Install (CPU only)
uv sync

# Install with GPU/CuPy support (Linux/Windows with CUDA 13)
uv sync --extra cuda

# Run interactive PySide6 GUI (entry point: turbo_main.main)
uv run turbulence
# turbulence also accepts CLI args: N K0 Re STEPS CFL BACKEND UPDATE ITERATIONS
# (see sim_loop.sh for a batch example)

# Run headless CLI solver (entry point: turbo_simulator.main)
#   args: N Re K0 STEPS CFL BACKEND   (BACKEND ∈ {cpu, gpu, auto})
uv run sim 256 10000 10 1001 0.75 cpu

# Post-process saved PGM dumps (entry point: turbo_postprocess.main)
uv run color

# View a saved case folder (u_velocity/v_velocity/kinetic/omega PGMs)
uv run python -m palinstrophy.turbo_viewcase /path/to/case_folder

# Batch parametric sweep across N and K0 (writes sim_metadata.csv)
bash sim_loop.sh
```

There are no automated tests. Validation is done by running simulations and inspecting the output fields visually or via the power spectrum.

Standalone app bundles are built with PyInstaller using `macos.spec` / `win32.spec` (see `make_dmg.sh` for the macOS dmg flow).

## Architecture

### Core solver — `palinstrophy/turbo_simulator.py`

All simulation state lives in `DnsState` (dataclass). Structural port of a CUDA/Fortran code — comments in the file map each Python function to its CUDA origin. Key entry points:

- `create_dns_state(N, Re, K0, CFL, backend, seed)` — builds grid, spectral operators, and PAO random velocity field
- `dns_step2b` / `dns_step3` / `dns_step2a` — three-stage Crank–Nicolson time integrator (matches the CUDA `STEP2B → STEP3 → STEP2A` ordering)
- `next_dt` — CFL-adaptive time step (uses a custom CuPy `ReductionKernel` on GPU)
- `dns_pao_host_init` — PAO random-field initialization (Fortran LCG RNG, optionally Numba-JIT)
- `get_xp(backend)` — returns `numpy` or `cupy`; every solver op goes through this

Array layouts mirror the CUDA code: compact `UR (NZ, NX, 3)` AoS, compact `UC (NZ, NK, 3)` spectral, 3/2-dealiased `UR_full (3, NZ_full, NX_full)` SoA, plus spectral vorticity `om2` and nonlinear term `fnm1`. 3/2 de-aliasing is applied in spectral space.

### GUI — `palinstrophy/turbo_main.py`

Large (~1900 line) PySide6 frontend. Drives the DNS loop via `QTimer`, renders the selected field through an Indexed8 palette (fast — avoids per-frame RGB conversion). GPU pixels are only pulled to host when a frame is about to be drawn; never inside the integrator. Single-key shortcuts control the run (H=stop, G=start, V=cycle variable, C=cycle colormap, etc.). Exports PNG frames and full-resolution PGM dumps.

### Wrapper — `palinstrophy/turbo_wrapper.py`

`DnsSimulator` bridges the GUI and core solver: init, single-step advancement, field extraction (U, V, kinetic energy, vorticity, stream function), and SciPy FFT worker-count control on CPU via `scipy.fft.set_workers`.

### Post-processor — `palinstrophy/turbo_postprocess.py`

Loads PGM dump folders and re-renders with the GUI's colormap controls.

## Key Implementation Details

- **Hot loop discipline**: fields stay in spectral space most of the time; physical-space IFFTs happen only when the nonlinear term or display output requires them. GPU → host transfers (`.get()`) are display-only.
- **Backend switching** is a single function (`get_xp`); do not import `cupy`/`numpy` directly in solver code — use the `xp` bound on `DnsState`.
- **Numba / CuPy are optional imports**; keep the try/except fallbacks intact when touching `turbo_simulator.py`.
- **FFT threading** on CPU is controlled explicitly via `spfft.set_workers(...)` in `DnsSimulator`.
