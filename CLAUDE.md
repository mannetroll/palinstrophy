# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Direct Numerical Simulation (DNS) of 2D homogeneous incompressible turbulence using spectral (Fourier) methods. The solver supports CPU (NumPy/SciPy) and GPU (CuPy/CUDA) backends with the same API.

## Setup and Commands

```bash
# Install (CPU only)
uv sync

# Install with GPU/CuPy support
uv sync --extra cuda

# Run interactive GUI simulator
uv run turbulence

# Run headless CLI simulation: N Re K0 STEPS CFL BACKEND
uv run sim 256 1000 4 10000 0.5 numpy

# Post-process saved PGM dumps
uv run color

# View a saved case folder
python -m palinstrophy.turbo_viewcase /path/to/folder

# Batch parametric runs
bash sim_loop.sh
```

There are no automated tests. Validation is done by running simulations and inspecting output fields visually or via power spectrum analysis.

## Architecture

### Core Solver — `turbo_simulator.py`

The heart of the project. All simulation state lives in `DnsState` (a dataclass). Key functions:

- `create_dns_state()` — initializes grid, spectral operators, and PAO-style random velocity field
- `dns_step2b()` / `dns_step3()` / `dns_step2a()` — three-stage Crank-Nicolson time integrator
- `next_dt()` — CFL-adaptive time step
- `dns_pao_host_init()` — PAO random initialization (Fortran LCG RNG, optionally Numba-JIT)

Backend switching is done via `get_xp(backend)` which returns either `numpy` or `cupy`. All array operations use `xp.*` so the same code runs on CPU or GPU.

### GUI Simulator — `turbo_main.py`

PySide6 interactive frontend (~1880 lines). Runs the DNS loop via `QTimer`. Renders fields to an Indexed8 palette for fast display — pixels are only pulled from GPU when a frame is needed. Keyboard shortcuts control simulation (H=stop, G=start, V=cycle variable, C=cycle colormap). Exports PNG frames and full-resolution PGM dumps.

### Wrapper — `turbo_wrapper.py`

`DnsSimulator` bridges the GUI and core solver. Handles initialization, single-step advancement, and field extraction (U, V, kinetic energy, vorticity, stream function). Also controls SciPy FFT worker count on CPU.

### Post-processor — `turbo_postprocess.py`

Reads PGM dump folders from prior simulations and displays them with the same colormap controls as the main GUI.

## Key Implementation Details

- **FFT layout**: fields live in spectral space most of the time; physical-space transforms happen only when needed (nonlinear term, output)
- **GPU memory**: GPU arrays are kept on device; `.get()` to CPU only for display, never inside the hot loop
- **Python version**: exactly 3.13 (pinned in `pyproject.toml`)
- **Optional deps**: CuPy (CUDA), Numba — code degrades gracefully if absent
