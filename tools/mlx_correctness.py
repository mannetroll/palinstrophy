#!/usr/bin/env python3
"""Capture or check deterministic MLX solver and rendered-frame references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from palinstrophy.turbo_wrapper import DnsSimulator


SEED = 4895
CHECKPOINTS = (4, 20)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def run_case() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    os.environ["SCIPYTURBO_SEED"] = str(SEED)
    sim = DnsSimulator(
        n=1024,
        re=27380.85174,
        k0=5,
        cfl=2.0,
        backend="mlx",
        start_spectrum="KM3",
        method="LS_IMEX_RK3",
    )
    sim.set_variable(sim.VAR_OMEGA)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "seed": SEED,
        "N": 1024,
        "Re": 27380.85174,
        "K0": 5,
        "CFL": 2.0,
        "method": "LS_IMEX_RK3",
        "checkpoints": {},
    }

    for step in range(1, CHECKPOINTS[-1] + 1):
        sim.step(2)
        if step not in CHECKPOINTS:
            continue
        pixels = sim.get_frame_pixels().copy()
        om2 = np.asarray(sim.state.om2).copy()
        fnm1 = np.asarray(sim.state.fnm1).copy()
        prefix = f"step_{step}"
        arrays[f"{prefix}_om2"] = om2
        arrays[f"{prefix}_fnm1"] = fnm1
        arrays[f"{prefix}_pixels"] = pixels
        metadata["checkpoints"][str(step)] = {
            "iteration": sim.get_iteration(),
            "t": sim.get_time(),
            "dt": float(sim.state.dt),
            "cn": float(sim.state.cn),
            "om2_sha256": _digest(om2),
            "fnm1_sha256": _digest(fnm1),
            "pixels_sha256": _digest(pixels),
            "pixels_sample": pixels[::127, ::131].astype(int).tolist(),
        }
    return metadata, arrays


def capture(path: Path) -> None:
    metadata, arrays = run_case()
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    print(json.dumps(metadata, indent=2))
    print(f"captured {path}")


def check(path: Path) -> None:
    actual_meta, actual_arrays = run_case()
    failures: list[str] = []
    with np.load(path, allow_pickle=False) as reference:
        expected_meta = json.loads(str(reference["metadata_json"]))
        for step in CHECKPOINTS:
            key = str(step)
            expected = expected_meta["checkpoints"][key]
            actual = actual_meta["checkpoints"][key]
            for scalar in ("iteration", "t"):
                if actual[scalar] != expected[scalar]:
                    failures.append(f"step {step}: {scalar} {expected[scalar]} -> {actual[scalar]}")
            for scalar in ("dt", "cn"):
                if not np.isclose(actual[scalar], expected[scalar], rtol=2e-6, atol=2e-7):
                    failures.append(f"step {step}: {scalar} {expected[scalar]} -> {actual[scalar]}")

            rtol = 2e-6 if step == CHECKPOINTS[0] else 2e-5
            atol = 2e-7 if step == CHECKPOINTS[0] else 2e-6
            for name in ("om2", "fnm1"):
                array_key = f"step_{step}_{name}"
                expected_array = reference[array_key]
                actual_array = actual_arrays[array_key]
                if not np.all(np.isfinite(actual_array)):
                    failures.append(f"step {step}: {name} contains NaN/Inf")
                if not np.allclose(actual_array, expected_array, rtol=rtol, atol=atol):
                    rel = np.linalg.norm(actual_array - expected_array) / max(np.linalg.norm(expected_array), 1e-30)
                    failures.append(f"step {step}: {name} relative L2 drift {rel:.3e}")

            pixel_key = f"step_{step}_pixels"
            expected_pixels = reference[pixel_key].astype(np.int16)
            actual_pixels = actual_arrays[pixel_key].astype(np.int16)
            delta = np.abs(actual_pixels - expected_pixels)
            if int(delta.max()) > 1 or float(np.mean(delta)) > 0.01:
                failures.append(
                    f"step {step}: pixels max/mean delta {int(delta.max())}/{float(np.mean(delta)):.4g}"
                )

    if failures:
        raise SystemExit("correctness check failed:\n  " + "\n  ".join(failures))
    print("correctness check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "check"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.path)
    else:
        check(args.path)


if __name__ == "__main__":
    main()
