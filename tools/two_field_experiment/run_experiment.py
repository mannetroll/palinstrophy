"""Isolated two-field experiment; never edits the production solver or references."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def install_two_field(dns):
    def products(state):
        xp = state.xp
        u, v = state.ur_full[0], state.ur_full[1]
        if state.backend == "mlx":
            state.ur_full = xp.stack((u * u - v * v, u * v), axis=0)
            transformed = state.fft.rfft2(
                state.ur_full, s=(state.NZ_full, state.NX_full), axes=(1, 2)
            )
        elif state.backend == "cpu":
            # Preserve the existing in-place products, then combine the squares.
            # Planes 0 and 2 are the two inputs to the batched forward transform.
            xp.multiply(u, v, out=state.ur_full[2])
            xp.multiply(u, u, out=state.ur_full[0])
            xp.multiply(v, v, out=state.ur_full[1])
            xp.subtract(state.ur_full[0], state.ur_full[1], out=state.ur_full[0])
            transformed = state.fft.rfft2(
                state.ur_full[::2], axes=(1, 2), overwrite_x=True,
                workers=state.fft_workers,
            )
        else:
            raise ValueError("This isolated experiment supports CPU and MLX only")
        # Keep allocation/reconstruction unchanged; the unused third spectral
        # plane is never consumed by the candidate nonlinear calculation.
        state.uc_full[0:2] = transformed
        state.uc_full[0:2, state.Nbase, :state.Nbase // 2] = dns._zero_c(xp)

    def nonlinear_cpu(state, out):
        xp = state.xp
        n = state.Nbase
        half = n // 2
        difference = state.step3_uc1_th
        cross = state.step3_uc3_th
        for target, source in ((difference, state.uc_full[0]), (cross, state.uc_full[1])):
            target[:half] = source[:half, :half]
            target[half:] = source[n:n + half, :half]
        xp.multiply(difference, state.step3_GA, out=out)
        xp.multiply(cross, state.step3_G2mA2, out=state.scratch2)
        xp.add(out, state.scratch2, out=out)
        xp.multiply(out, state.step3_divxz, out=out)

    def nonlinear_mlx(state):
        half = state.Nbase // 2

        def band(start, ga, g2ma2):
            stop = start + half
            value = state.uc_full[0, start:stop, :half] * ga
            value = value + state.uc_full[1, start:stop, :half] * g2ma2
            return value * state.step3_divxz

        positive = band(0, state.step3_GA[:half], state.step3_G2mA2[:half])
        negative = band(state.Nbase, state.step3_GA[half:], state.step3_G2mA2[half:])
        return state.xp.concatenate((positive, negative), axis=0)

    dns.dns_step2b = products
    dns._compute_nonlinear_vorticity_term = nonlinear_cpu
    dns._compute_nonlinear_vorticity_term_mlx = nonlinear_mlx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("baseline", "two-field"))
    parser.add_argument("backend", choices=("cpu", "mlx"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1001)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--update", type=int, default=100)
    parser.add_argument("--method", default="CNAB2")
    parser.add_argument("--spectrum", default="PAO")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    os.environ["SCIPYTURBO_SEED"] = str(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".txt").open("w") as log, contextlib.redirect_stdout(log):
        from palinstrophy import turbo_simulator as dns
        if args.variant == "two-field":
            install_two_field(dns)
        state = dns.run_dns(
            N=args.n, Re=10000, K0=10, STEPS=args.steps, CFL=args.cfl,
            backend=args.backend, start_spectrum=args.spectrum,
            UPDATE=args.update, method=args.method,
        )
        if state is not None:
            arrays = {name: np.asarray(getattr(state, name)).copy() for name in ("om2", "fnm1")}
            arrays["velocity"] = np.asarray(state.ur_full[:2]).copy()
            np.savez(args.output.with_suffix(".npz"), **arrays)
            summary = {
                "variant": args.variant, "backend": args.backend, "n": args.n,
                "steps": args.steps, "method": args.method, "seed": args.seed,
                "t": float(state.t), "dt": float(state.dt), "cn": float(state.cn),
                "finite": all(np.isfinite(a).all().item() for a in arrays.values()),
                "source_sha256": hashlib.sha256((repo / "palinstrophy/turbo_simulator.py").read_bytes()).hexdigest(),
                "array_sha256": {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in arrays.items()},
            }
            args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed {args.variant} {args.backend} N={args.n} {args.method}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
