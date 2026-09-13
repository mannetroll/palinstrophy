"""Test bounded MLX submission without editing numerical reference files."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def install_async(dns):
    """Replace scheduling only, preserving the existing evaluation roots."""
    original_eval = dns.DnsState.eval_state

    def submit(state):
        if state.backend != "mlx":
            return
        mx = dns._mx
        previous = getattr(state, "_mlx_pending_state", ())
        if previous:
            mx.eval(*previous)
        # Take descriptor snapshots: subsequent slice assignments mutate the
        # Python array objects, so retaining those objects is not a fence.
        pending = tuple(
            mx.array(a)
            for a in (state.om2, state.fnm1, state.uc_full,
                      state.ur_full, state.uc, state.ur)
            if a is not None
        )
        if pending:
            mx.async_eval(*pending)
        state._mlx_pending_state = pending

    def sync(state):
        if state.backend == "mlx":
            original_eval(state)
            dns._mx.synchronize()
            state._mlx_pending_state = ()
        elif state.backend == "gpu":
            state.xp.cuda.Stream.null.synchronize()

    dns.DnsState.eval_state = submit
    dns.DnsState.sync = sync


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def capture(state, arrays, prefix=""):
    state.sync()
    for name in ("om2", "fnm1", "uc_full", "ur_full", "uc", "ur"):
        arrays[prefix + name] = np.asarray(getattr(state, name)).copy()
    return {name: float(getattr(state, name)) for name in ("t", "dt", "cn", "cnm1")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("baseline", "async"))
    parser.add_argument("mode", choices=("cli", "frames", "benchmark"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("cpu", "mlx"), default="mlx")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1001)
    parser.add_argument("--update", type=int, default=100)
    parser.add_argument("--method", default="CNAB2")
    parser.add_argument("--spectrum", default="PAO")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--re", type=float, default=10000)
    parser.add_argument("--k0", type=float, default=10)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=60)
    args = parser.parse_args()
    os.environ["SCIPYTURBO_SEED"] = str(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    result = {**vars(args), "output": str(args.output), "platform": platform.platform()}
    result["source_sha256"] = hashlib.sha256(
        (REPO / "palinstrophy/turbo_simulator.py").read_bytes()).hexdigest()
    with args.output.with_suffix(".txt").open("w") as log, contextlib.redirect_stdout(log):
        from palinstrophy import turbo_simulator as dns
        if args.variant == "async":
            install_async(dns)
        elif args.variant == "baseline":
            def blocking_eval(state, **kwargs):
                if state.backend == "mlx":
                    dns._mx.eval(*(a for a in (
                        state.om2, state.fnm1, state.uc_full,
                        state.ur_full, state.uc, state.ur) if a is not None))
            dns.DnsState.eval_state = blocking_eval
        if args.backend == "mlx":
            result["mlx_version"] = dns._mx.__version__
            result["device"] = dns._mx.device_info()
        if args.mode == "cli":
            state = dns.run_dns(
                N=args.n, Re=args.re, K0=args.k0, STEPS=args.steps,
                CFL=args.cfl, backend=args.backend, start_spectrum=args.spectrum,
                UPDATE=args.update, method=args.method,
            )
            if state is not None:
                result["scalars"] = capture(state, arrays)
        else:
            from palinstrophy.turbo_wrapper import DnsSimulator
            sim = DnsSimulator(n=args.n, re=args.re, k0=args.k0,
                               cfl=args.cfl, backend=args.backend,
                               start_spectrum=args.spectrum, method=args.method)
            sim.set_variable(sim.VAR_OMEGA)
            if args.mode == "frames":
                result["checkpoints"] = {}
                for step in range(1, args.steps + 1):
                    sim.step(args.update)
                    if step % args.update == 0 or step == args.steps:
                        pixels = sim.get_frame_pixels().copy()
                        # Check every displayed frame; capture without waiting
                        # elsewhere so the actual submission pipeline is used.
                        prefix = f"step_{step}_"
                        arrays[prefix + "pixels"] = pixels
                        result["checkpoints"][str(step)] = capture(sim.state, arrays, prefix)
            else:
                if args.warmup % args.update or args.steps % args.update:
                    raise ValueError("Warmup and measurement must end at frame boundaries")
                for step in range(1, args.warmup + 1):
                    sim.step(args.update)
                    if step % args.update == 0:
                        sim.get_frame_pixels()
                sim.state.sync()
                if args.backend == "mlx":
                    dns._mx.reset_peak_memory()
                start_t = float(sim.state.t)
                start = frame_start = time.perf_counter()
                frame_ms = []
                for step in range(1, args.steps + 1):
                    sim.step(args.update)
                    if step % args.update == 0:
                        sim.get_frame_pixels()
                        now = time.perf_counter()
                        frame_ms.append((now - frame_start) * 1000)
                        frame_start = now
                sim.state.sync()
                elapsed = time.perf_counter() - start
                simulated = float(sim.state.t) - start_t
                result["measurement"] = {
                    "wall_seconds": elapsed, "simulation_time": simulated,
                    "T_R": simulated / elapsed, "steps_per_second": args.steps / elapsed,
                    "frame_ms": frame_ms,
                    "frame_median_ms": float(np.median(frame_ms)),
                    "frame_p95_ms": float(np.percentile(frame_ms, 95)),
                    "frame_max_ms": max(frame_ms),
                }
                result["scalars"] = capture(sim.state, arrays)
        result["process_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if args.backend == "mlx":
            result["memory_bytes"] = {
                name: getattr(dns._mx, f"get_{name}_memory")()
                for name in ("active", "peak", "cache")
            }
    result["finite"] = all(np.isfinite(a).all().item() for a in arrays.values())
    result["array_sha256"] = {name: digest(a) for name, a in arrays.items()}
    if args.mode != "benchmark":
        np.savez(args.output.with_suffix(".npz"), **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Completed {args.variant} {args.mode}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
