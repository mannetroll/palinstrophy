"""Run four alternating pairs per workload, sequentially on one GPU."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    runner = Path(__file__).with_name("run_experiment.py")
    for case in ("cnab2", "rk3"):
        for pair in range(1, 5):
            variants = ("baseline", "async") if pair % 2 else ("async", "baseline")
            for variant in variants:
                cmd = [sys.executable, str(runner), variant, "benchmark",
                       str(args.output_directory / f"bench-{case}-{pair}-{variant}"),
                       "--n", "1024", "--warmup", "100", "--steps", "1000"]
                if case == "rk3":
                    cmd += ["--method", "LS_IMEX_RK3", "--update", "2",
                            "--seed", "4895", "--re", "27380.85174", "--k0", "5",
                            "--cfl", "2", "--spectrum", "KM3"]
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
