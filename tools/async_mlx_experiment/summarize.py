"""Require exact correctness and summarize the recorded paired measurements."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


REPO = Path(__file__).resolve().parents[2]


def records(path):
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip().startswith(("[NEXTDT INIT]", "Initial DT=",
                                        "ITERATION ", "Final T="))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory

    def read(name):
        return json.loads((directory / (name + ".json")).read_text())

    baseline = read("baseline-mlx")
    candidate = read("candidate-mlx")
    reference = records(REPO / "tools/sim_512_1001_gpu_macOS.txt")
    assert len(reference) == 15
    assert reference == records(directory / "baseline-mlx.txt")
    assert reference == records(directory / "candidate-mlx.txt")
    assert baseline["scalars"] == candidate["scalars"]
    assert baseline["array_sha256"] == candidate["array_sha256"]
    assert candidate["finite"]
    assert records(REPO / "tools/sim_512_1001_cpu_macOS.txt") == records(directory / "candidate-cpu.txt")
    frames_b, frames_c = read("baseline-frames"), read("candidate-frames")
    assert frames_b["checkpoints"] == frames_c["checkpoints"]
    assert frames_b["array_sha256"] == frames_c["array_sha256"]
    assert frames_c["finite"]
    report = {
        "correctness": {"reference_records_exact": 15, "final_arrays_bitwise_equal": True,
                        "final_scalars_equal": True, "cpu_reference_exact": True,
                        "frame_checkpoints_exact": len(frames_b["checkpoints"]),
                        "frame_arrays_and_pixels_exact": len(frames_b["array_sha256"])},
        "benchmarks": {},
    }
    for case in ("cnab2", "rk3"):
        rows = []
        for pair in range(1, 5):
            b = read(f"bench-{case}-{pair}-baseline")
            c = read(f"bench-{case}-{pair}-async")
            assert b["array_sha256"] == c["array_sha256"]
            assert b["scalars"] == c["scalars"]
            assert b["finite"] and c["finite"]
            assert b["measurement"]["simulation_time"] == c["measurement"]["simulation_time"]
            rows.append({"pair": pair, "baseline": b["measurement"]["T_R"],
                         "async": c["measurement"]["T_R"],
                         "change_percent": 100 * (c["measurement"]["T_R"] / b["measurement"]["T_R"] - 1)})
        summary = {"pairs": rows, "all_final_arrays_and_scalars_exact": True}
        for variant in ("baseline", "async"):
            runs = [read(f"bench-{case}-{pair}-{variant}") for pair in range(1, 5)]
            summary[variant] = {
                key: statistics.median(run["measurement"][key] for run in runs)
                for key in ("T_R", "steps_per_second", "frame_median_ms", "frame_p95_ms", "frame_max_ms")
            }
            summary[variant]["peak_mib"] = statistics.median(run["memory_bytes"]["peak"] / 2**20 for run in runs)
        summary["median_T_R_change_percent"] = 100 * (summary["async"]["T_R"] / summary["baseline"]["T_R"] - 1)
        report["benchmarks"][case] = summary
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    files = sorted((REPO / "tools").glob("sim_512_1001_*.txt"))
    files += sorted((REPO / "palinstrophy").glob("*.py"))
    manifest = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "files": {}}
    for path in files:
        rel = str(path.relative_to(REPO))
        content = path.read_bytes()
        assert content == subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO)
        manifest["files"][rel] = hashlib.sha256(content).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
