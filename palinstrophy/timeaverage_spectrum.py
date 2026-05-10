"""
Plot a CUDA time-averaged energy spectrum from an output folder.

Reads:
    output_.../energy_spectrum.csv

The CSV is expected to have the spectrum metadata in the first two commented
lines followed by:
    normalized_radius,shell_sum_energy,count

Usage:
    uv run python -m tools.timeaverage_spectrum output_folder
    uv run python -m tools.timeaverage_spectrum output_folder -o /tmp/avg.png

Writes:
    timeaverage_spectrum.png
    timeaverage_spectrum_k3.png
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpectrumFile:
    path: Path
    metadata: dict[str, str]


@dataclass(frozen=True)
class AveragedSpectrum:
    radius: Any
    energy: Any
    source: SpectrumFile
    average_kind: str


def configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return

    matplotlib_config_dir = os.path.join(os.path.expanduser("~"), ".config", "matplotlib")
    if not os.access(matplotlib_config_dir, os.W_OK):
        os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the time-averaged energy spectrum from output_.../energy_spectrum.csv.",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Output folder containing energy_spectrum.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path. Defaults to timeaverage_spectrum.png in the input folder.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="PNG resolution in dots per inch.",
    )
    return parser.parse_args()


def spectrum_csv_path(folder: Path) -> Path:
    return folder / "energy_spectrum.csv"


def parse_metadata(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        header_line = f.readline()
        values_line = f.readline()

    if not header_line.startswith("#") or not values_line.startswith("#"):
        return {}

    header = next(csv.reader([header_line.lstrip("#").strip()]))
    values = next(csv.reader([values_line.lstrip("#").strip()]))
    return dict(zip(header, values, strict=False))


def read_spectrum(path: Path) -> tuple[Any, Any]:
    import numpy as np

    skip_header = 0
    with path.open(newline="") as f:
        first_line = f.readline()
        second_line = f.readline()
    if first_line.startswith("#") and second_line.startswith("#"):
        skip_header = 2

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        comments="#",
        skip_header=skip_header,
    )
    if data.size == 0:
        raise ValueError(f"No spectrum rows in {path}")

    data = np.atleast_1d(data)
    names = data.dtype.names or ()
    missing = {"normalized_radius", "shell_sum_energy"} - set(names)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in {path}: {missing_text}")

    radius = np.asarray(data["normalized_radius"], dtype=np.float64)
    energy = np.asarray(data["shell_sum_energy"], dtype=np.float64)
    good = np.isfinite(radius) & np.isfinite(energy) & (radius > 0.0)
    return radius[good], energy[good]


def metadata_float(metadata: dict[str, str], name: str) -> float | None:
    value = metadata.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_averaged_spectrum(folder: Path) -> AveragedSpectrum:
    path = spectrum_csv_path(folder)
    metadata = parse_metadata(path)
    radius, energy = read_spectrum(path)
    return AveragedSpectrum(
        radius=radius,
        energy=energy,
        source=SpectrumFile(path=path, metadata=metadata),
        average_kind=metadata["AVERAGE_KIND"],
    )


def compact_float(x: float) -> str:
    return f"{x:g}"


def compact_float32(x: float) -> str:
    x32 = struct.unpack("f", struct.pack("f", float(x)))[0]
    return f"{x32:.7g}"


def metadata_float32_text(metadata: dict[str, str], name: str) -> str:
    value = metadata_float(metadata, name)
    if value is None:
        return metadata.get(name, "")
    return compact_float32(value)


def metadata_values(average: AveragedSpectrum, name: str) -> list[float]:
    value = metadata_float(average.source.metadata, name)
    return [] if value is None else [value]


def metadata_mean(average: AveragedSpectrum, name: str) -> float | None:
    values = metadata_values(average, name)
    if not values:
        return None
    return sum(values) / float(len(values))


def k0_norm(average: AveragedSpectrum) -> float | None:
    n = metadata_mean(average, "N")
    k0 = metadata_mean(average, "K0")
    if n is None or k0 is None or n == 0.0:
        return None
    return 2.0 * k0 / n


def summary_text(average: AveragedSpectrum) -> str:
    metadata = average.source.metadata
    t_start = metadata_float32_text(metadata, "T_START")
    t_end = metadata_float32_text(metadata, "T_END")
    lines = [
        f"average={average.average_kind}",
        f"source={average.source.path.name}",
        f"spectrum samples={metadata['SPECTRUM_SAMPLES']}",
        f"spectrum steps={metadata['FIRST_STEP']}..{metadata['LAST_STEP']}",
        f"T={t_start}...{t_end}",
        f"total dt={metadata['TOTAL_DT']}",
    ]

    for name in ("N", "K0", "CFL", "SIG"):
        value = metadata_mean(average, name)
        if value is not None:
            lines.append(f"{name}={compact_float(value)}")

    re_value = metadata_mean(average, "Re")
    if re_value is not None:
        lines.append(f"Re(avg)={re_value:,.0f}")

    for name in ("VISC", "PALIN", "U", "L", "TAU_L", "T_OVER_TAU_L", "E(J)"):
        value = metadata_mean(average, name)
        if value is not None:
            lines.append(f"{name}(avg)={compact_float(value)}")

    lines.append(f"TS={metadata['TS']}")

    return "\n".join(lines)


def default_output_path(folder: Path) -> Path:
    return folder / "timeaverage_spectrum.png"


def compensated_output_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".png"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}_k3{suffix}")


def add_summary_box(ax: Any, average: AveragedSpectrum, location: str = "lower-left") -> None:
    positions = {
        "lower-left": (0.02, 0.02, "left", "bottom"),
        "upper-right": (0.72, 0.96, "left", "top"),
    }
    x, y, ha, va = positions[location]
    ax.text(
        x,
        y,
        summary_text(average),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        linespacing=1.5,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=(1, 1, 1, 0.9),
            edgecolor="none",
        ),
    )


def add_k0_marker(ax: Any, average: AveragedSpectrum) -> None:
    k0 = k0_norm(average)
    if k0 is None:
        return

    ax.axvline(k0)
    ax.text(k0, 0.92, "K0", transform=ax.get_xaxis_transform(), ha="center", va="center")


def save_timeaverage_energy_spectrum_pngs(
    folder: str | Path,
    output_path: str | Path | None = None,
    dpi: int = 100,
) -> tuple[str, str]:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    input_folder = Path(folder).expanduser().resolve()
    averaged = load_averaged_spectrum(input_folder)
    out_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_output_path(input_folder)
    )
    k3_out_path = compensated_output_path(out_path)

    radius = np.asarray(averaged.radius, dtype=np.float64)
    energy = np.asarray(averaged.energy, dtype=np.float64)
    good = np.isfinite(radius) & np.isfinite(energy) & (radius > 0.0) & (energy > 0.0)
    if not np.any(good):
        raise ValueError("No positive averaged spectrum values to plot.")

    plot_radius = radius[good]
    plot_energy = energy[good]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(plot_radius, plot_energy, label="time-averaged E(k)")
    ax.set_ylim(bottom=1)
    ax.set_title("Time-averaged energy spectrum estimate E(k) from vorticity (shell sum)")
    ax.set_xlabel("normalized radius  k / k_Nyquist")
    ax.set_ylabel("time-averaged shell-sum energy (unnormalized)")

    add_k0_marker(ax, averaged)

    x2 = 0.7
    if plot_radius.size > 0 and plot_energy.size > 0:
        i_peak = int(np.argmax(plot_energy))
        x1 = float(plot_radius[i_peak])
        y1 = float(plot_energy[i_peak])
        if x1 > 0.0 and y1 > 0.0 and x2 != x1:
            y2 = y1 * (x2 / x1) ** -3.0
            ax.loglog([x1, x2], [y1, y2], "--", linewidth=2)
            ax.text(x2, y2 * 2, "k^-3", fontsize=11, ha="left", va="center", color="black")

    add_summary_box(ax, averaged, location="lower-left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    k3_energy = plot_radius**3 * plot_energy
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(plot_radius, k3_energy, label="(k / k_Nyquist)^3 E(k)")
    ax.set_title("Compensated time-averaged energy spectrum k^3 E(k)")
    ax.set_xlabel("normalized radius  k / k_Nyquist")
    ax.set_ylabel("(k / k_Nyquist)^3 * time-averaged shell-sum energy")
    ax.grid(True, which="both", alpha=0.25)
    add_k0_marker(ax, averaged)
    add_summary_box(ax, averaged, location="upper-right")

    fig.tight_layout()
    k3_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(k3_out_path, dpi=dpi)
    plt.close(fig)

    return str(out_path), str(k3_out_path)


def save_timeaverage_energy_spectrum_png(
    folder: str | Path,
    output_path: str | Path | None = None,
    dpi: int = 100,
) -> str:
    out_path, _ = save_timeaverage_energy_spectrum_pngs(
        folder,
        output_path=output_path,
        dpi=dpi,
    )
    return str(out_path)


def main() -> None:
    args = parse_args()
    out_paths = save_timeaverage_energy_spectrum_pngs(
        args.folder,
        output_path=args.output,
        dpi=args.dpi,
    )
    for out_path in out_paths:
        print(out_path)


if __name__ == "__main__":
    main()
