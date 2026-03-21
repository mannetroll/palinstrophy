"""
Standalone PGM viewer for palinstrophy dump folders.

Reads .pgm files (P5 binary grayscale) saved by the main turbulence app
and displays them with matplotlib.

Usage:
    uv run python -m palinstrophy.turbo_viewcase /Users/drtobbe/Desktop/palinstrophy_512_15_2E03_0.3_6851


where <folder> contains u_velocity.pgm, v_velocity.pgm, kinetic.pgm, omega.pgm
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def read_pgm(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        magic = f.readline().strip()
        assert magic == b"P5", f"Not a P5 PGM file: {magic}"
        # skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        data = f.read(w * h)
    arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    return arr


def show_folder(folder: str) -> None:
    names = ["u_velocity.pgm", "v_velocity.pgm", "kinetic.pgm", "omega.pgm"]
    titles = ["u velocity", "v velocity", "kinetic energy", "vorticity (ω)"]

    found = [(n, t) for n, t in zip(names, titles) if os.path.isfile(os.path.join(folder, n))]
    if not found:
        print(f"No .pgm files found in {folder}")
        sys.exit(1)

    n_plots = len(found)
    cols = 2 if n_plots > 1 else 1
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    fig.suptitle(os.path.basename(os.path.normpath(folder)), fontsize=14)

    for idx, (name, title) in enumerate(found):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        arr = read_pgm(os.path.join(folder, name))
        im = ax.imshow(arr, cmap="inferno", origin="upper")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused axes
    for idx in range(n_plots, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)
    folder = sys.argv[1]
    show_folder(folder)


if __name__ == "__main__":
    main()
