import numpy as np
import matplotlib.pyplot as plt

from palinstrophy.turbo_main import Re_from_N_K0


def main():
    K0 = 15
    exps = np.arange(8, 14)  # 8..13 inclusive
    N = 2 ** exps

    Re = Re_from_N_K0(N, K0)

    plt.figure()
    plt.plot(N, Re, marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Re")
    plt.title(f"Re from N for K0={K0}")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()