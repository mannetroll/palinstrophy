#
# \log(\mathrm{Re}) \approx \frac{4}{3}\log(N) - \frac{4}{3}\log(k_0) + \text{const}
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Load CSV (expects header row with columns including: N, Re, K0)
data = np.genfromtxt("sim_metadata.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")

N = data["N"].astype(float)
Re = data["Re"].astype(float)
K0 = data["K0"].astype(float)

# Moderate subset: Re < 1e6  (keep this consistent with prints/titles)
Re_max = 1e6
mask = Re < Re_max
N = N[mask]
Re = Re[mask]
K0 = K0[mask]

# Fixed slopes:
a = 4.0 / 3.0
b = -4.0 / 3.0

x1 = np.log10(N)
x2 = np.log10(K0)
y = np.log10(Re)

# Fit ONLY intercept c (least squares with fixed a,b):
c = np.mean(y - a * x1 - b * x2)

yhat = a * x1 + b * x2 + c
ss_res = np.sum((y - yhat) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot

print(f"Using {len(y)} points where Re < {Re_max:.1e}")
print(f"log10(Re) ≈ {a:.6f}*log10(N) + {b:.6f}*log10(K0) + {c:.6f}")
print(f"c = {c:.6f}")
print(f"R^2 = {r2:.6f}")

# If you want the multiplicative constant in Re ≈ C * N^(4/3) * K0^(-4/3):
C = 10 ** c
print(f"C ≈ 10^c = {C:.6e}")
print("So: Re ≈ C * (N/K0)^(4/3)")

# ---- 3D plot: scatter + fixed-slope plane with fitted intercept ----
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x1, x2, y, s=60)

x1g = np.linspace(x1.min(), x1.max(), 30)
x2g = np.linspace(x2.min(), x2.max(), 30)
X1g, X2g = np.meshgrid(x1g, x2g)
Yg = a * X1g + b * X2g + c
ax.plot_surface(X1g, X2g, Yg, alpha=0.35)

ax.set_xlabel("log10(N)")
ax.set_ylabel("log10(K0)")
ax.set_zlabel("log10(Re)")
ax.set_title(
    f"log10(Re) = {a:.6f}*log10(N) + {b:.6f}*log10(K0) + {c:.6f}  (Re < {Re_max:.1e})"
)

plt.show()