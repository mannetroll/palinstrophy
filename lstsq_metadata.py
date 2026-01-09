# Kolmogorov
# \mathrm{Re} \propto \left(\frac{k_{\max}}{k_0}\right)^{4/3}
# \;\;\Rightarrow\;\;
# \log(\mathrm{Re}) \approx \frac{4}{3}\log(N) - \frac{4}{3}\log(k_0) + \text{const}

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Load CSV (expects header row with columns including: N, Re, K0)
# If your file has different column names, edit the names below.
data = np.genfromtxt("sim_metadata.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")

N = data["N"].astype(float)
Re = data["Re"].astype(float)
K0 = data["K0"].astype(float)

# Moderate subset: Re < 1e6
mask = Re < 1e6
N = N[mask]
Re = Re[mask]
K0 = K0[mask]

# Fit all:
# log10(Re) = a*log10(N) + b*log10(K0) + c  (least squares)
x1 = np.log10(N)
x2 = np.log10(K0)
y = np.log10(Re)

X = np.column_stack([x1, x2, np.ones_like(x1)])
beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)  # beta = [a, b, c]

a, b, c = beta
yhat = X @ beta
r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

print(f"Using {len(y)} points where Re < 1e7")
print(f"log10(Re) ≈ {a:.6f}*log10(N) + {b:.6f}*log10(K0) + {c:.6f}")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")
print(f"R^2 = {r2:.6f}")

# ---- 3D plot: scatter + fitted plane ----
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Scatter of data
ax.scatter(x1, x2, y, s=60)

# Plane surface over the plotted domain
x1g = np.linspace(x1.min(), x1.max(), 30)
x2g = np.linspace(x2.min(), x2.max(), 30)
X1g, X2g = np.meshgrid(x1g, x2g)
Yg = a * X1g + b * X2g + c
ax.plot_surface(X1g, X2g, Yg, alpha=0.35)

ax.set_xlabel("log10(N)")
ax.set_ylabel("log10(K0)")
ax.set_zlabel("log10(Re)")
ax.set_title(f"log10(Re) = {a:.6f}*log10(N) + {b:.6f}*log10(K0) + {c:.6f}  (Re < 1e7)")

plt.show()