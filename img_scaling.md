# Vorticity Image Scaling

Live rendering first converts the physical omega field $\Omega_{ij}$ to an
intermediate uint8 image $p_{ij}$, then applies a display normalization to get
the final palette index $I_{ij}$.

## Float omega to uint8

Omega is written to `S.ur_full[2, :, :]` by `dns_om2_phys(S)`.

$$
\Omega_{\min}=\min_{ij}\Omega_{ij}, \qquad
\Omega_{\max}=\max_{ij}\Omega_{ij}, \qquad
r=\Omega_{\max}-\Omega_{\min}
$$

If $|r|\le 10^{-12}$:

$$
p_{ij}=128
$$

Otherwise:

$$
p_{ij}
=\left\lfloor
\operatorname{clip}\left(
1+254\frac{\Omega_{ij}-\Omega_{\min}}{\Omega_{\max}-\Omega_{\min}},
1,
255
\right)
\right\rfloor
$$

Thus live min/max scaling maps $\Omega_{\min}\mapsto 1$ and
$\Omega_{\max}\mapsto 255$.

## Display normalization

The GUI then normalizes the uint8 image $p_{ij}$ using its current mean and
standard deviation:

$$
\mu=\operatorname{mean}(p), \qquad \sigma=\operatorname{std}(p)
$$

If $\sigma<1$, rendering stops for that frame. Otherwise, with $k=2.5$:

$$
lo=\mu-k\sigma, \qquad hi=\mu+k\sigma
$$

$$
I_{ij}
=\operatorname{uint8}\left(
\operatorname{clip}\left(
\operatorname{round}\left(
255\frac{p_{ij}-lo}{hi-lo}
\right),
0,
255
\right)
\right)
$$

Equivalently, since $hi-lo=5\sigma$:

$$
I_{ij}
=\operatorname{uint8}\left(
\operatorname{clip}\left(
\operatorname{round}\left(
255\frac{p_{ij}-\mu+2.5\sigma}{5\sigma}
\right),
0,
255
\right)
\right)
$$

$I_{ij}$ is the final `QImage.Format_Indexed8` palette index. The selected
colormap maps this index to RGB. Display resizing only repeats or strides
pixels; it does not change pixel values.
