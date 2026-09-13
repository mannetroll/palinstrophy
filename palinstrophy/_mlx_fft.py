"""Improve MLX FFT phase accuracy using kernels generated entirely from Python.

The installed MLX headers supply the Stockham butterflies (Copyright Apple
Inc., MIT license); their notices remain in the generated source. Only phase
factors and inverse normalization change. MLX compiles the kernels at runtime;
neither the installed headers nor the library are modified.
"""

from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
import math
import re
from types import ModuleType
import warnings

import mlx.core as mx
import numpy as np


# De-aliasing lengths for the usual N=64..2048 power-of-two solver grids.
# Keep other decompositions on MLX's native planner.
_LENGTHS = frozenset(3 * 2**power for power in range(5, 11))
_RADICES = (13, 11, 8, 7, 6, 5, 4, 3, 2)
_HEADER_HASHES = {
    "fft.h": "7b486dfc562a2f46b30de9e198cc1ed3c338b5f410dae4ba2d2080906639bb8f",
    "fft/radix.h": "3f3e8a9f42f464c6f39893fe972c2d7bfacaefc4b47e5ef8ea0205b662da869e",
    "fft/readwrite.h": "e26d49d0c9a86fefef643fd3a7c24c6f13b5a92cd0aac2352c8b03a07b7c1f35",
    "steel/defines.h": "b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a",
}


@lru_cache(maxsize=1)
def _load_headers() -> dict[str, str]:
    """Accept only the reviewed MLX 0.32.0 sources, including on newer wheels."""
    root = distribution("mlx").locate_file("mlx/include/mlx/backend/metal/kernels")
    headers = {}
    for name, expected in _HEADER_HASHES.items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"unrecognized MLX header {name}")
        # Dependencies are concatenated below so no compiler include path or
        # copied Metal source files are needed.
        headers[name] = re.sub(r'^#include "[^\n]+\n', '', data.decode(), flags=re.M)
    return headers


def _plan(n: int) -> tuple[dict[int, int], int]:
    """Match MLX's Stockham decomposition for the supported 3*2**k lengths."""
    steps = dict.fromkeys(_RADICES, 0)
    remaining = n
    for radix in _RADICES:
        while remaining % radix == 0:
            steps[radix] += 1
            remaining //= radix
    used = sorted(radix for radix, count in steps.items() if count)
    return steps, used[1] if len(used) > 1 else used[0]


@lru_cache(maxsize=12)
def _kernel_header(n: int, inverse: bool) -> str:
    headers = _load_headers()
    steps, elements_per_thread = _plan(n)

    # Compute phase constants once in double precision, then round once to
    # float32. No simulation fields leave the GPU for this calculation.
    angles = -2 * np.pi * np.arange(n, dtype=np.float64) / n
    factors = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    factors[0] = (1, 0)
    factors[n // 4] = (0, -1)
    factors[n // 2] = (-1, 0)
    factors[3 * n // 4] = (0, 1)
    table = "constant float2 fft_phases[] = {\n" + ",\n".join(
        f"float2({real:.9e}f, {imag:.9e}f)" for real, imag in factors
    ) + "\n};\n"
    radix = table + re.sub(
        r'METAL_FUNC float2 get_twiddle\(int k, int p\) \{.*?\n\}',
        f"METAL_FUNC float2 get_twiddle(int k, int p) {{\n"
        f"  return fft_phases[(k * ({n} / p)) % {n}];\n}}",
        headers["fft/radix.h"], flags=re.S,
    )

    # Both inverse axes are unscaled. Apply the requested normalization once
    # in Python; norm='forward' then needs no division or compensating multiply.
    readwrite = headers["fft/readwrite.h"].replace(
        "float2(elem.x / n, -elem.y / n)", "float2(elem.x, -elem.y)"
    ).replace(
        "seq_buf[index].x / n", "seq_buf[index].x"
    ).replace(
        "seq_buf[index].y / -n", "-seq_buf[index].y"
    )
    main = headers["fft.h"].split("// Each FFT is computed entirely")[0]
    constants = {
        "inv_": str(inverse).lower(), "is_power_of_2_": "false",
        "elems_per_thread_": elements_per_thread, "rader_m_": n,
        **{f"radix_{r}_steps_": steps[r] for r in _RADICES},
        **{f"rader_{r}_steps_": 0 for r in _RADICES},
    }
    main = re.sub(
        r'STEEL_CONST (bool|int) (\w+) \[\[function_constant\(\d+\)\]\];',
        lambda match: f"STEEL_CONST {match[1]} {match[2]} = {constants[match[2]]};",
        main,
    ).replace(
        "twiddle = complex_mul(twiddle, twiddle_1);",
        "twiddle = get_twiddle(t * k, radix * p);",
    )
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "#ifndef METAL_FUNC\n#define METAL_FUNC inline\n#endif\n"
        + headers["steel/defines.h"] + radix + readwrite + main
    )


@lru_cache(maxsize=64)
def _kernel(n: int, batch: int, kind: str):
    _, elements_per_thread = _plan(n)
    threads = (n + elements_per_thread - 1) // elements_per_thread
    group_batch = max(256 // n, 1)
    shared_size = 1 << (group_batch * n - 1).bit_length()
    groups = (batch + group_batch - 1) // group_batch
    if kind in ("rfft", "irfft"):
        groups = (groups + 1) // 2
    in_type = "float" if kind == "rfft" else "float2"
    out_type = "float" if kind == "irfft" else "float2"
    source = f"""
    threadgroup float2 shared_in[{shared_size}];
    uint3 elem = thread_position_in_grid;
    uint3 grid = uint3({groups}, {group_batch}, {threads});
    int n = {n};
    int batch_size = {batch};
    thread ReadWriter<{in_type}, {out_type}> rw(
        reinterpret_cast<const device {in_type}*>(inp), shared_in,
        reinterpret_cast<device {out_type}*>(out),
        n, batch_size, elems_per_thread_, elem, grid, inv_);
    if (rw.out_of_bounds()) return;
    rw.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int p = 1;
    perform_fft(elem.z, &p, {threads}, n, shared_in + elem.y * n);
    rw.write();
    """
    fn = mx.fast.metal_kernel(
        name=f"palinstrophy_fft_{n}_{batch}_{kind}",
        input_names=["inp"], output_names=["out"], source=source,
        header=_kernel_header(n, kind in ("ifft", "irfft")),
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"},
    )
    return fn, (groups, group_batch, threads), (1, group_batch, threads)


def _one_axis(a: mx.array, n: int, axis: int, kind: str) -> mx.array:
    a = mx.moveaxis(a, axis, -1)
    fn, grid, threadgroup = _kernel(n, math.prod(a.shape[:-1]), kind)
    length = n // 2 + 1 if kind == "rfft" else n
    result = fn(
        inputs=[a], output_shapes=[(*a.shape[:-1], length)],
        output_dtypes=[mx.float32 if kind == "irfft" else mx.complex64],
        grid=grid, threadgroup=threadgroup,
    )[0]
    return mx.moveaxis(result, -1, axis)


class _PatchedFFT:
    label = "MLX FFT (precomputed phase factors)"

    def _transform(self, a, s, axes, norm, inverse):
        # The solver needs only 2D fields and 3D component batches, with no
        # padding/truncation. Delegate other FFT requests to the native API.
        size = tuple(s) if s is not None else (
            (a.shape[-2], 2 * (a.shape[-1] - 1)) if inverse and a.ndim >= 2
            else a.shape[-2:]
        )
        eligible = (
            a.ndim in (2, 3) and len(axes) == 2
            and tuple(axes) in ((-2, -1), (a.ndim - 2, a.ndim - 1))
            and len(size) == 2 and all(n in _LENGTHS for n in size)
            and a.dtype == (mx.complex64 if inverse else mx.float32)
            and norm in (None, "backward", "forward", "ortho")
            and a.shape[-2:] == (size[0], size[1] // 2 + 1 if inverse else size[1])
            and a.size > 0
        )
        if not eligible:
            fn = mx.fft.irfft2 if inverse else mx.fft.rfft2
            return fn(a, s=s, axes=axes, norm="backward" if norm is None else norm)

        if inverse:
            result = _one_axis(a, size[0], -2, "ifft")
            # Preserve MLX's contiguous-copy order when it pairs real rows:
            # [component, z, kx] is processed as [z, component, kx].
            if a.ndim == 3:
                result = mx.swapaxes(result, 0, 1)
            result = _one_axis(result, size[1], -1, "irfft")
            if a.ndim == 3:
                result = mx.swapaxes(result, 0, 1)
        else:
            result = _one_axis(a, size[1], -1, "rfft")
            result = _one_axis(result, size[0], -2, "fft")

        count = math.prod(size)
        if norm == "ortho":
            return result * float(np.float32(1 / math.sqrt(count)))
        if (inverse and norm != "forward") or (not inverse and norm == "forward"):
            return result * float(np.float32(1 / count))
        return result

    def rfft2(
        self, a: mx.array, s: tuple[int, int] | None = None,
        axes: tuple[int, int] = (-2, -1), norm: str = "backward",
    ) -> mx.array:
        """Real forward FFT, retaining native behavior outside the patched scope."""
        return self._transform(a, s, axes, norm, inverse=False)

    def irfft2(
        self, a: mx.array, s: tuple[int, int] | None = None,
        axes: tuple[int, int] = (-2, -1), norm: str = "backward",
    ) -> mx.array:
        """Real inverse FFT with a single final normalization, when requested."""
        return self._transform(a, s, axes, norm, inverse=True)


@lru_cache(maxsize=1)
def _fft_module():
    try:
        _load_headers()
    except (OSError, ValueError, PackageNotFoundError) as exc:
        warnings.warn(
            f"MLX FFT precision patch unavailable ({exc}); using mlx.core.fft.",
            RuntimeWarning, stacklevel=2,
        )
        return mx.fft
    return _PatchedFFT()


def fft_for_shape(shape: tuple[int, int]) -> _PatchedFFT | ModuleType:
    """Select the precision patch for supported de-aliasing grid lengths."""
    return _fft_module() if all(n in _LENGTHS for n in shape) else mx.fft
