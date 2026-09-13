"""GPU FFT accuracy, normalization, batch layout, and compatibility checks."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.fft as fft

from palinstrophy import turbo_simulator as dns


@unittest.skipIf(dns._mx is None, "Apple GPU unavailable")
class MlxFFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from palinstrophy import _mlx_fft

        cls.module = _mlx_fft
        cls.mx = dns._mx
        cls.fft = _mlx_fft.fft_for_shape((96, 192))
        if cls.fft is cls.mx.fft:
            raise unittest.SkipTest("Installed MLX headers do not support the precision patch")

    def assert_relative_error(self, actual, reference, tolerance=3e-7):
        error = np.linalg.norm(np.asarray(actual) - reference) / np.linalg.norm(reference)
        self.assertLess(error, tolerance)
        return error

    def test_supported_lengths_and_batches(self):
        rng = np.random.default_rng(182)
        # Rectangular fields cover every supported length without allocating
        # production-size square grids. Include single fields and odd batches.
        for length in sorted(self.module._LENGTHS):
            for shape in sorted({(96, length), (length, 96)}):
                for batch_shape in ((), (3,)):
                    with self.subTest(shape=shape, batch=batch_shape):
                        real = rng.normal(size=(*batch_shape, *shape)).astype(np.float32)
                        spectral = self.fft.rfft2(self.mx.array(real))
                        reference = fft.rfft2(real.astype(np.float64))
                        self.assertEqual(spectral.dtype, self.mx.complex64)
                        self.assert_relative_error(spectral, reference)
                        restored = self.fft.irfft2(spectral, s=shape)
                        self.assertEqual(restored.dtype, self.mx.float32)
                        self.assert_relative_error(restored, real)

    def test_normalization_and_noncontiguous_inputs(self):
        rng = np.random.default_rng(711)
        real = rng.normal(size=(192, 3, 96)).astype(np.float32).transpose(1, 2, 0)
        device = self.mx.transpose(self.mx.array(real.transpose(2, 0, 1)), (1, 2, 0))
        for norm in ("backward", "forward", "ortho"):
            with self.subTest(norm=norm):
                spectral = self.fft.rfft2(device, axes=(1, 2), norm=norm)
                reference = fft.rfft2(real.astype(np.float64), axes=(1, 2), norm=norm)
                self.assert_relative_error(spectral, reference)
                inverse = self.fft.irfft2(spectral, s=(96, 192), axes=(1, 2), norm=norm)
                self.assert_relative_error(inverse, real)

    def test_dc_nyquist_and_impulse(self):
        z, x = np.indices((96, 192))
        real = np.stack((np.ones_like(x), (-1.)**x + 2 * (-1.)**z, (z == 11) & (x == 17)))
        real = real.astype(np.float32)
        spectral = self.fft.rfft2(self.mx.array(real))
        self.assert_relative_error(spectral, fft.rfft2(real.astype(np.float64)))
        self.assert_relative_error(self.fft.irfft2(spectral, s=(96, 192)), real)

    def test_768_inverse_precision(self):
        rng = np.random.default_rng(512)
        spectral = (rng.normal(size=(2, 768, 385)) + 1j * rng.normal(size=(2, 768, 385))).astype(np.complex64)
        device = self.mx.array(spectral)
        reference = fft.irfft2(spectral.astype(np.complex128), norm="forward")
        actual = self.fft.irfft2(device, norm="forward")
        error = self.assert_relative_error(actual, reference)
        native = np.asarray(self.mx.fft.irfft2(device, norm="forward"))
        native_error = np.linalg.norm(native - reference) / np.linalg.norm(reference)
        self.assertLess(error, native_error / 3)

    def test_native_fallback(self):
        rng = np.random.default_rng(144)
        real = self.mx.array(rng.normal(size=(96, 192)).astype(np.float32))
        # Native API handles unsupported lengths, padding, and other axes.
        for kwargs in ({"s": (98, 194)}, {"s": (192, 192)}, {"axes": (1, 0)}):
            with self.subTest(kwargs=kwargs):
                np.testing.assert_array_equal(
                    np.asarray(self.fft.rfft2(real, **kwargs)),
                    np.asarray(self.mx.fft.rfft2(real, **kwargs)),
                )
        self.assertIs(self.module.fft_for_shape((6144, 6144)), self.mx.fft)

    def test_incompatible_headers_fall_back(self):
        failures = (
            ValueError("changed header"), FileNotFoundError("missing header"),
            self.module.PackageNotFoundError("mlx"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.module._fft_module.cache_clear()
                try:
                    with patch.object(self.module, "_load_headers", side_effect=failure):
                        with self.assertWarnsRegex(RuntimeWarning, "using mlx.core.fft"):
                            self.assertIs(self.module.fft_for_shape((768, 768)), self.mx.fft)
                finally:
                    self.module._fft_module.cache_clear()


if __name__ == "__main__":
    unittest.main()
