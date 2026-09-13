"""Main-image sampling preserves full-field statistics and exact pixel values."""

import contextlib
import io
import unittest

import numpy as np

from palinstrophy import turbo_simulator as dns
from palinstrophy.turbo_wrapper import DnsSimulator


@unittest.skipIf(dns._mx is None, "Apple GPU unavailable")
class MlxFramePixelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.sim = DnsSimulator(n=32, re=10000, k0=5, cfl=.25, backend="mlx",
                                   start_spectrum="PAO", method="CNAB2")
        cls.sim._next_dt_pending = False

    def test_all_fields_and_full_resolution_default(self):
        sim = self.sim
        for variable in (sim.VAR_U, sim.VAR_V, sim.VAR_ENERGY, sim.VAR_OMEGA, sim.VAR_STREAM, -1):
            sim.set_variable(variable)
            full = sim.get_frame_pixels().copy()
            stats = sim.take_frame_stats()
            self.assertEqual(full.shape, (48, 48))
            for stride in (2, 3, 5):
                with self.subTest(variable=variable, stride=stride):
                    pixels = sim.get_frame_pixels(display_stride=stride)
                    np.testing.assert_array_equal(pixels, full[::stride, ::stride])
                    self.assertEqual(sim.take_frame_stats(), stats)
                    self.assertIsNone(sim.take_frame_stats())
                    self.assertTrue(pixels.flags.c_contiguous)
            np.testing.assert_array_equal(sim.get_frame_pixels(), full)

    def test_statistics_use_pixels_omitted_from_display(self):
        field = np.ones((9, 13), dtype=np.float32)
        field[::2, ::2] = 0
        field = dns._mx.array(field)
        full = self.sim._float_to_pixels_mlx(field).copy()
        stats = self.sim.take_frame_stats()
        sampled = self.sim._float_to_pixels_mlx(field, display_stride=2)
        np.testing.assert_array_equal(sampled, full[::2, ::2])
        self.assertEqual(float(sampled.std()), 0)
        self.assertGreater(stats[1], 0)
        self.assertEqual(self.sim.take_frame_stats(), stats)

    def test_constant_and_nondivisible_frames(self):
        field = dns._mx.full((9, 13), 7, dtype=dns._mx.float32)
        pixels = self.sim._float_to_pixels_mlx(field, display_stride=3)
        np.testing.assert_array_equal(pixels, np.full((3, 5), 128, dtype=np.uint8))
        self.assertEqual(self.sim.take_frame_stats(), (128.0, 0.0))

    def test_invalid_stride(self):
        for stride in (0, -1):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                self.sim.get_frame_pixels(display_stride=stride)


if __name__ == "__main__":
    unittest.main()
