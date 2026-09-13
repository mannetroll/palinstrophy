"""CPU/MLX arithmetic regressions; requires an accessible Apple GPU."""

import contextlib
import io
import math
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.fft as fft

from palinstrophy import turbo_simulator as dns


class _SharedFFT:
    """Use identical FFT results to isolate the two solvers' arithmetic."""

    @staticmethod
    def irfft2(array, **kwargs):
        return dns._mx.array(fft.irfft2(np.asarray(array), **kwargs))

    @staticmethod
    def rfft2(array, **kwargs):
        return dns._mx.array(fft.rfft2(np.asarray(array), **kwargs))


@unittest.skipIf(dns._mx is None, "Apple GPU unavailable")
class MlxArithmeticTests(unittest.TestCase):
    def test_cfl_scaling_matches_cpu(self):
        velocity = np.array([[[4.4227753, -1.0]], [[0.0, 2.0]]], dtype=np.float32)
        geometry = dict(NX_full=2, NZ_full=1, inv_dx=512 / (2 * math.pi))
        cpu = SimpleNamespace(
            xp=np, backend="cpu", ur_full=velocity,
            cfl_tmp=np.empty((1, 2), dtype=np.float32),
            cfl_absw=np.empty((1, 2), dtype=np.float32), **geometry,
        )
        mlx = SimpleNamespace(
            xp=dns._mx, backend="mlx", ur_full=dns._mx.array(velocity), **geometry,
        )
        self.assertEqual(dns.compute_cflm(cpu), dns.compute_cflm(mlx))

    def _check_integrator(self, method):
        states = []
        for backend in ("cpu", "mlx"):
            with contextlib.redirect_stdout(io.StringIO()):
                state = dns.create_dns_state(
                    N=64, Re=10000, K0=5, CFL=0.25, backend=backend,
                    seed=1, start_spectrum="PAO", populate_compact_ur=False,
                )
            if backend == "mlx":
                state.fft = _SharedFFT
            dns.dns_step2a(state)
            state.dt = state.cflnum / (dns.compute_cflm(state) * math.pi)
            states.append(state)

        cpu, mlx = states
        for step in range(1, 21):
            for state in states:
                state.it = step
                dt_old = state.dt
                if method == "CNAB2":
                    dns.dns_step2b(state)
                    dns.dns_step3(state)
                    dns.dns_step2a(state)
                else:
                    dns.dns_step_ls_imex_rk3(state)
                if step == 1 or step % 3 == 0:
                    dns.next_dt(state)
                state.t += dt_old
            with self.subTest(method=method, step=step):
                for name in ("om2", "fnm1"):
                    np.testing.assert_array_equal(
                        np.asarray(getattr(mlx, name)), getattr(cpu, name),
                        err_msg=name,
                    )
                np.testing.assert_array_equal(np.asarray(mlx.ur_full[:2]), cpu.ur_full[:2])
                self.assertEqual(mlx.t, cpu.t)
                self.assertEqual(mlx.dt, cpu.dt)
                self.assertEqual(mlx.cn, cpu.cn)

    def test_cnab2_matches_cpu_with_shared_fft(self):
        self._check_integrator("CNAB2")

    def test_rk3_matches_cpu_with_shared_fft(self):
        self._check_integrator("LS_IMEX_RK3")


if __name__ == "__main__":
    unittest.main()
