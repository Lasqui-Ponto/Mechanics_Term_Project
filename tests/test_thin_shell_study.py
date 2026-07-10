import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Thin_Shell_Study.thin_shell_model import (
    ThinShellPoint,
    q_linear_sphere,
    q_matched_shell,
    q_thin_shell,
    q_yukawa_matched,
    solve_charge,
)
from Thin_Shell_Study.run_study import _solve_payload


class ThinShellStudyVerificationTests(unittest.TestCase):
    @patch("Thin_Shell_Study.run_study.solve_charge")
    def test_scan_worker_rejects_unresolved_estimator_disagreement(self, mocked_solve):
        mocked_solve.return_value = SimpleNamespace(estimator_disagreement=0.03)
        record = _solve_payload((0, 0, 100.0, 0.1))
        self.assertEqual(record["success"], 0)
        self.assertIn("estimator disagreement", record["error"])
        self.assertEqual(mocked_solve.call_count, 3)

    def test_uniform_yukawa_charge_has_correct_small_mu_limit(self):
        values = q_linear_sphere(np.array((1.0e-4, 1.0e-3, 1.0e-2)))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertLess(abs(values[0] - 1.0), 2.0e-4)
        self.assertTrue(np.all(np.diff(values) < 0.0))

    def test_approximations_have_required_limits(self):
        unscreened = ThinShellPoint(100.0, 1.0e-3)
        screened = ThinShellPoint(1.0e5, 0.1)
        self.assertEqual(q_thin_shell(unscreened), 1.0)
        self.assertEqual(q_matched_shell(unscreened), 1.0)
        self.assertAlmostEqual(
            q_yukawa_matched(unscreened),
            float(q_linear_sphere(unscreened.mu_out)),
            places=11,
        )
        self.assertLess(screened.epsilon, 0.01)
        self.assertLess(abs(q_matched_shell(screened) / (3.0 * screened.epsilon) - 1.0), 0.01)

    def test_nonlinear_bvp_reproduces_weak_source_analytic_charge(self):
        point = ThinShellPoint(1.01, 0.1)
        solution = solve_charge(
            point,
            x_max=80.0,
            tol=2.0e-6,
            mesh_points=300,
            max_nodes=50_000,
        )
        exact = (
            (point.density_contrast - 1.0)
            / point.density_contrast
            * float(q_linear_sphere(point.mu_out))
        )
        self.assertLess(abs(solution.q_num / exact - 1.0), 5.0e-4)
        self.assertLess(solution.estimator_disagreement, 2.0e-3)

    def test_screened_solution_is_positive_and_has_charge_plateau(self):
        point = ThinShellPoint(1.0e5, 0.1)
        solution = solve_charge(
            point,
            x_max=60.0,
            tol=1.0e-5,
            mesh_points=300,
            max_nodes=50_000,
        )
        radius = np.linspace(0.0, 60.0, 600)
        self.assertGreater(float(np.min(solution.field(radius))), 0.0)
        self.assertLess(solution.estimator_disagreement, 0.01)
        self.assertGreater(solution.q_num, 0.0)
        self.assertLess(solution.q_num, 0.02)


if __name__ == "__main__":
    unittest.main()
