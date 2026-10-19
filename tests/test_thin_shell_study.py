import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Thin_Shell_Study.thin_shell_model import (
    ThinShellPoint,
    q_linear_sphere,
    q_matched_shell,
    q_point_matched_shell,
    q_thin_shell,
    q_yukawa_matched,
    force_ratio,
    solve_charge,
)
from Thin_Shell_Study.run_study import _solve_payload, StaleScanError, validate_scan
from Thin_Shell_Study.study_analysis import (
    threshold_summary, sampled_validity_radii, tail_force, tail_force_cube,
)


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
        # An independent analytic force checks the physical normalization and
        # the distinction between local differentiation and tail substitution.
        self.assertLess(abs(float(solution.direct_force_ratio()) / float(force_ratio(exact, point)) - 1), 5e-4)

    def test_local_nonlinear_force_differs_from_the_asymptotic_tail(self):
        point = ThinShellPoint(1e5, .3)
        solution = solve_charge(point, x_max=160, tol=2e-6,
                                mesh_points=480, max_nodes=120_000)
        direct = float(solution.direct_force_ratio())
        self.assertAlmostEqual(direct, .001616276, delta=1e-8)
        self.assertGreater(abs(float(force_ratio(solution.q_num, point)) / direct - 1), .15)
        far_point = ThinShellPoint(1e5, .3, probe_radius=10)
        self.assertLess(abs(float(force_ratio(solution.q_num, far_point)) /
                            float(solution.direct_force_ratio(10)) - 1), .02)

    def test_point_volume_control_has_the_correct_long_range_limit(self):
        point = ThinShellPoint(1e5, .001)
        self.assertLess(abs(q_point_matched_shell(point) / q_matched_shell(point) - 1), .0011)
        # Positive spherical radial weights increase the same-shell point model.
        point = ThinShellPoint(1e4, .1)
        self.assertGreater(q_yukawa_matched(point), q_point_matched_shell(point))

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


class LocalForceAnalysisTests(unittest.TestCase):
    def data(self):
        mu = np.array([[.01, .02], [.01, .02]])
        data = {"mus": np.array([.01, .02]), "mu_grid": mu,
                "probe_radii": np.array([1.5]),
                "alpha_direct": np.array([[[.2], [.1]], [[.4], [.3]]])}
        data["q_num"] = np.ones((2, 2))
        for key in ("q_kw", "q_center", "q_ms", "q_point_ms", "q_yms"):
            data[key] = np.array([[.19, .12], [.41, .28]]) / tail_force(np.ones((2,2)), data)
        return data

    def test_no_threshold_crossing_is_a_valid_result(self):
        for threshold in (.001, 100.):
            result = threshold_summary(self.data(), threshold)
            model = result["models"]["surface_amplitude"]
            self.assertEqual(model["rows_with_both_contours"], 0)
            self.assertIsNone(model["median_contour_shift_dex"])
            self.assertIsNone(model["max_contour_shift_dex"])

    def test_threshold_decisions_use_the_direct_gradient(self):
        data = self.data()
        result = threshold_summary(data, .3)
        # Equality counts as above; reference and simple model differ at .3.
        self.assertEqual(result["direct_above_points"], 2)
        self.assertEqual(result["models"]["surface_amplitude"]["false_below_points"], 1)
        self.assertEqual(result["both_simple_ansatzes_wrong_points"], 1)

    def test_invalid_thresholds_are_rejected(self):
        for threshold in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                threshold_summary(self.data(), threshold)

    def test_validity_radius_requires_every_later_sample_to_pass(self):
        data = {"probe_radii": np.array([1.1, 1.5, 2., 5.]),
                "mu_grid": np.array([[.1]]), "q_num": np.array([[.5]])}
        tail = tail_force_cube(data)
        data["alpha_direct"] = tail / (1 + np.array([.12, .08, .11, .04]))
        self.assertEqual(float(sampled_validity_radii(data)[0,0]), 5.)

    def test_charge_only_cache_is_rejected(self):
        with self.assertRaises(StaleScanError):
            validate_scan({"q_num": np.ones((2,2))})


if __name__ == "__main__":
    unittest.main()
