import unittest

import numpy as np

from chameleon_numerics import (
    MODEL,
    harmonic_analytic,
    phi_min,
    rk4_fixed,
    solve_particle_rk45,
    solve_planar_field,
)


class NumericalVerificationTests(unittest.TestCase):
    def test_phi_min_satisfies_equilibrium(self):
        for density in (0.01, 1.0, 10.0):
            value = phi_min(density)
            self.assertLess(abs(float(MODEL.source(value, density))), 1.0e-10)

    def test_uniform_density_bvp_matches_constant_solution(self):
        density = 2.5
        exact = phi_min(density)
        grid = np.linspace(-4.0, 4.0, 101)
        solution = solve_planar_field(
            grid,
            lambda x: np.full_like(np.asarray(x), density, dtype=float),
            exact,
            exact,
            tol=1.0e-8,
        )
        self.assertLess(np.max(np.abs(solution.sol(grid)[0] - exact)), 1.0e-8)
        self.assertLess(np.max(np.abs(solution.sol(grid)[1])), 1.0e-8)

    def test_rk45_matches_harmonic_analytic_solution(self):
        gravity_k = 0.15
        initial = (-2.0, 0.7)

        def rhs(_time, state):
            return np.array((state[1], -gravity_k * state[0]))

        times = np.linspace(0.0, 40.0, 2001)
        result = solve_particle_rk45(
            rhs,
            (times[0], times[-1]),
            initial,
            t_eval=times,
            rtol=1.0e-9,
            atol=1.0e-11,
            max_step=0.2,
        )
        error = np.max(np.linalg.norm(result.y - harmonic_analytic(times, initial, gravity_k), axis=0))
        self.assertLess(error, 1.0e-7)

    def test_direct_rk4_has_fourth_order_convergence(self):
        gravity_k = 0.15
        initial = (-2.0, 0.7)

        def rhs(_time, state):
            return np.array((state[1], -gravity_k * state[0]))

        steps = np.array((0.4, 0.2, 0.1, 0.05))
        errors = []
        for step in steps:
            times, states = rk4_fixed(rhs, (0.0, 40.0), initial, float(step))
            exact = harmonic_analytic(times, initial, gravity_k)
            errors.append(np.max(np.linalg.norm(states - exact, axis=0)))
        order = np.polyfit(np.log(steps), np.log(errors), 1)[0]
        self.assertGreater(order, 3.8)
        self.assertLess(order, 4.2)


if __name__ == "__main__":
    unittest.main()
