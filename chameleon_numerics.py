"""Shared numerical tools for the dimensionless chameleon toy model.

The code in this repository is deliberately dimensionless.  It demonstrates
screening and orbit deformation, but it is not a direct constraint calculation
for a particular physical (M, n, beta) parameter point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq


Array = np.ndarray


@dataclass(frozen=True)
class DimensionlessModel:
    """Coefficients of the normalized n=1 model used in the project."""

    n: int = 1
    beta: float = 1.0
    gravity_k: float = 0.15

    def source(self, phi: Array | float, density: Array | float) -> Array:
        """Return dV_eff/dphi in normalized variables."""

        phi_safe = np.clip(np.asarray(phi, dtype=float), 1.0e-8, 50.0)
        return -1.0 / phi_safe**2 + np.asarray(density) * np.exp(phi_safe)

    def source_derivative(self, phi: Array | float, density: Array | float) -> Array:
        """Return d^2 V_eff/dphi^2 = m_eff^2 in normalized variables."""

        phi_safe = np.clip(np.asarray(phi, dtype=float), 1.0e-8, 50.0)
        return 2.0 / phi_safe**3 + np.asarray(density) * np.exp(phi_safe)


MODEL = DimensionlessModel()


def phi_min(density: float) -> float:
    """Find the positive minimum of V_eff robustly with a bracketed root."""

    if density <= 0:
        raise ValueError("density must be positive")

    def equation(value: float) -> float:
        return float(MODEL.source(value, density))

    lower = 1.0e-8
    upper = max(2.0, density ** -0.5)
    while equation(upper) < 0.0:
        upper *= 2.0
        if upper > 50.0:
            raise RuntimeError("failed to bracket phi_min")
    return float(brentq(equation, lower, upper, xtol=1.0e-13, rtol=1.0e-13))


def solve_planar_field(
    x_grid: Array,
    density: Callable[[Array], Array],
    phi_left: float,
    phi_right: float,
    *,
    tol: float = 1.0e-6,
    max_nodes: int = 30_000,
):
    """Solve the static planar chameleon BVP on a supplied grid."""

    x_grid = np.asarray(x_grid, dtype=float)

    def ode(x: Array, y: Array) -> Array:
        return np.vstack((y[1], MODEL.source(y[0], density(x))))

    def boundary(ya: Array, yb: Array) -> Array:
        return np.array((ya[0] - phi_left, yb[0] - phi_right))

    guess = np.zeros((2, x_grid.size))
    guess[0] = np.linspace(phi_left, phi_right, x_grid.size)
    guess[1] = np.gradient(guess[0], x_grid)
    result = solve_bvp(ode, boundary, x_grid, guess, tol=tol, max_nodes=max_nodes)
    if not result.success:
        raise RuntimeError(f"BVP failed: {result.message}")
    return result


def particle_rhs(
    field_solution,
    *,
    gravity_k: float = MODEL.gravity_k,
    beta: float = MODEL.beta,
) -> Callable[[float, Array], Array]:
    """Build the autonomous particle equation y'=(v, -kx-beta*phi')."""

    def rhs(_time: float, state: Array) -> Array:
        position, velocity = state
        dphi_dx = float(field_solution.sol(position)[1])
        return np.array((velocity, -gravity_k * position - beta * dphi_dx))

    return rhs


def solve_particle_rk45(
    rhs: Callable[[float, Array], Array],
    t_span: tuple[float, float],
    initial_state: tuple[float, float] | Array,
    *,
    t_eval: Array | None = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    max_step: float = 0.05,
):
    """Adaptive Dormand-Prince 5(4) integration through SciPy RK45."""

    result = solve_ivp(
        rhs,
        t_span,
        np.asarray(initial_state, dtype=float),
        method="RK45",
        t_eval=t_eval,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not result.success:
        raise RuntimeError(f"IVP failed: {result.message}")
    return result


def rk4_fixed(
    rhs: Callable[[float, Array], Array],
    t_span: tuple[float, float],
    initial_state: tuple[float, float] | Array,
    step: float,
) -> tuple[Array, Array]:
    """A direct, dependency-free implementation of classical fourth-order RK."""

    start, stop = map(float, t_span)
    if step <= 0 or stop <= start:
        raise ValueError("require step > 0 and t_span[1] > t_span[0]")

    times = [start]
    states = [np.asarray(initial_state, dtype=float)]
    time = start
    state = states[0].copy()
    while time < stop - 10 * np.finfo(float).eps:
        h = min(step, stop - time)
        k1 = np.asarray(rhs(time, state))
        k2 = np.asarray(rhs(time + h / 2, state + h * k1 / 2))
        k3 = np.asarray(rhs(time + h / 2, state + h * k2 / 2))
        k4 = np.asarray(rhs(time + h, state + h * k3))
        state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        time += h
        times.append(time)
        states.append(state.copy())
    return np.asarray(times), np.asarray(states).T


def harmonic_analytic(
    times: Array,
    initial_state: tuple[float, float] | Array,
    gravity_k: float = MODEL.gravity_k,
) -> Array:
    """Closed-form harmonic-oscillator state used as an IVP benchmark."""

    times = np.asarray(times, dtype=float)
    x0, v0 = np.asarray(initial_state, dtype=float)
    omega = np.sqrt(gravity_k)
    position = x0 * np.cos(omega * times) + v0 * np.sin(omega * times) / omega
    velocity = -x0 * omega * np.sin(omega * times) + v0 * np.cos(omega * times)
    return np.vstack((position, velocity))


def mechanical_energy(
    states: Array,
    field_solution,
    *,
    gravity_k: float = MODEL.gravity_k,
    beta: float = MODEL.beta,
) -> Array:
    """Energy per unit mass for the static one-dimensional conservative model."""

    position, velocity = states
    phi = field_solution.sol(position)[0]
    return 0.5 * velocity**2 + 0.5 * gravity_k * position**2 + beta * phi


def cumulative_lyapunov(
    acceleration: Callable[[float], float],
    acceleration_derivative: Callable[[float], float],
    initial_state: tuple[float, float] | Array,
    t_stop: float,
    *,
    renormalize_interval: float = 1.0,
    delta0: float = 1.0e-8,
) -> tuple[Array, Array]:
    """Estimate the largest finite-time Lyapunov exponent by tangent renormalization."""

    state = np.asarray(initial_state, dtype=float)
    tangent = np.array((delta0, 0.0))
    log_sum = 0.0
    times: list[float] = []
    exponents: list[float] = []
    time = 0.0

    def variational_rhs(_time: float, combined: Array) -> Array:
        x, v, dx, dv = combined
        return np.array((v, acceleration(x), dv, acceleration_derivative(x) * dx))

    while time < t_stop - 1.0e-12:
        next_time = min(time + renormalize_interval, t_stop)
        combined0 = np.concatenate((state, tangent))
        segment = solve_ivp(
            variational_rhs,
            (time, next_time),
            combined0,
            method="RK45",
            rtol=2.0e-10,
            atol=2.0e-12,
            max_step=0.03,
        )
        if not segment.success:
            raise RuntimeError(f"Lyapunov integration failed: {segment.message}")
        state = segment.y[:2, -1]
        tangent = segment.y[2:, -1]
        norm = float(np.linalg.norm(tangent))
        if not np.isfinite(norm) or norm == 0:
            raise RuntimeError("invalid tangent norm in Lyapunov calculation")
        log_sum += np.log(norm / delta0)
        tangent *= delta0 / norm
        time = next_time
        times.append(time)
        exponents.append(log_sum / time)

    return np.asarray(times), np.asarray(exponents)
