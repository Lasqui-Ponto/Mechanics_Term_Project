"""Spherical chameleon thin-shell model and scalar-charge estimators.

The main scan uses the controlled weak-coupling limit of

    V(phi) = Lambda**5 / phi,
    A(phi) = exp(beta phi / M_Pl) ~= 1 + beta phi / M_Pl.

With x=r/R, u=phi/phi_inf, delta=rho_in/rho_out, and
mu=m_inf R, the n=1 field equation becomes

    u'' + 2 u'/x = mu**2/2 * (rho/rho_out - u**-2).

The code solves the full nonlinear spherical boundary-value problem.  It does
not assume a thin shell.  A charge-scaled dependent variable makes both weak
and strongly screened sources numerically well conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import solve_bvp


Array = np.ndarray


class PiecewiseBVPSolution:
    """Expose a two-domain collocation solution as a function of physical x."""

    def __init__(self, raw: object, x_max: float):
        self.raw = raw
        self.x_max = float(x_max)
        self.success = raw.success
        self.message = raw.message
        self.status = raw.status
        inner_x = raw.x
        outer_x = 1.0 + (self.x_max - 1.0) * raw.x
        self.x = np.concatenate((inner_x, outer_x[1:]))
        self.y = np.concatenate((raw.y[:2], raw.y[2:, 1:]), axis=1)

    def sol(self, x: Array | float) -> Array:
        scalar = np.ndim(x) == 0
        radius = np.atleast_1d(np.asarray(x, dtype=float))
        if np.any(radius < 0.0) or np.any(radius > self.x_max):
            raise ValueError("evaluation radius lies outside the BVP domain")
        values = np.empty((2, radius.size), dtype=float)
        inside = radius <= 1.0
        if np.any(inside):
            values[:, inside] = self.raw.sol(radius[inside])[:2]
        if np.any(~inside):
            outer_t = (radius[~inside] - 1.0) / (self.x_max - 1.0)
            values[:, ~inside] = self.raw.sol(outer_t)[2:]
        return values[:, 0] if scalar else values


@dataclass(frozen=True)
class ThinShellPoint:
    """Dimensionless parameters for a uniform spherical source."""

    density_contrast: float
    mu_out: float
    beta: float = 1.0
    probe_radius: float = 1.5

    def __post_init__(self) -> None:
        if self.density_contrast <= 1.0:
            raise ValueError("density_contrast must exceed one")
        if self.mu_out <= 0.0:
            raise ValueError("mu_out must be positive")
        if self.probe_radius <= 1.0:
            raise ValueError("probe_radius is measured in R and must exceed one")

    @property
    def phi_core_ratio(self) -> float:
        """Equilibrium phi_core/phi_out for n=1 in the weak-coupling limit."""

        return self.density_contrast ** -0.5

    @property
    def b_coefficient(self) -> float:
        """Matter/source coefficient B=mu_out^2/(n+1), with n=1."""

        return 0.5 * self.mu_out**2

    @property
    def charge_scale(self) -> float:
        """Exterior amplitude scale normalized by the total body mass."""

        return self.b_coefficient * self.density_contrast / 3.0

    @property
    def epsilon(self) -> float:
        """Potential-defined thin-shell parameter epsilon_th.

        It approximates a geometric shell thickness only in an additional
        asymptotic regime.  Finite interior-mass corrections need not be small
        merely because m_in R exceeds one.
        """

        return (1.0 - self.phi_core_ratio) / (
            self.b_coefficient * self.density_contrast
        )

    @property
    def mu_in(self) -> float:
        """Interior Compton parameter m_in R for n=1."""

        return self.mu_out * self.density_contrast ** 0.75


@dataclass
class ChargeSolution:
    """A converged nonlinear BVP and its scalar-charge diagnostics."""

    point: ThinShellPoint
    result: object
    q_amplitude: float
    q_gradient: float
    q_scatter: float
    fit_x_min: float
    fit_x_max: float
    formulation: str

    @property
    def q_num(self) -> float:
        return self.q_amplitude

    @property
    def estimator_disagreement(self) -> float:
        denominator = max(abs(self.q_amplitude), 1.0e-15)
        return abs(self.q_gradient - self.q_amplitude) / denominator

    def field(self, x: Array | float) -> Array:
        """Return u=phi/phi_out."""

        x_array = np.asarray(x, dtype=float)
        values = self.result.sol(x_array)
        ratio = _v_over_x(x_array, values[0], values[1])
        if self.formulation == "log":
            return np.exp(self.point.charge_scale * ratio)
        return 1.0 + self.point.charge_scale * ratio

    def field_gradient(self, x: Array | float) -> Array:
        """Return du/dx."""

        x_array = np.asarray(x, dtype=float)
        values = self.result.sol(x_array)
        v, vp = values
        gradient = np.empty_like(x_array)
        nonzero = x_array != 0.0
        prefactor = np.ones_like(x_array)
        if self.formulation == "log":
            log_field = self.point.charge_scale * _v_over_x(x_array, v, vp)
            prefactor = np.exp(log_field)
        gradient[nonzero] = (
            prefactor[nonzero]
            * self.point.charge_scale
            * (x_array[nonzero] * vp[nonzero] - v[nonzero])
            / x_array[nonzero] ** 2
        )
        if np.any(~nonzero):
            # Regularity makes the first derivative vanish at the origin.
            gradient[~nonzero] = 0.0
        return gradient

    def local_charge_estimators(self, x: Array | float) -> tuple[Array, Array]:
        """Return exterior amplitude and gradient charge estimators."""

        radius = np.asarray(x, dtype=float)
        if np.any(radius <= 1.0):
            raise ValueError("local charge estimators require x>1")
        u = self.field(radius)
        up = self.field_gradient(radius)
        exponential = np.exp(self.point.mu_out * (radius - 1.0))
        amplitude = radius * (1.0 - u) * exponential / self.point.charge_scale
        gradient = (
            radius**2
            * up
            * exponential
            / (self.point.charge_scale * (1.0 + self.point.mu_out * radius))
        )
        return amplitude, gradient

    def direct_force_ratio(
        self, x: Array | float | None = None, *, q_probe: Array | float = 1.0
    ) -> Array:
        """Evaluate |a_5/a_N| from the nonlinear local field gradient.

        The Newtonian reference is the total source mass, as in charge_scale.
        The probe is small enough not to backreact on the source solution;
        q_probe is a supplied response factor rather than a two-body solution.
        No exterior Yukawa linearization is used in this evaluation.
        """

        radius = np.asarray(self.point.probe_radius if x is None else x, dtype=float)
        if np.any(~np.isfinite(radius)) or np.any(radius <= 1.0):
            raise ValueError("force-ratio evaluation requires finite x>1")
        return (
            2.0 * self.point.beta**2 * np.asarray(q_probe) * radius**2
            * self.field_gradient(radius) / self.point.charge_scale
        )


def q_thin_shell(point: ThinShellPoint) -> float:
    """Long-range thin-shell charge, clipped through the transition."""

    return float(min(3.0 * point.epsilon, 1.0))


def q_center_point_continuation(point: ThinShellPoint) -> float:
    """Alternative finite-range ansatz in the same surface-tail convention.

    This changes the predicted amplitude; it is not a reparameterization of
    q_thin_shell and does not imply an ambiguity in the underlying field theory.
    """

    return float(np.exp(-point.mu_out) * q_thin_shell(point))


def q_matched_shell(point: ThinShellPoint) -> float:
    """Matched massless-shell interpolation across the transition regime."""

    core_fraction = np.sqrt(max(1.0 - 2.0 * point.epsilon, 0.0))
    return float(1.0 - core_fraction**3)


def q_point_matched_shell(point: ThinShellPoint) -> float:
    """Ablation control: finite shell volume times a point Yukawa factor.

    Comparing this with q_yukawa_matched isolates the additional spherical
    radial weighting while keeping the same assumed core radius.
    """

    return float(np.exp(-point.mu_out) * q_matched_shell(point))


def _g_yukawa(z: Array | float) -> Array:
    """Stable evaluation of z cosh(z)-sinh(z)."""

    z_array = np.asarray(z, dtype=float)
    absolute = np.abs(z_array)
    direct = z_array * np.cosh(z_array) - np.sinh(z_array)
    series = z_array**3 / 3.0 + z_array**5 / 30.0 + z_array**7 / 840.0
    return np.where(absolute < 0.08, series, direct)


def q_linear_sphere(mu_out: Array | float) -> Array:
    """Surface-referenced Yukawa form factor of a uniform linear sphere."""

    mu = np.asarray(mu_out, dtype=float)
    return 3.0 * np.exp(-mu) * _g_yukawa(mu) / mu**3


def q_yukawa_matched(point: ThinShellPoint) -> float:
    """Active-shell ansatz using the standard spherical Yukawa radial weight."""

    mu = point.mu_out
    core_fraction = np.sqrt(max(1.0 - 2.0 * point.epsilon, 0.0))
    numerator = _g_yukawa(mu) - _g_yukawa(mu * core_fraction)
    return float(3.0 * np.exp(-mu) * numerator / mu**3)


def force_ratio(
    q_source: Array | float,
    point: ThinShellPoint,
    *,
    q_probe: Array | float = 1.0,
) -> Array:
    """Yukawa-tail prediction of |a_5/a_N|, not the nonlinear local force.

    Use ChargeSolution.direct_force_ratio for the numerical reference.  Even an
    accurately extracted asymptotic charge can mispredict the force near R.
    """

    x_probe = point.probe_radius
    yukawa = (1.0 + point.mu_out * x_probe) * np.exp(
        -point.mu_out * (x_probe - 1.0)
    )
    return 2.0 * point.beta**2 * np.asarray(q_source) * np.asarray(q_probe) * yukawa


def _v_over_x(x: Array, v: Array, vp: Array) -> Array:
    ratio = np.empty_like(x, dtype=float)
    nonzero = x != 0.0
    ratio[nonzero] = v[nonzero] / x[nonzero]
    ratio[~nonzero] = vp[~nonzero]
    return ratio


def _initial_guess(point: ThinShellPoint, x: Array, *, formulation: str) -> Array:
    q_guess = q_matched_shell(point)
    v = np.empty_like(x)
    vp = np.empty_like(x)
    inside = x <= 1.0

    core_radius = np.sqrt(max(1.0 - 2.0 * point.epsilon, 0.0))
    if core_radius == 0.0:
        # Exact massless Poisson shape, scaled by the expected charge.
        xi = x[inside]
        v[inside] = -q_guess * xi * (1.5 - 0.5 * xi**2)
        vp[inside] = -1.5 * q_guess * (1.0 - xi**2)
    else:
        xi = x[inside]
        if formulation == "log":
            core_slope = np.log(point.phi_core_ratio) / point.charge_scale
        else:
            core_slope = (point.phi_core_ratio - 1.0) / point.charge_scale
        core = xi <= core_radius
        v_inside = core_slope * xi
        vp_inside = np.full_like(xi, core_slope)
        if np.any(~core):
            shell_x = xi[~core]
            width = 1.0 - core_radius
            tau = (shell_x - core_radius) / width
            value_left = core_slope * core_radius
            slope_left = core_slope * width
            value_right = -q_guess
            slope_right = point.mu_out * q_guess * width
            h00 = 2.0 * tau**3 - 3.0 * tau**2 + 1.0
            h10 = tau**3 - 2.0 * tau**2 + tau
            h01 = -2.0 * tau**3 + 3.0 * tau**2
            h11 = tau**3 - tau**2
            v_inside[~core] = (
                h00 * value_left
                + h10 * slope_left
                + h01 * value_right
                + h11 * slope_right
            )
            dh00 = 6.0 * tau**2 - 6.0 * tau
            dh10 = 3.0 * tau**2 - 4.0 * tau + 1.0
            dh01 = -dh00
            dh11 = 3.0 * tau**2 - 2.0 * tau
            vp_inside[~core] = (
                dh00 * value_left
                + dh10 * slope_left
                + dh01 * value_right
                + dh11 * slope_right
            ) / width
        v[inside] = v_inside
        vp[inside] = vp_inside

    xo = x[~inside]
    exterior = np.exp(-point.mu_out * (xo - 1.0))
    v[~inside] = -q_guess * exterior
    vp[~inside] = point.mu_out * q_guess * exterior
    return np.vstack((v, vp))


def _default_mesh(points: int) -> Array:
    """Shared collocation coordinate for the exact inner and outer domains."""

    points = max(points, 120)
    # Cosine spacing resolves the interior side at t=1 and the exterior side
    # at t=0 without smoothing the physical density discontinuity.
    angle = np.linspace(0.0, np.pi, points)
    return 0.5 * (1.0 - np.cos(angle))


def solve_charge(
    point: ThinShellPoint,
    *,
    x_max: float = 20.0,
    tol: float = 2.0e-5,
    mesh_points: int = 360,
    max_nodes: int = 50_000,
    initial_solution: object | None = None,
    formulation: str = "auto",
) -> ChargeSolution:
    """Solve the nonlinear spherical BVP and extract its exterior charge.

    The density step is kept sharp and x=1 is inserted explicitly in the
    initial mesh.  Convergence tests in the study compare this result against
    tighter tolerances, larger domains, and denser meshes.
    """

    if x_max <= 2.0:
        raise ValueError("x_max must exceed two source radii")
    if formulation not in {"auto", "linear", "log"}:
        raise ValueError("formulation must be 'auto', 'linear', or 'log'")
    requested_formulation = formulation
    if formulation == "auto":
        formulation = "linear" if point.epsilon < 0.05 else "log"

    t = _default_mesh(mesh_points)
    x_inner = t
    x_outer = 1.0 + (x_max - 1.0) * t
    if initial_solution is None:
        guess = np.vstack(
            (
                _initial_guess(point, x_inner, formulation=formulation),
                _initial_guess(point, x_outer, formulation=formulation),
            )
        )
    else:
        guess = np.vstack(
            (
                np.asarray(initial_solution.sol(x_inner), dtype=float),
                np.asarray(initial_solution.sol(x_outer), dtype=float),
            )
        )

    delta = point.density_contrast
    scale = point.charge_scale

    def physical_vpp(radius: Array, v: Array, vp: Array, density: float) -> Array:
        ratio = _v_over_x(radius, v, vp)
        if formulation == "log":
            log_u = scale * ratio
            log_u_safe = np.clip(log_u, -200.0, 200.0)
            source = density * np.exp(-log_u_safe) - np.exp(-3.0 * log_u_safe)
            vpp = 3.0 * radius * source / delta
            combination = radius * vp - v
            gradient_square = np.zeros_like(radius)
            nonzero = radius != 0.0
            with np.errstate(over="ignore", invalid="ignore"):
                gradient_square[nonzero] = (
                    scale * combination[nonzero] ** 2 / radius[nonzero] ** 3
                )
            return vpp - gradient_square

        u = 1.0 + scale * ratio
        u_safe = np.maximum(u, 1.0e-12)
        return 3.0 * radius * (density - u_safe**-2) / delta

    def ode(coordinate: Array, y: Array) -> Array:
        vi, vpi, vo, vpo = y
        radius_inner = coordinate
        radius_outer = 1.0 + (x_max - 1.0) * coordinate
        vppi = physical_vpp(radius_inner, vi, vpi, delta)
        vppo = physical_vpp(radius_outer, vo, vpo, 1.0)
        outer_scale = x_max - 1.0
        return np.vstack((vpi, vppi, outer_scale * vpo, outer_scale * vppo))

    def boundary(ya: Array, yb: Array) -> Array:
        return np.array(
            (
                ya[0],
                yb[0] - ya[2],
                yb[1] - ya[3],
                yb[3] + point.mu_out * yb[2],
            )
        )

    raw_result = solve_bvp(
        ode,
        boundary,
        t,
        guess,
        tol=tol,
        max_nodes=max_nodes,
        verbose=0,
    )
    if not raw_result.success and requested_formulation == "auto":
        alternate = "log" if formulation == "linear" else "linear"
        return solve_charge(
            point,
            x_max=x_max,
            tol=tol,
            mesh_points=mesh_points,
            max_nodes=max_nodes,
            initial_solution=None,
            formulation=alternate,
        )
    if not raw_result.success:
        raise RuntimeError(
            f"BVP failed for delta={delta:.5g}, mu={point.mu_out:.5g}: "
            f"{raw_result.message}"
        )
    result = PiecewiseBVPSolution(raw_result, x_max)

    diagnostic_x = np.linspace(1.05, x_max - 0.2, 900)
    values = result.sol(diagnostic_x)
    v, vp = values
    if formulation == "log":
        u = np.exp(scale * v / diagnostic_x)
        gradient_prefactor = u
    else:
        u = 1.0 + scale * v / diagnostic_x
        gradient_prefactor = np.ones_like(u)

    # Choose points for which the exact exterior source is close to its
    # linear Yukawa expansion.  A fallback keeps the analysis deterministic
    # for the most nonlinear profiles.
    exact_source = point.b_coefficient * (1.0 - u**-2)
    linear_source = point.mu_out**2 * (u - 1.0)
    nonlinearity = np.abs(exact_source - linear_source) / (
        np.abs(linear_source) + 1.0e-14
    )
    signal = np.abs(v) > 1.0e-10
    fit_mask = (nonlinearity < 0.02) & signal
    if np.count_nonzero(fit_mask) < 20:
        fit_mask = (diagnostic_x >= max(2.0, 0.55 * diagnostic_x[-1])) & signal
    if np.count_nonzero(fit_mask) < 10:
        raise RuntimeError("insufficient exterior points for scalar-charge fit")

    x_fit = diagnostic_x[fit_mask]
    v_fit = v[fit_mask]
    vp_fit = vp[fit_mask]
    exponential = np.exp(point.mu_out * (x_fit - 1.0))
    u_fit = u[fit_mask]
    prefactor_fit = gradient_prefactor[fit_mask]
    q_amp_samples = x_fit * (1.0 - u_fit) * exponential / scale
    q_grad_samples = (
        exponential
        * prefactor_fit
        * (x_fit * vp_fit - v_fit)
        / (1.0 + point.mu_out * x_fit)
    )

    q_amplitude = float(np.median(q_amp_samples))
    q_gradient = float(np.median(q_grad_samples))
    q_scatter = float(
        1.4826 * np.median(np.abs(q_amp_samples - q_amplitude))
        / max(abs(q_amplitude), 1.0e-15)
    )
    if q_amplitude <= 0.0 or not np.isfinite(q_amplitude):
        raise RuntimeError("extracted scalar charge is not positive and finite")

    field_coordinate = scale * _v_over_x(result.x, result.y[0], result.y[1])
    if formulation == "log":
        minimum_u = float(np.min(np.exp(field_coordinate)))
    else:
        minimum_u = float(np.min(1.0 + field_coordinate))
    if minimum_u <= 0.0:
        raise RuntimeError("converged BVP contains non-positive field values")

    return ChargeSolution(
        point=point,
        result=result,
        q_amplitude=q_amplitude,
        q_gradient=q_gradient,
        q_scatter=q_scatter,
        fit_x_min=float(x_fit.min()),
        fit_x_max=float(x_fit.max()),
        formulation=formulation,
    )


def solve_sequence(
    points: Iterable[ThinShellPoint],
    **kwargs,
) -> list[ChargeSolution]:
    """Solve a continuation sequence, reusing the previous converged profile."""

    solutions: list[ChargeSolution] = []
    previous = None
    for point in points:
        solution = solve_charge(point, initial_solution=previous, **kwargs)
        solutions.append(solution)
        previous = solution.result
    return solutions
