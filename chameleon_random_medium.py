"""Static random-medium experiment with quantitative chaos diagnostics.

Important: a time-independent one-dimensional Hamiltonian is integrable.  The
random force distorts the orbit, but a positive asymptotic Lyapunov exponent is
not expected.  This script checks that statement instead of inferring chaos
from a complicated-looking phase portrait.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from chameleon_numerics import (
    cumulative_lyapunov,
    mechanical_energy,
    particle_rhs,
    phi_min,
    solve_particle_rk45,
    solve_planar_field,
)


X_MIN, X_MAX = -15.0, 15.0
GRAVITY_K = 0.15
X_START, V_START = -7.0, 2.2
SEED = 137
OUTPUT = Path(__file__).with_name("chameleon_random_medium_results.png")
METRICS_OUTPUT = Path(__file__).with_name("random_medium_metrics.json")


def build_density(x_grid: np.ndarray):
    rng = np.random.default_rng(SEED)
    core_grid = np.linspace(-10.0, 10.0, 700)
    noise = gaussian_filter1d(rng.normal(0.0, 1.0, core_grid.size), sigma=20)
    noise /= np.std(noise)
    density_core = np.clip(2.5 + 1.8 * noise, 0.05, 8.0)
    density_values = np.interp(x_grid, core_grid, density_core)

    def density(x):
        return np.interp(x, x_grid, density_values)

    return density_values, density


def positive_crossing_periods(times: np.ndarray, states: np.ndarray) -> np.ndarray:
    x, v = states
    indices = np.where((x[:-1] < 0.0) & (x[1:] >= 0.0) & (v[1:] > 0.0))[0]
    crossings = []
    for index in indices:
        fraction = -x[index] / (x[index + 1] - x[index])
        crossings.append(times[index] + fraction * (times[index + 1] - times[index]))
    return np.diff(np.asarray(crossings))


def main() -> None:
    x_grid = np.linspace(X_MIN, X_MAX, 1100)
    density_values, density = build_density(x_grid)
    field = solve_planar_field(
        x_grid,
        density,
        phi_min(float(density_values[0])),
        phi_min(float(density_values[-1])),
        tol=2.0e-6,
    )
    rhs = particle_rhs(field, gravity_k=GRAVITY_K)

    times = np.linspace(0.0, 220.0, 9000)
    orbit = solve_particle_rk45(rhs, (times[0], times[-1]), (X_START, V_START), t_eval=times)
    if np.max(np.abs(orbit.y[0])) >= 0.98 * X_MAX:
        raise RuntimeError("particle approached the BVP boundary")

    energy = mechanical_energy(orbit.y, field, gravity_k=GRAVITY_K)
    relative_energy = (energy - energy[0]) / energy[0]
    max_energy_drift = float(np.max(np.abs(relative_energy)))
    periods = positive_crossing_periods(orbit.t, orbit.y)
    period_cv = float(np.std(periods) / np.mean(periods)) if periods.size >= 2 else float("nan")

    def acceleration(position: float) -> float:
        return -GRAVITY_K * position - float(field.sol(position)[1])

    def acceleration_derivative(position: float) -> float:
        return -GRAVITY_K - float(field.sol(position, 1)[1])

    lyapunov_t, lyapunov = cumulative_lyapunov(
        acceleration,
        acceleration_derivative,
        (X_START, V_START),
        t_stop=220.0,
        renormalize_interval=0.5,
    )
    final_ftle = float(lyapunov[-1])

    plot_grid = np.linspace(-12.0, 12.0, 850)
    phi_values, gradients = field.sol(plot_grid)
    fifth_force = -gradients

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=140)
    fig.suptitle(
        "Static Disordered Medium: nonlinear orbit deformation, not chaos\n"
        "Energy conservation and finite-time Lyapunov diagnostic",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0, 0]
    density_line = ax.plot(plot_grid, density(plot_grid), color="#ff7675", lw=1.8, label=r"Density $\hat\rho_R$")
    ax.set(xlabel=r"Position $\hat x$", ylabel=r"Density $\hat\rho_R$", title="(a) Reproducible random density (seed 137)")
    ax.tick_params(axis="y", labelcolor="#ff7675")
    twin = ax.twinx()
    field_line = twin.plot(plot_grid, phi_values, color="#55efc4", lw=2.2, label=r"Field $\hat\phi$")
    twin.set_ylabel(r"Field $\hat\phi$", color="#55efc4")
    twin.tick_params(axis="y", labelcolor="#55efc4")
    lines = density_line + field_line
    ax.legend(lines, [line.get_label() for line in lines], loc="upper center")
    ax.grid(alpha=0.16, ls="--")

    ax = axes[0, 1]
    ax.plot(plot_grid, fifth_force, color="#d6a2e8", lw=2.0)
    ax.axhline(0.0, color="white", lw=0.8, alpha=0.35)
    ax.set(xlabel=r"Position $\hat x$", ylabel=r"Acceleration $\hat a_5$", title="(b) Static fifth-force landscape")
    ax.grid(alpha=0.16, ls="--")

    ax = axes[1, 0]
    ax.plot(orbit.y[0], orbit.y[1], color="#fdcb6e", lw=1.3)
    ax.scatter([X_START], [V_START], color="#ff5252", s=50, zorder=4)
    ax.set(xlabel=r"Position $\hat x$", ylabel=r"Velocity $\hat v$", title="(c) One closed energy contour in phase space")
    ax.text(
        0.03,
        0.04,
        "A 1D autonomous conservative system\nhas no deterministic chaos.",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#17202a", "alpha": 0.85},
    )
    ax.grid(alpha=0.16, ls="--")

    ax = axes[1, 1]
    ax.semilogy(orbit.t, np.maximum(np.abs(relative_energy), 1.0e-16), color="#74b9ff", lw=1.5, label=r"$|\Delta E/E_0|$")
    ax.set(xlabel=r"Time $\hat t$", ylabel="Relative energy error", title="(d) Quantitative diagnostics")
    ax.grid(alpha=0.16, ls="--")
    twin = ax.twinx()
    twin.plot(lyapunov_t, lyapunov, color="#ff7675", lw=1.8, label=r"Cumulative $\lambda_{FT}$")
    twin.axhline(0.0, color="#ff7675", ls=":", alpha=0.5)
    twin.set_ylabel(r"Finite-time exponent $\lambda_{FT}$", color="#ff7675")
    twin.tick_params(axis="y", labelcolor="#ff7675")
    ax.text(
        0.03,
        0.80,
        f"max |ΔE/E₀| = {max_energy_drift:.1e}\nλFT(t={lyapunov_t[-1]:.0f}) = {final_ftle:.2e}\nperiod CV = {period_cv:.1e}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#17202a", "alpha": 0.85},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "model": "static_1d_autonomous_conservative",
        "seed": SEED,
        "max_relative_energy_drift": max_energy_drift,
        "final_finite_time_lyapunov": final_ftle,
        "lyapunov_evaluation_time": float(lyapunov_t[-1]),
        "positive_crossing_period_mean": float(np.mean(periods)),
        "positive_crossing_period_cv": period_cv,
        "interpretation": "nonlinear orbit deformation; no evidence of deterministic chaos",
    }
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
