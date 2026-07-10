"""Reproducible verification suite and figure for the term project."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chameleon_numerics import (
    harmonic_analytic,
    mechanical_energy,
    particle_rhs,
    phi_min,
    rk4_fixed,
    solve_particle_rk45,
    solve_planar_field,
)


OUTPUT_FIGURE = Path(__file__).with_name("verification_results.png")
OUTPUT_METRICS = Path(__file__).with_name("verification_metrics.json")


def transition_density(x):
    return 5.005 - 4.995 * np.tanh(np.asarray(x) / 1.5)


def compute_metrics() -> dict:
    gravity_k = 0.15
    initial = (-2.0, 0.7)
    benchmark_times = np.linspace(0.0, 40.0, 4001)

    def harmonic_rhs(_time, state):
        return np.array((state[1], -gravity_k * state[0]))

    analytic = harmonic_analytic(benchmark_times, initial, gravity_k)
    rk45_tolerances = np.array((1.0e-5, 1.0e-7, 1.0e-9))
    rk45_errors = []
    for tolerance in rk45_tolerances:
        solution = solve_particle_rk45(
            harmonic_rhs,
            (benchmark_times[0], benchmark_times[-1]),
            initial,
            t_eval=benchmark_times,
            rtol=tolerance,
            atol=tolerance * 1.0e-2,
            max_step=2.0,
        )
        rk45_errors.append(float(np.max(np.linalg.norm(solution.y - analytic, axis=0))))

    rk4_steps = np.array((0.4, 0.2, 0.1, 0.05))
    rk4_errors = []
    for step in rk4_steps:
        times, states = rk4_fixed(harmonic_rhs, (0.0, 40.0), initial, float(step))
        exact = harmonic_analytic(times, initial, gravity_k)
        rk4_errors.append(float(np.max(np.linalg.norm(states - exact, axis=0))))
    rk4_order = float(np.polyfit(np.log(rk4_steps), np.log(rk4_errors), 1)[0])

    constant_density = 2.5
    exact_phi = phi_min(constant_density)
    x_uniform = np.linspace(-5.0, 5.0, 201)
    uniform_field = solve_planar_field(
        x_uniform,
        lambda x: np.full_like(np.asarray(x), constant_density, dtype=float),
        exact_phi,
        exact_phi,
        tol=1.0e-8,
    )
    bvp_error = float(np.max(np.abs(uniform_field.sol(x_uniform)[0] - exact_phi)))
    bvp_gradient_error = float(np.max(np.abs(uniform_field.sol(x_uniform)[1])))

    x_grid = np.linspace(-10.0, 10.0, 600)
    transition_field = solve_planar_field(
        x_grid,
        transition_density,
        phi_min(10.0),
        phi_min(0.01),
    )
    transition_rhs = particle_rhs(transition_field, gravity_k=gravity_k)
    full_initial = (-8.0, 2.0)
    full_rk45 = solve_particle_rk45(
        transition_rhs,
        (0.0, 35.0),
        full_initial,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.03,
    )
    rk4_times, rk4_states = rk4_fixed(transition_rhs, (0.0, 35.0), full_initial, 0.005)
    full_cross_error = float(
        np.max(np.linalg.norm(rk4_states - full_rk45.sol(rk4_times), axis=0))
    )
    energy = mechanical_energy(full_rk45.y, transition_field, gravity_k=gravity_k)
    energy_drift = float(np.max(np.abs((energy - energy[0]) / energy[0])))

    return {
        "rk45_tolerances": rk45_tolerances.tolist(),
        "rk45_max_state_errors": rk45_errors,
        "rk4_steps": rk4_steps.tolist(),
        "rk4_max_state_errors": rk4_errors,
        "rk4_observed_order": rk4_order,
        "uniform_density_bvp_max_phi_error": bvp_error,
        "uniform_density_bvp_max_gradient_error": bvp_gradient_error,
        "full_model_rk4_rk45_max_state_difference": full_cross_error,
        "full_model_max_relative_energy_drift": energy_drift,
    }


def make_figure(metrics: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), dpi=150)

    ax = axes[0]
    tolerances = np.asarray(metrics["rk45_tolerances"])
    errors = np.asarray(metrics["rk45_max_state_errors"])
    ax.loglog(tolerances, errors, "o-", color="#0984e3", lw=2)
    ax.invert_xaxis()
    ax.set(xlabel="RK45 relative tolerance", ylabel="Maximum state error", title="(a) Analytic harmonic benchmark")

    ax = axes[1]
    steps = np.asarray(metrics["rk4_steps"])
    rk4_errors = np.asarray(metrics["rk4_max_state_errors"])
    ax.loglog(steps, rk4_errors, "o-", color="#00b894", lw=2, label="measured")
    reference = rk4_errors[-1] * (steps / steps[-1]) ** 4
    ax.loglog(steps, reference, "--", color="#636e72", label=r"$h^4$ reference")
    ax.set(
        xlabel="Fixed RK4 step h",
        ylabel="Maximum state error",
        title=f"(b) Direct RK4: observed order {metrics['rk4_observed_order']:.2f}",
    )
    ax.legend()

    ax = axes[2]
    labels = ("uniform BVP\nfield error", "RK4 vs RK45\nfull orbit", "full-orbit\nenergy drift")
    values = (
        metrics["uniform_density_bvp_max_phi_error"],
        metrics["full_model_rk4_rk45_max_state_difference"],
        metrics["full_model_max_relative_energy_drift"],
    )
    bars = ax.bar(labels, np.maximum(values, 1.0e-16), color=("#6c5ce7", "#fdcb6e", "#e17055"))
    ax.set_yscale("log")
    ax.set(ylabel="Error metric", title="(c) Independent consistency checks")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, max(value, 1.0e-16) * 1.5, f"{value:.1e}", ha="center", fontsize=9)

    fig.suptitle("Numerical verification: analytic solution, convergence, and conservation", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = compute_metrics()
    OUTPUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    make_figure(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"saved {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
