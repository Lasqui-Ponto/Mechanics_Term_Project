"""Planar density-transition experiment for the dimensionless toy model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chameleon_numerics import (
    MODEL,
    mechanical_energy,
    particle_rhs,
    phi_min,
    rk4_fixed,
    solve_particle_rk45,
    solve_planar_field,
)


X_MIN, X_MAX = -20.0, 20.0
RHO_HIGH, RHO_LOW = 10.0, 0.01
TRANSITION_WIDTH = 1.5
GRAVITY_K = 0.15
X_START, V_START = -8.0, 2.0
OUTPUT = Path(__file__).with_name("chameleon_simulation_results.png")


def density(x):
    x = np.asarray(x)
    return 0.5 * (RHO_HIGH + RHO_LOW) - 0.5 * (RHO_HIGH - RHO_LOW) * np.tanh(
        x / TRANSITION_WIDTH
    )


def main() -> None:
    x_grid = np.linspace(X_MIN, X_MAX, 1000)
    left, right = phi_min(RHO_HIGH), phi_min(RHO_LOW)
    field = solve_planar_field(x_grid, density, left, right)
    rhs = particle_rhs(field, gravity_k=GRAVITY_K)

    times = np.linspace(0.0, 35.0, 1600)
    initial = (X_START, V_START)
    chameleon = solve_particle_rk45(rhs, (times[0], times[-1]), initial, t_eval=times)

    def newton_rhs(_time, state):
        return np.array((state[1], -GRAVITY_K * state[0]))

    newton = solve_particle_rk45(newton_rhs, (times[0], times[-1]), initial, t_eval=times)
    rk4_t, rk4_y = rk4_fixed(rhs, (times[0], times[-1]), initial, step=0.005)
    rk45_on_rk4 = chameleon.sol(rk4_t)
    rk4_difference = float(np.max(np.linalg.norm(rk4_y - rk45_on_rk4, axis=0)))

    energy = mechanical_energy(chameleon.y, field, gravity_k=GRAVITY_K)
    relative_energy_drift = float(np.max(np.abs((energy - energy[0]) / energy[0])))
    if np.max(np.abs(chameleon.y[0])) >= 0.98 * X_MAX:
        raise RuntimeError("particle approached the BVP boundary; enlarge the domain or reduce energy")

    plot_grid = np.linspace(-12.0, 12.0, 700)
    phi_values = field.sol(plot_grid)[0]
    fifth_force = -MODEL.beta * field.sol(plot_grid)[1]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=140)
    fig.suptitle(
        "Dimensionless Chameleon Screening Toy Model\n"
        "Static BVP field + adaptive RK45 particle orbit",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0, 0]
    density_line = ax.plot(plot_grid, density(plot_grid), color="#ff7675", lw=2.4, label=r"Density $\hat\rho$")
    ax.set(xlabel=r"Position $\hat x$", ylabel=r"Density $\hat\rho$", title="(a) Density and field response")
    ax.tick_params(axis="y", labelcolor="#ff7675")
    twin = ax.twinx()
    field_line = twin.plot(plot_grid, phi_values, color="#55efc4", lw=2.5, label=r"Field $\hat\phi$")
    twin.set_ylabel(r"Field $\hat\phi$", color="#55efc4")
    twin.tick_params(axis="y", labelcolor="#55efc4")
    lines = density_line + field_line
    ax.legend(lines, [line.get_label() for line in lines], loc="upper center")
    ax.grid(alpha=0.18, ls="--")

    ax = axes[0, 1]
    ax.plot(plot_grid, fifth_force, color="#d6a2e8", lw=2.5)
    ax.axvspan(-2 * TRANSITION_WIDTH, 2 * TRANSITION_WIDTH, color="#ffeaa7", alpha=0.08)
    ax.axhline(0.0, color="white", lw=0.8, alpha=0.35)
    ax.set(
        xlabel=r"Position $\hat x$",
        ylabel=r"Acceleration $\hat a_5=-d\hat\phi/d\hat x$",
        title="(b) Localized fifth-force acceleration",
    )
    ax.grid(alpha=0.18, ls="--")

    ax = axes[1, 0]
    ax.plot(newton.t, newton.y[0], "--", color="#74b9ff", lw=1.8, label="Harmonic control")
    ax.plot(chameleon.t, chameleon.y[0], color="#00cec9", lw=2.2, label="Control + chameleon")
    ax.set(xlabel=r"Time $\hat t$", ylabel=r"Position $\hat x$", title="(c) Orbit comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.18, ls="--")

    ax = axes[1, 1]
    ax.plot(newton.y[0], newton.y[1], "--", color="#74b9ff", lw=1.8, label="Harmonic control")
    ax.plot(chameleon.y[0], chameleon.y[1], color="#fdcb6e", lw=2.1, label="Chameleon orbit")
    ax.scatter([X_START], [V_START], s=55, color="#ff5252", zorder=4, label="Initial state")
    ax.set(xlabel=r"Position $\hat x$", ylabel=r"Velocity $\hat v$", title="(d) Phase-space deformation")
    ax.text(
        0.03,
        0.03,
        f"max |ΔE/E₀| = {relative_energy_drift:.1e}\nRK4–RK45 max Δstate = {rk4_difference:.1e}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#17202a", "alpha": 0.8},
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.18, ls="--")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)

    print(f"BVP converged with {field.x.size} adaptive nodes")
    print(f"phi_min(high)={left:.8f}, phi_min(low)={right:.8f}")
    print(f"max relative energy drift={relative_energy_drift:.3e}")
    print(f"max RK4-RK45 state difference={rk4_difference:.3e}")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
