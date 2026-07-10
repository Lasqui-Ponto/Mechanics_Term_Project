"""Illustrative vacuum-chamber profile for atom versus bulk sensors.

The field profile is a property of the chamber and source.  A test sensor does
not get a different phi(r); instead, its scalar charge Q_test multiplies the
same field gradient.  The charge values below are illustrative, not a fit to a
specific experiment or a new exclusion limit.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

from chameleon_numerics import MODEL, phi_min


SOURCE_RADIUS = 0.18
CHAMBER_RADIUS = 1.0
WALL_START = 0.92
RHO_SOURCE = 20.0
RHO_VACUUM = 0.01
RHO_WALL = 30.0
Q_ATOM = 1.0
Q_BULK = 0.03
OUTPUT = Path(__file__).with_name("sensor_detectability.png")


def smooth_step(x, center, width=0.018):
    return 0.5 * (1.0 + np.tanh((np.asarray(x) - center) / width))


def density(radius):
    source_to_vacuum = smooth_step(radius, SOURCE_RADIUS)
    vacuum_to_wall = smooth_step(radius, WALL_START)
    source_vacuum = RHO_SOURCE * (1.0 - source_to_vacuum) + RHO_VACUUM * source_to_vacuum
    return source_vacuum * (1.0 - vacuum_to_wall) + RHO_WALL * vacuum_to_wall


def solve_profile():
    grid = np.linspace(0.0, CHAMBER_RADIUS, 500)
    wall_value = phi_min(RHO_WALL)
    source_value = phi_min(RHO_SOURCE)
    guess = np.zeros((2, grid.size))
    guess[0] = source_value + (wall_value - source_value) * (grid / CHAMBER_RADIUS) ** 2
    guess[1] = np.gradient(guess[0], grid)

    def ode(radius, values):
        return np.vstack((values[1], MODEL.source(values[0], density(radius))))

    def boundary(left, right):
        return np.array((left[1], right[0] - wall_value))

    result = solve_bvp(ode, boundary, grid, guess, tol=1.0e-6, max_nodes=30_000)
    if not result.success:
        raise RuntimeError(result.message)
    return result


def main() -> None:
    profile = solve_profile()
    radius = np.linspace(0.0, CHAMBER_RADIUS, 1000)
    phi, gradient = profile.sol(radius)
    atom_acceleration = -Q_ATOM * gradient
    bulk_acceleration = -Q_BULK * gradient

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    fig.suptitle(
        "Why an atom can be a more sensitive chameleon probe than bulk matter\n"
        "Illustrative dimensionless vacuum-chamber cross section",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0]
    phi_line = ax.plot(radius, phi, color="#00a884", lw=2.8, label=r"Common field $\hat\phi(r)$")
    ax.set(xlabel=r"Radius $r/R_{ch}$", ylabel=r"Field $\hat\phi$", title="(a) Source + vacuum + dense chamber wall")
    ax.axvspan(0.0, SOURCE_RADIUS, color="#636e72", alpha=0.15, label="dense source")
    ax.axvspan(WALL_START, CHAMBER_RADIUS, color="#2d3436", alpha=0.25, label="dense wall")
    twin = ax.twinx()
    density_line = twin.semilogy(radius, density(radius), "--", color="#d63031", lw=1.8, label=r"Density $\hat\rho$")
    twin.set_ylabel(r"Density $\hat\rho$", color="#d63031")
    twin.tick_params(axis="y", labelcolor="#d63031")
    lines = phi_line + density_line
    ax.legend(lines, [line.get_label() for line in lines], loc="upper center")
    ax.text(
        0.50,
        0.06,
        "The sensor does not create a separate φ(r).",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    ax = axes[1]
    ax.plot(radius, atom_acceleration, color="#0984e3", lw=2.7, label=r"atom: $Q_{test}\simeq1$")
    ax.plot(radius, bulk_acceleration, color="#e17055", lw=2.7, label=r"bulk: $Q_{test}=0.03$ (illustrative)")
    ax.axhline(0.0, color="#636e72", lw=0.8)
    ax.axvspan(0.0, SOURCE_RADIUS, color="#636e72", alpha=0.15)
    ax.axvspan(WALL_START, CHAMBER_RADIUS, color="#2d3436", alpha=0.25)
    ax.set(
        xlabel=r"Radius $r/R_{ch}$",
        ylabel=r"Measured response $\hat a_5=-Q_{test}\,d\hat\phi/dr$",
        title="(b) Same gradient, different scalar charge",
    )
    ax.legend(loc="lower right")
    ax.text(
        0.04,
        0.91,
        r"$Q_{test}=\min(3\Delta R/R,1)$",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.text(
        0.04,
        0.76,
        "Actual Q depends on (M, n, β),\nobject size/density, and chamber geometry.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
