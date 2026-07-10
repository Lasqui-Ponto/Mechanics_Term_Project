"""Run the nonlinear thin-shell scan and generate all paper figures.

Usage from the repository root:

    python -m Thin_Shell_Study.run_study --workers 4 --force

The script writes only inside ``Thin_Shell_Study/results`` and
``Thin_Shell_Study/figures``.  A cached scan is reused unless ``--force`` is
given.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Use a headless backend and keep the font cache inside the project.  This is
# required for reproducible execution in CI and other GUI-free environments.
_MODULE_DIR = Path(__file__).resolve().parent
_MPL_CONFIG = _MODULE_DIR / "results" / ".mplconfig"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

from .thin_shell_model import (
    ThinShellPoint,
    force_ratio,
    q_center_point_continuation,
    q_linear_sphere,
    q_matched_shell,
    q_thin_shell,
    q_yukawa_matched,
    solve_charge,
)


ROOT = _MODULE_DIR
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SCAN_FILE = RESULTS / "thin_shell_scan.npz"
CSV_FILE = RESULTS / "thin_shell_scan.csv"
METRICS_FILE = RESULTS / "study_metrics.json"


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.grid": False,
        }
    )


def _solve_payload(payload: tuple[int, int, float, float]) -> dict[str, float | int | str]:
    i, j, density, mu = payload
    point = ThinShellPoint(density, mu)
    attempts = (
        dict(x_max=160.0, tol=2.0e-4, mesh_points=220, max_nodes=40_000),
        dict(x_max=240.0, tol=3.0e-4, mesh_points=300, max_nodes=70_000),
        dict(x_max=320.0, tol=5.0e-4, mesh_points=360, max_nodes=100_000),
    )
    last_error = ""
    solution = None
    accepted = False
    for options in attempts:
        try:
            solution = solve_charge(point, **options)
            if solution.estimator_disagreement <= 0.02:
                accepted = True
                break
            last_error = (
                "estimator disagreement remained above 2%: "
                f"{solution.estimator_disagreement:.6g} at x_max={options['x_max']}"
            )
        except Exception as error:  # captured in the reproducibility table
            last_error = repr(error)
            solution = None
    if solution is None or not accepted:
        return {"i": i, "j": j, "success": 0, "error": last_error}
    return {
        "i": i,
        "j": j,
        "success": 1,
        "q_num": solution.q_num,
        "q_gradient": solution.q_gradient,
        "q_scatter": solution.q_scatter,
        "estimator_disagreement": solution.estimator_disagreement,
        "nodes": solution.result.raw.x.size,
        "formulation": solution.formulation,
        "x_max": solution.result.x_max,
        "error": last_error,
    }


def run_scan(workers: int) -> dict[str, np.ndarray]:
    densities = np.geomspace(1.0e2, 1.0e5, 25)
    mus = np.geomspace(1.0e-3, 3.0e-1, 29)
    shape = (densities.size, mus.size)
    q_num = np.full(shape, np.nan)
    q_gradient = np.full(shape, np.nan)
    q_scatter = np.full(shape, np.nan)
    estimator_disagreement = np.full(shape, np.nan)
    nodes = np.zeros(shape, dtype=int)
    formulations = np.full(shape, "", dtype="U6")
    domains = np.full(shape, np.nan)
    success = np.zeros(shape, dtype=bool)
    errors: list[dict[str, object]] = []

    payloads = [
        (i, j, float(density), float(mu))
        for i, density in enumerate(densities)
        for j, mu in enumerate(mus)
    ]

    if workers == 1:
        iterator = map(_solve_payload, payloads)
        for completed, record in enumerate(iterator, start=1):
            _store_record(
                record,
                q_num,
                q_gradient,
                q_scatter,
                estimator_disagreement,
                nodes,
                formulations,
                domains,
                success,
                errors,
            )
            if completed % 25 == 0 or completed == len(payloads):
                print(f"scan {completed}/{len(payloads)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_solve_payload, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                _store_record(
                    record,
                    q_num,
                    q_gradient,
                    q_scatter,
                    estimator_disagreement,
                    nodes,
                    formulations,
                    domains,
                    success,
                    errors,
                )
                if completed % 25 == 0 or completed == len(payloads):
                    print(f"scan {completed}/{len(payloads)}", flush=True)

    density_grid, mu_grid = np.meshgrid(densities, mus, indexing="ij")
    phi_core = density_grid**-0.5
    b_grid = 0.5 * mu_grid**2
    epsilon = (1.0 - phi_core) / (b_grid * density_grid)
    mu_in = mu_grid * density_grid**0.75
    q_kw = np.minimum(3.0 * epsilon, 1.0)
    q_center = np.exp(-mu_grid) * q_kw
    core_fraction = np.sqrt(np.maximum(1.0 - 2.0 * epsilon, 0.0))
    q_ms = 1.0 - core_fraction**3
    q_yms = np.empty(shape)
    for i, density in enumerate(densities):
        for j, mu in enumerate(mus):
            q_yms[i, j] = q_yukawa_matched(ThinShellPoint(density, mu))

    data = {
        "densities": densities,
        "mus": mus,
        "density_grid": density_grid,
        "mu_grid": mu_grid,
        "epsilon": epsilon,
        "mu_in": mu_in,
        "q_num": q_num,
        "q_gradient": q_gradient,
        "q_scatter": q_scatter,
        "estimator_disagreement": estimator_disagreement,
        "q_kw": q_kw,
        "q_center": q_center,
        "q_ms": q_ms,
        "q_yms": q_yms,
        "nodes": nodes,
        "formulations": formulations,
        "domains": domains,
        "success": success,
    }
    np.savez_compressed(SCAN_FILE, **data)
    _write_scan_csv(data)
    if errors:
        (RESULTS / "scan_failures.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )
    return data


def _store_record(
    record: dict[str, object],
    q_num: np.ndarray,
    q_gradient: np.ndarray,
    q_scatter: np.ndarray,
    estimator_disagreement: np.ndarray,
    nodes: np.ndarray,
    formulations: np.ndarray,
    domains: np.ndarray,
    success: np.ndarray,
    errors: list[dict[str, object]],
) -> None:
    i, j = int(record["i"]), int(record["j"])
    if not record["success"]:
        errors.append(record)
        return
    success[i, j] = True
    q_num[i, j] = float(record["q_num"])
    q_gradient[i, j] = float(record["q_gradient"])
    q_scatter[i, j] = float(record["q_scatter"])
    estimator_disagreement[i, j] = float(record["estimator_disagreement"])
    nodes[i, j] = int(record["nodes"])
    formulations[i, j] = str(record["formulation"])
    domains[i, j] = float(record["x_max"])


def _write_scan_csv(data: dict[str, np.ndarray]) -> None:
    fields = [
        "density_contrast",
        "mu_out",
        "epsilon",
        "mu_in",
        "q_num",
        "q_kw",
        "q_center",
        "q_ms",
        "q_yms",
        "q_scatter",
        "estimator_disagreement",
        "nodes",
        "formulation",
        "x_max",
        "success",
    ]
    with CSV_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, density in enumerate(data["densities"]):
            for j, mu in enumerate(data["mus"]):
                writer.writerow(
                    {
                        "density_contrast": density,
                        "mu_out": mu,
                        "epsilon": data["epsilon"][i, j],
                        "mu_in": data["mu_in"][i, j],
                        "q_num": data["q_num"][i, j],
                        "q_kw": data["q_kw"][i, j],
                        "q_center": data["q_center"][i, j],
                        "q_ms": data["q_ms"][i, j],
                        "q_yms": data["q_yms"][i, j],
                        "q_scatter": data["q_scatter"][i, j],
                        "estimator_disagreement": data["estimator_disagreement"][i, j],
                        "nodes": data["nodes"][i, j],
                        "formulation": data["formulations"][i, j],
                        "x_max": data["domains"][i, j],
                        "success": int(data["success"][i, j]),
                    }
                )


def load_scan() -> dict[str, np.ndarray]:
    with np.load(SCAN_FILE, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def plot_profiles() -> dict[str, object]:
    cases = [
        ("unscreened", ThinShellPoint(1.0e2, 1.0e-1)),
        ("transition", ThinShellPoint(1.0e4, 3.2e-2)),
        ("screened", ThinShellPoint(1.0e5, 1.0e-1)),
    ]
    colors = ("#2b8cbe", "#f39c34", "#b2182b")
    solutions = [
        solve_charge(point, x_max=160.0, tol=3.0e-6, mesh_points=500, max_nodes=100_000)
        for _, point in cases
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x_field = np.concatenate((np.linspace(0.0, 0.98, 300), np.linspace(0.98, 3.0, 500)))
    for (label, point), solution, color in zip(cases, solutions, colors):
        axes[0].plot(x_field, solution.field(x_field), lw=2.2, color=color, label=label)
        x_charge = np.linspace(max(1.03, solution.fit_x_min), solution.fit_x_max, 320)
        amplitude, gradient = solution.local_charge_estimators(x_charge)
        axes[1].plot(x_charge, amplitude, lw=2.0, color=color, label=f"{label}: amplitude")
        axes[1].plot(x_charge, gradient, lw=1.3, ls="--", color=color, alpha=0.8)
        axes[1].axhline(solution.q_num, lw=0.8, color=color, alpha=0.35)

    axes[0].axvline(1.0, color="0.3", lw=1.0, ls=":")
    axes[0].set(xlabel=r"radius $x=r/R$", ylabel=r"field $u=\phi/\phi_\infty$", xlim=(0, 3.0))
    axes[0].legend(frameon=False)
    axes[0].set_title("Full nonlinear spherical solutions")
    axes[1].set(
        xlabel=r"radius $x=r/R$",
        ylabel=r"local scalar charge $Q(x)$",
        yscale="log",
    )
    axes[1].set_title("Exterior Yukawa plateau (solid / gradient dashed)")
    axes[1].legend(frameon=False, fontsize=8)
    for axis, label in zip(axes, ("a", "b")):
        _panel_label(axis, label)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES / "profiles_and_charge_extraction.png")
    plt.close(figure)

    return {
        label: {
            "density_contrast": point.density_contrast,
            "mu_out": point.mu_out,
            "epsilon": point.epsilon,
            "mu_in": point.mu_in,
            "q_num": solution.q_num,
            "q_kw": q_thin_shell(point),
            "q_center": q_center_point_continuation(point),
            "q_yms": q_yukawa_matched(point),
            "estimator_disagreement": solution.estimator_disagreement,
        }
        for (label, point), solution in zip(cases, solutions)
    }


def plot_accuracy_atlas(data: dict[str, np.ndarray]) -> None:
    density = data["density_grid"]
    mu = data["mu_grid"]
    q_num = data["q_num"]
    approximations = (
        (data["q_kw"], r"surface-shell continuation $Q_{\rm surf}$"),
        (data["q_center"], r"center-point continuation $Q_{\rm ctr}$"),
        (data["q_yms"], r"active-shell integral $Q_{\rm AS}$"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.15), sharex=True, sharey=True)
    normalization = TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=0.35)
    image = None
    for axis, (approximation, title), label in zip(axes, approximations, ("a", "b", "c")):
        signed = np.log10(approximation / q_num)
        image = axis.pcolormesh(mu, density, signed, shading="auto", cmap="RdBu_r", norm=normalization)
        relative = np.abs(approximation / q_num - 1.0)
        axis.contour(mu, density, relative, levels=[0.1], colors="black", linewidths=1.5)
        axis.contour(
            mu,
            density,
            data["epsilon"],
            levels=[0.01, 0.1, 0.5],
            colors=["#fdd49e", "#fc8d59", "#b30000"],
            linewidths=0.9,
            linestyles="--",
        )
        axis.set(xscale="log", yscale="log", xlabel=r"$\mu=m_\infty R$", title=title)
        axis.grid(alpha=0.15, which="both")
        _panel_label(axis, label)
    axes[0].set_ylabel(r"density contrast $\delta=\rho_c/\rho_\infty$")
    color_axis = figure.add_axes([0.915, 0.18, 0.015, 0.68])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(r"signed error $\log_{10}(Q_{\rm approx}/Q_{\rm num})$")
    figure.text(0.5, -0.01, "Black: 10% error boundary; dashed: epsilon = 0.01, 0.1, 0.5", ha="center")
    figure.subplots_adjust(left=0.075, right=0.895, bottom=0.18, top=0.9, wspace=0.08)
    figure.savefig(FIGURES / "accuracy_atlas.png")
    plt.close(figure)


def plot_charge_comparison(data: dict[str, np.ndarray]) -> None:
    epsilon = data["epsilon"].ravel()
    q_num = data["q_num"].ravel()
    approximations = (
        (data["q_kw"].ravel(), "surface continuation", "#b2182b"),
        (data["q_center"].ravel(), "center continuation", "#762a83"),
        (data["q_ms"].ravel(), "finite shell only", "#ef8a62"),
        (data["q_yms"].ravel(), "active-shell integral", "#2166ac"),
    )
    bins = np.geomspace(max(epsilon.min(), 1.0e-5), epsilon.max(), 17)
    centers = np.sqrt(bins[:-1] * bins[1:])

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    sample = np.arange(epsilon.size)
    axes[0].scatter(q_num[sample], data["q_kw"].ravel()[sample], s=10, alpha=0.28, color="#b2182b", label="surface continuation")
    axes[0].scatter(q_num[sample], data["q_center"].ravel()[sample], s=10, alpha=0.28, color="#762a83", label="center continuation")
    axes[0].scatter(q_num[sample], data["q_yms"].ravel()[sample], s=10, alpha=0.35, color="#2166ac", label="active-shell integral")
    identity = np.geomspace(q_num.min(), 1.1, 200)
    axes[0].plot(identity, identity, color="black", lw=1.2, ls="--")
    axes[0].set(xscale="log", yscale="log", xlabel=r"numerical charge $Q_{\rm num}$", ylabel=r"predicted charge", title="Pointwise prediction")
    axes[0].legend(frameon=False)

    for approximation, label, color in approximations:
        relative = np.abs(approximation / q_num - 1.0)
        median = np.full_like(centers, np.nan)
        p90 = np.full_like(centers, np.nan)
        for index in range(centers.size):
            mask = (epsilon >= bins[index]) & (epsilon < bins[index + 1])
            if np.any(mask):
                median[index] = np.median(relative[mask])
                p90[index] = np.quantile(relative[mask], 0.9)
        axes[1].plot(centers, median, color=color, lw=2.2, label=label)
        axes[1].fill_between(centers, median, p90, color=color, alpha=0.14)
    axes[1].axhline(0.1, color="black", ls="--", lw=1.1, label="10% target")
    axes[1].set(xscale="log", yscale="log", xlabel=r"thin-shell parameter $\epsilon=\Delta R/R$", ylabel="relative charge error", title="Median and 90th-percentile error")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    for axis, label in zip(axes, ("a", "b")):
        axis.grid(alpha=0.2, which="both")
        _panel_label(axis, label)
    figure.tight_layout()
    figure.savefig(FIGURES / "charge_comparison.png")
    plt.close(figure)


def _force_map(q_source: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    mu = data["mu_grid"]
    x_probe = 1.5
    yukawa = (1.0 + mu * x_probe) * np.exp(-mu * (x_probe - 1.0))
    return 2.0 * q_source * yukawa


def plot_detectability(data: dict[str, np.ndarray], threshold: float) -> dict[str, float | int]:
    density = data["density_grid"]
    mu = data["mu_grid"]
    alpha_num = _force_map(data["q_num"], data)
    alpha_surface = _force_map(data["q_kw"], data)
    alpha_center = _force_map(data["q_center"], data)
    detect_num = alpha_num >= threshold
    detect_surface = alpha_surface >= threshold
    detect_center = alpha_center >= threshold
    continuation_ambiguity = detect_surface != detect_center
    robust_mismatch = (~continuation_ambiguity) & (detect_num != detect_surface)
    classification = np.zeros(alpha_num.shape, dtype=int)
    classification[(~continuation_ambiguity) & detect_num & detect_surface] = 1
    classification[continuation_ambiguity] = 2
    classification[robust_mismatch] = 3

    normalized_bias = np.maximum(
        np.abs(alpha_surface - alpha_num), np.abs(alpha_center - alpha_num)
    ) / threshold

    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.25), sharex=True, sharey=True)
    image0 = axes[0].pcolormesh(mu, density, np.log10(alpha_num), shading="auto", cmap="viridis")
    axes[0].contour(mu, density, alpha_num, levels=[threshold], colors="white", linewidths=2.0)
    axes[0].contour(mu, density, alpha_surface, levels=[threshold], colors="#ffcc33", linewidths=1.8, linestyles="--")
    axes[0].contour(mu, density, alpha_center, levels=[threshold], colors="#66c2ff", linewidths=1.8, linestyles=":")
    axes[0].plot([], [], color="white", lw=2.0, label="numerical threshold")
    axes[0].plot([], [], color="#ffcc33", lw=1.8, ls="--", label="surface continuation")
    axes[0].plot([], [], color="#66c2ff", lw=1.8, ls=":", label="center continuation")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7.5)
    cbar0 = figure.colorbar(image0, ax=axes[0], pad=0.02)
    cbar0.set_label(r"$\log_{10}|a_5/a_N|$ (unscreened probe)")

    class_colors = ListedColormap(["#d9d9d9", "#41ab5d", "#ef8a62", "#67a9cf"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], class_colors.N)
    image1 = axes[1].pcolormesh(mu, density, classification, shading="auto", cmap=class_colors, norm=norm)
    cbar1 = figure.colorbar(image1, ax=axes[1], ticks=[0, 1, 2, 3], pad=0.02)
    cbar1.ax.set_yticklabels(["all below", "all above", "continuations differ", "both miss"])

    image2 = axes[2].pcolormesh(
        mu,
        density,
        np.log10(np.maximum(normalized_bias, 1.0e-5)),
        shading="auto",
        cmap="magma",
    )
    axes[2].contour(mu, density, normalized_bias, levels=[1.0], colors="white", linewidths=1.5)
    cbar2 = figure.colorbar(image2, ax=axes[2], pad=0.02)
    cbar2.set_label(r"$\log_{10}(\max|\alpha_{\rm naive}-\alpha_{\rm num}|/\alpha_\star)$")

    titles = (
        "Dimensionless force-ratio proxy",
        "Continuation ambiguity at one threshold",
        "Naive-continuation error in threshold units",
    )
    for axis, title, label in zip(axes, titles, ("a", "b", "c")):
        axis.set(xscale="log", yscale="log", xlabel=r"$\mu=m_\infty R$", title=title)
        axis.grid(alpha=0.12, which="both")
        _panel_label(axis, label)
    axes[0].set_ylabel(r"density contrast $\delta=\rho_c/\rho_\infty$")
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.15, top=0.9, wspace=0.35)
    figure.savefig(FIGURES / "detectability_map.png")
    plt.close(figure)

    surface_shifts: list[float] = []
    center_shifts: list[float] = []
    log_mu = np.log10(data["mus"])
    log_threshold = np.log10(threshold)

    def crossing(row: np.ndarray) -> float | None:
        log_row = np.log10(row)
        for index in range(row.size - 1):
            if (row[index] - threshold) * (row[index + 1] - threshold) <= 0.0:
                if log_row[index] == log_row[index + 1]:
                    return float(log_mu[index])
                fraction = (log_threshold - log_row[index]) / (
                    log_row[index + 1] - log_row[index]
                )
                return float(log_mu[index] + fraction * (log_mu[index + 1] - log_mu[index]))
        return None

    for row_num, row_surface, row_center in zip(alpha_num, alpha_surface, alpha_center):
        crossing_num = crossing(row_num)
        crossing_surface = crossing(row_surface)
        crossing_center = crossing(row_center)
        if crossing_num is not None and crossing_surface is not None:
            surface_shifts.append(abs(crossing_surface - crossing_num))
        if crossing_num is not None and crossing_center is not None:
            center_shifts.append(abs(crossing_center - crossing_num))
    return {
        "threshold": threshold,
        "surface_mismatch_points": int(np.count_nonzero(detect_num != detect_surface)),
        "center_mismatch_points": int(np.count_nonzero(detect_num != detect_center)),
        "continuation_ambiguity_points": int(np.count_nonzero(continuation_ambiguity)),
        "robust_mismatch_points": int(np.count_nonzero(robust_mismatch)),
        "surface_rows_with_both_contours": len(surface_shifts),
        "surface_median_contour_shift_dex": float(np.median(surface_shifts)),
        "surface_max_contour_shift_dex": float(np.max(surface_shifts)),
        "center_rows_with_both_contours": len(center_shifts),
        "center_median_contour_shift_dex": float(np.median(center_shifts)),
        "center_max_contour_shift_dex": float(np.max(center_shifts)),
    }


def plot_verification() -> dict[str, object]:
    weak_mus = np.array([0.01, 0.03, 0.1, 0.3])
    weak_density = 1.01
    weak_numeric = []
    for mu in weak_mus:
        solution = solve_charge(
            ThinShellPoint(weak_density, float(mu)),
            x_max=160.0,
            tol=2.0e-7,
            mesh_points=450,
            max_nodes=100_000,
        )
        weak_numeric.append(solution.q_num)
    weak_numeric = np.asarray(weak_numeric)
    weak_analytic = (weak_density - 1.0) / weak_density * q_linear_sphere(weak_mus)

    benchmark = ThinShellPoint(1.0e5, 1.0e-2)
    domains = np.array([40.0, 80.0, 160.0, 240.0])
    domain_charge = np.array(
        [
            solve_charge(
                benchmark,
                x_max=float(domain),
                tol=2.0e-6,
                mesh_points=500,
                max_nodes=100_000,
            ).q_num
            for domain in domains
        ]
    )
    tolerances = np.array([3.0e-4, 3.0e-5, 3.0e-6])
    tolerance_charge = np.array(
        [
            solve_charge(
                benchmark,
                x_max=240.0,
                tol=float(tolerance),
                mesh_points=500,
                max_nodes=100_000,
            ).q_num
            for tolerance in tolerances
        ]
    )

    plateau_solution = solve_charge(
        benchmark, x_max=240.0, tol=2.0e-6, mesh_points=520, max_nodes=100_000
    )
    plateau_x = np.linspace(20.0, 239.0, 600)
    plateau_amp, plateau_grad = plateau_solution.local_charge_estimators(plateau_x)

    representation_point = ThinShellPoint(1.0e4, 0.063)
    representation_linear = solve_charge(
        representation_point,
        x_max=160.0,
        tol=2.0e-6,
        mesh_points=420,
        max_nodes=100_000,
        formulation="linear",
    ).q_num
    representation_log = solve_charge(
        representation_point,
        x_max=160.0,
        tol=2.0e-6,
        mesh_points=420,
        max_nodes=100_000,
        formulation="log",
    ).q_num

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.95))
    axes[0].plot(weak_mus, weak_analytic, color="black", lw=1.5, label="analytic uniform sphere")
    axes[0].scatter(weak_mus, weak_numeric, color="#2b8cbe", s=38, zorder=3, label="nonlinear BVP")
    axes[0].set(xscale="log", xlabel=r"$\mu=m_\infty R$", ylabel=r"unscreened charge $Q$", title="Analytic-limit benchmark")
    axes[0].legend(frameon=False)

    reference = domain_charge[-1]
    category_x = np.array([0.0, 1.0, 2.0])
    axes[1].plot(category_x, np.abs(domain_charge[:3] / reference - 1.0), marker="o", lw=1.8, label=r"domain: $x_{\max}=40,80,160$")
    axes[1].plot(
        category_x[:2],
        np.abs(tolerance_charge[:2] / tolerance_charge[-1] - 1.0),
        marker="s",
        lw=1.8,
        label=r"tolerance: $3\times10^{-4},3\times10^{-5}$",
    )
    axes[1].set(yscale="log", xlabel="resolution level", ylabel="relative change from finest $Q$", title="Numerical convergence")
    axes[1].set_xticks(category_x, ["coarse", "medium", "fine"])
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(plateau_x, plateau_amp, lw=1.8, label="amplitude estimator")
    axes[2].plot(plateau_x, plateau_grad, lw=1.5, ls="--", label="gradient estimator")
    axes[2].axhline(plateau_solution.q_num, color="black", lw=0.9, ls=":")
    axes[2].set(xlabel=r"radius $x=r/R$", ylabel=r"$Q(x)$", title="Independent charge extraction")
    axes[2].legend(frameon=False, fontsize=8)
    for axis, label in zip(axes, ("a", "b", "c")):
        axis.grid(alpha=0.2, which="both")
        _panel_label(axis, label)
    figure.tight_layout()
    figure.savefig(FIGURES / "verification.png")
    plt.close(figure)

    return {
        "weak_source_max_relative_error": float(
            np.max(np.abs(weak_numeric / weak_analytic - 1.0))
        ),
        "domain_relative_change_40_to_240": float(abs(domain_charge[0] / domain_charge[-1] - 1.0)),
        "domain_relative_change_80_to_240": float(abs(domain_charge[1] / domain_charge[-1] - 1.0)),
        "domain_relative_change_160_to_240": float(abs(domain_charge[2] / domain_charge[-1] - 1.0)),
        "tolerance_relative_change_coarse_to_fine": float(
            abs(tolerance_charge[0] / tolerance_charge[-1] - 1.0)
        ),
        "plateau_estimator_disagreement": float(plateau_solution.estimator_disagreement),
        "linear_log_representation_disagreement": float(
            abs(representation_linear / representation_log - 1.0)
        ),
        "weak_mus": weak_mus.tolist(),
        "weak_numeric": weak_numeric.tolist(),
        "weak_analytic": weak_analytic.tolist(),
        "domains": domains.tolist(),
        "domain_charge": domain_charge.tolist(),
        "tolerances": tolerances.tolist(),
        "tolerance_charge": tolerance_charge.tolist(),
    }


def run_stratified_domain_audit(data: dict[str, np.ndarray]) -> dict[str, object]:
    """Re-solve high-leverage atlas points on a longer exterior domain.

    The selection is deliberately deterministic: the four parameter-space
    corners, the transition-profile and convergence-benchmark points, and the
    four atlas points nearest the 10% error boundary for both the surface-shell
    and active-shell predictions.  This does not certify every grid point, but
    it directly tests the corners and the classification boundary on which the
    headline 10% statements depend.
    """

    densities = data["densities"]
    mus = data["mus"]
    selected: dict[tuple[int, int], set[str]] = {}

    def add(i: int, j: int, reason: str) -> None:
        selected.setdefault((int(i), int(j)), set()).add(reason)

    for i, j in ((0, 0), (0, mus.size - 1), (densities.size - 1, 0),
                 (densities.size - 1, mus.size - 1)):
        add(i, j, "parameter-space corner")

    def nearest_index(density: float, mu: float) -> tuple[int, int]:
        i = int(np.argmin(np.abs(np.log(densities / density))))
        j = int(np.argmin(np.abs(np.log(mus / mu))))
        return i, j

    add(*nearest_index(1.0e4, 3.2e-2), "transition profile")
    add(*nearest_index(1.0e5, 1.0e-2), "convergence benchmark")

    for label, approximation in (
        ("surface 10% boundary", data["q_kw"]),
        ("active-shell 10% boundary", data["q_yms"]),
    ):
        relative_error = np.abs(approximation / data["q_num"] - 1.0)
        order = np.argsort(np.abs(relative_error.ravel() - 0.1))
        for flat_index in order[:4]:
            add(*np.unravel_index(int(flat_index), relative_error.shape), label)

    rows: list[dict[str, object]] = []
    for (i, j), reasons in sorted(selected.items()):
        point = ThinShellPoint(float(densities[i]), float(mus[j]))
        solution_160 = solve_charge(
            point,
            x_max=160.0,
            tol=2.0e-6,
            mesh_points=480,
            max_nodes=120_000,
        )
        solution_240 = solve_charge(
            point,
            x_max=240.0,
            tol=2.0e-6,
            mesh_points=560,
            max_nodes=140_000,
        )
        approximations = {
            "surface": q_thin_shell(point),
            "center": q_center_point_continuation(point),
            "active_shell": q_yukawa_matched(point),
        }
        row: dict[str, object] = {
            "density_contrast": point.density_contrast,
            "mu_out": point.mu_out,
            "selection_reason": "; ".join(sorted(reasons)),
            "q_160": solution_160.q_num,
            "q_240": solution_240.q_num,
            "relative_domain_change": abs(solution_160.q_num / solution_240.q_num - 1.0),
            "estimator_disagreement_160": solution_160.estimator_disagreement,
            "estimator_disagreement_240": solution_240.estimator_disagreement,
        }
        for name, approximation in approximations.items():
            error_160 = abs(approximation / solution_160.q_num - 1.0)
            error_240 = abs(approximation / solution_240.q_num - 1.0)
            row[f"{name}_error_160"] = error_160
            row[f"{name}_error_240"] = error_240
            row[f"{name}_10_percent_classification_changed"] = int(
                (error_160 < 0.1) != (error_240 < 0.1)
            )
        rows.append(row)

    path = RESULTS / "stratified_domain_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    domain_changes = np.array([float(row["relative_domain_change"]) for row in rows])
    return {
        "selection": (
            "four grid corners, transition profile, convergence benchmark, "
            "and four points nearest each of the surface and active-shell 10% boundaries"
        ),
        "audited_points": len(rows),
        "median_relative_domain_change": float(np.median(domain_changes)),
        "max_relative_domain_change": float(np.max(domain_changes)),
        "surface_10_percent_classification_changes": int(
            sum(int(row["surface_10_percent_classification_changed"]) for row in rows)
        ),
        "center_10_percent_classification_changes": int(
            sum(int(row["center_10_percent_classification_changed"]) for row in rows)
        ),
        "active_shell_10_percent_classification_changes": int(
            sum(int(row["active_shell_10_percent_classification_changed"]) for row in rows)
        ),
        "output_csv": "results/stratified_domain_audit.csv",
    }


def _approximation_metrics(approximation: np.ndarray, numerical: np.ndarray) -> dict[str, float]:
    relative = np.abs(approximation / numerical - 1.0)
    return {
        "median_relative_error": float(np.median(relative)),
        "p90_relative_error": float(np.quantile(relative, 0.9)),
        "max_relative_error": float(np.max(relative)),
        "fraction_within_10_percent": float(np.mean(relative < 0.1)),
        "fraction_within_20_percent": float(np.mean(relative < 0.2)),
    }


def validate_simple_gate(data: dict[str, np.ndarray]) -> dict[str, object]:
    """Test the stated validity filter on log-midpoints excluded from the atlas."""

    density_midpoints = np.sqrt(data["densities"][:-1] * data["densities"][1:])
    mu_midpoints = np.sqrt(data["mus"][:-1] * data["mus"][1:])
    rows: list[dict[str, float]] = []
    for density in density_midpoints:
        for mu in mu_midpoints:
            point = ThinShellPoint(float(density), float(mu))
            if not (point.epsilon < 0.1 and point.mu_in > 10.0 and point.mu_out < 0.035):
                continue
            solution = solve_charge(
                point,
                x_max=160.0,
                tol=1.0e-4,
                mesh_points=240,
                max_nodes=50_000,
            )
            relative_error = abs(q_thin_shell(point) / solution.q_num - 1.0)
            rows.append(
                {
                    "density_contrast": point.density_contrast,
                    "mu_out": point.mu_out,
                    "epsilon": point.epsilon,
                    "mu_in": point.mu_in,
                    "q_num": solution.q_num,
                    "q_surface": q_thin_shell(point),
                    "relative_error": relative_error,
                }
            )
    path = RESULTS / "gate_validation_midpoints.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = np.array([row["relative_error"] for row in rows])
    return {
        "conditions": "epsilon < 0.1, mu_in > 10, mu_out < 0.035",
        "held_out_midpoint_count": len(rows),
        "fraction_within_10_percent": float(np.mean(errors < 0.1)),
        "max_relative_error": float(np.max(errors)),
        "median_relative_error": float(np.median(errors)),
    }


def compute_metrics(
    data: dict[str, np.ndarray],
    profile_metrics: dict[str, object],
    verification: dict[str, object],
    detectability: dict[str, float | int],
    gate_validation: dict[str, object],
) -> dict[str, object]:
    valid = data["success"] & np.isfinite(data["q_num"])
    numerical = data["q_num"][valid]
    metrics: dict[str, object] = {
        "grid": {
            "density_points": int(data["densities"].size),
            "mu_points": int(data["mus"].size),
            "total_points": int(valid.size),
            "successful_points": int(np.count_nonzero(valid)),
            "success_fraction": float(np.mean(valid)),
            "density_range": [float(data["densities"].min()), float(data["densities"].max())],
            "mu_range": [float(data["mus"].min()), float(data["mus"].max())],
            "max_estimator_disagreement": float(
                np.nanmax(data["estimator_disagreement"])
            ),
        },
        "surface_shell_continuation": _approximation_metrics(data["q_kw"][valid], numerical),
        "center_point_continuation": _approximation_metrics(data["q_center"][valid], numerical),
        "finite_shell_only": _approximation_metrics(data["q_ms"][valid], numerical),
        "active_shell_integral": _approximation_metrics(data["q_yms"][valid], numerical),
        "profiles": profile_metrics,
        "verification": verification,
        "detectability": detectability,
    }

    asymptotic = valid & (data["epsilon"] < 0.1) & (data["mu_in"] > 10.0)
    metrics["surface_continuation_asymptotic"] = _approximation_metrics(
        data["q_kw"][asymptotic], data["q_num"][asymptotic]
    )
    metrics["asymptotic_points"] = int(np.count_nonzero(asymptotic))

    candidates_epsilon = np.geomspace(0.003, 0.5, 30)
    candidates_mu_in = np.geomspace(1.0, 100.0, 25)
    candidates_mu_out = np.geomspace(0.01, 0.3, 20)
    best: dict[str, float] | None = None
    accurate = np.abs(data["q_kw"] / data["q_num"] - 1.0) < 0.1
    for epsilon_max in candidates_epsilon:
        for mu_in_min in candidates_mu_in:
            for mu_out_max in candidates_mu_out:
                gate = (
                    valid
                    & (data["epsilon"] <= epsilon_max)
                    & (data["mu_in"] >= mu_in_min)
                    & (data["mu_grid"] <= mu_out_max)
                )
                count = int(np.count_nonzero(gate))
                if count < 20:
                    continue
                precision = float(np.mean(accurate[gate]))
                coverage = float(count / np.count_nonzero(valid))
                if precision >= 0.9 and (best is None or coverage > best["coverage"]):
                    best = {
                        "epsilon_max": float(epsilon_max),
                        "mu_in_min": float(mu_in_min),
                        "mu_out_max": float(mu_out_max),
                        "precision": precision,
                        "coverage": coverage,
                        "points": count,
                    }
    metrics["empirical_10_percent_gate"] = best
    simple_gate = (
        valid
        & (data["epsilon"] < 0.1)
        & (data["mu_in"] > 10.0)
        & (data["mu_grid"] < 0.035)
    )
    simple_error = np.abs(data["q_kw"][simple_gate] / data["q_num"][simple_gate] - 1.0)
    metrics["simple_sufficient_gate"] = {
        "conditions": "epsilon < 0.1, mu_in > 10, mu_out < 0.035",
        "points": int(np.count_nonzero(simple_gate)),
        "fraction_within_10_percent": float(np.mean(simple_error < 0.1)),
        "max_relative_error": float(np.max(simple_error)),
        "median_relative_error": float(np.median(simple_error)),
        "held_out_validation": gate_validation,
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold", type=float, default=1.0e-2)
    arguments = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    _configure_plotting()

    if arguments.force or not SCAN_FILE.exists():
        data = run_scan(max(1, arguments.workers))
    else:
        data = load_scan()
        _write_scan_csv(data)
    if not np.all(data["success"]):
        failures = int(np.count_nonzero(~data["success"]))
        raise RuntimeError(f"parameter scan contains {failures} failed points")

    profiles = plot_profiles()
    plot_accuracy_atlas(data)
    plot_charge_comparison(data)
    detectability = plot_detectability(data, arguments.threshold)
    verification = plot_verification()
    verification["stratified_domain_audit"] = run_stratified_domain_audit(data)
    gate_validation = validate_simple_gate(data)
    metrics = compute_metrics(data, profiles, verification, detectability, gate_validation)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
