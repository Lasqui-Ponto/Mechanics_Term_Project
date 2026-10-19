"""Reproduce the spherical charge/local-force study without hand-edited numbers.

    python -m Thin_Shell_Study.run_study --workers 4 --force

The versioned scan contains nonlinear force gradients at every probe radius.
Stale charge-only caches are rejected and recomputed.  English and Korean
figures share the same numerical inputs and threshold.  PDF tables and numeric
macros are generated from study_metrics.json in the same run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / ".mplconfig"))

import matplotlib
import numpy as np
import scipy

from . import thin_shell_model
from .study_analysis import (APPROXIMATIONS, PROBE_RADII, distance_summary,
                             probe_index, relative_error_metrics,
                             sampled_validity_radii, tail_force, threshold_summary)
from .thin_shell_model import (ThinShellPoint, q_center_point_continuation,
                               q_linear_sphere, q_matched_shell,
                               q_point_matched_shell, q_thin_shell,
                               q_yukawa_matched, solve_charge)

SCAN_FILE = RESULTS / "thin_shell_scan.npz"
METRICS_FILE = RESULTS / "study_metrics.json"
REFERENCE_FILE = RESULTS / "reference_profiles.npz"
SCHEMA_VERSION = 3
SCAN_ATTEMPTS = (
    dict(x_max=160.0, tol=2e-4, mesh_points=220, max_nodes=40_000),
    dict(x_max=240.0, tol=3e-4, mesh_points=300, max_nodes=70_000),
    dict(x_max=320.0, tol=5e-4, mesh_points=360, max_nodes=100_000),
)


class StaleScanError(ValueError):
    """The cached data do not describe the current numerical experiment."""


def _solve_payload(payload: tuple[int, int, float, float]) -> dict:
    i, j, density, mu = payload
    point = ThinShellPoint(density, mu)
    last_error = ""
    for options in SCAN_ATTEMPTS:
        try:
            solution = solve_charge(point, **options)
            if solution.estimator_disagreement > .02:
                last_error = f"estimator disagreement above 2%: {solution.estimator_disagreement:.6g}"
                continue
            local = solution.direct_force_ratio(PROBE_RADII)
            if not np.all(np.isfinite(local) & (local > 0)):
                raise RuntimeError("local force is not positive and finite")
            return {
                "i": i, "j": j, "success": 1, "q_num": solution.q_num,
                "q_gradient": solution.q_gradient, "q_scatter": solution.q_scatter,
                "estimator_disagreement": solution.estimator_disagreement,
                "nodes": solution.result.raw.x.size, "formulation": solution.formulation,
                "x_max": solution.result.x_max, "alpha_direct": local,
                "fit_x_min": solution.fit_x_min, "fit_x_max": solution.fit_x_max,
            }
        except Exception as error:
            last_error = repr(error)
    return {"i": i, "j": j, "success": 0, "error": last_error}


def scan_fingerprint() -> str:
    """Hash the model, worker algorithm, and sampling/solver settings."""

    settings = {"schema": SCHEMA_VERSION, "densities": [100., 1e5, 25],
                "mus": [.001, .3, 29], "radii": PROBE_RADII.tolist(),
                "attempts": SCAN_ATTEMPTS,
                "numerical_runtime": {"python": platform.python_version(),
                                      "numpy": np.__version__, "scipy": scipy.__version__}}
    # File content, rather than machine-specific absolute paths, determines data.
    digest = hashlib.sha256(Path(thin_shell_model.__file__).read_text(encoding="utf-8").encode())
    digest.update(inspect.getsource(_solve_payload).encode())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def _approximation_grids(data: dict) -> None:
    density, mu = data["density_grid"], data["mu_grid"]
    epsilon = 2 * (1 - density**-.5) / (mu**2 * density)
    q_core = np.sqrt(np.maximum(1 - 2 * epsilon, 0))
    data.update(epsilon=epsilon, mu_in=mu * density**.75,
                q_kw=np.minimum(3 * epsilon, 1), q_ms=1 - q_core**3)
    data["q_center"] = np.exp(-mu) * data["q_kw"]
    data["q_point_ms"] = np.exp(-mu) * data["q_ms"]
    data["q_yms"] = np.asarray([
        [q_yukawa_matched(ThinShellPoint(float(d), float(m))) for m in data["mus"]]
        for d in data["densities"]
    ])


def validate_scan(data: dict) -> None:
    required = {"schema_version", "fingerprint", "probe_radii", "alpha_direct",
                "q_num", "success", "density_grid", "mu_grid", "densities", "mus"}
    if not required.issubset(data):
        raise StaleScanError("scan lacks the nonlinear local-force data or provenance")
    if int(data["schema_version"]) != SCHEMA_VERSION or str(data["fingerprint"]) != scan_fingerprint():
        raise StaleScanError("scan schema, model, or solver settings have changed")
    shape = (data["densities"].size, data["mus"].size)
    if data["alpha_direct"].shape != shape + (PROBE_RADII.size,):
        raise StaleScanError("local-force array has the wrong shape")
    if not np.array_equal(data["probe_radii"], PROBE_RADII):
        raise StaleScanError("probe-radius samples have changed")
    if not np.all(data["success"]) or not np.all(np.isfinite(data["alpha_direct"]) & (data["alpha_direct"] > 0)):
        raise StaleScanError("scan contains failed or nonphysical local-force samples")


def run_scan(workers: int = 1) -> dict:
    densities = np.geomspace(100., 1e5, 25)
    mus = np.geomspace(.001, .3, 29)
    shape = (densities.size, mus.size)
    density_grid, mu_grid = np.meshgrid(densities, mus, indexing="ij")
    data = {"densities": densities, "mus": mus, "density_grid": density_grid,
            "mu_grid": mu_grid, "probe_radii": PROBE_RADII,
            "schema_version": np.asarray(SCHEMA_VERSION),
            "fingerprint": np.asarray(scan_fingerprint()),
            "success": np.zeros(shape, dtype=bool),
            "nodes": np.zeros(shape, dtype=int),
            "formulations": np.full(shape, "", dtype="U6"),
            "alpha_direct": np.full(shape + (PROBE_RADII.size,), np.nan)}
    for name in ("q_num", "q_gradient", "q_scatter", "estimator_disagreement",
                 "domains", "fit_x_min", "fit_x_max"):
        data[name] = np.full(shape, np.nan)
    payloads = [(i, j, float(d), float(m)) for i, d in enumerate(densities)
                for j, m in enumerate(mus)]
    failures = []

    def store(record):
        i, j = record["i"], record["j"]
        if not record["success"]:
            failures.append(record)
            return
        for key in ("q_num", "q_gradient", "q_scatter", "estimator_disagreement",
                    "nodes", "alpha_direct", "fit_x_min", "fit_x_max", "success"):
            data[key][i, j] = record[key]
        data["domains"][i, j] = record["x_max"]
        data["formulations"][i, j] = record["formulation"]

    if workers == 1:
        for n, record in enumerate(map(_solve_payload, payloads), 1):
            store(record)
            if n % 50 == 0 or n == len(payloads):
                print(f"scan {n}/{len(payloads)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_solve_payload, p) for p in payloads]
            for n, future in enumerate(as_completed(futures), 1):
                store(future.result())
                if n % 50 == 0 or n == len(payloads):
                    print(f"scan {n}/{len(payloads)}", flush=True)
    if failures:
        (RESULTS / "scan_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} scan points failed; previous cache was not replaced")
    _approximation_grids(data)
    validate_scan(data)
    np.savez_compressed(SCAN_FILE, **data)
    return data


def load_scan() -> dict:
    with np.load(SCAN_FILE, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_scan(data)
    # Predictions are cheap and are always derived from the current formulas.
    _approximation_grids(data)
    return data


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("a CSV export requires at least one row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_scan_tables(data: dict, metrics: dict) -> None:
    validity = sampled_validity_radii(data)
    direct = data["alpha_direct"][..., probe_index(data)]
    tail = tail_force(data["q_num"], data)
    rows, local_rows = [], []
    threshold = metrics["detectability"]["threshold"]
    for i, density in enumerate(data["densities"]):
        for j, mu in enumerate(data["mus"]):
            row = {"density_contrast": density, "mu_out": mu,
                   "epsilon_th": data["epsilon"][i, j], "mu_in": data["mu_in"][i, j]}
            for key in ("q_num", "q_gradient", "q_kw", "q_center", "q_ms", "q_point_ms",
                        "q_yms", "q_scatter", "estimator_disagreement", "nodes"):
                row[key] = data[key][i, j]
            row.update(formulation=data["formulations"][i, j], x_max=data["domains"][i, j], success=1)
            rows.append(row)
            local_rows.append({"density_contrast": density, "mu_out": mu,
                               "alpha_direct_1p5R": direct[i, j],
                               "alpha_tail_1p5R": tail[i, j],
                               "relative_tail_force_error": abs(tail[i, j] / direct[i, j] - 1),
                               "threshold": threshold, "direct_above": int(direct[i, j] >= threshold),
                               "tail_above": int(tail[i, j] >= threshold),
                               "first_sampled_10_percent_radius": validity[i, j]})
    _write_csv(RESULTS / "thin_shell_scan.csv", rows)
    _write_csv(RESULTS / "local_force_scan.csv", local_rows)
    _write_csv(RESULTS / "force_distance_summary.csv", metrics["distance_validity"]["by_radius"])
    _write_csv(RESULTS / "threshold_sensitivity.csv", [
        {"threshold": item["threshold"],
         "tail_mismatch_points": item["models"]["numerical_charge_tail"]["mismatch_points"],
         "surface_mismatch_points": item["models"]["surface_amplitude"]["mismatch_points"],
         "center_mismatch_points": item["models"]["center_point"]["mismatch_points"],
         "both_simple_ansatzes_wrong_points": item["both_simple_ansatzes_wrong_points"]}
        for item in metrics["threshold_sensitivity"]
    ])


def compute_reference_profiles() -> tuple[dict, dict]:
    references, profiles = {}, {}
    x_field = np.unique(np.concatenate((np.linspace(0, .98, 300), np.linspace(.98, 3, 500))))
    references["profile_x"] = x_field
    for label, density, mu in (("unscreened", 100., .1), ("transition", 1e4, .032), ("screened", 1e5, .1)):
        point = ThinShellPoint(density, mu)
        sol = solve_charge(point, x_max=160, tol=3e-6, mesh_points=500, max_nodes=100_000)
        x_charge = np.linspace(max(1.03, sol.fit_x_min), sol.fit_x_max, 320)
        a, g = sol.local_charge_estimators(x_charge)
        references[f"{label}_u"] = sol.field(x_field)
        references[f"{label}_charge_x"] = x_charge
        references[f"{label}_charge_A"] = a
        references[f"{label}_charge_G"] = g
        profiles[label] = {"density_contrast": density, "mu_out": mu, "epsilon_th": point.epsilon,
                           "mu_in": point.mu_in, "q_num": sol.q_num,
                           "alpha_direct_1p5R": float(sol.direct_force_ratio()),
                           "estimator_disagreement": sol.estimator_disagreement}
    return references, profiles


def compute_verification(references: dict) -> dict:
    weak_mus = np.array([.01, .03, .1, .3])
    weak_numeric = np.asarray([solve_charge(ThinShellPoint(1.01, float(mu)), x_max=160,
                                            tol=2e-7, mesh_points=450, max_nodes=100_000).q_num
                               for mu in weak_mus])
    weak_analytic = (.01 / 1.01) * q_linear_sphere(weak_mus)
    benchmark = ThinShellPoint(1e5, .01)
    domains = np.array([40., 80., 160., 240.])
    domain_charge = np.asarray([solve_charge(benchmark, x_max=float(L), tol=2e-6,
                                             mesh_points=500, max_nodes=100_000).q_num for L in domains])
    tolerances = np.array([3e-4, 3e-5, 3e-6])
    tolerance_charge = np.asarray([solve_charge(benchmark, x_max=240, tol=float(tol),
                                                mesh_points=500, max_nodes=100_000).q_num for tol in tolerances])
    plateau = solve_charge(benchmark, x_max=240, tol=2e-6, mesh_points=520, max_nodes=100_000)
    references["plateau_x"] = np.linspace(20, 239, 600)
    references["plateau_A"], references["plateau_G"] = plateau.local_charge_estimators(references["plateau_x"])
    point = ThinShellPoint(1e4, .063)
    linear = solve_charge(point, x_max=160, tol=2e-6, mesh_points=420, max_nodes=100_000, formulation="linear")
    logarithmic = solve_charge(point, x_max=160, tol=2e-6, mesh_points=420, max_nodes=100_000, formulation="log")
    return {"weak_source_max_relative_error": float(np.max(abs(weak_numeric / weak_analytic - 1))),
            "weak_mus": weak_mus.tolist(), "weak_numeric": weak_numeric.tolist(),
            "weak_analytic": weak_analytic.tolist(), "domains": domains.tolist(),
            "domain_charge": domain_charge.tolist(), "tolerances": tolerances.tolist(),
            "tolerance_charge": tolerance_charge.tolist(),
            "domain_relative_change_160_to_240": float(abs(domain_charge[2] / domain_charge[-1] - 1)),
            "tolerance_relative_change_coarse_to_fine": float(abs(tolerance_charge[0] / tolerance_charge[-1] - 1)),
            "plateau_estimator_disagreement": plateau.estimator_disagreement,
            "linear_log_representation_disagreement": float(abs(linear.q_num / logarithmic.q_num - 1))}


def run_stratified_domain_audit(data: dict, threshold: float) -> dict:
    selected = {}

    def add(i, j, reason):
        selected.setdefault((int(i), int(j)), set()).add(reason)

    n, m = data["q_num"].shape
    for i, j in ((0, 0), (0, m-1), (n-1, 0), (n-1, m-1)):
        add(i, j, "grid corner")
    for density, mu, label in ((1e4, .032, "transition profile"), (1e5, .01, "convergence benchmark")):
        add(np.argmin(abs(np.log(data["densities"] / density))),
            np.argmin(abs(np.log(data["mus"] / mu))), label)
    for name, key in (("surface", "q_kw"), ("active shell", "q_yms")):
        order = np.argsort(abs(abs(data[key] / data["q_num"] - 1).ravel() - .1))
        for index in order[:4]:
            add(*np.unravel_index(index, (n, m)), f"{name} 10% boundary")
    direct = data["alpha_direct"][..., probe_index(data)]
    tail = tail_force(data["q_num"], data)
    for i, j in np.argwhere((direct >= threshold) != (tail >= threshold)):
        add(i, j, "tail/direct threshold flip")
    i, j = np.unravel_index(np.argmax(abs(tail / direct - 1)), (n, m))
    add(i, j, "largest local-force error")
    rows = []
    for count, ((i, j), reasons) in enumerate(sorted(selected.items()), 1):
        p = ThinShellPoint(float(data["densities"][i]), float(data["mus"][j]))
        a = solve_charge(p, x_max=160, tol=2e-6, mesh_points=480, max_nodes=120_000)
        b = solve_charge(p, x_max=240, tol=2e-7, mesh_points=560, max_nodes=140_000)
        fa, fb = a.direct_force_ratio(PROBE_RADII), b.direct_force_ratio(PROBE_RADII)
        row = {"density_contrast": p.density_contrast, "mu_out": p.mu_out,
               "selection_reason": "; ".join(sorted(reasons)), "q_scan": data["q_num"][i, j],
               "q_160": a.q_num, "q_240": b.q_num,
               "relative_domain_change": abs(a.q_num / b.q_num - 1),
               "max_local_force_change_scan_to_fine": float(np.max(abs(data["alpha_direct"][i,j] / fb - 1))),
               "max_local_force_change_160_to_240": float(np.max(abs(fa / fb - 1))),
               "direct_threshold_changed_scan_to_fine": int((direct[i,j] >= threshold) != (float(b.direct_force_ratio()) >= threshold))}
        for name, approximation in (("surface", q_thin_shell(p)), ("center", q_center_point_continuation(p)),
                                    ("active_shell", q_yukawa_matched(p))):
            ea, eb = abs(approximation / a.q_num - 1), abs(approximation / b.q_num - 1)
            row[f"{name}_error_160"] = ea
            row[f"{name}_error_240"] = eb
            row[f"{name}_10_percent_classification_changed"] = int((ea <= .1) != (eb <= .1))
        rows.append(row)
        print(f"domain/local-force audit {count}/{len(selected)}", flush=True)
    _write_csv(RESULTS / "stratified_domain_audit.csv", rows)
    return {"audited_points": len(rows),
            "selection": "corners, benchmarks, charge-error boundaries, tail/direct threshold flips, largest local-force error",
            "max_relative_domain_change": max(row["relative_domain_change"] for row in rows),
            "max_local_force_change_scan_to_fine": max(row["max_local_force_change_scan_to_fine"] for row in rows),
            "max_local_force_change_160_to_240": max(row["max_local_force_change_160_to_240"] for row in rows),
            "direct_threshold_classification_changes": sum(row["direct_threshold_changed_scan_to_fine"] for row in rows),
            "charge_10_percent_classification_changes": sum(row[f"{name}_10_percent_classification_changed"]
                for row in rows for name in ("surface", "center", "active_shell"))}


def validate_simple_gate(data: dict) -> dict:
    rows = []
    for density in np.sqrt(data["densities"][:-1] * data["densities"][1:]):
        for mu in np.sqrt(data["mus"][:-1] * data["mus"][1:]):
            p = ThinShellPoint(float(density), float(mu))
            if p.epsilon < .1 and p.mu_in > 10 and p.mu_out < .035:
                sol = solve_charge(p, x_max=160, tol=2e-5, mesh_points=300, max_nodes=60_000)
                rows.append({"density_contrast": density, "mu_out": mu, "epsilon_th": p.epsilon,
                             "mu_in": p.mu_in, "q_num": sol.q_num,
                             "relative_error": abs(q_thin_shell(p) / sol.q_num - 1)})
    _write_csv(RESULTS / "gate_validation_midpoints.csv", rows)
    gate = (data["epsilon"] < .1) & (data["mu_in"] > 10) & (data["mu_grid"] < .035)
    error = abs(data["q_kw"][gate] / data["q_num"][gate] - 1)
    return {"conditions": "epsilon_th<0.1, mu_in>10, mu_out<0.035; empirical within the sampled domain",
            "points": int(np.count_nonzero(gate)), "coverage": float(np.mean(gate)),
            "max_relative_error": float(error.max()),
            "held_out_midpoint_count": len(rows),
            "held_out_max_relative_error": max(row["relative_error"] for row in rows)}


def compute_metrics(data: dict, profiles: dict, verification: dict,
                    gate: dict, threshold: float) -> dict:
    direct = data["alpha_direct"][..., probe_index(data)]
    tail = tail_force(data["q_num"], data)
    changed = np.argwhere((direct >= threshold) != (tail >= threshold))
    return {
        "grid": {"total_points": int(data["q_num"].size), "successful_points": int(data["success"].sum()),
                 "density_points": len(data["densities"]), "mu_points": len(data["mus"]),
                 "density_range": [float(data["densities"][0]), float(data["densities"][-1])],
                 "mu_range": [float(data["mus"][0]), float(data["mus"][-1])],
                 "domains": np.unique(data["domains"]).tolist(),
                 "max_estimator_disagreement": float(data["estimator_disagreement"].max())},
        "charge_approximations": {name: relative_error_metrics(data[key], data["q_num"])
                                  for name, key in APPROXIMATIONS.items()},
        "ablation": {"median_extra_radial_weight": float(np.median(data["q_yms"] / data["q_point_ms"] - 1)),
                     "max_extra_radial_weight": float(np.max(data["q_yms"] / data["q_point_ms"] - 1))},
        "profiles": profiles, "verification": verification, "empirical_charge_gate": gate,
        "distance_validity": distance_summary(data),
        "detectability": threshold_summary(data, threshold),
        "threshold_sensitivity": [threshold_summary(data, t) for t in (.001, .003, .01, .03, .1, .3, 1.)],
        "tail_direct_threshold_examples": [
            {"density_contrast": float(data["density_grid"][i,j]), "mu_out": float(data["mu_grid"][i,j]),
             "alpha_tail": float(tail[i,j]), "alpha_direct": float(direct[i,j])} for i,j in changed],
    }


def write_document_numbers(metrics: dict) -> None:
    """Make every quantitative PDF table share the machine-readable results."""

    charge = metrics["charge_approximations"]
    distance = metrics["distance_validity"]
    detection = metrics["detectability"]
    audit = metrics["verification"]["stratified_domain_audit"]
    radii = {float(row["radius_over_R"]): row for row in distance["by_radius"]}
    pct = lambda value: f"{100 * value:.3g}"
    def sci(value):
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"{float(mantissa):g}\times10^{{{int(exponent)}}}"
    values = {
        "Points": metrics["grid"]["total_points"],
        "SurfaceWithin": pct(charge["surface_amplitude"]["fraction_within_10_percent"]),
        "CenterWithin": pct(charge["center_point"]["fraction_within_10_percent"]),
        "ASWithin": pct(charge["active_shell_integral"]["fraction_within_10_percent"]),
        "ASMedian": pct(charge["active_shell_integral"]["median_relative_error"]),
        "ASMax": pct(charge["active_shell_integral"]["max_relative_error"]),
        "ASLocalMax": pct(distance["local_force_approximations_at_1p5R"]["active_shell_integral"]["max_relative_error"]),
        "ASLocalWithin": pct(distance["local_force_approximations_at_1p5R"]["active_shell_integral"]["fraction_within_10_percent"]),
        "PointMedian": pct(charge["point_matched_shell"]["median_relative_error"]),
        "PointMax": pct(charge["point_matched_shell"]["max_relative_error"]),
        "RadialMax": pct(metrics["ablation"]["max_extra_radial_weight"]),
        "ForceNearMax": pct(radii[1.5]["max_relative_error"]),
        "ForceNearFailures": metrics["grid"]["total_points"] - radii[1.5]["points_within_10_percent"],
        "ForceFiveMax": pct(radii[5.]["max_relative_error"]),
        "ForceTenMax": pct(radii[10.]["max_relative_error"]),
        "ValidityRadiusMax": f"{distance['max_first_sampled_10_percent_radius']:.3g}",
        "RadiusSamples": distance["sampled_radius_count"],
        "TailFlips": detection["models"]["numerical_charge_tail"]["mismatch_points"],
        "SurfaceFlips": detection["models"]["surface_amplitude"]["mismatch_points"],
        "CenterFlips": detection["models"]["center_point"]["mismatch_points"],
        "AnsatzDisagreement": detection["ansatz_disagreement_points"],
        "BothWrong": detection["both_simple_ansatzes_wrong_points"],
        "AuditPoints": audit["audited_points"],
        "AuditChargeMax": pct(audit["max_relative_domain_change"]),
        "AuditForceMax": pct(audit["max_local_force_change_scan_to_fine"]),
        "AuditForceFlips": audit["direct_threshold_classification_changes"],
        "GatePoints": metrics["empirical_charge_gate"]["points"],
        "GateCoverage": pct(metrics["empirical_charge_gate"]["coverage"]),
        "GateHeldout": metrics["empirical_charge_gate"]["held_out_midpoint_count"],
        "WeakError": sci(metrics["verification"]["weak_source_max_relative_error"]),
    }
    exponent = np.log10(detection["threshold"])
    values["Threshold"] = (f"10^{{{int(round(exponent))}}}" if abs(exponent-round(exponent)) < 1e-10
                           else f"{detection['threshold']:.4g}")
    for name in ("surface_amplitude", "center_point"):
        shift = detection["models"][name]["median_contour_shift_dex"]
        values["SurfaceShift" if name == "surface_amplitude" else "CenterShift"] = (
            f"{shift:.3g}" if shift is not None else r"\mathrm{N/A}")
    content = "% Generated by run_study.py; do not edit individual numbers.\n"
    content += "".join(f"\\newcommand{{\\Study{name}}}{{{value}}}\n" for name, value in values.items())
    (RESULTS / "study_numbers.tex").write_text(content, encoding="utf-8")
    labels = {
        "en": ["Surface-amplitude model", "Center-point model", "Finite shell volume", "Point + finite volume", "Active-shell integral"],
        "ko": ["표면 진폭 모형", "중심점 모형", "유한 껍질 부피", "점원천+유한 부피", "활성 껍질 적분"],
    }
    for language in ("en", "ko"):
        table, local_table = [], []
        for (name, _), label in zip(APPROXIMATIONS.items(), labels[language]):
            row = charge[name]
            table.append(f"{label} & {pct(row['median_relative_error'])}\\% & "
                         f"{pct(row['max_relative_error'])}\\% & "
                         f"{row['points_within_10_percent']}/{metrics['grid']['total_points']} \\\\")
            row = distance["local_force_approximations_at_1p5R"][name]
            local_table.append(f"{label} & {pct(row['median_relative_error'])}\\% & "
                               f"{pct(row['max_relative_error'])}\\% & "
                               f"{row['points_within_10_percent']}/{metrics['grid']['total_points']} \\\\")
        suffix = language.upper()
        (RESULTS / f"charge_table_{language}.tex").write_text(
            f"\\newcommand{{\\StudyChargeRows{suffix}}}{{%\n" + "\n".join(table)+"\n}\n", encoding="utf-8")
        (RESULTS / f"local_force_table_{language}.tex").write_text(
            f"\\newcommand{{\\StudyLocalForceRows{suffix}}}{{%\n" + "\n".join(local_table)+"\n}\n", encoding="utf-8")
    distance_table = [f"{r:g} & {pct(radii[r]['median_relative_error'])}\\% & "
                      f"{pct(radii[r]['max_relative_error'])}\\% & "
                      f"{metrics['grid']['total_points']-radii[r]['points_within_10_percent']} \\\\"
                      for r in (1.05, 1.5, 2., 5., 10.)]
    (RESULTS / "distance_table.tex").write_text(
        "\\newcommand{\\StudyDistanceRows}{%\n"+"\n".join(distance_table)+"\n}\n", encoding="utf-8")


def collect_provenance(data: dict) -> dict:
    """Describe the actual source state, including uncommitted revisions."""
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    hashes = {name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8").encode()).hexdigest()
              for name in ("thin_shell_model.py", "study_analysis.py", "run_study.py", "study_figures.py")}
    return {"schema_version": SCHEMA_VERSION, "scan_fingerprint": str(data["fingerprint"]),
            "git_revision_at_run": revision, "git_worktree_dirty_at_run": dirty,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashes, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "force_reference": "direct nonlinear gradient; beta=Q_t=1; total source mass"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold", type=float, default=.01)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be positive and finite")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / ".mplconfig").mkdir(exist_ok=True)
    try:
        if args.force:
            raise StaleScanError("fresh scan requested")
        data = load_scan()
        print("current scan cache validated", flush=True)
    except (FileNotFoundError, StaleScanError) as error:
        print(f"recomputing scan: {error}", flush=True)
        data = run_scan(max(1, args.workers))
    references, profiles = compute_reference_profiles()
    verification = compute_verification(references)
    verification["stratified_domain_audit"] = run_stratified_domain_audit(data, args.threshold)
    gate = validate_simple_gate(data)
    metrics = compute_metrics(data, profiles, verification, gate, args.threshold)
    metrics["provenance"] = collect_provenance(data)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(REFERENCE_FILE, **references)
    write_scan_tables(data, metrics)
    write_document_numbers(metrics)
    from .study_figures import render_figures
    render_figures(data, metrics, references, "en")
    render_figures(data, metrics, references, "ko")
    print(json.dumps({"charge": metrics["charge_approximations"], "threshold": metrics["detectability"],
                      "audit": verification["stratified_domain_audit"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
