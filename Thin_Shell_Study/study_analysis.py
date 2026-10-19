"""Pure analysis of charge, local force, and threshold decisions.

All force references in this module are gradients of the nonlinear BVP.
The functions do not write files or generate figures, so threshold edge cases
and physical limits can be checked independently of the presentation layer.
"""

from __future__ import annotations

import numpy as np


PROBE_RADII = np.unique(np.concatenate((
    np.geomspace(1.02, 20.0, 65), np.array([1.05, 1.5, 2.0, 3.0, 5.0, 10.0])
)))
APPROXIMATIONS = {
    "surface_amplitude": "q_kw",
    "center_point": "q_center",
    "finite_shell_volume": "q_ms",
    "point_matched_shell": "q_point_ms",
    "active_shell_integral": "q_yms",
}


def probe_index(data: dict, radius: float = 1.5) -> int:
    index = np.flatnonzero(np.isclose(data["probe_radii"], radius, rtol=1e-12, atol=0))
    if index.size != 1:
        raise ValueError("requested radius is not a directly sampled probe radius")
    return int(index[0])


def tail_force(q_source: np.ndarray, data: dict, radius: float = 1.5) -> np.ndarray:
    """Yukawa prediction for beta=Q_t=1 at the supplied radius."""

    mu = data["mu_grid"]
    return 2 * np.asarray(q_source) * (1 + mu * radius) * np.exp(-mu * (radius - 1))


def relative_error_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict:
    error = np.abs(np.asarray(prediction) / np.asarray(reference) - 1)
    return {
        "median_relative_error": float(np.median(error)),
        "p90_relative_error": float(np.quantile(error, .9)),
        "max_relative_error": float(np.max(error)),
        "points_within_10_percent": int(np.count_nonzero(error <= .1)),
        "fraction_within_10_percent": float(np.mean(error <= .1)),
        "fraction_within_20_percent": float(np.mean(error <= .2)),
    }


def tail_force_cube(data: dict) -> np.ndarray:
    radii = data["probe_radii"][None, None, :]
    mu = data["mu_grid"][..., None]
    return 2 * data["q_num"][..., None] * (1 + mu * radii) * np.exp(-mu * (radii - 1))


def sampled_validity_radii(data: dict, tolerance: float = .1) -> np.ndarray:
    """First sampled radius meeting the error target at every later sample.

    This is a discrete diagnostic on [1.02,20]R, not a proof of a continuous
    boundary or of behavior beyond 20R.  NaN means no such sample was found.
    """

    error = np.abs(tail_force_cube(data) / data["alpha_direct"] - 1)
    good = np.logical_and.accumulate((error <= tolerance)[..., ::-1], axis=-1)[..., ::-1]
    first = np.argmax(good, axis=-1)
    return np.where(np.any(good, axis=-1), data["probe_radii"][first], np.nan)


def distance_summary(data: dict) -> dict:
    tail = tail_force_cube(data)
    direct = data["alpha_direct"]
    rows = []
    for k, radius in enumerate(data["probe_radii"]):
        rows.append({"radius_over_R": float(radius),
                     **relative_error_metrics(tail[..., k], direct[..., k])})
    validity = sampled_validity_radii(data)
    index = probe_index(data)
    direct_probe = direct[..., index]
    return {
        "definition": "relative error of the numerical-charge Yukawa tail against the local nonlinear gradient",
        "beta": 1.0, "q_probe": 1.0,
        "sampled_radius_range": [float(data["probe_radii"][0]), float(data["probe_radii"][-1])],
        "sampled_radius_count": int(data["probe_radii"].size),
        "by_radius": rows,
        "max_first_sampled_10_percent_radius": float(np.nanmax(validity)),
        "unresolved_validity_points": int(np.count_nonzero(~np.isfinite(validity))),
        "local_force_approximations_at_1p5R": {
            name: relative_error_metrics(tail_force(data[key], data), direct_probe)
            for name, key in APPROXIMATIONS.items()
        },
    }


def _first_crossing(log_mu: np.ndarray, row: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero((row[:-1] - threshold) * (row[1:] - threshold) <= 0)
    if not indices.size:
        return None
    i = int(indices[0])
    y0, y1 = np.log10(row[i:i+2])
    if y0 == y1:
        return float(log_mu[i])
    fraction = (np.log10(threshold) - y0) / (y1 - y0)
    return float(log_mu[i] + fraction * (log_mu[i+1] - log_mu[i]))


def _contour_shifts(reference: np.ndarray, prediction: np.ndarray, mus: np.ndarray,
                    threshold: float) -> dict:
    shifts = []
    for actual, predicted in zip(reference, prediction):
        a = _first_crossing(np.log10(mus), actual, threshold)
        b = _first_crossing(np.log10(mus), predicted, threshold)
        if a is not None and b is not None:
            shifts.append(abs(a - b))
    return {
        "rows_with_both_contours": len(shifts),
        "median_contour_shift_dex": float(np.median(shifts)) if shifts else None,
        "max_contour_shift_dex": float(np.max(shifts)) if shifts else None,
    }


def threshold_summary(data: dict, threshold: float = .01, radius: float = 1.5) -> dict:
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    direct = data["alpha_direct"][..., probe_index(data, radius)]
    truth = direct >= threshold
    predictions = {name: tail_force(data[key], data, radius)
                   for name, key in APPROXIMATIONS.items()}
    predictions["numerical_charge_tail"] = tail_force(data["q_num"], data, radius)
    models = {}
    for name, prediction in predictions.items():
        exceeds = prediction >= threshold
        models[name] = {
            "mismatch_points": int(np.count_nonzero(exceeds != truth)),
            "false_above_points": int(np.count_nonzero(exceeds & ~truth)),
            "false_below_points": int(np.count_nonzero(~exceeds & truth)),
            **_contour_shifts(direct, prediction, data["mus"], threshold),
        }
    surface = predictions["surface_amplitude"] >= threshold
    center = predictions["center_point"] >= threshold
    return {
        "threshold": float(threshold), "probe_radius_over_R": float(radius),
        "reference": "direct nonlinear gradient; total-source-mass Newtonian reference",
        "beta": 1.0, "q_probe": 1.0,
        "direct_above_points": int(np.count_nonzero(truth)),
        "ansatz_disagreement_points": int(np.count_nonzero(surface != center)),
        "both_simple_ansatzes_wrong_points": int(np.count_nonzero((surface == center) & (surface != truth))),
        "models": models,
    }
