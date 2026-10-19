"""Render Korean figures from the same scan, metrics, and threshold as English.

Run ``python -m Thin_Shell_Study.run_study`` first to generate current products.
This command performs no BVP re-solves and cannot silently reset the threshold.
"""

import json
import numpy as np

from .run_study import METRICS_FILE, REFERENCE_FILE, load_scan
from .study_figures import render_figures


def main():
    data = load_scan()
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    if metrics["provenance"]["scan_fingerprint"] != str(data["fingerprint"]):
        raise ValueError("metrics and scan differ; rerun Thin_Shell_Study.run_study")
    with np.load(REFERENCE_FILE, allow_pickle=False) as archive:
        references = {key: archive[key] for key in archive.files}
    render_figures(data, metrics, references, "ko")


if __name__ == "__main__":
    main()
