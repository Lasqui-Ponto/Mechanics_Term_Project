# Chameleon Screening Dynamics — mechanics term project

This repository contains a reproducible, dimensionless mechanics study of a static chameleon scalar field and the motion of a test particle in its fifth-force landscape.

Repository: [github.com/Lasqui-Ponto/Mechanics_Term_Project](https://github.com/Lasqui-Ponto/Mechanics_Term_Project)

## New independent study (2026-07-10)

The repository now also contains a separate research package, [`Thin_Shell_Study/`](Thin_Shell_Study/README.md), which leaves the original report and slides unchanged. It asks a narrower quantitative question:

> How accurately does the long-range thin-shell law predict the full nonlinear spherical solution, and how should it be continued when the source has finite Yukawa range?

The new study solves 725 spherical BVPs at an outer radius of $160R$, extracts a numerical scalar charge from the exterior Yukawa tail, compares surface-shell and center-point continuations, and benchmarks an active-shell radial integral. A 14-point stratified $160R\to240R$ audit checks grid corners and the 10% decision boundary. The study also shows how continuation ambiguity propagates into an illustrative force-ratio threshold. Its report, 16:9 slides, Korean speaker notes, scan data, held-out checks, tests, and generation scripts are self-contained in that directory.

## What was improved

- The numerical method is now explicit: the static field is solved as a collocation BVP and the particle orbit is integrated with adaptive Dormand–Prince RK45 (`rtol=1e-9`, `atol=1e-11`, `max_step=0.05`).
- A classical fixed-step RK4 integrator is implemented directly in `chameleon_numerics.py` and cross-checked against RK45.
- The code is verified against two analytic limits: a harmonic oscillator IVP and the constant-field solution of a uniform-density BVP.
- Energy drift, RK4 convergence order, RK4–RK45 agreement, return-period variation, and a finite-time Lyapunov diagnostic are reported as machine-readable JSON.
- The earlier “chaos” interpretation was corrected. A static one-dimensional autonomous Hamiltonian is integrable; the random medium deforms the energy contour but does not establish deterministic chaos.
- A vacuum-chamber figure distinguishes the common field profile from the sensor-dependent scalar charge, explaining why an unscreened atom can respond more strongly than screened bulk matter.

## Scope and units

The scripts solve the normalized equation

\[
\frac{d^2\hat\phi}{d\hat x^2}=-\frac{1}{\hat\phi^2}+\hat\rho(\hat x)e^{\hat\phi},
\qquad
\frac{d^2\hat x}{d\hat t^2}=-\hat k\hat x-\frac{d\hat\phi}{d\hat x}.
\]

For physical variables \(x=L_0\hat x\), \(\phi=\Phi_0\hat\phi\), and \(\rho=\rho_0\hat\rho\), the normalized coefficients are

\[
A=\frac{nM^{4+n}L_0^2}{\Phi_0^{n+2}},\qquad
B=\frac{\beta\rho_0L_0^2}{M_{\rm Pl}\Phi_0},\qquad
C=\frac{\beta\Phi_0T_0^2}{M_{\rm Pl}L_0^2}.
\]

The toy model sets \(A=B=C=1\). Therefore `BETA = 1` in the scripts is a normalized coefficient, not a claim that a particular dimensional parameter point is allowed by laboratory or astrophysical bounds. A physical exclusion plot requires selecting \((M,n,\beta)\), chamber geometry, source/test-body screening factors, and SI/natural-unit conversion together.

## Reproduce

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m unittest -v
.venv\Scripts\python.exe verify_simulation.py
.venv\Scripts\python.exe chameleon_simulation.py
.venv\Scripts\python.exe chameleon_field_visualization.py
.venv\Scripts\python.exe chameleon_random_medium.py
.venv\Scripts\python.exe sensor_detectability.py
latexmk -xelatex term_project_paper.tex
latexmk -xelatex presentation.tex
```

The random medium is reproducible with seed `137`.

## Verification results

The current run records:

| Check | Result |
|---|---:|
| Direct RK4 observed convergence order | 4.0038 |
| Uniform-density BVP maximum field error | \(5.55\times10^{-17}\) |
| Full-model RK4–RK45 maximum state difference | \(4.97\times10^{-11}\) |
| Full-model maximum relative energy drift | \(1.09\times10^{-11}\) |

See `verification_metrics.json`, `random_medium_metrics.json`, and `verification_results.png` for the complete machine-readable results and plots.

## Main files

- `chameleon_numerics.py`: shared model, BVP, RK45, direct RK4, energy, and Lyapunov tools
- `verify_simulation.py`: analytic and convergence verification suite
- `chameleon_simulation.py`: smooth high-to-low-density transition
- `chameleon_field_visualization.py`: spherical screening visualization
- `chameleon_random_medium.py`: static disordered-medium diagnostics
- `sensor_detectability.py`: atom-versus-bulk sensor response schematic
- `term_project_paper.tex`, `presentation.tex`: report and slides

## Physical context

The original mechanism is due to Khoury and Weltman ([PRL 93, 171104](https://doi.org/10.1103/PhysRevLett.93.171104); [PRD 69, 044026](https://doi.org/10.1103/PhysRevD.69.044026)). For experimental context, see the review by Burrage and Sakstein ([Living Reviews in Relativity 21, 1](https://doi.org/10.1007/s41114-018-0011-x)), atom-interferometer searches by Hamilton et al. ([Science 349, 849](https://doi.org/10.1126/science.aaa8883)) and Sabulsky et al. ([PRL 123, 061102](https://doi.org/10.1103/PhysRevLett.123.061102)), and the levitated-sensor result of Yin et al. ([Nature Physics 18, 1181](https://doi.org/10.1038/s41567-022-01706-9)).
