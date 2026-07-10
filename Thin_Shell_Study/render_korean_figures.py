"""영문 그림을 보존하면서 논문·발표용 한국어 그림을 생성한다.

사용법(저장소 루트에서 실행)::

    python -m Thin_Shell_Study.render_korean_figures

725점 매개변수 스캔은 ``results/thin_shell_scan.npz``에서 읽는다. 프로파일과
검증 그림에 필요한 소수의 기준점만 다시 계산하며, 결과는 기존 영문 파일을
건드리지 않고 ``Thin_Shell_Study/figures_ko``에 저장한다.
"""

from __future__ import annotations

import os
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_MPL_CONFIG = _MODULE_DIR / "results" / ".mplconfig"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

from .thin_shell_model import ThinShellPoint, q_linear_sphere, solve_charge


RESULTS = _MODULE_DIR / "results"
FIGURES_KO = _MODULE_DIR / "figures_ko"
SCAN_FILE = RESULTS / "thin_shell_scan.npz"


def _configure_plotting() -> str:
    """한국어 글꼴을 명시적으로 등록하고 원본 그림의 스타일을 재현한다."""

    candidates = (
        Path(r"C:\Windows\Fonts\NotoSansCJKkr-Regular.otf"),
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    )
    selected: str | None = None
    for font_path in candidates:
        if not font_path.exists():
            continue
        font_manager.fontManager.addfont(str(font_path))
        if selected is None:
            selected = font_manager.FontProperties(fname=str(font_path)).get_name()
    if selected is None:
        available = {entry.name for entry in font_manager.fontManager.ttflist}
        for name in ("Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic"):
            if name in available:
                selected = name
                break
    if selected is None:
        raise RuntimeError("Noto Sans KR 또는 Malgun Gothic 글꼴을 찾지 못했습니다.")

    mpl.rcParams.update(
        {
            "font.family": selected,
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )
    return selected


def _load_scan() -> dict[str, np.ndarray]:
    if not SCAN_FILE.exists():
        raise FileNotFoundError(
            f"스캔 캐시가 없습니다: {SCAN_FILE}\n"
            "먼저 `python -m Thin_Shell_Study.run_study`를 실행하세요."
        )
    with np.load(SCAN_FILE, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="normal",
        va="top",
    )


def plot_profiles() -> None:
    cases = [
        ("비차폐", ThinShellPoint(1.0e2, 1.0e-1)),
        ("전이 영역", ThinShellPoint(1.0e4, 3.2e-2)),
        ("차폐", ThinShellPoint(1.0e5, 1.0e-1)),
    ]
    colors = ("#2b8cbe", "#f39c34", "#b2182b")
    solutions = [
        solve_charge(point, x_max=160.0, tol=3.0e-6, mesh_points=500, max_nodes=100_000)
        for _, point in cases
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x_field = np.concatenate((np.linspace(0.0, 0.98, 300), np.linspace(0.98, 3.0, 500)))
    for (label, _point), solution, color in zip(cases, solutions, colors):
        axes[0].plot(x_field, solution.field(x_field), lw=2.2, color=color, label=label)
        x_charge = np.linspace(max(1.03, solution.fit_x_min), solution.fit_x_max, 320)
        amplitude, gradient = solution.local_charge_estimators(x_charge)
        axes[1].plot(x_charge, amplitude, lw=2.0, color=color, label=f"{label}: 진폭")
        axes[1].plot(x_charge, gradient, lw=1.3, ls="--", color=color, alpha=0.8)
        axes[1].axhline(solution.q_num, lw=0.8, color=color, alpha=0.35)

    axes[0].axvline(1.0, color="0.3", lw=1.0, ls=":")
    axes[0].set(
        xlabel=r"반지름 $x=r/R$",
        ylabel=r"장 $u=\phi/\phi_\infty$",
        xlim=(0, 3.0),
        title="완전 비선형 구대칭 해",
    )
    axes[0].legend(frameon=False)
    axes[1].set(
        xlabel=r"반지름 $x=r/R$",
        ylabel=r"국소 스칼라 전하 $Q(x)$",
        yscale="log",
        title="외부 유카와 평탄부 (실선: 진폭, 점선: 기울기)",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis, label in zip(axes, ("a", "b")):
        _panel_label(axis, label)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES_KO / "profiles_and_charge_extraction.png")
    plt.close(figure)


def plot_accuracy_atlas(data: dict[str, np.ndarray]) -> None:
    density = data["density_grid"]
    mu = data["mu_grid"]
    q_num = data["q_num"]
    approximations = (
        (data["q_kw"], r"표면 기준 연장 $Q_{\rm surf}$"),
        (data["q_center"], r"중심점 기준 연장 $Q_{\rm ctr}$"),
        (data["q_yms"], r"활성 껍질 적분 $Q_{\rm AS}$"),
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
    axes[0].set_ylabel(r"밀도 대비 $\delta=\rho_c/\rho_\infty$")
    color_axis = figure.add_axes([0.915, 0.18, 0.015, 0.68])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(r"부호 있는 오차 $\log_{10}(Q_{\rm approx}/Q_{\rm num})$")
    figure.text(
        0.5,
        -0.01,
        r"검은 실선: 10% 오차 경계; 점선: $\epsilon=0.01,\ 0.1,\ 0.5$",
        ha="center",
    )
    figure.subplots_adjust(left=0.075, right=0.895, bottom=0.18, top=0.9, wspace=0.08)
    figure.savefig(FIGURES_KO / "accuracy_atlas.png")
    plt.close(figure)


def plot_charge_comparison(data: dict[str, np.ndarray]) -> None:
    epsilon = data["epsilon"].ravel()
    q_num = data["q_num"].ravel()
    approximations = (
        (data["q_kw"].ravel(), "표면 연장", "#b2182b"),
        (data["q_center"].ravel(), "중심 연장", "#762a83"),
        (data["q_ms"].ravel(), "유한 두께 껍질", "#ef8a62"),
        (data["q_yms"].ravel(), "활성 껍질 적분", "#2166ac"),
    )
    bins = np.geomspace(max(epsilon.min(), 1.0e-5), epsilon.max(), 17)
    centers = np.sqrt(bins[:-1] * bins[1:])

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    sample = np.arange(epsilon.size)
    axes[0].scatter(q_num[sample], data["q_kw"].ravel()[sample], s=10, alpha=0.28, color="#b2182b", label="표면 연장")
    axes[0].scatter(q_num[sample], data["q_center"].ravel()[sample], s=10, alpha=0.28, color="#762a83", label="중심 연장")
    axes[0].scatter(q_num[sample], data["q_yms"].ravel()[sample], s=10, alpha=0.35, color="#2166ac", label="활성 껍질 적분")
    identity = np.geomspace(q_num.min(), 1.1, 200)
    axes[0].plot(identity, identity, color="black", lw=1.2, ls="--")
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel=r"수치 전하 $Q_{\rm num}$",
        ylabel="예측 전하",
        title="점별 예측",
    )
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
    axes[1].axhline(0.1, color="black", ls="--", lw=1.1, label="10% 목표")
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel=r"얇은 껍질 매개변수 $\epsilon=\Delta R/R$",
        ylabel="상대 전하 오차",
        title="중앙값과 90백분위 오차",
    )
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    for axis, label in zip(axes, ("a", "b")):
        axis.grid(alpha=0.2, which="both")
        _panel_label(axis, label)
    figure.tight_layout()
    figure.savefig(FIGURES_KO / "charge_comparison.png")
    plt.close(figure)


def _force_map(q_source: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    mu = data["mu_grid"]
    x_probe = 1.5
    yukawa = (1.0 + mu * x_probe) * np.exp(-mu * (x_probe - 1.0))
    return 2.0 * q_source * yukawa


def plot_detectability(data: dict[str, np.ndarray], threshold: float = 0.01) -> None:
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
    axes[0].plot([], [], color="white", lw=2.0, label="수치해 임계선")
    axes[0].plot([], [], color="#ffcc33", lw=1.8, ls="--", label="표면 연장")
    axes[0].plot([], [], color="#66c2ff", lw=1.8, ls=":", label="중심 연장")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7.5)
    cbar0 = figure.colorbar(image0, ax=axes[0], pad=0.02)
    cbar0.set_label(r"$\log_{10}|a_5/a_N|$ (비차폐 시험입자)")

    class_colors = ListedColormap(["#d9d9d9", "#41ab5d", "#ef8a62", "#67a9cf"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], class_colors.N)
    image1 = axes[1].pcolormesh(mu, density, classification, shading="auto", cmap=class_colors, norm=norm)
    cbar1 = figure.colorbar(image1, ax=axes[1], ticks=[0, 1, 2, 3], pad=0.02)
    cbar1.ax.set_yticklabels(["모두 미만", "모두 이상", "연장법 불일치", "둘 다 오판"])

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
        "무차원 힘 비율 대리지표",
        "한 임계값에서의 연장법 모호성",
        "임계값 단위의 단순 연장 오차",
    )
    for axis, title, label in zip(axes, titles, ("a", "b", "c")):
        axis.set(xscale="log", yscale="log", xlabel=r"$\mu=m_\infty R$", title=title)
        axis.grid(alpha=0.12, which="both")
        _panel_label(axis, label)
    axes[0].set_ylabel(r"밀도 대비 $\delta=\rho_c/\rho_\infty$")
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.15, top=0.9, wspace=0.35)
    figure.savefig(FIGURES_KO / "detectability_map.png")
    plt.close(figure)


def plot_verification() -> None:
    weak_mus = np.array([0.01, 0.03, 0.1, 0.3])
    weak_density = 1.01
    weak_numeric = np.asarray(
        [
            solve_charge(
                ThinShellPoint(weak_density, float(mu)),
                x_max=160.0,
                tol=2.0e-7,
                mesh_points=450,
                max_nodes=100_000,
            ).q_num
            for mu in weak_mus
        ]
    )
    weak_analytic = (weak_density - 1.0) / weak_density * q_linear_sphere(weak_mus)

    benchmark = ThinShellPoint(1.0e5, 1.0e-2)
    domains = np.array([40.0, 80.0, 160.0, 240.0])
    domain_charge = np.asarray(
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
    tolerance_charge = np.asarray(
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

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.95))
    axes[0].plot(weak_mus, weak_analytic, color="black", lw=1.5, label="균일 구 해석해")
    axes[0].scatter(weak_mus, weak_numeric, color="#2b8cbe", s=38, zorder=3, label="비선형 BVP")
    axes[0].set(
        xscale="log",
        xlabel=r"$\mu=m_\infty R$",
        ylabel=r"비차폐 전하 $Q$",
        title="해석적 극한 검증",
    )
    axes[0].legend(frameon=False)

    reference = domain_charge[-1]
    category_x = np.array([0.0, 1.0, 2.0])
    axes[1].plot(
        category_x,
        np.abs(domain_charge[:3] / reference - 1.0),
        marker="o",
        lw=1.8,
        label=r"계산 영역: $x_{\max}=40,80,160$",
    )
    axes[1].plot(
        category_x[:2],
        np.abs(tolerance_charge[:2] / tolerance_charge[-1] - 1.0),
        marker="s",
        lw=1.8,
        label=r"허용오차: $3\times10^{-4},3\times10^{-5}$",
    )
    axes[1].set(
        yscale="log",
        xlabel="해상도 단계",
        ylabel=r"최고 해상도 $Q$ 대비 상대 변화",
        title="수치 수렴",
    )
    axes[1].set_xticks(category_x, ["낮음", "중간", "높음"])
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(plateau_x, plateau_amp, lw=1.8, label="진폭 추정량")
    axes[2].plot(plateau_x, plateau_grad, lw=1.5, ls="--", label="기울기 추정량")
    axes[2].axhline(plateau_solution.q_num, color="black", lw=0.9, ls=":")
    axes[2].set(
        xlabel=r"반지름 $x=r/R$",
        ylabel=r"$Q(x)$",
        title="독립적 전하 추출",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis, label in zip(axes, ("a", "b", "c")):
        axis.grid(alpha=0.2, which="both")
        _panel_label(axis, label)
    figure.tight_layout()
    figure.savefig(FIGURES_KO / "verification.png")
    plt.close(figure)


def main() -> None:
    selected_font = _configure_plotting()
    FIGURES_KO.mkdir(parents=True, exist_ok=True)
    data = _load_scan()
    plot_profiles()
    plot_accuracy_atlas(data)
    plot_charge_comparison(data)
    plot_detectability(data)
    plot_verification()
    outputs = (
        "profiles_and_charge_extraction.png",
        "accuracy_atlas.png",
        "charge_comparison.png",
        "detectability_map.png",
        "verification.png",
    )
    print(f"한국어 그림 5종 생성 완료 (글꼴: {selected_font})")
    for name in outputs:
        print(FIGURES_KO / name)


if __name__ == "__main__":
    main()
