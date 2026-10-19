"""Shared English/Korean scientific figures from one numerical data product."""

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np

from .study_analysis import probe_index, sampled_validity_radii, tail_force


ROOT = Path(__file__).resolve().parent
CASES = ("unscreened", "transition", "screened")
COLORS = ("#2b8cbe", "#f39c34", "#b2182b")
MODEL_COLORS = ("#b2182b", "#762a83", "#ef8a62", "#555555", "#2166ac")
MODEL_KEYS = ("q_kw", "q_center", "q_ms", "q_point_ms", "q_yms")


def _configure(language):
    family = "DejaVu Sans"
    if language == "ko":
        for font in (Path(r"C:\Windows\Fonts\NotoSansCJKkr-Regular.otf"),
                     Path(r"C:\Windows\Fonts\malgun.ttf")):
            if font.exists():
                fm.fontManager.addfont(str(font))
        available = {font.name for font in fm.fontManager.ttflist}
        family = next((name for name in ("Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic")
                       if name in available), None)
        if family is None:
            raise RuntimeError("Korean figures require Noto Sans CJK KR or Malgun Gothic")
    mpl.rcParams.update({"font.family": family, "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8, "figure.dpi": 140,
                         "savefig.dpi": 220, "savefig.bbox": "tight",
                         "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})


def _label(axis, label):
    axis.text(-.12, 1.04, label, transform=axis.transAxes, va="top", fontsize=11)


def _contour(axis, x, y, values, level, **kwargs):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size and finite.min() < level < finite.max():
        axis.contour(x, y, values, levels=[level], **kwargs)


def render_figures(data, metrics, references, language="en"):
    if language not in ("en", "ko"):
        raise ValueError("language must be en or ko")
    _configure(language)
    ko = language == "ko"
    out = ROOT / ("figures_ko" if ko else "figures")
    out.mkdir(exist_ok=True)
    density, mu = data["density_grid"], data["mu_grid"]
    density_label = r"밀도 대비 $\delta=\rho_c/\rho_\infty$" if ko else r"density contrast $\delta=\rho_c/\rho_\infty$"
    radius_label = r"반지름 $x=r/R$" if ko else r"radius $x=r/R$"
    model_labels = (["표면 진폭", "중심점", "유한 부피", "점원천+유한 부피", "활성 껍질 적분"] if ko else
                    ["surface amplitude", "center point", "finite volume", "point + finite volume", "active-shell integral"])
    case_labels = ["비차폐", "전이", "차폐"] if ko else list(CASES)
    description = json.dumps({"language": language,
                              "scan_fingerprint": str(data["fingerprint"]),
                              "threshold": metrics["detectability"]["threshold"],
                              "force_reference": "nonlinear local gradient"})

    def save(figure, name):
        figure.savefig(out / name, metadata={"Description": description})
        plt.close(figure)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for case, label, color in zip(CASES, case_labels, COLORS):
        axes[0].plot(references["profile_x"], references[f"{case}_u"], color=color, lw=2, label=label)
        x = references[f"{case}_charge_x"]
        axes[1].plot(x, references[f"{case}_charge_A"], color=color, lw=2, label=label)
        axes[1].plot(x, references[f"{case}_charge_G"], color=color, lw=1.3, ls="--")
    axes[0].axvline(1, ls=":", color=".4")
    axes[0].set(xlim=(0, 3), xlabel=radius_label, ylabel=r"$u=\phi/\phi_\infty$",
                title="완전한 비선형 장" if ko else "Full nonlinear field")
    axes[1].set(xlabel=radius_label, ylabel=r"$Q(x)$", yscale="log",
                title="꼬리 추출: 진폭(실선), 기울기(점선)" if ko else "Tail extraction: amplitude / gradient dashed")
    for axis, label in zip(axes, "ab"):
        axis.legend(frameon=False)
        axis.grid(alpha=.2)
        _label(axis, label)
    fig.tight_layout()
    save(fig, "profiles_and_charge_extraction.png")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.15), sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-.35, vcenter=0, vmax=.35)
    for axis, key, title, label in zip(axes, ("q_kw", "q_center", "q_yms"),
                                       ([r"표면 진폭 $Q_{\rm surf}$", r"중심점 $Q_{\rm ctr}$", r"활성 껍질 $Q_{\rm AS}$"] if ko else
                                        [r"Surface amplitude $Q_{\rm surf}$", r"Center point $Q_{\rm ctr}$", r"Active shell $Q_{\rm AS}$"]), "abc"):
        image = axis.pcolormesh(mu, density, np.log10(data[key]/data["q_num"]), shading="auto", cmap="RdBu_r", norm=norm)
        _contour(axis, mu, density, abs(data[key]/data["q_num"]-1), .1, colors="black", linewidths=1.4)
        axis.contour(mu, density, data["epsilon"], levels=[.01,.1,.5],
                     colors=["#fdd49e", "#fc8d59", "#b30000"], linewidths=.8, linestyles="--")
        axis.set(xscale="log", yscale="log", xlabel=r"$\mu=m_\infty R$", title=title)
        _label(axis, label)
    axes[0].set_ylabel(density_label)
    cb_axis = fig.add_axes([.918,.18,.014,.68])
    fig.colorbar(image, cax=cb_axis).set_label(r"$\log_{10}(Q_{\rm model}/Q_{\rm num})$")
    fig.text(.49,-.01, r"검은 선: 10% 오차; 점선: $\epsilon_{\rm th}=0.01,0.1,0.5$" if ko else
             r"Black: 10% error; dashed: $\epsilon_{\rm th}=0.01,0.1,0.5$", ha="center")
    fig.subplots_adjust(left=.075,right=.895,bottom=.18,top=.9,wspace=.08)
    save(fig, "accuracy_atlas.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    for key, label, color in zip(MODEL_KEYS, model_labels, MODEL_COLORS):
        if key == "q_ms":
            continue
        axes[0].scatter(data["q_num"], data[key], s=9, alpha=.3, color=color, label=label)
    identity = np.geomspace(data["q_num"].min(),1.1,200)
    axes[0].plot(identity,identity,"k--",lw=1)
    axes[0].set(xscale="log",yscale="log",xlabel=r"$Q_{\rm num}$",
                ylabel="예측 전하" if ko else "predicted charge",
                title="전하 예측 비교" if ko else "Charge prediction")
    epsilon = data["epsilon"].ravel()
    bins = np.geomspace(epsilon.min(),epsilon.max()*(1+1e-12),18)
    centers = np.sqrt(bins[:-1]*bins[1:])
    for key,label,color in zip(MODEL_KEYS,model_labels,MODEL_COLORS):
        error = abs((data[key]/data["q_num"]-1).ravel())
        median, p90 = np.full(centers.size,np.nan), np.full(centers.size,np.nan)
        for i in range(centers.size):
            mask = (epsilon>=bins[i]) & (epsilon<bins[i+1])
            if mask.any():
                median[i],p90[i] = np.median(error[mask]),np.quantile(error[mask],.9)
        axes[1].plot(centers,median,color=color,lw=1.8,ls="--" if key=="q_point_ms" else "-",label=label)
        axes[1].fill_between(centers,median,p90,color=color,alpha=.08)
    axes[1].axhline(.1,color="black",ls=":",lw=1)
    axes[1].set(xscale="log",yscale="log",xlabel=r"$\epsilon_{\rm th}$",
                ylabel="상대 전하 오차" if ko else "relative charge error",
                title="보정 요소를 분리한 대조군" if ko else "Ablation of volume and radial weight")
    for axis,label in zip(axes,"ab"):
        axis.legend(frameon=False,fontsize=7.5)
        axis.grid(alpha=.18,which="both")
        _label(axis,label)
    fig.tight_layout()
    save(fig,"charge_comparison.png")

    direct = data["alpha_direct"][...,probe_index(data)]
    tail = tail_force(data["q_num"],data)
    error = abs(tail/direct-1)
    validity = sampled_validity_radii(data)
    fig,axes = plt.subplots(1,3,figsize=(13.3,4.25))
    image = axes[0].pcolormesh(mu,density,100*error,shading="auto",cmap="YlOrRd",vmin=0)
    _contour(axes[0],mu,density,error,.1,colors="black",linewidths=1.3)
    fig.colorbar(image,ax=axes[0],pad=.02).set_label("국소 힘 오차 (%)" if ko else "local-force error (%)")
    image = axes[1].pcolormesh(mu,density,validity,shading="auto",cmap="viridis",vmin=1,vmax=max(2.5,np.nanmax(validity)))
    fig.colorbar(image,ax=axes[1],pad=.02).set_label(r"$r_{10,\rm sampled}/R$")
    titles = ([r"$1.5R$에서 점근 꼬리의 힘 오차", "10% 목표를 만족하는 첫 표본 거리"] if ko else
              [r"Tail-force error at $1.5R$", "First sampled 10% validity radius"])
    for axis,title,label in zip(axes[:2],titles,"ab"):
        axis.set(xscale="log",yscale="log",xlabel=r"$\mu=m_\infty R$",title=title)
        _label(axis,label)
    axes[0].set_ylabel(density_label)
    rows = metrics["distance_validity"]["by_radius"]
    r = [row["radius_over_R"] for row in rows]
    for key,label,color in (("max_relative_error","최대" if ko else "maximum","#b2182b"),
                             ("p90_relative_error","90백분위" if ko else "90th percentile","#ef8a62"),
                             ("median_relative_error","중앙값" if ko else "median","#2166ac")):
        axes[2].plot(r,[100*row[key] for row in rows],label=label,color=color,lw=2)
    axes[2].axhline(10,color="black",ls="--",lw=1,label="10% 목표" if ko else "10% target")
    axes[2].set(xscale="log",yscale="log",xlabel=r"$r/R$",
                ylabel="꼬리 공식의 힘 오차 (%)" if ko else "tail-force error (%)",
                title="거리별 오차" if ko else "Error versus distance")
    axes[2].legend(frameon=False,fontsize=8)
    axes[2].grid(alpha=.2,which="both")
    _label(axes[2],"c")
    fig.subplots_adjust(left=.065,right=.98,bottom=.17,top=.89,wspace=.42)
    save(fig,"force_validity_map.png")

    threshold = metrics["detectability"]["threshold"]
    surface,center = tail_force(data["q_kw"],data),tail_force(data["q_center"],data)
    truth,s,c = direct>=threshold,surface>=threshold,center>=threshold
    classification = np.zeros(truth.shape,dtype=int)
    classification[(s==c)&truth&s] = 1
    classification[s!=c] = 2
    classification[(s==c)&(s!=truth)] = 3
    bias = np.maximum(abs(surface-direct),abs(center-direct))/threshold
    fig,axes = plt.subplots(1,3,figsize=(13.4,4.25),sharex=True,sharey=True)
    image = axes[0].pcolormesh(mu,density,np.log10(direct),shading="auto",cmap="viridis")
    lines = ((direct,"white","-","직접 기울기" if ko else "direct gradient"),
             (tail,"#ff88dd","-.","수치 전하의 꼬리" if ko else "numerical-charge tail"),
             (surface,"#ffcc33","--",model_labels[0]),(center,"#66c2ff",":",model_labels[1]))
    for values,color,style,label in lines:
        _contour(axes[0],mu,density,values,threshold,colors=color,linewidths=1.5,linestyles=style)
        axes[0].plot([],[],color=color,ls=style,lw=1.5,label=label)
    axes[0].legend(frameon=False,loc="lower left",fontsize=6.8)
    fig.colorbar(image,ax=axes[0],pad=.02).set_label(r"$\log_{10}\alpha_{\rm direct}$")
    cmap = ListedColormap(["#d9d9d9","#41ab5d","#ef8a62","#67a9cf"])
    image = axes[1].pcolormesh(mu,density,classification,shading="auto",cmap=cmap,
                              norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],cmap.N))
    cb = fig.colorbar(image,ax=axes[1],ticks=[0,1,2,3],pad=.02)
    cb.ax.set_yticklabels(["모두 미만","모두 이상","두 근사 불일치","두 근사 오판"] if ko else
                          ["all below","all above","models differ","both wrong"])
    image = axes[2].pcolormesh(mu,density,np.log10(np.maximum(bias,1e-5)),shading="auto",cmap="magma")
    _contour(axes[2],mu,density,bias,1,colors="white",linewidths=1.3)
    fig.colorbar(image,ax=axes[2],pad=.02).set_label(r"$\log_{10}(\max|\alpha_{\rm model}-\alpha_{\rm direct}|/\alpha_\star)$")
    titles = (["직접 계산한 국소 힘비", "직접 힘을 기준으로 한 판정", "임계값 단위의 근사 편향"] if ko else
              ["Direct nonlinear local force", "Decisions against direct force", "Approximation bias / threshold"])
    for axis,title,label in zip(axes,titles,"abc"):
        axis.set(xscale="log",yscale="log",xlabel=r"$\mu=m_\infty R$",title=title)
        _label(axis,label)
    axes[0].set_ylabel(density_label)
    fig.subplots_adjust(left=.065,right=.98,bottom=.16,top=.9,wspace=.35)
    save(fig,"detectability_map.png")

    v = metrics["verification"]
    fig,axes = plt.subplots(1,3,figsize=(13,3.95))
    axes[0].plot(v["weak_mus"],v["weak_analytic"],color="black",label="해석적 균일 구" if ko else "analytic sphere")
    axes[0].scatter(v["weak_mus"],v["weak_numeric"],color=COLORS[0],s=30,zorder=3,label="비선형 BVP" if ko else "nonlinear BVP")
    axes[0].set(xscale="log",xlabel=r"$\mu=m_\infty R$",ylabel=r"$Q$",
                title="약한 원천의 해석적 극한" if ko else "Weak-source analytic limit")
    charge = np.asarray(v["domain_charge"])
    tol = np.asarray(v["tolerance_charge"])
    axes[1].plot([0,1,2],abs(charge[:3]/charge[-1]-1),"o-",label=r"$x_{\max}=40,80,160$")
    axes[1].plot([0,1],abs(tol[:2]/tol[-1]-1),"s-",label=r"$\mathrm{tol}=3\cdot10^{-4},3\cdot10^{-5}$")
    axes[1].set_xticks([0,1,2],["낮음","중간","높음"] if ko else ["coarse","medium","fine"])
    axes[1].set(yscale="log",xlabel="해상도 단계" if ko else "resolution level",
                ylabel="최고 해상도 대비 상대 변화" if ko else "relative change from finest",
                title="수치 수렴" if ko else "Numerical convergence")
    axes[2].plot(references["plateau_x"],references["plateau_A"],label="진폭" if ko else "amplitude")
    axes[2].plot(references["plateau_x"],references["plateau_G"],ls="--",label="기울기" if ko else "gradient")
    axes[2].set(xlabel=radius_label,ylabel=r"$Q(x)$",title="꼬리 추정량 교차검사" if ko else "Tail-estimator cross-check")
    for axis,label in zip(axes,"abc"):
        axis.legend(frameon=False,fontsize=8)
        axis.grid(alpha=.2)
        _label(axis,label)
    fig.tight_layout()
    save(fig,"verification.png")
    print(f"rendered 6 figures: {out}",flush=True)
