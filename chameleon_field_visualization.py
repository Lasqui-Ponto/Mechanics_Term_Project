import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from pathlib import Path

from chameleon_numerics import phi_min

# =====================================================================
# 1. 구형 대칭 천체 모델 물리 매개변수 설정
# =====================================================================
N_POWER = 1            # Ratra-Peebles 지수 (n)
MASS_SCALE = 1.0       # 포텐셜 에너지 스케일 (M)
BETA = 1.0             # 결합 상수 (beta)
M_PL = 1.0             # 축소 플랑크 질량 (M_Pl)

# 구형 대칭 천체 반지름 및 밀도 설정
BODY_RADIUS = 3.5      # 천체 반지름 (R)
RHO_BODY = 8.0         # 천체 내부 밀도 (고밀도, rho_body)
RHO_VACUUM = 0.01      # 천체 외부 우주 밀도 (저밀도, rho_vacuum)
TRANSITION_WIDTH = 0.5 # 표면 전이 영역 두께 (w)

# r 격자 설정 (특이점 r=0을 방지하기 위해 1e-4에서 시작)
R_MIN = 1e-4
R_MAX = 10.0
R_GRID_POINTS = 600
r_span = np.linspace(R_MIN, R_MAX, R_GRID_POINTS)

# 구형 천체의 반경별 밀도 함수 rho(r)
def rho_spherical(r):
    return 0.5 * (RHO_BODY + RHO_VACUUM) - 0.5 * (RHO_BODY - RHO_VACUUM) * np.tanh((r - BODY_RADIUS) / TRANSITION_WIDTH)

# =====================================================================
# 2. 경계 조건용 유효 포텐셜 극솟값 구하기
# =====================================================================
def get_phi_min(density):
    return phi_min(density)

phi_body_center = get_phi_min(RHO_BODY)
phi_vacuum_far = get_phi_min(RHO_VACUUM)

# =====================================================================
# 3. 구형 대칭 카멜레온 BVP 방정식 풀이
# =====================================================================
# 구형 대칭 하에서 클라인-고든 방정식:
# d2phi/dr2 + (2/r)*dphi/dr = -1/phi^2 + rho(r)*e^phi
# y = [phi, dphi/dr] 라 두면:
# dy/dr = [ y[1], -(2/r)*y[1] - 1/y[0]^2 + rho(r)*e^y[0] ]
def spherical_chameleon_ode(r, y):
    phi = y[0]
    dphi_dr = y[1]
    
    phi_safe = np.clip(phi, 1e-5, None)
    
    # 2/r 항 연산 시 r=0 부근 특이점 방지를 위해 r을 최솟값 이상으로 클리핑
    r_safe = np.clip(r, R_MIN, None)
    
    d2phi_dr2 = -(2.0 / r_safe) * dphi_dr - 1.0 / (phi_safe**2) + rho_spherical(r) * np.exp(phi_safe)
    return np.vstack((dphi_dr, d2phi_dr2))

# 구형 대칭 경계 조건:
# 1) 천체 중심 r=0 에서 대칭성에 의해 dphi/dr = 0
# 2) 우주 원거리 r=R_MAX 에서 phi = phi_vacuum_far (저밀도 진공 평탄해)
def spherical_boundary_conditions(ya, yb):
    return np.array([
        ya[1] - 0.0,                 # r_min 에서 dphi/dr = 0
        yb[0] - phi_vacuum_far       # r_max 에서 phi = phi_min(rho_low)
    ])

# 초기 추정 프로파일 설정
y_guess = np.zeros((2, R_GRID_POINTS))
y_guess[0, :] = np.linspace(phi_body_center, phi_vacuum_far, R_GRID_POINTS)
y_guess[1, :] = (phi_vacuum_far - phi_body_center) / (R_MAX - R_MIN)

# BVP 연산 수행
bvp_result = solve_bvp(spherical_chameleon_ode, spherical_boundary_conditions, r_span, y_guess)

if bvp_result.success:
    print("[성공] 구형 대칭 카멜레온 장의 수치 솔버가 완벽하게 수렴하였습니다.")
else:
    print("[실패] 수치 솔버가 수렴하지 못했습니다. 메시지:", bvp_result.message)

phi_sol_func = bvp_result.sol

# =====================================================================
# 4. 정밀 시각화 그래프 그리기 (1D 상세 분석 & 2D Radial Heatmap)
# =====================================================================
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=120)
fig.suptitle("Radial Distribution and Screening of the Chameleon Scalar Field", 
             fontsize=16, fontweight='bold', color='#E0E0E0', y=0.96)

# ---------------------------------------------------------------------
# [Graph 1] 1D Radial Profile with Physical Annotations
# ---------------------------------------------------------------------
r_eval = np.linspace(R_MIN, R_MAX, 1000)
phi_eval = phi_sol_func(r_eval)[0]
rho_eval = rho_spherical(r_eval)

# 좌측 y축: 카멜레온 장 phi(r)
color_phi = '#10AC84' # Emerald Green
ax1.plot(r_eval, phi_eval, color=color_phi, linewidth=3.0, label=r"Chameleon Field $\phi(r)$")
ax1.fill_between(r_eval, phi_eval, color=color_phi, alpha=0.1)
ax1.set_xlabel("Radial Distance from Center $r$", fontsize=12, color='#CCCCCC')
ax1.set_ylabel(r"Chameleon Field Value $\phi$", color=color_phi, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_phi)
ax1.grid(True, alpha=0.15, linestyle='--')

# 우측 y축: 밀도 분포 rho(r)
ax1_twin = ax1.twinx()
color_rho = '#FF6B6B' # Coral Red
ax1_twin.plot(r_eval, rho_eval, color=color_rho, linewidth=2.0, linestyle=':', label=r"Density Profile $\rho(r)$")
ax1_twin.set_ylabel(r"Matter Density $\rho$", color=color_rho, fontsize=12)
ax1_twin.tick_params(axis='y', labelcolor=color_rho)

# 천체 반경 경계선 시각화
ax1.axvline(BODY_RADIUS, color='#FFEEAA', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.text(BODY_RADIUS - 0.2, 1.5, "Star Surface\n(r = R)", color='#FFEEAA', 
         fontsize=10, fontweight='bold', horizontalalignment='right')

# 물리 현상 상세 주석(Annotation) 추가
# 1) 천체 내부 차폐 영역
ax1.annotate("Screened Core (High Mass)\n" + r"$m_{\mathrm{eff}} \gg 1 \rightarrow \lambda_c \approx 0$",
             xy=(1.5, phi_body_center + 0.1), xytext=(1.0, 0.9),
             arrowprops=dict(facecolor='#54A0FF', shrink=0.08, width=1.5, headwidth=6),
             color='#54A0FF', fontsize=10, fontweight='bold')

# 2) 천체 외부 자유 영역
ax1.annotate("Free Space (Light Mass)\n" + r"$m_{\mathrm{eff}} \ll 1 \rightarrow \lambda_c \rightarrow \infty$",
             xy=(8.0, phi_vacuum_far - 0.1), xytext=(5.2, 2.0),
             arrowprops=dict(facecolor='#FF9F43', shrink=0.08, width=1.5, headwidth=6),
             color='#FF9F43', fontsize=10, fontweight='bold')

ax1.set_title("1. Radial Field Profile & Screening Boundaries", fontsize=13, fontweight='bold', pad=12)

# ---------------------------------------------------------------------
# [Graph 2] 2D Spherical Screened Field Map (2차원 단면 시각화)
# ---------------------------------------------------------------------
# 2D 그리드 생성
grid_size = 300
x_coord = np.linspace(-R_MAX, R_MAX, grid_size)
y_coord = np.linspace(-R_MAX, R_MAX, grid_size)
X, Y = np.meshgrid(x_coord, y_coord)
R_2d = np.sqrt(X**2 + Y**2)

# BVP 1D 솔루션을 2D 반지름 격자에 매핑
# R_2d 격자 중 R_MAX를 초과하는 외부는 최대 반경 해로 클리핑하여 보간 에러 방지
R_2d_clipped = np.clip(R_2d, R_MIN, R_MAX)
PHI_2d = phi_sol_func(R_2d_clipped)[0]

# 등고선 채우기(contourf)를 활용한 2차원 히트맵 플로팅
contour = ax2.contourf(X, Y, PHI_2d, levels=80, cmap='viridis')
cbar = fig.colorbar(contour, ax=ax2, pad=0.03)
cbar.set_label(r"Chameleon Field Strength $\phi$", color='#E0E0E0', fontsize=11)
cbar.ax.tick_params(labelsize=9)

# 구형 천체의 경계면을 동그란 실선으로 플로팅
star_circle = plt.Circle((0, 0), BODY_RADIUS, color='#FFFFFF', fill=False, 
                         linestyle='--', linewidth=2.0, alpha=0.8, label="Star Surface")
ax2.add_patch(star_circle)

ax2.set_xlim(-R_MAX, R_MAX)
ax2.set_ylim(-R_MAX, R_MAX)
ax2.set_aspect('equal')
ax2.set_xlabel("Spatial Coordinate $x$", fontsize=11, color='#CCCCCC')
ax2.set_ylabel("Spatial Coordinate $y$", fontsize=11, color='#CCCCCC')
ax2.set_title("2. 2D Sectional Heatmap of Spherical Screening", fontsize=13, fontweight='bold', pad=12)
ax2.legend(loc='upper right', framealpha=0.8, edgecolor='#545454')

# 중앙 텍스트 라벨 추가 (천체 내부와 외부)
ax2.text(0, 0, "Core\n(Screened)", color='#FFFFFF', fontsize=11, fontweight='bold', 
         horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='black', alpha=0.4, boxstyle='round'))
ax2.text(7.0, 7.0, "Vacuum\n(Unscreened)", color='#10AC84', fontsize=11, fontweight='bold', 
         horizontalalignment='center', verticalalignment='center')

# 전체 레이아웃 조정 및 저장
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
save_path = Path(__file__).with_name("chameleon_field_distribution.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"[성공] 카멜레온 장의 공간 상세 시각화 이미지가 다음 경로에 저장되었습니다:\n -> {save_path}")
