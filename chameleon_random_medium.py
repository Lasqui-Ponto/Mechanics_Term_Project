import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter1d

# =====================================================================
# 1. 무작위 불균일 매질 물리 매개변수 설정 (무차원 단위계)
# =====================================================================
N_POWER = 1            # Ratra-Peebles 지수
BETA = 1.0             # 결합 상수
M_PL = 1.0             # 플랑크 질량

# 공간 도메인 설정
X_MIN = -10.0
X_MAX = 10.0
X_GRID_POINTS = 600
x_span = np.linspace(X_MIN, X_MAX, X_GRID_POINTS)

# 가우시안 무작위 밀도 분포(Gaussian Disordered Medium) 생성
np.random.seed(137) # 파인만 상수를 기리는 재현성 시드 고정
raw_noise = np.random.normal(0.0, 1.2, size=X_GRID_POINTS)

# 미시 노이즈를 물리적으로 타당한 공간 요동(Clumps)으로 변환하기 위해 가우시안 평활화(sigma=18) 적용
smooth_noise = gaussian_filter1d(raw_noise, sigma=18)

# 밀도 값을 0.01~15.0 사이로 조정 (평균 밀도 약 2.5)
rho_values = np.clip(2.5 + smooth_noise * 12.0, 0.01, 15.0)

# 보간용 연속 밀도 함수 정의
def rho_random(x):
    # 격자상 해와 매칭
    return np.interp(x, x_span, rho_values)

# =====================================================================
# 2. 경계 조건용 유효 포텐셜 극솟값 구하기
# =====================================================================
def get_phi_min(density):
    func = lambda phi: -1.0 / (phi**2) + density * np.exp(phi)
    initial_guess = (1.0 / density) ** (1.0 / (N_POWER + 1.0))
    phi_sol = fsolve(func, initial_guess)
    return phi_sol[0]

# 양끝 경계에서의 밀도값에 따른 경계치 도출
phi_left = get_phi_min(rho_values[0])
phi_right = get_phi_min(rho_values[-1])

print(f"[무작위 불균일계 경계 조건]")
print(f" - 좌측 경계 밀도 rho(-10) = {rho_values[0]:.4f} => phi_min = {phi_left:.5f}")
print(f" - 우측 경계 밀도 rho(10) = {rho_values[-1]:.4f} => phi_min = {phi_right:.5f}")

# =====================================================================
# 3. scipy.integrate.solve_bvp를 활용한 Chameleon Field BVP 풀이
# =====================================================================
# dy/dx = [y[1], -1/y[0]^2 + rho_random(x) * e^y[0]]
def random_chameleon_ode(x, y):
    phi = y[0]
    dphi_dx = y[1]
    phi_safe = np.clip(phi, 1e-5, None)
    
    d2phi_dx2 = -1.0 / (phi_safe**2) + rho_random(x) * np.exp(phi_safe)
    return np.vstack((dphi_dx, d2phi_dx2))

def boundary_conditions(ya, yb):
    return np.array([
        ya[0] - phi_left,
        yb[0] - phi_right
    ])

# 초기 추정 프로파일
y_guess = np.zeros((2, X_GRID_POINTS))
y_guess[0, :] = np.linspace(phi_left, phi_right, X_GRID_POINTS)
y_guess[1, :] = (phi_right - phi_left) / (X_MAX - X_MIN)

bvp_result = solve_bvp(random_chameleon_ode, boundary_conditions, x_span, y_guess)

if bvp_result.success:
    print("[성공] 무작위 매질에서의 카멜레온 장 BVP 수치 솔버가 정상 수렴하였습니다.")
else:
    print("[실패] 수치 수렴에 실패하였습니다. 오류:", bvp_result.message)

phi_interpolated = bvp_result.sol

# =====================================================================
# 4. 제5의 힘 F5(x) 및 카오스 입자 궤적 시뮬레이션
# =====================================================================
def get_fifth_force(x):
    y_val = phi_interpolated(x)
    dphi_dx = y_val[1]
    return -BETA * dphi_dx

# 배경 조화 진동 복원력 상수 (k=0.15)
GRAVITY_K = 0.15

def particle_dynamics(t, u, enable_fifth_force=True):
    pos_x = u[0]
    vel_v = u[1]
    
    pos_x_clipped = np.clip(pos_x, X_MIN, X_MAX)
    
    a_gravity = -GRAVITY_K * pos_x
    a_fifth = get_fifth_force(pos_x_clipped) if enable_fifth_force else 0.0
    
    return [vel_v, a_gravity + a_fifth]

# 입자 초기값 설정 (x = -7.5에서 속도 4.2로 발사)
X_START = -7.5
V_START = 4.2
t_span = (0.0, 45.0)
t_eval = np.linspace(t_span[0], t_span[1], 1200)
u_init = [X_START, V_START]

# 1) 대조군 (뉴턴 조화 진동자만 작동)
sol_newton = solve_ivp(
    fun=lambda t, u: particle_dynamics(t, u, enable_fifth_force=False),
    t_span=t_span,
    y0=u_init,
    t_eval=t_eval,
    method='RK45'
)

# 2) 실험군 (뉴턴 + 무작위 매질 카멜레온 제5의 힘 융합)
sol_chameleon = solve_ivp(
    fun=lambda t, u: particle_dynamics(t, u, enable_fifth_force=True),
    t_span=t_span,
    y0=u_init,
    t_eval=t_eval,
    method='RK45'
)

print("[성공] 무작위 매질 속 카오스 궤적 입자 동역학 연산을 완료하였습니다.")

# =====================================================================
# 5. 미려한 4분할 시각화 대시보드 플로팅
# =====================================================================
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=120)
fig.suptitle("Dynamical Chaos in Disordered Medium\nChameleon Screening Effect with Random Density Fields", 
             fontsize=18, fontweight='bold', color='#E0E0E0', y=0.96)

# [Plot 1] 무작위 공간 밀도 rho(x) 및 반응하는 카멜레온 장 phi(x)
color_rho = '#FF7675'
color_phi = '#55E6C1'

ax1 = axs[0, 0]
ax1.set_title("1. Disordered Matter Density vs. Chameleon Field", fontsize=12, fontweight='bold', pad=10)
line1 = ax1.plot(x_span, rho_values, color=color_rho, linewidth=2.0, alpha=0.9, label=r"Random Density $\rho_R(x)$")
ax1.set_xlabel("Spatial Coordinate $x$", color='#CCCCCC')
ax1.set_ylabel(r"Disordered Density $\rho$", color=color_rho, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_rho)
ax1.grid(True, alpha=0.12, linestyle='--')

ax1_twin = ax1.twinx()
phi_eval = phi_interpolated(x_span)[0]
line2 = ax1_twin.plot(x_span, phi_eval, color=color_phi, linewidth=2.5, label=r"Chameleon Field $\phi(x)$")
ax1_twin.set_ylabel(r"Chameleon Field $\phi$", color=color_phi, fontsize=11)
ax1_twin.tick_params(axis='y', labelcolor=color_phi)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', framealpha=0.8, edgecolor='#444444')

# [Plot 2] 무작위 요동 매질이 만드는 제5의 힘 F5(x) 장벽들
ax2 = axs[0, 1]
ax2.set_title("2. Disordered Barrier of the Fifth Force $F_5(x)$", fontsize=12, fontweight='bold', pad=10)
f5_vals = [get_fifth_force(xi) for xi in x_span]
ax2.plot(x_span, f5_vals, color='#D284FF', linewidth=2.2, label=r"Disordered Force $F_5(x) \propto -\nabla\phi$")
ax2.axhline(0.0, color='#666666', linestyle=':', alpha=0.5)
ax2.set_xlabel("Spatial Coordinate $x$", color='#CCCCCC')
ax2.set_ylabel("Force Magnitude $F_5$", color='#E0E0E0', fontsize=11)
ax2.grid(True, alpha=0.12, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.8, edgecolor='#444444')

# [Plot 3] 시간 경과에 따른 입자 궤적 x(t) 대조 (산란 및 카오스적 섭동)
ax3 = axs[1, 0]
ax3.set_title("3. Test Particle Position Over Time $x(t)$", fontsize=12, fontweight='bold', pad=10)
ax3.plot(sol_newton.t, sol_newton.y[0], color='#54A0FF', linewidth=1.8, linestyle='--', label="Newton Harmonic Orbit")
ax3.plot(sol_chameleon.t, sol_chameleon.y[0], color='#FF7675', linewidth=2.2, label="Chameleon Random Medium Orbit")
ax3.axhline(0.0, color='#666666', linestyle=':', alpha=0.5)
ax3.set_xlabel("Time $t$", color='#CCCCCC')
ax3.set_ylabel("Particle Position $x$", color='#E0E0E0', fontsize=11)
ax3.grid(True, alpha=0.12, linestyle='--')
ax3.legend(loc='lower right', framealpha=0.8, edgecolor='#444444')

# [Plot 4] 위상 공간 (Phase Space, x-v) 상의 일그러진 카오스적 궤도
ax4 = axs[1, 1]
ax4.set_title("4. Orbit Deformation and Chaos in Phase Space ($x - v$)", fontsize=12, fontweight='bold', pad=10)
ax4.plot(sol_newton.y[0], sol_newton.y[1], color='#54A0FF', linewidth=1.8, linestyle='--', label="Newton Harmonic (Stable)")
ax4.plot(sol_chameleon.y[0], sol_chameleon.y[1], color='#F1C40F', linewidth=2.2, label="Chameleon Perturbed (Chaotic)")
ax4.scatter(X_START, V_START, color='#FF5252', s=80, zorder=5, label="Launch State ($x_0, v_0$)")
ax4.set_xlabel("Position $x$", color='#CCCCCC')
ax4.set_ylabel("Velocity $v$", color='#E0E0E0', fontsize=11)
ax4.grid(True, alpha=0.12, linestyle='--')
ax4.legend(loc='upper right', framealpha=0.8, edgecolor='#444444')

# 레이아웃 정리 및 저장
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
save_path = "c:/Users/k0104/Downloads/역학_Term_Project/chameleon_random_medium_results.png"
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"[성공] 무작위 매질 시뮬레이션 결과 대시보드 이미지가 다음 경로에 저장되었습니다:\n -> {save_path}")
