import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve

# =====================================================================
# 1. 시뮬레이션 물리 매개변수 설정 (무차원 단위계)
# =====================================================================
N_POWER = 1            # Ratra-Peebles 포텐셜 지수 (n)
MASS_SCALE = 1.0       # 포텐셜 에너지 스케일 (M)
BETA = 1.0             # 물질 결합 상수 (beta)
M_PL = 1.0             # 축소 플랑크 질량 (M_Pl)

# 공간 도메인 설정 (x = -10에서 10까지)
X_MIN = -10.0
X_MAX = 10.0
X_GRID_POINTS = 500
x_span = np.linspace(X_MIN, X_MAX, X_GRID_POINTS)

# 밀도 구배 (Density Gradient) 매개변수
RHO_HIGH = 10.0        # 고밀도 영역 (좌측: 예: 별 내부, 지구 표면)
RHO_LOW = 0.01         # 저밀도 영역 (우측: 예: 우주 공간)
TRANSITION_WIDTH = 1.5 # 전이 영역 두께 (w)
TRANSITION_CENTER = 0.0# 전이 중심점

# 1차원 공간 밀도 함수 rho(x)
def rho(x):
    return 0.5 * (RHO_HIGH + RHO_LOW) - 0.5 * (RHO_HIGH - RHO_LOW) * np.tanh((x - TRANSITION_CENTER) / TRANSITION_WIDTH)

# =====================================================================
# 2. 경계 조건용 유효 포텐셜 극솟값 구하기 (fsolve 사용)
# =====================================================================
def get_phi_min(density):
    # dV_eff/d_phi = -n * M^(4+n) / phi^(n+1) + (beta * rho / M_Pl) * e^(beta * phi / M_Pl) = 0
    # 무차원화 식: -1/phi^2 + density * e^phi = 0
    func = lambda phi: -1.0 / (phi**2) + density * np.exp(phi)
    
    # 초깃값 추정 (밀도에 따라 역비례하는 분석적 근사치)
    initial_guess = (1.0 / density) ** (1.0 / (N_POWER + 1.0))
    phi_sol = fsolve(func, initial_guess)
    return phi_sol[0]

# 경계 조건용 필드값 계산
phi_left = get_phi_min(RHO_HIGH)
phi_right = get_phi_min(RHO_LOW)

print(f"[물리 경계 조건 결정]")
print(f" - 좌측 고밀도 경계 (x={X_MIN}): rho = {RHO_HIGH} => phi_min = {phi_left:.5f}")
print(f" - 우측 저밀도 경계 (x={X_MAX}): rho = {RHO_LOW} => phi_min = {phi_right:.5f}")

# =====================================================================
# 3. scipy.integrate.solve_bvp를 활용한 Chameleon Field 계산
# =====================================================================
# 2차 ODE인 phi'' = -1/phi^2 + rho(x) * e^phi 를 1차 연립 미분방정식으로 변환
# y = [phi, phi'] 라 두면:
# dy/dx = [y[1], -1/y[0]^2 + rho(x) * e^y[0]]
def chameleon_ode(x, y):
    phi = y[0]
    dphi_dx = y[1]
    
    # 0 근처 또는 음수로 떨어지는 것을 방지하여 안정성 확보
    phi_safe = np.clip(phi, 1e-5, None)
    
    d2phi_dx2 = -1.0 / (phi_safe**2) + rho(x) * np.exp(phi_safe)
    return np.vstack((dphi_dx, d2phi_dx2))

# 경계 조건 정의
# bc(y(a), y(b)) = 0 형태로 제공해야 함
def boundary_conditions(ya, yb):
    return np.array([
        ya[0] - phi_left,   # x = X_MIN 에서의 phi 조건
        yb[0] - phi_right   # x = X_MAX 에서의 phi 조건
    ])

# BVP 솔버 초기화 및 실행
# 초기 기댓값(Guess)은 경계 조건 사이를 매끄럽게 연결하는 선형 함수로 설정
y_guess = np.zeros((2, X_GRID_POINTS))
y_guess[0, :] = np.linspace(phi_left, phi_right, X_GRID_POINTS)
y_guess[1, :] = (phi_right - phi_left) / (X_MAX - X_MIN)

bvp_result = solve_bvp(chameleon_ode, boundary_conditions, x_span, y_guess)

if bvp_result.success:
    print("[성공] 카멜레온 장 경계값 문제(BVP)의 수치 수렴에 성공하였습니다.")
else:
    print("[실패] 수치 수렴에 실패하였습니다. 메시지:", bvp_result.message)

# 보간 함수 생성 (입자 궤적 ODE 솔버가 임의의 x 위치에서 필드와 기울기를 읽을 수 있도록 함)
phi_interpolated = bvp_result.sol

# =====================================================================
# 4. 제5의 힘 F5(x) 및 입자 운동 시뮬레이션
# =====================================================================
# 제5의 힘 F5(x) = - beta * dphi/dx
def get_fifth_force(x):
    # solve_bvp 솔버의 보간해로부터 phi' = y[1] 값을 직접 호출
    y_val = phi_interpolated(x)
    dphi_dx = y_val[1]
    return -BETA * dphi_dx

# 배경 고전적 인력 (조화 진동자 퍼텐셜, k=0.15)
GRAVITY_K = 0.15

# 입자의 운동 방정식 정의 (IVP)
# u = [position_x, velocity_v]
# du/dt = [v, -k*x + F5(x)]
def particle_dynamics(t, u, enable_fifth_force=True):
    pos_x = u[0]
    vel_v = u[1]
    
    # 도메인 탈출 시 경계값에 고정하여 에러 방지
    pos_x_clipped = np.clip(pos_x, X_MIN, X_MAX)
    
    # 가속도 계산
    a_gravity = -GRAVITY_K * pos_x
    a_fifth = get_fifth_force(pos_x_clipped) if enable_fifth_force else 0.0
    
    total_accel = a_gravity + a_fifth
    return [vel_v, total_accel]

# 입자 초기 상태 설정 및 시간 축 정의
X_START = -8.0         # 고밀도(스크리닝) 영역 내부에서 출발
V_START = 3.5          # 우측 저밀도 영역으로 돌파하기 위한 양의 초속도
t_span = (0.0, 35.0)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
u_init = [X_START, V_START]

# 1) 대조군 시뮬레이션: 오직 뉴턴 중력(조화 진동자)만 존재할 때
sol_newton = solve_ivp(
    fun=lambda t, u: particle_dynamics(t, u, enable_fifth_force=False),
    t_span=t_span,
    y0=u_init,
    t_eval=t_eval,
    method='RK45'
)

# 2) 실험군 시뮬레이션: 뉴턴 중력 + 카멜레온 제5의 힘이 융합되었을 때
sol_chameleon = solve_ivp(
    fun=lambda t, u: particle_dynamics(t, u, enable_fifth_force=True),
    t_span=t_span,
    y0=u_init,
    t_eval=t_eval,
    method='RK45'
)

print("[성공] 시험 입자의 두 가지 대조 시나리오 궤적 연산을 완료하였습니다.")

# =====================================================================
# 5. 미려한 4분할 시각화 대시보드 플로팅
# =====================================================================
# 고급스러운 다크 모드 스타일 설정
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=120)
fig.suptitle("Environmental Density-Dependent Dynamics Simulation\nModeling the Fifth Force through 'Chameleon Screening'", 
             fontsize=18, fontweight='bold', color='#E0E0E0', y=0.96)

# [Plot 1] 공간 밀도 분포 rho(x) 및 카멜레온 장 phi(x) 대조
color_rho = '#FF6B6B' # Coral Red
color_phi = '#10AC84' # Emerald Green

ax1 = axs[0, 0]
ax1.set_title("1. Density Profile vs. Chameleon Scalar Field", fontsize=12, fontweight='bold', pad=10)
line1 = ax1.plot(x_span, rho(x_span), color=color_rho, linewidth=2.5, label=r"Density $\rho(x)$")
ax1.set_xlabel("Spatial Coordinate $x$", color='#CCCCCC')
ax1.set_ylabel(r"Density $\rho$", color=color_rho, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_rho)
ax1.grid(True, alpha=0.15, linestyle='--')

# 우측 y축을 활용하여 장의 분포 phi(x) 시각화
ax1_twin = ax1.twinx()
# solve_bvp가 내부적으로 노드를 추가하므로, 통일된 x_span에서의 보간해를 플로팅합니다.
line2 = ax1_twin.plot(x_span, phi_interpolated(x_span)[0], color=color_phi, linewidth=2.5, label=r"Chameleon Field $\phi(x)$")
ax1_twin.set_ylabel(r"Chameleon Field $\phi$", color=color_phi, fontsize=11)
ax1_twin.tick_params(axis='y', labelcolor=color_phi)

# 범례(Legend) 통합
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', framealpha=0.8, edgecolor='#444444')

# [Plot 2] 공간 좌표에 따른 제5의 힘 F5(x) 프로파일
ax2 = axs[0, 1]
ax2.set_title("2. Manifestation of the Fifth Force $F_5(x)$", fontsize=12, fontweight='bold', pad=10)
f5_vals = [get_fifth_force(xi) for xi in x_span]
ax2.plot(x_span, f5_vals, color='#FFD2FF', linewidth=2.5, label=r"Fifth Force $F_5(x) \propto -\nabla\phi$")
# 경계 영역을 시각적으로 강조
ax2.axvspan(-TRANSITION_WIDTH*2, TRANSITION_WIDTH*2, color='#FFEEAA', alpha=0.07, label="Transition Zone")
ax2.set_xlabel("Spatial Coordinate $x$", color='#CCCCCC')
ax2.set_ylabel("Force Magnitude $F_5$", color='#E0E0E0', fontsize=11)
ax2.grid(True, alpha=0.15, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.8, edgecolor='#444444')

# [Plot 3] 시간에 따른 시험 입자의 1차원 위치 x(t) 변화 대조
ax3 = axs[1, 0]
ax3.set_title("3. Test Particle Position Over Time $x(t)$", fontsize=12, fontweight='bold', pad=10)
ax3.plot(sol_newton.t, sol_newton.y[0], color='#54A0FF', linewidth=2.0, linestyle='--', label="Classical Gravity Only (Newton)")
ax3.plot(sol_chameleon.t, sol_chameleon.y[0], color='#00D2D3', linewidth=2.5, label="Newton + Chameleon Fifth Force")
ax3.axhline(0.0, color='#666666', linestyle=':', alpha=0.5)
ax3.text(-0.5, -4, 'Screened Zone (High Density)', rotation=90, verticalalignment='center', color='#888888', fontsize=9)
ax3.text(1.5, 4, 'Unscreened Zone (Low Density)', rotation=90, verticalalignment='center', color='#888888', fontsize=9)
ax3.set_xlabel("Time $t$", color='#CCCCCC')
ax3.set_ylabel("Particle Position $x$", color='#E0E0E0', fontsize=11)
ax3.grid(True, alpha=0.15, linestyle='--')
ax3.legend(loc='lower right', framealpha=0.8, edgecolor='#444444')

# [Plot 4] 위상 공간 (Phase Space, x - v) 상에서의 궤적 왜곡 양상
ax4 = axs[1, 1]
ax4.set_title("4. Dynamical Trajectory in Phase Space ($x - v$)", fontsize=12, fontweight='bold', pad=10)
ax4.plot(sol_newton.y[0], sol_newton.y[1], color='#54A0FF', linewidth=2.0, linestyle='--', label="Newton (Harmonic Orbit)")
ax4.plot(sol_chameleon.y[0], sol_chameleon.y[1], color='#FF9F43', linewidth=2.5, label="Chameleon Perturbed Orbit")
# 시작점과 운동 방향 표시
ax4.scatter(X_START, V_START, color='#FF5252', s=80, zorder=5, label="Start State ($x_0, v_0$)")
ax4.set_xlabel("Position $x$", color='#CCCCCC')
ax4.set_ylabel("Velocity $v$", color='#E0E0E0', fontsize=11)
ax4.grid(True, alpha=0.15, linestyle='--')
ax4.legend(loc='upper right', framealpha=0.8, edgecolor='#444444')

# 레이아웃 정돈 및 이미지 저장
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
save_path = "c:/Users/k0104/Downloads/역학_Term_Project/chameleon_simulation_results.png"
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"[성공] 고해상도 시뮬레이션 결과 대시보드 이미지가 다음 경로에 저장되었습니다:\n -> {save_path}")
