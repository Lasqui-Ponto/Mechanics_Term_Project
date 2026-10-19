# Scalar Charge and Local-Force Validity in Chameleon Screening

기존 1차원 궤도 프로젝트와 별도로 작성한 구대칭 수치 연구입니다.

> 전하 근사식은 점근 스칼라 전하를 얼마나 정확히 예측하며, 그 전하로 국소 힘을 예측하려면 원천에서 얼마나 떨어져야 하는가?

## 2026-10-19 개정의 핵심

- 힘비 지도의 수치적 기준값은 비선형 BVP의 **직접 기울기**입니다. 먼 거리에서 추출한 전하를 근거리 유카와 공식에 넣은 값은 별도의 근사로 비교합니다.
- 725개 BVP마다 `1.02R`에서 `20R`까지 71개 거리의 국소 힘을 저장합니다. 점근 전하 자체를 사용해도 `1.5R`의 힘은 최대 약 16.6% 차이가 나며, 100개 점에서 오차가 10%를 넘습니다. 최대 오차는 `5R`에서 약 3.85%, `10R`에서 약 1.84%입니다.
- 활성 껍질 적분과 `exp(-mu) * Q_MS` 대조군은 모두 723/725개 점에서 전하를 10% 이내로 예측합니다. 추가 반지름 가중치가 정확도 향상에 필수적이라고 주장하지 않습니다.
- 얇은 껍질 매개변수는 `epsilon_th`로 표기합니다. 이를 실제 기하학적 껍질 두께와 정확히 동일시하지 않습니다.
- 기본 임계값 `alpha_star=0.01`, `r=1.5R`에서 점근 꼬리와 직접 힘의 분류가 두 점에서 다릅니다. 임계값을 바꾸는 민감도 표도 함께 생성합니다.
- 정밀 검증 16점은 모서리, 전하 오차 경계, 근거리 힘 오차와 임계값 판정 변화를 포함합니다. 명목 계산과 정밀 계산의 국소 힘 차이는 최대 약 0.0026%이고, 직접 힘 판정은 모두 유지됩니다.

최신 정확한 값과 실행 환경은 `results/study_metrics.json`을 확인하세요. 위 수치는 기본 임계값으로 재현한 개정 결과이며, 로그 격자의 표본 통계입니다.

## 물리 모형과 해석 범위

`V(phi)=Lambda^5/phi`, `D=phi_infinity/M_c -> 0`, 정적 구대칭 계를 사용합니다.
입력은 `delta=rho_c/rho_infinity`와 `mu=m_infinity R`입니다. 전하 및 뉴턴 힘은 원천 물체의 총질량에 대해 정규화합니다. 힘 지도는 `beta=Q_t=1`인 작은 비되먹임 탐침을 가정합니다.

표면 진폭 모형과 중심점 모형은 **같은 꼬리 정규화에서 서로 다른 근사 가설**입니다. 좌표 기준을 바꾸는 것만으로 물리가 달라지거나 이론에 모호성이 생긴다고 해석하지 않습니다.

이 지도는 물리적 검출 가능성이나 배제곡선이 아닙니다. 실제 실험에는 원천의 절대 크기·질량, 챔버 벽, 시험 물체의 반응, 원천 변조, 잡음과 계통 오차가 추가로 필요합니다. 최초성이나 다른 퍼텐셜·형상에서의 보편적 정확도는 주장하지 않습니다.

## 코드와 자료

- `thin_shell_model.py`: 두 영역 BVP, 점근 전하, 직접 국소 힘, 전하 근사식
- `study_analysis.py`: 전하·거리별 힘 오차, 표본 유효성 거리, 임계값 판정
- `run_study.py`: 주사, 정밀 검증, 데이터·지표·문서 수치 생성
- `study_figures.py`: 영문·한국어 공통 그림 생성기
- `render_korean_figures.py`: 기존 결과의 임계값을 그대로 읽는 한국어 재렌더링
- `thin_shell_report.tex/.pdf`, `thin_shell_report_ko.tex/.pdf`: 영문·한국어 보고서
- `thin_shell_presentation.tex/.pdf`, `thin_shell_presentation_ko.tex/.pdf`: 영문·한국어 16:9 발표자료
- `presentation_notes_ko.md`: 한국어 발표 노트
- `report_preamble.tex`, `slide_preamble.tex`, `study_references.tex`: 공통 문서 설정과 참고문헌
- `../tests/test_thin_shell_study.py`: 해석적 극한, 근거리 힘, 임계값 및 캐시 회귀 테스트

주요 데이터:

- `results/thin_shell_scan.npz`: 725개 점의 전하 및 71개 거리의 직접 힘, 스키마·설정 해시
- `results/thin_shell_scan.csv`: 전하·근사식·수렴 진단
- `results/local_force_scan.csv`: `1.5R`의 직접 힘과 꼬리 힘, 기본 임계값 분류
- `results/force_distance_summary.csv`: 거리별 힘 오차 통계
- `results/threshold_sensitivity.csv`: 여러 임계값의 판정 변화
- `results/stratified_domain_audit.csv`: 긴 영역 및 정밀 계산과의 비교
- `results/gate_validation_midpoints.csv`: 경험적 전하 필터의 별도 검증점
- `results/reference_profiles.npz`: 두 언어 그림이 공유하는 장 분포·검증 곡선
- `results/study_metrics.json`: 지표와 실행 환경·소스 해시
- `results/study_numbers.tex`, `results/*table*.tex`: 지표로부터 자동 생성하는 문서 수치

## 재현

저장소 루트에서 가상환경과 의존성을 준비한 후 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m Thin_Shell_Study.run_study --workers 4 --force
.\.venv\Scripts\python.exe -m unittest discover -v
```

첫 명령이 두 언어의 그림과 문서 수치까지 모두 생성합니다. `--force`를 빼면 현재 스키마·모형·주사 설정과 일치하는 캐시만 재사용하고, 일치하지 않으면 자동 재계산합니다. 언어별로 기본 임계값을 따로 두지 않습니다.

임계값 변경 예:

```powershell
.\.venv\Scripts\python.exe -m Thin_Shell_Study.run_study --threshold 0.001
```

모든 점이 임계값 위나 아래에 있어도 정상적으로 완료됩니다. 교차 등고선이 없으면 이동량은 JSON에서 `null`입니다. 이 명령은 현재 그림·지표·문서 수치를 선택한 임계값으로 갱신하므로, 기본 결과로 되돌리려면 `--threshold 0.01`로 다시 실행합니다.

한국어 그림만 재렌더링할 때:

```powershell
.\.venv\Scripts\python.exe -m Thin_Shell_Study.render_korean_figures
```

PDF는 `Thin_Shell_Study` 디렉터리에서 빌드합니다. TeX Live와 XeLaTeX, 한국어판에는 Noto Serif CJK KR 및 Noto Sans CJK KR 글꼴이 필요합니다.

```powershell
Set-Location Thin_Shell_Study
latexmk -xelatex -interaction=nonstopmode -halt-on-error thin_shell_report.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error thin_shell_report_ko.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error thin_shell_presentation.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error thin_shell_presentation_ko.tex
```

수치가 바뀌면 두 보고서와 발표자료를 모두 다시 빌드해야 PDF가 갱신됩니다. 원래 과목 제출물은 이 디렉터리 밖에 별도로 유지합니다.

## 유효성 경계를 읽을 때

`first_sampled_10_percent_radius`는 `20R`까지 더 먼 모든 **표본 거리**에서도 10% 목표를 유지하는 첫 표본입니다. 표본 사이의 연속적인 오차 한계나 `20R` 바깥의 보증이 아닙니다. 전하 필터의 17개 격자점·12개 중점 검사도 해당 영역의 경험적 결과입니다.
