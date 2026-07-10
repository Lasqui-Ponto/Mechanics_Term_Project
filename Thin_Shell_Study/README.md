# Thin-Shell Validity and Force-Ratio Proxy Study

기존 1차원 궤도 프로젝트와 별개로 작성한 독립 연구 패키지입니다. 핵심 질문은 다음과 같습니다.

> long-range thin-shell 법칙을 finite range로 연장할 때 scalar charge를 얼마나 정확히 예측하며, source 위치의 모호성이 force-ratio threshold 판단에 어떤 영향을 주는가?

## 핵심 결과

- $10^2\leq\rho_c/\rho_\infty\leq10^5$, $10^{-3}\leq m_\infty R\leq0.3$의 725개 점을 $x_{\max}=160$에서 모두 수렴시켰습니다.
- long-range charge를 source 표면에 둔 연장은 격자의 68.8%, 중심에 둔 연장은 91.3%에서 10% 이내였습니다. 즉 finite-range “표준 연장”은 하나로 정해지지 않습니다.
- active shell에 표준 Yukawa radial weight를 적분한 $Q_{\rm AS}$는 같은 격자의 99.7%에서 10% 이내였습니다(중앙오차 0.39%, 최대 13.1%).
- 예시 threshold $|a_5/a_N|_\star=10^{-2}$에서 두 naive continuation은 4개 점의 판정을 다르게 했고, convention과 무관하게 수치해와 불일치한 점은 없었습니다.
- 보수적 validity filter는 atlas 17점뿐 아니라 별도의 log-midpoint 12점에서도 모두 10% 이내였습니다.
- 네 모서리와 10% 오차 경계 인근을 포함한 14점의 $x_{\max}=160\to240$ 층화 감사에서 charge 변화는 최대 0.0560%였고, 세 근사식의 10% 판정은 하나도 바뀌지 않았습니다.
- 이 수치는 구형 (n=1) 무차원 모형의 방법론 결과이며, 실제 실험 배제곡선이나 제5힘 검출 주장이 아닙니다.

## 파일

- `thin_shell_model.py`: 두 영역 구형 BVP, scalar-charge 추출, finite-range continuation
- `run_study.py`: 725점 scan, 검증, 그림, 지표 생성
- `thin_shell_report.tex`: 새 독립 보고서
- `thin_shell_presentation.tex`: 16:9 새 발표자료
- `thin_shell_report_ko.tex`, `thin_shell_report_ko.pdf`: 한국어 보고서 원문과 PDF
- `thin_shell_presentation_ko.tex`, `thin_shell_presentation_ko.pdf`: 한국어 16:9 발표자료 원문과 PDF
- `presentation_notes_ko.md`: 한국어 슬라이드별 발표 노트
- `results/thin_shell_scan.csv`: 전체 parameter scan
- `results/gate_validation_midpoints.csv`: atlas에서 제외한 validity-filter 검증점
- `results/stratified_domain_audit.csv`: 모서리·오차경계 인근의 longer-domain 검증점
- `results/study_metrics.json`: 보고서에 사용한 정량 결과
- `figures/`: 보고서 및 발표 그림
- `render_korean_figures.py`, `figures_ko/`: 한국어 그래프 생성기와 출력 그림
- `../tests/test_thin_shell_study.py`: 해석해·양의 장·charge plateau 검증

## 재현

레포지터리 루트에서:

```powershell
.\.venv\Scripts\python.exe -m Thin_Shell_Study.run_study --workers 4 --force
.\.venv\Scripts\python.exe -m Thin_Shell_Study.render_korean_figures
.\.venv\Scripts\python.exe -m unittest discover -v
```

PDF 빌드:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error thin_shell_report.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error thin_shell_presentation.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error thin_shell_report_ko.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error thin_shell_presentation_ko.tex
```

영문 PDF와 한국어 PDF는 서로 별도 파일이며, 한국어판 빌드는 기존 영문 산출물을 덮어쓰지 않습니다.

## 해석할 때의 주의

- 격자 비율은 parameter prior나 이론이 참일 확률이 아닙니다.
- `force-ratio threshold map`은 무차원 proxy입니다. source의 절대 질량·반지름과 장비 잡음, chamber wall, 3차원 형상은 포함하지 않습니다.
- $Q_{\rm AS}$는 표준 Yukawa shell integral을 이 프로젝트에서 구성·benchmark한 ansatz입니다. 문헌상 최초성이나 다른 potential·비구형 계에 대한 보편성을 주장하지 않습니다.
- 2022년 levitated-force-sensor 실험의 결론을 고려하여, 이 연구는 기본 chameleon dark-energy model을 되살리는 연구가 아니라 screening 근사와 실험 설계의 방법론 연구로 서술합니다.
