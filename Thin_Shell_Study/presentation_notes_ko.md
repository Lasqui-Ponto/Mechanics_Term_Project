# 발표 노트 — Thin-shell 근사는 언제 실패하는가?

약 12–15분 발표를 기준으로 작성했습니다. 마지막 참고문헌 슬라이드는 질의응답 때만 보여도 됩니다.

## 1. 제목

기존 프로젝트에서 카멜레온 screening 자체를 시각화했다면, 이번 연구는 long-range thin-shell 법칙을 finite range로 어떻게 연장해야 하는지 묻습니다. 표면에 유효 전하를 둔 경우와 중심에 둔 경우가 달라지며, 실제 비선형 수치해가 이 모호성을 어떻게 해소하는지가 핵심입니다.

## 2. 연구 공백

(Q_{\rm KW}=\min(3\Delta R/R,1))은 전체 BVP를 한 숫자로 바꾸는 매우 유용한 shortcut입니다. 하지만 이 식 자체에는 error bar가 없습니다. 따라서 이번 연구의 niche는 screening의 존재를 다시 보여주는 것이 아니라, 근사 정확도에서 관측 판단까지 연결하는 것입니다.

## 3. 모형

(n=1) inverse-power potential과 균일 구형 source를 사용했습니다. 입력은 density contrast (delta)와 외부 Compton 변수 (mu=m_\infty R) 두 개입니다. (epsilon), (m_cR)은 입력에서 따라오는 진단량입니다. 작은 (epsilon), 큰 (m_cR), 작은 (mu)가 서로 다른 가정임을 강조합니다.

## 4. 세 근사식

표준 long-range 식은 shell mass를 $3\epsilon$으로 선형화하며 $m_\infty R\ll1$을 전제합니다. finite range에서는 같은 charge를 표면에 둘지 중심에 둘지에 따라 두 continuation이 생깁니다. active-shell 식은 표준 Yukawa radial weight를 shell에 적분한 비교용 ansatz이며 수치해의 입력이 아닙니다.

## 5. 수치 방법

구 내부와 외부를 별도 collocation domain으로 두어 sharp density boundary를 smoothing하지 않았습니다. 중심 regularity와 외부 Yukawa Robin 조건을 사용했습니다. 외부 tail의 amplitude와 gradient에서 (Q)를 독립적으로 추출합니다. 25×29, 총 725개 점이 모두 수렴했습니다.

## 6. 검증

약한 source에서는 알려진 uniform Yukawa sphere 해와 최대 $2.8\times10^{-5}$까지 일치합니다. atlas는 $x_{\max}=160$을 사용합니다. 네 모서리, 대표점, 10% 오차 경계 인근을 포함한 14점을 240까지 다시 풀었을 때 charge 변화는 최대 0.0560%였고, 세 근사식의 10% 판정은 하나도 바뀌지 않았습니다. charge 두 추정량의 전체 격자 최대 불일치는 0.0481%입니다.

## 7. 세 regime

왼쪽은 full nonlinear field입니다. unscreened에서는 내부 전체가 반응하고, transition에서는 유한한 표면층, screened에서는 거의 고정된 core가 생깁니다. 오른쪽 plateau는 scalar charge가 근사식으로 입력된 값이 아니라 외부 수치해에서 측정된 값이라는 증거입니다.

## 8. 정확도 지도

첫 panel은 표면 continuation, 둘째는 중심 continuation, 셋째는 active-shell integral입니다. 두 naive continuation의 오차 부호와 10% 경계가 다릅니다. 즉 long-range 식과 $\epsilon$만으로는 finite-range source의 위치를 정할 수 없습니다.

## 9. 정량 결과

표면 continuation은 68.8%, 중심 continuation은 91.3%의 grid에서 10% 이내입니다. active-shell integral은 99.7%에서 10% 이내이고 중앙오차는 0.39%입니다. 보수적 filter $\epsilon<0.1$, $m_cR>10$, $m_\infty R<0.035$는 atlas 17점과 별도 midpoint 12점에서 모두 10% 목표를 만족했습니다.

## 10. force-ratio proxy의 정의

지도는 실제 discovery forecast나 장비에 필요한 sensitivity가 아니라, 임의로 정한 무차원 force-ratio proxy threshold map입니다. 과거 null result에는 reach ($\mathcal R$), 미래 기기에는 SNR proxy ($\mathcal D$)를 따로 써야 합니다. 원자와 bulk의 scalar charge만 보고 장비 성능을 단정하지 않습니다.

## 11. threshold 판단 지도

예시 threshold $10^{-2}$에서 표면과 중심 continuation은 4개 grid point를 서로 다르게 분류합니다. 그러나 그 네 점에서 수치해는 항상 둘 중 하나와 일치하므로 convention-independent mismatch는 0개입니다. 표면 contour shift의 median은 0.032 dex, 중심은 0.0066 dex입니다. 핵심은 ‘false detection’이 아니라 finite-range continuation 선택의 모호성입니다.

## 12. 주장 범위

의미 있는 결과는 transition error atlas, 독립 검증된 charge, active-shell benchmark, threshold ambiguity map입니다. 반대로 실제 제5힘 검출, 새로운 $(\Lambda,M_c)$ exclusion, chamber 예측, $Q_{\rm AS}$의 문헌상 최초성은 주장하지 않습니다.

## 13. 결론

기억할 숫자는 표면 68.8%, 중심 91.3%, active shell 99.7%입니다. 기억할 물리는 long-range law만으로 finite-range source 위치가 정해지지 않는다는 점입니다. 이 연구는 근사를 버리는 것이 아니라 radial source structure를 명시하고 검증하자는 결론입니다.

## 예상 질문

### $Q_{\rm AS}$가 정말 새로운 보편식인가?

아닙니다. 표준 균일구 Yukawa radial weight를 active shell에 적용해 이 프로젝트에서 구성·benchmark한 ansatz입니다. 문헌상 최초성을 주장하지 않으며, 다른 $n$, chamber, 비구형 source에서 별도 검증해야 합니다.

### 왜 실제 ((\Lambda,M_c)) exclusion map을 만들지 않았나?

실제 laboratory map에는 chamber wall이 정하는 환경장, source modulation, probe charge, 장비별 noise가 필요합니다. 이를 생략한 채 exclusion을 그리면 근사 자체의 오차와 geometry 오차가 섞입니다. 이번 연구는 먼저 isolated-sphere validity를 분리했습니다.

### 725개 점 중 비율을 확률로 해석할 수 있나?

없습니다. 로그 간격 계산 grid의 coverage일 뿐이며 parameter prior가 아닙니다.
