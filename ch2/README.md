<img width="1607" alt="1" src="https://github.com/junyong1111/Bigdata/assets/79856225/126e8525-4dd3-478f-a352-623ca3624756">

<img width="1557" alt="2" src="https://github.com/junyong1111/Bigdata/assets/79856225/c6660269-64b7-47c8-ad2b-d87d1ec9d1c6">

### 이항분포

- N : 횟수
- P : 확률
- 평균 : N * P
- 분산 : N * P * (1-P)

### 확률변수

- 확률 = 일어날 경우의 수(사건) / 전체경우의 수 (표본공간)
- 확률 변수 ⇒ 표본공간의 집합(숫자로 이루어짐)
- 기댓값 ⇒ 평균적으로 ~정도 얻음
- 이산형 ⇒ 확률분포표/ 그래프
- 연속형 ⇒ 확률밀도 함수(전체 면적 1)

### 평균, 분산

- 분산 → 평균으로부터 얼마나 떨어져 있는가
    - 분산 0 : 평균이랑 같음
    - 편차^2의 평균
    - **각각의 변량의 제곱의 평균 - 변량의 평균**
- 편차 → 개인-평균(이상치에 영향을 받음)
    - 편차의 합은 0
- 표준편차
    - 분산의 제곱근

<aside>
💡 편차의 합은 모두 0 → 거리를 구해야 하므로 제곱을 함 → 숫자가 너무 크므로 다시 평균을 냄 ⇒ 분산
분산의 제곱근 ⇒ 표준편차

</aside>

### 기대값

- E(x) ⇒ 평균(확률분포 X를 따를 )
- V(x) ⇒ 분산 (확률분포 X를 따를 때)
    - E(x^2) - {E(x)}^2
    - **제곱의 평균 - 평균의 제곱**

### 정규분포

X ~ N(평균, 분산)

- 표준정규분포 X~N(0, 1)
- 정규화 Z → X-평균/표준편차
- 표준정규분포에서 계산 → 원래 분포의 면적을 계산
    - **평균+표준편차 - 평균 / 표준편차**

### 모평균/표본평균

어떠한 모집단의 모평균/모분산/모표준편차

- N만큼 샘플한 표본평균에서 구하는 방법
    - 모평균 ⇒ 표본평균과 같다
    - 모분산 ⇒ 모분산 / N개의 샘플
    - 모표준편차 ⇒ 모표준편차 / N개의 샘플의 제곱근

### 중심극한정리

### 중심극한정리

- 모분포의 모양에 상관없이 N개의 샘플로 분포를 만들면 정규분포
    - 단 N≥30인 경우

<img width="1630" alt="3" src="https://github.com/junyong1111/Bigdata/assets/79856225/14023dab-3b0f-4030-b9d9-448441588343">

<img width="1636" alt="4" src="https://github.com/junyong1111/Bigdata/assets/79856225/063df471-a55a-4228-8c90-e4af0b92089b">

### 신뢰구간, 모평균 추정

모분포의 평균을 구하기는 현실적으로 어렵기 때문에 표본평균을 통해서 **역으로 모집단을 추정함**

- A : 우리나라 키 평균은 100 ~ 200cm 이다.
- B : 우리나라 키 평균은 150 ~ 190cm 이다.
- C : 우리나라 키 평균은 160 ~ 175cm 이다.

**정확도가 높으면서 적당한 구간을 찾는 것 ⇒ 신뢰구간**

**구간을 너무 줄이면 정확도가 떨어짐**

<img width="1648" alt="5" src="https://github.com/junyong1111/Bigdata/assets/79856225/200279ac-9f94-4920-be7f-e747a5c3347d">


<img width="1637" alt="6" src="https://github.com/junyong1111/Bigdata/assets/79856225/864715ca-8624-42bd-af06-465f1fffbc96">

<img width="1655" alt="7" src="https://github.com/junyong1111/Bigdata/assets/79856225/73c02093-4016-4c37-ac9d-50ae1929d783">

<img width="1633" alt="8" src="https://github.com/junyong1111/Bigdata/assets/79856225/5677c11f-eac9-4e23-b213-0caacab08b47">
<img width="1645" alt="9" src="https://github.com/junyong1111/Bigdata/assets/79856225/d8d736c9-ce0f-4800-8aa4-73599475e3a0">
