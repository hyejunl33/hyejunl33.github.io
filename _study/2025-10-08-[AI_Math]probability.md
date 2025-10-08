---
title: "[AI_Math][2주차_기본과제_3]확률론_기본문제"
date: 2025-10-08
tags:
  - 기본3_확률론_기본문제
  - 2주차 과제
  - AI_Math
  - 과제
excerpt: "확률론_기본문제"
math: true
---

# [AI_Math]기본3_확률론_기본문제

- 베이즈정리의 원리를 이해하고, 조건부 확룰이 어떻게 새로운 정보를 통합하여 사후 확률을 업데이트하는지 이해하고, 코드를 통해 구현
- 몬테카를로 샘플링을 통해 불확실성 하에서 최적의 결정을 내리는 방법을 코드를 통해 구현

$$P(\theta|D) = P(\theta)\frac{P(D|\theta)}{P(D)}$$

## Gemini가 설명하는 사전확률, 가능도, 사후확률, evidence.

사전 확률 (Prior Probability)

- **의미:** **사건이 일어나기 전(데이터를 관찰하기 전)에** 우리가 이미 가지고 있는 어떤 사건에 대한 **믿음 또는 확률**입니다.
- **설명:** 이것은 개인적인 믿음, 과거의 통계, 또는 아무런 정보가 없다면 균등한 확률 등, 현재 데이터를 보기 전의 **출발점**이라고 할 수 있습니다. "사전"이라는 이름 그대로 사전에 가지고 있는 확률이죠.
- **간단한 예시:** 어떤 공장에서 만든 제품의 불량률에 대해 아무 정보가 없다면, "이 제품이 불량일 사전 확률은 50%일 것이다"라고 가정하거나, 과거 통계에 따라 "이 공장의 평균 불량률은 5%이므로, 불량일 사전 확률은 5%이다"라고 설정할 수 있습니다.

가능도 (Likelihood)

- **의미:** 어떤 **사건이 이미 일어났다고 가정했을 때**, 우리가 가진 **데이터가 관찰될 확률**입니다.
- **설명:** 가능도는 **'이 가정이 얼마나 그럴듯한가(likely)'**를 나타내는 척도입니다. 즉, "만약 제품이 정상이라면, 이런 데이터가 나올 가능성"과 "만약 제품이 불량이라면, 이런 데이터가 나올 가능성"을 비교하는 데 사용됩니다. **주의할 점은 가능도 자체는 확률이 아니라는 것입니다.**
- **간단한 예시:** 제품을 검사했는데 '살짝 흠집이 있는' 데이터가 관찰되었다고 합시다.
    - **정상 제품이라는 가정 하의 가능도:** "제품이 정상일 때, 흠집이 있을 가능도는 1%이다."
    - **불량 제품이라는 가정 하의 가능도:** "제품이 불량일 때, 흠집이 있을 가능도는 60%이다."

---

사후 확률 (Posterior Probability)

- **의미:** **데이터를 관찰한 후**에 갱신된(update) 사건의 확률입니다.
- **설명:** 이것이 바로 우리가 최종적으로 알고 싶은 결과입니다. **사전 확률(기존 믿음)**에서 출발하여, **가능도(데이터의 증언)**를 반영했을 때의 **최종 결론**이죠. "사후"라는 이름처럼 데이터를 본 이후의 확률입니다.
- **간단한 예시:** '살짝 흠집이 있는' 데이터를 관찰한 후, "이 제품이 실제로 불량일 사후 확률은 95%이다" 와 같이 결론을 내릴 수 있습니다. 이는 기존의 믿음(사전 확률)보다 훨씬 더 강한 확신을 갖게 된 것입니다.

---

증거 (Evidence)

- **의미:** 현재 **관찰된 데이터가 나타날 전체 확률**입니다.
- **설명:** 증거는 사후 확률을 실제 확률 값(0과 1 사이)으로 만들어주기 위한 **'정규화 상수(normalization constant)'** 역할을 합니다. 즉, 모든 가능한 가설(정상일 경우, 불량일 경우 등)에 대한 가능도를 각각의 사전 확률과 곱하여 모두 더한 값입니다.
- **간단한 예시:** '흠집이 있는' 데이터가 관찰될 전체 확률을 계산합니다.
    - (정상일 사전 확률 × 정상일 때 흠집이 있을 가능도) + (불량일 사전 확률 × 불량일 때 흠집이 있을 가능도)
    - 이 계산 결과가 바로 증거(Evidence) 값입니다.

이 네 가지 요소는 **베이즈 정리**라는 하나의 공식으로 아름답게 연결됩니다.

> 사후 확률 = (사전 확률 × 가능도) / 증거
 

즉 베이즈 정리를 애용해서 사전확률과, 가능도, evidence를 이용해서 사후확률을 구할 수 있다.

# 스마트폰 피드백 분석

$$P(A)$$: `p_software` : 소프트웨어 관련 피드백일 확률

$$P(B)$$ : `p_critical` : 비판적인 피드백일 확률

$$P(A|B)$$: `p_software_given_critical` : 비판적 리뷰일때 소프트웨어 관련 피드백일 확률

$$P(A|B^C)$$: `p_software_given_positive` :긍정적 리뷰일때 소프트웨어 관련 피드백일 확률

위의 4가지 확률이 주어졌을떄 아래의 확률을 구해보자.

(1) 소프트웨어 관련 피드백이라면, 비판적인 리뷰가 포함되었을 확률 $$𝑃(𝐵|𝐴)$$

(2) 하드웨어 관련 피드백이라면, 비판적인 리뷰가 포함되었을 확률 $$𝑃(𝐵|𝐴^𝐶)$$

(3) 소프트웨어 관련 피드백이라면, 긍정적인 리뷰가 포함되었을 확률 $$𝑃(𝐵^𝐶|𝐴)$$

(4) 하드웨어 관련 피드백이라면, 긍정적인 리뷰가 포함되었을 확률 $$𝑃(𝐵^𝐶|𝐴^𝐶)$$

```python
def prob(prior, likelihood, evidence):
	#사전확률, 가능도, evidence가 주어졌을때 사후확률 구하기
    posterior = prior * likelihood / evidence
    return posterior

def prob_calculate(p_software, p_critical, p_software_given_critical, p_software_given_positive):

    # P(B|A)
    p_critical_given_software = prob(p_critical,p_software_given_critical,p_software)
		print(f"P(B|A): {p_critical_given_software:.4f}")

    # P(B|A^C)
    p_critical_given_hardware = prob(p_critical,1-p_software_given_critical,1-p_software)
    print(f"P(B|A^C): {p_critical_given_hardware:.4f}")

    # P(B^C|A)
    p_positive_given_software = prob(1-p_critical,p_software_given_positive,p_software)
    print(f"P(B^C|A): {p_positive_given_software:.4f}")

    # P(B^C|A^C)
    p_positive_given_hardware = prob(1-p_critical,1-p_software_given_positive,1-p_software)
    print(f"P(B^C|A^C): {p_positive_given_hardware:.4f}")

    return (p_critical_given_software, p_critical_given_hardware, p_positive_given_software, p_positive_given_hardware)

ex1_p_software = 40/100
ex1_p_critical = 50/100
ex1_p_software_given_critical = 25/50
ex1_p_software_given_positive = 15/50

ex2_p_software = 40/100
ex2_p_critical = 30/100
ex2_p_software_given_critical = 13/30
ex2_p_software_given_positive = 27/70

print("---------------사례1---------------")
ex1_probs = prob_calculate(ex1_p_software, ex1_p_critical, ex1_p_software_given_critical, ex1_p_software_given_positive)

print("---------------사례2---------------")
ex2_probs = prob_calculate(ex2_p_software, ex2_p_critical, ex2_p_software_given_critical, ex2_p_software_given_positive)
```

![image](/assets/images/2025-10-08-13-32-55.png)

베이즈정리에 의해 사전확률, 가능도, evidence를 알면 사후확률을 구할 수 있다.

## Monte Calro Sampling

몬테카를로 샘플링은 확률분포의 명시적인 형태를 모르는 상황(비모수)에서 기댓값과 같은 통계적 수치를 추정하는 효과적인 방법이다. → 무작위 샘플링을 통해 확률분포의 특성을 수치적으로 모사하고, 그 결과를 통계적으로 분석할 수 있고, 원하는 값을 추정할 수 있다.

특히 machine Learning에서 모델이 생성하는 데이터의 분포를 직접적으로 알 수 없을때 몬테카를로 샘플링을 사용한다.

$$E[f(x)] \approx \frac{1}{N} \sum_{i=1}^N f\left(x^{(i)}\right)$$

$$E[f(x)]$$: $$f(x)$$의 기댓값

$$x^{(i)}$$: 확률분포 $$P(x)$$에서 추출된 N개의 i.i.d.(독립적이고 동일하게 분포된)무작위 샘플 → $$P(x)$$를 명시적으로 알지 못하므로 이 샘플들을 이용해서 $$f(x)$$의 기댓값을 근사한다.

몬테카를로 샘플링의 정확도는 샘플의 크기 $$N$$에 의존한다. 대수에 법칙에 의해서 $$N$$이 커질수록 추정값이 실제 기대값에 가까워지며, 고차원 데이터나 복잡한 확률분포에서 유용하다.

## 예시1 적분값 계산

일반적인 적분법으로는 적분하기 힘든 함수를 적분해보자.

가우시안함수: $$f(x) = e^{-x^2}$$

1. 적분 구간 사이에서 난수를 생성해서, 2. 함수에 난수를 넣고, 3. 함수값들의 평균값을 계산하여, 4. 구간의 길이로 나눠서 적분을 근사한다.

```python
import numpy as np

# fun: 적분할 함수
# low, high: 적분 구간의 하한과 상한
# sample_size: 샘플의 크기, 즉 난수의 개수
# repeat: 적분을 반복할 횟수
def mc_int(fun, low, high, sample_size=100, repeat=10):
    int_len = np.abs(high - low)  # 적분 구간의 길이 계산
    stat = []  # 적분값을 저장할 리스트
    for _ in range(repeat):
        x = np.random.uniform(low = low, high = high, size = sample_size) 
        fun_x = fun(x)  # 생성된 난수에 대한 함수 값 계산
        int_val = int_len * np.mean(fun_x)
        stat.append(int_val)  # 근사값을 리스트에 추가
    # 근사값들의 평균과 표준편차 반환
    return np.mean(stat), np.std(stat)

# 적분할 함수 정의
def f_x(x):
    return np.exp(-x**2)
# mc_int 함수를 사용하여 적분 근사값 계산 및 출력.
result = mc_int(f_x, low=-1, high=1, sample_size=10000, repeat=100)
print(result)
```

## 예시2: 원주율 추정

$$\pi \approx 4 \times \frac{\text { 원 안에 생성된 점의 수 }}{\text { 전체 생성된 점의 수 }}$$

1. 단위 정사각형내에서 무작위 점을 생성하고, 2. 생성된 점중 원의 방정식 안에 있는 점을 Count한다. 3. 위의 공식을 통해서 전체점중에 원안의 생성된 점의 수를 분수로 나타내고 4를 곱해서 원주율을 추정한다. 

원 넓이의 공식이 $$\pi r^2$$임을 이용해서 단위 정사각형 넓이와 원의 넓이 비는 $$\pi$$임을 이용함.

```python
import numpy as np
import matplotlib.pyplot as plt

# 시드 설정
np.random.seed(0)

# 무작위 점(샘플)의 수 설정
num_samples = 10000

# 정사각형 [0,1] x [0,1] 내에서 무작위 점 생성
x_random = np.random.uniform(0, 1, num_samples)
y_random = np.random.uniform(0, 1, num_samples)

# 단위 원 내부에 있는 점의 수 계산
inside_circle = (x_random**2 + y_random**2) <= 1

# 공식에 따라 π의 값 추정
pi_estimate = 4 * np.sum(inside_circle) / num_samples
print(f"π의 추정값: {pi_estimate}")

# 점들과 원을 시각화
plt.figure(figsize=(6, 6))
plt.scatter(x_random[inside_circle], y_random[inside_circle], color='blue', s=1, label='Inside Circle')
plt.scatter(x_random[~inside_circle], y_random[~inside_circle], color='red', s=1, label='Outside Circle')
circle = plt.Circle((0, 0), 1, edgecolor='green', facecolor='none')
plt.gca().add_artist(circle)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(f"Monte Carlo Estimation of π (estimate: {pi_estimate})")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.show()
```

![image](/assets/images/2025-10-08-13-34-05.png)