---
title: "[RecSys][과제1] 기본통계학"
date: 2025-10-19
tags:
  - RecSys
  - 과제
  - 기본 통계학
  - CLT
  - Conjugate Prior
excerpt: "[RecSys][과제1] 기본통계학"
math: true
---

# 과제1_기본통계학

- 중심 극한 정리의 개념을 실습을 통해 이해할 수 있다.
- conjugate prior의 의미를 실습으로 체득할 수 있다.
- 확률분포를 코드로 구현하는 방법과 plot하는 방법을 배울 수 있다.

# 중심극한정리(CLT)

$\lim_{{n \to \infty}} P\left(\frac{{X_1 + X_2 + \ldots + X_n - n\mu}}{{\sigma \sqrt{n}}} \leq x\right) = \Phi(x)$

중심극한정리는 독립적인 확률 변수들의 합이나 평균이 표본의 크기가 커질수록 정규분포에 근사하는 현상이다.

$X_n$은 독립적이고 동일한 분포를 가진($i$$.i.d$)확률변수

$n$:표본의 크기

$\mu$: 확률변수의 평균

$\sigma$: 확률변수의 표준편차

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm

# 균등 분포로부터 표본 추출하는 함수
def sample_uniform_distribution(n, sample_size):
    samples =  np.random.rand(n, sample_size) 
    sample_means = np.mean(samples, axis=1) 
    return sample_means

# 히스토그램을 통한 정규 분포 확인
def plot_normal_distribution(sample_means, n):
    plt.figure(figsize=(8, 6))
    plt.hist(sample_means, bins=20, density=True, alpha=0.6, color='g')

    # 표준 정규 분포를 평균과 표준 편차 계산하여 함께 플롯
    mu = np.mean(sample_means)
    sigma = np.std(sample_means)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2)

    plt.title(f'Distribution of Sample Means (n={n})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend(['Normal Distribution', 'Sample Means'])
    plt.show()

# 표본 크기와 추출 횟수 설정
sample_size = 100  # 각 표본의 크기
n = [10, 50, 1000]  # 표본 추출 횟수

for size in n:
    sample_means = sample_uniform_distribution(size, sample_size)
    plot_normal_distribution(sample_means, size)

```

![image](/assets/images/2025-10-19-19-53-09.png)
![image](/assets/images/2025-10-19-19-53-16.png)
![image](/assets/images/2025-10-19-19-53-21.png)

`sample_size` 를 100으로 고정하고 표본추출 횟수 n을 10, 50, 1000으로 늘려보면 표본 추출 횟수 $n$이 커질수록 표본평균은 정규분포와 가까워짐을 확인할 수 있다.

# Conjugate Prior(켤레 사전 확률)

사전확률분포(Prior)와 사후 확률 분포(Posterior)가 같은 종류의 확률분포가 되도록 만들어주는 특별한 관계의 사전분포를 의미한다.

즉 베이즈정리에서 Prior와 likelihood의 곱은 Posterior와 비례관계에 있는데, prior와 likelihood의 곱이 prior의 확률분포꼴을 띠게 되는 관계이다.

이 관계에 있는 확률분포들은 복잡한 계산 없이 파라미터만 업데이트 하면 바로 Posterior를 얻을 수 있다.

### 왜사용할까?

원래 Posterior를 정확히 계산하려면 베이즈정리의 분모에 해당하는 복잡한 적분계산이 필요하다. 하지만 켤레 관계를 이용하면 이 과정을 생략하고, 사전 분포의 파라미터를 데이터의 정보를 이용해서 업데이트하는것 만으로도 Posterior를 얻을 수 있다.

 $P(\theta|D) = \frac{P(\theta) \cdot P(D|\theta)}{P(D)}$

즉 분모의 $P(D)$를 구하려면 복잡한 적분과정을 거쳐야 하는데, 그러한 과정 없이 파라미터 업데이트만으로 Posterior를 구할 수 있음.

## 베르누이 - 베타분포

베르누이 분포: 이항분포에서 시행횟수가 한번의 시행에서 성공할 확률이 p인경우를 모델링

베타분포: 0과 1사이의 값을 가지는 연속 확률분포

베타분포를 prior로, 베르누이분포를 likelihood로 모델링하면 사후분포또한 베타분포이다.

$\text{사후분포 파라미터 추정식 (}\alpha, \beta \text{)} \\\alpha_{\text{new}} = \alpha + \sum_{i=1}^{n} x_i, \quad \beta_{\text{new}}=\beta + n - \sum_{i=1}^{n} x_i$

베타분포의 PDF는 $\alpha, \beta$에 의해서 결정된다.

$f(y;α,β)=\frac{y^{α−1}(1−y)^{β−1}}{B(α,β)}​$

$B(α,β)=∫_0^1t^{α−1}(1−t)^{β−1}dt$

따라서 알파와 베타를 파라미터 추정식으로 업데이트 해준다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 데이터 생성 (동전 던지기 시뮬레이션)
np.random.seed(42)  # 재현성을 위한 랜덤 시드 설정

data = np.random.binomial(n=1, p=0.7, size=20)  # 동전 던지기를 통한 데이터 생성 (20번 던진 경우)

# 사전분포 (베타 분포)
alpha_prior = 1  # 베타 분포의 알파 파라미터
beta_prior = 1   # 베타 분포의 베타 파라미터
prior_distribution = beta(alpha_prior, beta_prior)

# 데이터를 이용하여 사후분포 갱신
alpha_posterior = alpha_prior + np.sum(data)  # 사전분포의 알파 파라미터 갱신
beta_posterior = beta_prior + len(data) - np.sum(data)  # 사전분포의 베타 파라미터 갱신
posterior_distribution = beta(alpha_posterior, beta_posterior)

# 사전분포와 사후분포 시각화
x = np.linspace(0, 1, 1000)
plt.plot(x, prior_distribution.pdf(x), label='Prior', color='blue')
plt.plot(x, posterior_distribution.pdf(x), label='Posterior', color='red')
plt.title('Conjugate Prior: Beta Prior and Posterior for Bernoulli Distribution (Coin Flip Example)')
plt.xlabel('p (Probability of heads)')
plt.ylabel('Density')
plt.legend()
plt.show()

```

![image](/assets/images/2025-10-19-19-53-38.png)

prior의 초기상태에서 Posterior를 업데이트 해서 베르누이 분포의 0.7에서의 확률이 가장 높게 파라미터가 업데이트 됐음을 알 수 있다.

## 포아송 분포 - 감마분포

**포아송분포**: 주어진 시간간격동안 발생한 사건의 횟수를 모델링하는 이산 확률 분포

$\text{포아송 분포 확률 밀도 함수} \\f(x|\lambda)=\frac{\lambda ^ x}{x!}e^{-\lambda}\;,\;x=0,1,\dots,\infty$

$\lambda$: 시간간격동안 사건이 발생할 횟수의 기댓값

$x$: 사건 발생 횟수

$\lambda$번 사건이 발생할떄 $x$번의 사건 발생이 관측될 확률을 모델링한다.

감마분포: 시간을 모델링하는 연속 확률 분포

$f(x;\alpha,\beta) = \frac{e^{(-\beta x)}\beta^\alpha}{\Gamma(\alpha)}x^{(\alpha-1)}$

$\alpha$는 shape parameter로 분포의 기본적인 형태를 결정한다. $\alpha$가 증가할 수록 대칭적인 형태에 가까워지며 $\alpha$가 커질수록 분포의 평균과 분산이 증가한다.

$\beta$는 Rate Parameter로 분포의 규모를 조정한다.

아래의 수식을 통해 파라미터를 업데이트한다.

$\\\alpha_{\text{new}} =\alpha + \sum_{i} x_i, \quad \beta_{\text{new}}= \beta + n$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson

# 데이터 생성 (가상의 포아송 분포 데이터)
np.random.seed(42)  # 재현성을 위한 랜덤 시드 설정
data = np.random.poisson(lam=3, size=50)  # 포아송 분포를 따르는 데이터 생성 (평균 λ=3, 데이터 수=50)

# 사전분포 (감마 분포)
alpha_prior = 2  # 감마 분포의 알파 파라미터
beta_prior = 1   # 감마 분포의 베타 파라미터
prior_distribution = gamma(alpha_prior, scale=1/beta_prior)

# 데이터를 이용하여 사후분포 갱신
alpha_posterior = alpha_prior + np.sum(data)        
beta_posterior =  beta_prior + len(data)            
posterior_distribution = gamma(alpha_posterior, beta_posterior)     

# 시각화
x = np.linspace(0, 15, 1000)
plt.plot(x, prior_distribution.pdf(x), label='Prior', color='blue')
plt.plot(x, posterior_distribution.pdf(x), label='Posterior', color='red')
plt.title('Conjugate Prior: Gamma Prior and Posterior for Poisson Distribution')
plt.xlabel('λ (Rate parameter)')
plt.ylabel('Density')
plt.legend()
plt.show()
```

![image](/assets/images/2025-10-19-19-53-53.png)

## 정규분포 - 정규분포

정규분포는 CLT의 성질을 활용할 수 있는 이점이 있기 때문에 모델링할때 자주 사용되는 연속확률 분포이다.

$\mu, \sigma^2$를 파라미토로 이용해서 모델링한다.

정규분포의 PDF:$f(x;\mu,\sigma^2)=\frac{1}{\sigma \sqrt{2\pi}} \times exp(-\frac{(x-\mu)^2}{2\sigma^2}$

Prior를 정규분포로 가정했을때 사후분포도 정규분포가 된다.

$\mu_{\text{new}}=\frac{1}{ \frac{1}{ \sigma_0^2} + \frac{n}{\sigma^2} }\left( \frac{\mu_0}{\sigma_0^2} + \frac{\sum_{i=1}^{n} x_i}{\sigma^2} \right) ,\quad  \sigma^2_{\text{new}}=\left( \frac{1}{ \sigma_0^2} + \frac{n}{\sigma^2} \right)^{-1}$

위의 식을 이용해서 파라미터를 업데이트 한다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 데이터 생성 (가상의 정규 분포 데이터)
np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=10)  # 평균이 10이고 표준편차가 2인 정규 분포를 따르는 데이터 생성

# 사전분포 선택 (정규 분포)
mu_prior = 0  # 평균의 초기값
sigma_prior = 1  # 표준편차의 초기값
prior_distribution = norm(mu_prior, sigma_prior)
data_sigma = 2

# 데이터를 이용한 사후분포 갱신
mu_posterior = (1 / ((1 / sigma_prior**2) + (len(data) / data_sigma**2))) * ((mu_prior / sigma_prior**2) + (np.sum(data) / data_sigma**2))
sigma_posterior = np.sqrt(1 / ((1 / sigma_prior**2) + (len(data) / data_sigma**2)))
posterior_distribution = norm(mu_posterior, sigma_posterior)   # FILL HERE #

# 시각화
x = np.linspace(-5, 15, 1000)
plt.plot(x, prior_distribution.pdf(x), label='Prior', color='blue')
plt.plot(x, posterior_distribution.pdf(x), label='Posterior', color='red')
plt.title('Conjugate Prior: Normal Prior and Posterior for Normal Distribution')
plt.xlabel('μ (Mean)')
plt.ylabel('Density')
plt.legend()
plt.show()

```

![image](/assets/images/2025-10-19-19-54-05.png)

prior와 posterior 둘다 모두 정규분포임을 볼 수 있다.

# 분포간의 근사관계

## 베타분포 - UniForm분포

beta분포는 $\alpha, \beta$가 1일때 Uniform분포가 된다.

$f(x; \alpha, \beta) = \frac{{x^{\alpha - 1} \cdot (1 - x)^{\beta - 1}}}{{B(\alpha, \beta)}}$

Beta분포의 PDF식에서 $\alpha = 1, \beta = 1$로두면

$f(x; \alpha=1, \beta=1) = \frac{1}{{B(1, 1)}}$

여기서 분모를 계산하면

$B(\alpha,\beta)=\int _0^1 t^{\alpha -1}(1-t)^{\beta-1}dt\\B(1,1)=\int _0^1 t^0(1-t)^0dt=\int_0^11dt=1$

따라서 PDF = 1이다.$f(x;1,1)=1$

Uniform분포$U(0,1)$는 [0,1]구간에서 모든 값이 균등한 확률을 가지며, PDF는 아래와 같다.

$f(x; a, b) = \frac{1}{{b - a}}, \quad \text{for } a \leq x \leq b\\f(x;0,1)=\frac{1}{{1-0}}=1$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform

# Beta 분포의 모수 설정
alpha_prior = 1
beta_prior = 1

alpha_prior2 = 0.5
beta_prior2 = 0.5

alpha_prior3 = 2
beta_prior3 = 3

# Uniform 분포의 범위 설정
x = np.linspace(0, 1, 1000)

# Beta 분포와 Uniform 분포의 확률 밀도 함수 계산
beta_pdf = beta(alpha_prior, beta_prior).pdf(x)
beta_pdf2 = beta(alpha_prior2, beta_prior2).pdf(x)
beta_pdf3 = beta(alpha_prior3, beta_prior3).pdf(x)
uniform_pdf = uniform.pdf(x)

# 시각화
plt.plot(x, beta_pdf, label='Beta Distribution (α=1, β=1)')
plt.plot(x, beta_pdf2, label='Beta Distribution (α=0.5, β=0.5)')
plt.plot(x, beta_pdf3, label='Beta Distribution (α=2, β=3)')
plt.plot(x, uniform_pdf, label='Uniform Distribution', linestyle='--', color='red')
plt.title('Comparison of Beta and Uniform Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

```

![image](/assets/images/2025-10-19-19-54-13.png)

$\alpha = 1, \beta = 1$에서 파란색과 빨간색 그래프가 일치함을 확인할 수 있다.

## Gamma분포 - Exponential분포

감마 분포는 두개의 파라미터를 가지며 확률 밀도 함수(PDF)는 아래와 같다.

$f(x; \alpha, \beta) = \frac{{x^{\alpha - 1} \cdot e^{-x/\beta}}}{{\beta^\alpha \cdot \Gamma(\alpha)}}$

지수분포는 감마분포에서 shape parameter인 $\alpha = 1$일때의 분포이다. 지수분포는 $\lambda$(rare parameter)평균 발생률을 매개변수로 가지며, 확률 밀도 함수는 아래와 같다.

$f(x; \lambda) = \lambda e^{-\lambda x}, \quad \text{for } x \geq 0$

$f(x; \alpha=1, \beta) = \frac{{x^{1 - 1} \cdot e^{-x/\beta}}}{{\beta^1 \cdot \Gamma(1)}}=\frac{1}{\beta}e^{-x/\beta}$

여기서 $\lambda = \frac{1}{\beta}$로 치환하면 지수분포의 확률 밀도 함수가 된다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon

# Gamma 분포의 모수 설정
alpha1 = 1
beta1 = 1

alpha2 = 2
beta2 = 3

# Exponential 분포의 모수 설정
lambda_ = 1

# 범위 설정
x = np.linspace(0, 5, 1000)

# Gamma 분포와 Exponential 분포의 확률 밀도 함수 계산
gamma_pdf1 = gamma.pdf(x, alpha1, scale=1/beta1)
gamma_pdf2 = gamma.pdf(x, alpha2, scale=1/beta2)
exponential_pdf = expon.pdf(x, scale=1/lambda_)

# 시각화
plt.plot(x, gamma_pdf1, label='Gamma Distribution (α=1, β=1)')
plt.plot(x, gamma_pdf2, label='Gamma Distribution (α=2, β=3)')
plt.plot(x, exponential_pdf, label='Exponential Distribution', linestyle='--', color='red')
plt.title('Comparison of Gamma and Exponential Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

```

![image](/assets/images/2025-10-19-19-54-25.png)

감마분포에서 $\alpha=1$로 주었을때 지수분포와 같음을 확인할 수 있다..

## Binomial분포 - Normal분포

이항분포의 PDF는 다음과 같다.

$f(x; n, p) = \binom{n}{x} p^x (1-p)^{n-x}$

이항분포에서 시행횟수 n이 충분히 크고 성공 확률 p가 0또는 1에 가깝지 않을때 CLT(중심 극한 정리)에 의하여 정규분포에 가까워 진다.

$X \sim Binomial(n, p)\\\mathbb E[X]=np, \quad Var(X)=np(1-p)\\X \approx N(np,np(1-p))$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

# 이항 분포의 모수 설정
n = 20  # 시행 횟수
p = 0.5  # 성공 확률

# 정규 분포의 모수 설정 (이항 분포의 기댓값과 분산을 사용)
mean = n*p    
variance = mean*(1-p) 
std_dev =  np.sqrt(variance) 

# 범위 설정
x = np.arange(0, 20)

# 이항 분포의 확률 질량 함수 계산
binomial_pmf = binom.pmf(x, n, p)

# 정규 분포의 확률 밀도 함수 계산
normal_pdf = norm.pdf(x, mean, std_dev)

# 시각화
plt.plot(x, binomial_pmf, label='Binomial Distribution (n=20, p=0.5)', color='blue')
plt.plot(x, normal_pdf, label='Normal Distribution', linestyle='--', color='red')
plt.title('Comparison of Binomial and Normal Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

```

![image](/assets/images/2025-10-19-19-54-33.png)

$n=20$이고 $p=0.5$인 이항분포는 정규분포를 따름을 확인할 수 있다.