---
title: "[논문 리뷰] Attention Is All You need (2017) 심층 분석"
date: 2025-09-15
tags:
  - NLP
  - Transformer
  - Attention
  - 논문리뷰
excerpt: "순환 신경망(RNN)의 한계를 극복하고 자연어 처리의 새로운 패러다임을 제시한 Transformer 모델의 아키텍처부터 Self-Attention 메커니즘, 실험 결과까지 PPT 내용을 기반으로 상세하게 분석."
math: true
---

## 목차

1.  [서론: 기존 RNN 기반 모델의 한계](#1-서론-기존-rnn-기반-모델의-한계)
    -   [순차적 처리와 병렬화의 부재](#순차적-처리와-병렬화의-부재)
    -   [장기 의존성 문제](#장기-의존성-문제-long-range-dependency)
2.  [Transformer: Attention만을 이용한 새로운 접근](#2-transformer-attention만을-이용한-새로운-접근)
3.  [모델 아키텍처 (Model Architecture) 심층 분석](#3-모델-아키텍처-model-architecture-심층-분석)
    -   [전체 구조: 인코더-디코더](#전체-구조-인코더-디코더)
    -   [인코더(Encoder) 상세 구조](#인코더encoder-상세-구조)
    -   [디코더(Decoder) 상세 구조](#디코더decoder-상세-구조)
4.  [어텐션 메커니즘 분석](#4-어텐션-메커니즘-분석)
    -   [Scaled Dot-Product Attention](#scaled-dot-product-attention)
    -   [Multi-Head Attention](#multi-head-attention)
    -   [Positional Encoding: 순서 정보의 주입](#positional-encoding-순서-정보의-주입)
    -   [Add & Norm: 잔차 연결과 층 정규화](#add--norm-잔차-연결과-층-정규화)
5.  [학습 및 실험 결과](#5-학습-및-실험-결과)
    -   [학습 방법](#학습-방법)
    -   [성능 평가 (BLEU Score)](#성능-평가-bleu-score)
    -   [모델 복잡도 비교](#모델-복잡도-비교)
6.  [결론](#6-결론)

---

## 1. 서론: 기존 RNN 기반 모델의 한계

`Attention Is All You Need` 논문이 등장하기 전, 자연어 처리(NLP), 특히 기계 번역 분야는 **RNN(Recurrent Neural Network)** 과 이를 개선한 LSTM, GRU 기반의 Seq2Seq 모델이 주를 이루었습니다. 이 모델들은 시퀀스 데이터 처리에 효과적이었지만, 다음과 같은 명확한 한계를 지니고 있었습니다.

### 순차적 처리와 병렬화의 부재
RNN은 이전 타임스텝의 은닉 상태($h_{t-1}$)를 현재 타임스텝($t$)의 입력으로 사용하는 순환 구조를 가집니다. 이는 데이터의 순서 정보를 자연스럽게 처리할 수 있는 장점이 있지만, **순차적으로만 계산이 가능**하다는 치명적인 단점을 야기합니다. 이로 인해 GPU의 장점인 병렬 연산을 활용하기 어려워 대규모 데이터셋 학습에 많은 시간이 소요되었습니다.

### 장기 의존성 문제 (Long-Range Dependency)
시퀀스의 길이가 길어질수록 문장의 앞부분 정보가 뒤로 전달되면서 희석되거나 소실되는 문제가 발생합니다.
- **기울기 소실/폭발 (Vanishing/Exploding Gradient)**: 역전파 과정에서 기울기가 반복적으로 곱해지면서 0에 가깝게 사라지거나(sigmoid, tanh 활성화 함수 사용 시) 무한대로 발산하는(ReLU 활성화 함수 사용 시) 문제가 발생하여 효과적인 학습이 어려웠습니다.



이러한 문제들을 해결하기 위해 Attention 메커니즘이 RNN 모델에 보조적으로 도입되었으나, 근본적인 순환 구조의 한계는 여전히 남아있었습니다.

---

## 2. Transformer: Attention만을 이용한 새로운 접근

본 논문은 RNN의 순환 구조를 완전히 배제하고, **오직 Attention 메커니즘만으로** 입력과 출력 시퀀스 간의 전역적인 의존성을 모델링하는 **트랜스포머(Transformer)** 구조를 제안했습니다. 이를 통해 다음과 같은 혁신을 이루었습니다.

-   **완전한 병렬 처리**: 순환 구조가 없으므로 문장 내 모든 단어에 대한 연산을 동시에 처리할 수 있어, 학습 효율성과 속도를 극대화했습니다.
-   **전역적 의존성 학습**: 아무리 멀리 떨어진 단어 사이의 관계도 직접적인 경로를 통해 한 번에 계산할 수 있어, 장기 의존성 문제를 근본적으로 해결했습니다.

---

## 3. 모델 아키텍처 (Model Architecture) 심층 분석

### 전체 구조: 인코더-디코더
트랜스포머는 기계 번역을 위해 설계된 전통적인 **인코더-디코더** 구조를 따릅니다.
-   **인코더 (Encoder)**: 입력 문장 시퀀스를 받아 문맥 정보를 함축한 연속적인 표현(representation)으로 변환합니다.
-   **디코더 (Decoder)**: 인코더의 출력과 이전에 생성된 출력 단어들을 입력으로 받아, 다음 단어를 예측합니다.

논문에서는 인코더와 디코더 모두 **N=6개의 동일한 레이어**를 쌓아 구성했습니다.



### 인코더(Encoder) 상세 구조
각 인코더 레이어는 두 개의 주요 서브 레이어(sub-layer)로 구성됩니다.
1.  **Multi-Head Self-Attention**: 입력 문장 내에서 단어들 간의 관계를 파악하는 역할을 합니다.
2.  **Position-wise Feed-Forward Network**: 어텐션을 통해 얻은 정보를 바탕으로 비선형 변환을 적용하는 완전 연결 신경망입니다.

각 서브 레이어의 출력에는 **잔차 연결(Residual Connection)**과 **층 정규화(Layer Normalization)**가 적용됩니다.

### 디코더(Decoder) 상세 구조
각 디코더 레이어는 세 개의 서브 레이어로 구성됩니다.
1.  **Masked Multi-Head Self-Attention**: 출력 시퀀스 내에서 단어들 간의 관계를 파악합니다. 이때, 현재 위치의 단어가 미래 위치의 단어를 참고하지 못하도록 **마스킹(Masking)**을 적용하여 **자가 회귀(auto-regressive)** 속성을 보장합니다.
2.  **Multi-Head Encoder-Decoder Attention**: 디코더가 단어를 예측할 때, 인코더가 출력한 입력 문장의 어떤 부분에 집중(Attention)해야 할지를 결정합니다.
3.  **Position-wise Feed-Forward Network**: 인코더와 동일한 역할을 수행합니다.



---

## 4. 어텐션 메커니즘 분석

### Scaled Dot-Product Attention
트랜스포머의 핵심 연산으로, **Query(Q), Key(K), Value(V)** 라는 세 개의 벡터를 입력으로 받습니다.
1.  하나의 단어를 대표하는 **Query 벡터**가 문장 내 모든 단어를 대표하는 **Key 벡터**들과 내적(dot-product)을 통해 **유사도(Attention Score)**를 계산합니다.
2.  이 점수를 Key 벡터의 차원($d_k$)의 제곱근($\sqrt{d_k}$)으로 나누어 **스케일링(Scaling)**합니다. 이는 $d_k$가 클 때 내적 값이 너무 커져 Softmax 함수의 기울기가 0에 가까워지는 것을 방지하여 학습을 안정화시키는 효과가 있습니다.
3.  스케일링된 점수에 **Softmax** 함수를 적용하여 합이 1인 가중치(Attention Weight)를 구합니다.
4.  이 가중치를 각 단어의 **Value 벡터**에 곱하여 가중합(weighted sum)을 구합니다. 이것이 바로 어텐션의 최종 출력값입니다.

이 모든 과정은 다음의 수식으로 압축됩니다.
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$



### Multi-Head Attention
한 번의 어텐션을 수행하는 대신, **여러 개의 어텐션을 병렬적으로 수행**하는 방식입니다.
-   입력 임베딩 벡터를 **h개의 헤드**로 나누어 각각 다른 Q, K, V 행렬을 학습시킵니다.
-   각 헤드는 독립적으로 Scaled Dot-Product Attention을 수행합니다. 이를 통해 모델은 "The animal didn't cross the street because **it** was too tired" 와 같은 문장에서 'it'이 'animal'을 가리키는 관계, 'tired'와 관련 있는 관계 등 다양한 관점의 정보를 동시에 포착할 수 있습니다.
-   각 헤드의 출력 값들을 모두 연결(concatenate)한 후, 최종 가중치 행렬 $W^O$를 곱하여 최종 출력을 생성합니다.

논문에서는 $d_{model}=512$ 차원을 $h=8$개의 헤드로 나누어, 각 헤드는 $d_k=d_v=512/8=64$ 차원을 갖도록 설정했습니다.

### Positional Encoding: 순서 정보의 주입
RNN을 제거하면서 단어의 순서 정보를 잃어버리는 문제가 발생합니다. 트랜스포머는 이를 해결하기 위해 **위치 인코딩(Positional Encoding)** 값을 단어의 임베딩 벡터에 더해줍니다. 사인(sine)과 코사인(cosine) 함수를 이용하여 각 위치마다 고유한 벡터 값을 생성하며, 이를 통해 모델은 단어의 상대적인 위치 관계를 학습할 수 있습니다.

$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}}) $$

### Add & Norm: 잔차 연결과 층 정규화
각 서브 레이어의 입력은 출력을 거친 후, 입력 자신과 더해지는 **잔차 연결(Residual Connection)** 구조를 가집니다. 이는 깊은 신경망에서 기울기가 소실되는 문제를 방지하는 효과적인 방법입니다. 이후 **층 정규화(Layer Normalization)**를 통해 데이터의 분포를 안정시켜 학습 효율을 높입니다.

---

## 5. 학습 및 실험 결과

### 학습 방법
-   **하드웨어**: NVIDIA P100 GPU 8개를 사용.
-   **Optimizer**: Adam Optimizer를 사용했으며, 학습 초기에는 learning rate를 선형적으로 증가시키고 이후에는 역제곱근에 비례하여 감소시키는 스케줄링을 적용했습니다.
-   **정규화**: 잔차 연결 외에 Dropout과 Label Smoothing을 사용하여 과적합을 방지했습니다.

### 성능 평가 (BLEU Score)
-   WMT 2014 영어-독일어 번역 태스크에서 기존 앙상블 모델보다 2.0 BLEU 포인트 높은 28.4를 기록하며 **SOTA(State-of-the-art)**를 달성했습니다.
-   영어-프랑스어 번역 태스크에서도 기존 단일 모델들의 성능을 모두 뛰어넘었습니다.
-   이러한 성능을 기존 모델의 **1/4 미만의 훈련 비용**으로 달성했습니다.

### 모델 복잡도 비교
| 모델 | 레이어당 복잡도 | 순차 연산 | 최대 경로 길이 |
| :--- | :---: | :---: | :---: |
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent (RNN) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |

- **레이어당 복잡도**: Self-Attention은 시퀀스 길이($n$)가 차원($d$)보다 작을 때 RNN보다 효율적입니다.
- **순차 연산 및 최대 경로 길이**: Self-Attention은 순차 연산이 $O(1)$로 병렬화에 유리하며, 단어 간 경로 길이가 $O(1)$로 장기 의존성 문제에 강합니다.

---

## 6. 결론

**Attention Is All You Need** 논문은 다음 세 가지 측면에서 NLP 분야에 혁신을 가져왔습니다.

1.  **RNN 대체**: 순환 구조 없이 Attention만으로 더 뛰어난 성능을 달성하며, NLP 모델의 패러다임을 전환했습니다.
2.  **성능 및 효율성**: 병렬 처리를 통해 기존 모델보다 훨씬 빠르고 효율적으로 훈련하면서도 더 높은 정확도를 기록했습니다.
3.  **확장 가능성**: 본 논문에서 제안된 트랜스포머 아키텍처는 이후 BERT, GPT 등 수많은 후속 모델의 기반이 되었으며, 자연어 처리를 넘어 **컴퓨터 비전(Vision Transformer, ViT)**, 음성 인식 등 AI의 거의 모든 분야로 확장되었습니다.