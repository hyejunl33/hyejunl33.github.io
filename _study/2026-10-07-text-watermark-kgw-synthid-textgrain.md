---
layout: modern-single
title: "텍스트에 워터마크를 어떻게 넣을까: KGW에서 SynthID-Text와 textGrain까지"
date: 2026-10-07
tags:
  - Study
excerpt: "뤼튼에서 인턴으로 일하던 올해 여름, 같은 IAB에서 PM들이 사용할 PRD Generator Agent를 만드는 동료 인턴이 있었다. 8월쯤 그 동료에게 모델을 바꿨다는 이야기를 들었다. Claude Fable API 출력에 텍스트 워터마크가 들어가기 시작한 뒤 단어 선택이 눈에 띄게 어색해졌고, PM들이 결국 A…"
math: true
notion_source_id: "sha256:9dc20ab41d934721bd65"
---

## PRD의 단어 선택이 어색해졌다는 이야기

뤼튼에서 인턴으로 일하던 올해 여름, 같은 IAB에서 PM들이 사용할 PRD Generator Agent를 만드는 동료 인턴이 있었다. 8월쯤 그 동료에게 모델을 바꿨다는 이야기를 들었다. Claude Fable API 출력에 텍스트 워터마크가 들어가기 시작한 뒤 단어 선택이 눈에 띄게 어색해졌고, PM들이 결국 Agent에서 Fable 대신 GPT Sol을 쓰게 됐다는 내용이었다.

이 글을 쓰게 된 출발점은 그 이야기다. PRD는 누가 읽어도 요구사항을 같은 뜻으로 이해해야 하는 문서다. 문장은 얼핏 그럴듯한데 단어가 어색하면, 읽는 사람이 의도를 다시 해석해야 한다. 문서 작성을 도와주는 Agent가 오히려 문장을 고치는 일을 늘린다면 사용하는 입장에서는 모델을 바꿀 이유가 충분하다.

그런데 텍스트에 워터마크를 넣는다는 말이 잘 이해되지 않았다. 이미지에는 픽셀을 조금 바꾸면 되는데, 글자나 단어를 바꾸면 의미까지 달라지지 않을까? 어떤 방식으로 표시를 남기길래 단어 선택과 연결되는지 궁금해서 논문을 찾아봤다.

동료가 전달한 설명은 워터마크 때문에 품질이 떨어졌다는 것이었다. 다만 나는 당시 API의 워터마크를 켜고 끈 대조 실험이나 전후 출력 원문을 가지고 있지는 않다. 이 경험을 출발점으로 삼되, 논문의 실험 결과와 내가 들은 사례를 같은 종류의 증거로 다루지는 않으려 한다.

## 시점과 읽을 자료부터 정리

내가 기억하는 8월과 공식 문서의 날짜도 구분할 필요가 있었다. Anthropic이 제시한 신규 모델의 적용 기준은 2026년 8월 2일이고, 워터마크의 원리를 설명한 공식 글은 8월 14일에 공개됐다. 기존 모델에는 별도 전환 기간이 있다. 따라서 8월 2일을 동료가 호출하던 Fable API에 실제로 워터마크가 들어간 날짜라고 단정할 수는 없다. [Anthropic 공식 설명](https://www.anthropic.com/news/claude-text-watermark), [모델별 적용 범위](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)

단어 선택과 품질을 둘러싼 논쟁은 8월 기사에서도 다뤄졌다. 이 보도는 당시 어떤 문제의식이 있었는지 이해하는 데 참고했고, 구현 원리는 공식 자료와 논문에서 확인했다. [PCWorld, 2026년 8월 17일](https://www.pcworld.com/article/3214192/claude-text-watermarks-nudge-its-word-choices-should-we-care.html)

원리를 이해할 때는 임커밋님의 [「텍스트 워터마크..? 어떻게?」](https://www.youtube.com/watch?v=8CV9GHclD0c)도 참고했다. 2026년 9월 10일 공개된 영상으로, 생성 후보에 점수를 매기는 과정과 그 점수로 워터마크를 검출하는 과정을 연결해 설명한다. 아래 논문 리뷰에서는 그 설명에 분포 보존 조건과 실제 실험의 범위를 덧붙였다.

<div style="width:100%;aspect-ratio:16/9;">
<iframe src="https://www.youtube.com/embed/8CV9GHclD0c" title="임커밋 — 텍스트 워터마크..? 어떻게?" loading="lazy" style="width:100%;height:100%;border:0;" allow="encrypted-media; picture-in-picture" allowfullscreen></iframe>
</div>

| 대상 | 읽을 자료 | 자료의 성격 |
| --- | --- | --- |
| Claude | [Scalable watermarking for identifying large language model outputs](https://www.nature.com/articles/s41586-024-08025-4) | Google DeepMind의 SynthID-Text 논문, Nature, 2024. Anthropic이 기반 방식으로 명시 |
| OpenAI | [textGrain: Entropy-Calibrated Watermarking for Language Model Text](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf) | 2026년 10월 5일 공개된 기술보고서. Nature 논문과 출판 형태가 다름 |
| 비교용 기초 | [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) | Kirchenbauer et al., 2023. 토큰 확률에 편향을 주는 방식을 이해할 때 참고 |

이 글은 KGW의 Green–Red List로 기본 원리를 이해한 다음, 공개된 SynthID-Text와 textGrain 논문을 각각 읽고 차이를 비교하는 순서로 정리했다. KGW는 워터마크 신호를 어떻게 남기고 검출하는지 이해하기 좋은 출발점이다. 이후 두 논문에서는 원래 토큰 분포를 보존하는 조건과 응답 다양성까지 살펴본다.

세 방식은 문맥에서 재현할 수 있는 난수를 토큰 선택에 연결한다는 공통점이 있다. 다만 SynthID-Text와 textGrain이 KGW의 Green 토큰 보너스를 그대로 사용하는 것은 아니다. Claude는 SynthID-Text의 변형을 사용한다고 설명하고, OpenAI는 textGrain을 공개했다. 공개 논문의 파라미터를 각 회사의 실제 서비스 설정으로 그대로 읽어서도 안 된다.

## 먼저 LLM이 다음 토큰을 고르는 방식

LLM은 문장을 한 번에 완성하지 않는다. 지금까지의 문맥을 보고 다음 토큰의 확률을 계산한 뒤, 그 분포에서 하나를 고른다. 토큰은 단어 전체일 수도 있고 단어의 일부일 수도 있다. 아래 예제에서는 설명을 쉽게 하려고 단어 하나를 토큰 하나처럼 다뤘다.


<div markdown="1" style="max-width:100%;overflow-x:auto;">

$$
p_t(v)=\operatorname{softmax}(z_t/T)_v,\qquad x_t\sim p_t(\cdot\mid x_{<t})
$$

</div>


z는 모델의 logit, T는 temperature다. pₜ는 시점 t에서의 다음 토큰 분포이고, xₜ는 그 분포에서 뽑은 토큰이다. 실제 생성에는 top-p 같은 설정도 들어갈 수 있다. 핵심은 보통의 샘플링이 항상 확률 1위 토큰만 고르는 방식은 아니라는 점이다.

예를 들어 ‘변경 이력은 \_\_\_’ 다음에 ‘보관한다’가 0.60, ‘저장한다’가 0.30, ‘기록한다’가 0.10이라고 해보자. 이 숫자는 설명을 위해 만든 값이며 실제 PRD 출력에서 측정한 값은 아니다. 워터마크가 없어도 0.30짜리 후보가 나올 수 있다. 그러므로 ‘최고 확률의 단어가 선택되지 않았다’는 사실만으로 품질 저하나 워터마크를 판정할 수는 없다.

텍스트 워터마크는 이 선택 과정에 비밀키와 연결된 통계적 패턴을 남긴다. 검출기는 나중에 결과 텍스트를 읽으며 같은 키로 패턴을 재구성한다. 글 뒤에 표시 문장을 붙이거나 보이지 않는 공백을 넣는 방식과는 작동 위치가 다르다.

![synthid overview](/assets/images/notion/9dc20ab41d934721bd65/01-synthid-overview.png)

그림 1. SynthID-Text 논문의 Fig. 1을 캡처했다. 생성할 때 샘플링에 키가 개입하고, 검출할 때 텍스트와 키를 사용한다. 출처: [Dathathri et al., 2024](https://www.nature.com/articles/s41586-024-08025-4).

## KGW: Green–Red List로 기본 원리 이해하기

![KGW Algorithm 1의 데이터 흐름: 입력 logits와 문맥에서 Green 마스크, 보너스, 토큰 샘플링으로 이어지는 과정](/assets/images/notion/9dc20ab41d934721bd65/kgw-paper-flow.gif)

KGW 논문의 Algorithm 1에서 워터마크 생성에 필요한 입력은 모델의 logits와 앞선 문맥이다. 애니메이션은 문맥으로 만든 Green 마스크가 입력 logits의 B 성분에 δ를 더하고, 바뀐 벡터가 softmax를 거쳐 다음 토큰으로 선택되는 과정을 보여준다. logits와 후보는 설명용 수치이며 실제 출력에서 측정한 값은 아니다. [Algorithm 1의 데이터 흐름 크게 보기](/assets/images/notion/9dc20ab41d934721bd65/kgw-paper-flow.gif)

KGW는 Kirchenbauer et al.의 「A Watermark for Large Language Models」에서 제안한 방식이다. 다음 토큰을 고를 때 어휘 집합 V를 Green List Gₜ와 나머지 Red List Rₜ로 나눈다. 앞선 토큰 문맥으로 의사난수 생성기의 시드를 정하기 때문에, 검출기도 같은 설정으로 그 시점의 분할을 재현할 수 있다. 시드 구성이나 문맥 길이는 구현에 따라 달라질 수 있다.

Green은 좋은 단어, Red는 나쁜 단어라는 뜻이 아니다. 같은 토큰도 문맥이 바뀌면 다른 집합에 들어갈 수 있다. 핵심은 생성기가 알고 검출기가 다시 만들 수 있는 임의의 구분이다.

일반적으로 설명하는 soft KGW는 Green 토큰의 logit에 δ를 더한 뒤 샘플링한다. Red 토큰의 확률을 0으로 만드는 것은 아니다. 논문의 hard 방식과 구분해서 읽어야 한다. 아래 식은 temperature 등을 적용한 생성용 logit을 z로 표기한 soft 방식이다.


<div markdown="1" style="max-width:100%;overflow-x:auto;">

$$
\widetilde p_t(v)=\frac{\exp(z_t(v)+\delta\,\mathbf{1}\{v\in G_t\})}{\sum_{u\in V}\exp(z_t(u)+\delta\,\mathbf{1}\{u\in G_t\})}
$$

</div>


Gₜ는 그 시점의 green list, δ는 보너스 크기다. 표시된 집합에 속하면 logit에 δ를 더하고 softmax를 다시 계산한다. ‘저장한다’에 보너스가 붙으면 원래보다 선택될 가능성이 높아질 수 있다. δ를 크게 잡을수록 표시는 강해지지만 원래 분포와의 차이도 커진다. [Kirchenbauer et al., 2023](https://arxiv.org/abs/2301.10226)

### Green 토큰이 우연보다 많이 나왔는지 검출한다


<div markdown="1" style="max-width:100%;overflow-x:auto;">

$$
z_{\mathrm{detect}}=\frac{C-\gamma N}{\sqrt{N\gamma(1-\gamma)}}
$$

</div>


검출기는 출력의 각 위치에서 Green List를 다시 만들고, 실제 토큰이 Green에 속한 횟수 C를 센다. Green이 전체 어휘에서 차지하는 비율을 γ, 검사한 토큰 수를 N이라고 하면, 워터마크가 없는 텍스트의 기준 모형에서는 Green 횟수를 대략 γN으로 기대한다. 표준화 통계량은 z = (C − γN) / √(Nγ(1 − γ))다. 이 z는 위 식의 모델 logit과 다른 값이다.

예를 들어 γ = 0.5이고 N = 100이면 기준 기대 횟수는 50, 표준편차는 5다. Green 토큰이 70개 관측되면 z = 4가 된다. 이 숫자는 검출 원리를 보여주기 위한 예시다. 특정 임계값을 통과했다고 작성자의 신원이나 사용 모델까지 확정되는 것은 아니다. 반복 문맥 때문에 관측이 독립적이지 않을 수 있어, 실제 검출에서는 반복 처리와 오검출률 보정이 필요하다. [KGW 논문의 생성·검출 알고리즘](https://arxiv.org/abs/2301.10226)

### 확률을 밀어주는 것과 분포를 보존하는 것

앞의 PRD 예제에서 ‘저장한다’만 Green이고 δ = log 2라고 해보자. 원래 확률 0.60, 0.30, 0.10에 곱해지는 가중치는 1, 2, 1이다. 정규화하면 ‘보관한다’는 약 0.462, ‘저장한다’는 약 0.462, ‘기록한다’는 약 0.077이 된다. Green 후보가 더 자주 나오도록 원래 분포가 바뀐 것이다. 좋은 후보가 여럿인 문맥에서는 선택 여지가 있지만, 특정 표현이 필요한 문맥에서는 다른 후보를 밀어주는 효과를 더 주의해서 봐야 한다.

KGW는 이런 분포 변화와 검출력을 함께 다룬다. δ를 높이면 신호가 강해지는 대신 분포 변화도 커질 수 있다. 두 후속 논문에서 내가 더 보고 싶었던 것은 이 관계를 다루는 방법이었다. SynthID-Text 논문은 KGW를 Soft Red List 비교 기준으로 사용하고, textGrain의 Appendix C도 Green List 방식과 자신의 접근을 구분한다. 인용하고 비교한다는 사실이 같은 알고리즘을 사용한다는 뜻은 아니다.

## Claude의 기반 방식: SynthID-Text

### 후보 토큰끼리 토너먼트를 시킨다

KGW에서는 Green 토큰에 보너스를 더하고 확률을 다시 계산했다. SynthID-Text의 Tournament Sampling은 원래 LLM 분포에서 후보들을 뽑은 뒤, 문맥과 비밀키로 만든 의사난수 점수에 따라 두 후보씩 겨룬다. 동점은 무작위로 처리하고, 여러 층을 거쳐 남은 토큰을 출력한다. 높은 워터마크 점수는 좋은 문장이라는 뜻이 아니다. 키와 연결된 임의의 점수다.

점수에 0과 1을 사용하면 두 색으로 나누는 것처럼 보일 수 있다. 그래도 생성 절차는 다르다. KGW의 soft 방식은 전체 어휘의 logit을 수정하고, 여기서는 원래 분포에서 나온 후보들의 대결이 선택 확률을 정한다. 논문은 이 토너먼트가 특정 조건에서 원래 분포를 보존하는 이유를 분석한다.

![synthid tournament](/assets/images/notion/9dc20ab41d934721bd65/03-synthid-tournament.png)

그림 2. SynthID-Text 논문의 Fig. 2. 위는 문맥·키로 만드는 점수, 아래는 후보 토큰의 토너먼트다. 출처: [Dathathri et al., 2024](https://www.nature.com/articles/s41586-024-08025-4).

그림에는 mango, lychee, papaya, durian이 등장한다. 확률이 높은 토큰은 처음 후보를 뽑을 때 더 자주 나온다. 이후 각 대결에서는 해당 층의 점수가 승자를 정한다. 후보를 여러 개 만든다고 모델에게 문장을 여러 번 끝까지 작성하게 시키는 것은 아니다. 다음 토큰의 샘플링 단계에서 일어나는 과정이다.

![SynthID-Text Fig. 2의 데이터 흐름: 논문의 과일 후보들이 층별 점수로 선택되어 mango가 출력되는 과정](/assets/images/notion/9dc20ab41d934721bd65/synthid-paper-flow.gif)

Fig. 2의 예제를 움직임으로 옮겼다. mango·lychee·papaya·durian의 확률은 각각 0.50·0.30·0.15·0.05다. 그 분포에서 나온 후보 8개에 문맥·키로 만든 g₁ 점수가 붙고, 후보가 4개, 2개로 줄면서 g₂와 g₃를 차례로 적용한다. 동점에서는 무작위 선택이 일어나고, 그림의 경로에서는 mango가 다음 토큰으로 남는다. 같은 문맥·층에서 같은 토큰은 같은 점수를 받는다. 논문의 설명용 예시를 따른 것이며 실제 Claude 출력이 아니다. [Fig. 2의 데이터 흐름 크게 보기](/assets/images/notion/9dc20ab41d934721bd65/synthid-paper-flow.gif)

### 키 점수를 보고 고르면서 원래 분포를 유지할 수 있을까?

내가 가장 궁금했던 부분이다. 이를 이해하려고 두 토큰만 있는 장난감 예제를 만들어봤다. 원래 분포가 A: 0.8, B: 0.2이고, 두 후보를 독립적으로 뽑는다고 하자. 같은 토큰끼리 붙으면 그 토큰이 남고, 서로 다르면 키의 점수가 승자를 정한다.

| 후보 쌍 | 확률 | 승자 |
| --- | --- | --- |
| A, A | 0.8² = 0.64 | A |
| A, B 또는 B, A | 2 × 0.8 × 0.2 = 0.32 | 키 점수에 따라 달라짐 |
| B, B | 0.2² = 0.04 | B |

키 점수의 분포가 A와 B를 대칭적으로 취급한다면, 서로 다른 후보가 붙었을 때 A가 이길 확률은 키에 대해 평균내어 1/2이다. 따라서 A의 최종 확률은 0.64 + 0.32 × 0.5 = 0.80이다. B도 0.20으로 돌아온다. 키 점수에 영향을 받는데도 평균적인 토큰 분포는 유지되는 것이다.

그런데 같은 문맥에서 키 점수를 고정해 A가 항상 이긴다면 A의 확률은 0.64 + 0.32 = 0.96이다. B가 항상 이기면 A의 확률은 0.64다. 둘을 평균내면 다시 0.80이 된다. 이 계산은 특정 문맥·키에서의 조건부 분포와 키에 대해 평균낸 분포가 다를 수 있다는 것을 보여준다. 실제 Claude의 내부 설정을 재현한 실험은 아니다.


<div markdown="1" style="max-width:100%;overflow-x:auto;">

$$
\mathbb{E}_{r}\!\left[q(v\mid x_{<t},r)\right]=p_{\mathrm{LM}}(v\mid x_{<t})
$$

</div>


논문이 구분하는 한 토큰·한 문장·여러 응답의 분포 보존은 보장 범위가 서로 다르다. 반복 문맥 처리도 영향을 준다. 평균 분포가 같다는 한 문장만 읽고, 고정된 키를 쓰는 실제 서비스의 모든 출력이 동일하거나 응답 다양성이 그대로라고 받아들이면 안 된다. 반대로 위 예제에서 조건부 분포가 달라진다는 이유만으로 모든 워터마크가 평균 품질을 반드시 떨어뜨린다고 결론낼 수도 없다.

### 검출은 점수가 우연보다 높은지 보는 일

검출기는 결과 글의 문맥과 키로 점수 함수를 다시 만들고, 관측된 토큰들의 점수를 모은다. 생성 때 점수가 높은 후보가 더 자주 살아남았다면, 충분히 긴 글에서는 우연으로 기대되는 점수와 차이가 생긴다.


<div markdown="1" style="max-width:100%;overflow-x:auto;">

$$
\operatorname{Score}(x)=\frac{1}{mN}\sum_{t=1}^{N}\sum_{\ell=1}^{m}g_\ell(x_t,r_t)
$$

</div>


이 식은 논문에 제시된 평균 점수다. m은 점수 함수의 층 수, N은 점수를 계산한 토큰 수다. 실제 검출에는 반복 문맥 처리와 임계값 보정이 필요하다. 점수가 높다는 것은 선택한 기준 아래에서 그 키의 워터마크와 일치하는 증거가 강하다는 뜻이지, 작성자의 신원이나 글의 정확성을 증명하는 것은 아니다.

## OpenAI의 textGrain: 키와 토큰의 의존성을 조절한다

OpenAI는 2026년 10월 5일 textGrain 보고서를 공개했다. API는 당시 기본적으로 워터마크가 꺼져 있고, 지원 모델에 대해 고객이 선택해서 켤 수 있다고 발표했다. 이 현재 정책만으로 여름에 동료가 사용한 GPT Sol의 당시 설정까지 알 수는 없다. [OpenAI 발표](https://openai.com/index/eu-text-provenance/)

![textgrain embedding](/assets/images/notion/9dc20ab41d934721bd65/05-textgrain-embedding.png)

그림 3. textGrain 보고서의 Fig. 1. 토큰을 키 기반 블록으로 묶고, 엔트로피 예산 아래에서 블록의 조건부 분포를 구성한 뒤 토큰을 뽑는다. 출처: [OpenAI 기술보고서, 2026](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf).

### 어휘 블록을 만들되, Green 보너스를 주지는 않는다

![textGrain Fig. 1의 데이터 흐름: 여섯 토큰이 키 기반 블록으로 모이고 선택된 열과 블록을 거쳐 cold가 출력되는 과정](/assets/images/notion/9dc20ab41d934721bd65/textgrain-paper-flow.gif)

Fig. 1의 여섯 토큰을 그대로 따라갔다. warm·calm은 확률 0.40인 블록, cold·sunny는 0.35인 블록, mild·bright는 0.25인 블록으로 합쳐진다. 문맥·키에서 나온 비용표는 Sinkhorn과 엔트로피 예산 갱신을 거쳐 결합분포를 만들고, 키 기반으로 고른 열은 그림에서 블록 선택 확률 0.04·0.69·0.27을 준다. 선택된 {cold, sunny} 안에서는 원래 비율 0.25:0.10으로 샘플링해 cold가 출력된다. 애니메이션의 비용표와 결합분포 도형은 계산의 연결을 보여주는 표현이며, 최적수송의 수치해를 재현한 것은 아니다. [Fig. 1의 데이터 흐름 크게 보기](/assets/images/notion/9dc20ab41d934721bd65/textgrain-paper-flow.gif)

textGrain도 키와 문맥을 이용해 어휘를 나누지만, 여러 블록에 토큰을 배정한다. 각 블록의 원래 확률은 그 안에 있는 토큰 확률의 합이다. 블록을 선택한 다음에는 블록 내부의 원래 상대 확률로 토큰을 뽑는다. 이 부분을 KGW의 Green/Red 분할과 혼동하기 쉬웠다.

블록 선택을 정하는 것은 optimal transport, 즉 최적수송으로 구성한 결합분포다. 블록과 키 기반 난수의 조합에 Gumbel 값으로 비용을 정하고, 선호되는 조합이 더 자주 선택되게 한다. 동시에 주변분포 제약을 두어 난수에 대해 평균내면 원래 블록 확률로 돌아오게 한다. 독립분포에서 벗어나는 정도에는 KL 페널티를 붙여 의존성을 제한한다. 어휘 전체 대신 블록에서 계산하면 최적화 차원도 줄일 수 있다.

예를 들어 확률이 0.60인 토큰 A와 0.10인 C가 같은 블록이라면, 그 블록을 고른 뒤에는 A:C = 6:1로 샘플링한다. 블록 선택은 키에 의존하지만 내부 비율은 유지한다. 키 기반 난수에 대해 평균낸 블록 선택 확률까지 원래 0.70으로 맞추면, A와 C의 전체 평균 확률도 각각 0.60과 0.10으로 돌아온다. 이는 원리를 설명한 예이며 실제 서비스의 블록 구성이 아니다.

### 워터마크 강도를 엔트로피 예산으로 정한다

분포를 평균적으로 보존해도 고정된 키 아래에서 선택이 거의 결정적일 수 있다. textGrain은 그래서 평균적인 조건부 샘플링 엔트로피 손실에 예산을 둔다. 아래 식의 KL 발산은 키 기반 난수와 토큰의 상호정보량이며, 키를 알게 됐을 때 줄어든 샘플링 불확실성과 같다.


<div markdown="1" style="max-width:100%;overflow-x:auto;">

$$
\begin{aligned}D_{\mathrm{KL}}(\pi\|P\otimes\mu)&=I(W;\Xi)=H(W)-H(W\mid\Xi)\\ I(W;\Xi)&\leq\beta H(W)\end{aligned}
$$

</div>


W는 선택한 토큰, Ξ는 키 기반 난수, π는 둘의 결합분포다. 이 식은 두 변수의 의존성이 커질수록 키를 안 상태에서 남는 샘플링 엔트로피가 줄어든다는 뜻이다. H는 엔트로피로, H(W) = −Σᵥ p(v) log p(v)다. β는 허용 손실의 비율이다. 예를 들어 H(W)가 2이고 β가 0.1이면 평균 손실을 0.2 이하로 제한하는 식이다. 이 숫자는 원리를 설명하기 위한 예이며 서비스 설정이 아니다.

여기서 ‘엔트로피 손실 10%’를 ‘문장 품질 10% 하락’으로 해석하면 안 된다. 측정하는 것은 선택의 불확실성이며 문장의 자연스러움 점수가 아니다. 원래 확률분포에서 매우 좁게 샘플링하는 모델도 좋은 문장을 만들 수 있고, 반대로 다양하게 생성한다고 PRD에 적합한 표현이 보장되지도 않는다.

![textgrain detection](/assets/images/notion/9dc20ab41d934721bd65/07-textgrain-detection.png)

그림 4. textGrain 보고서의 Fig. 2. 관측한 토큰의 블록과 키 기반 점수를 복원해 검정 통계량을 만든다. 출처: [OpenAI 기술보고서, 2026](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf).

### 검출기는 생성 때의 확률분포를 몰라도 된다

검출기는 출력 토큰과 앞선 문맥, 키를 이용해 각 토큰의 블록과 Gumbel 기반 점수를 재구성한다. 생성 과정에서 키와 연결된 조합을 더 자주 선택했다면, 관측 점수를 누적했을 때 기준 분포와 차이가 생긴다. 보고서는 이를 통계적 증거로 바꾸는 검출 절차와 보장 조건을 설명한다. 같은 키·토크나이저·설정은 필요하지만 생성 모델의 다음 토큰 확률이나 생성 시 엔트로피 예산은 필요하지 않다.

KGW는 Green 횟수, SynthID-Text는 토너먼트 점수, textGrain은 블록과 난수의 관계를 관측한다. 서로 다른 통계량을 사용해도, 생성 때 남긴 패턴이 우연으로 얼마나 설명되는지 묻는다는 점은 이어진다. 짧은 글은 증거가 적고, 편집으로 토큰과 문맥이 바뀌면 신호가 약해질 수 있다. 평균 분포 보존은 모든 재작성 공격에 대한 검출 보장이 아니다.

## KGW를 기준으로 두 논문의 차이 비교하기

| 비교 항목 | KGW · soft 방식 | SynthID-Text · 비왜곡 설정 | textGrain |
| --- | --- | --- | --- |
| 생성 | Green 토큰 logit에 δ 추가 | 원래 분포에서 뽑은 후보의 점수 토너먼트 | 블록과 키 기반 난수의 최적수송 결합 후 토큰 샘플링 |
| 검출 | Green 토큰의 초과 출현 | 출력 토큰의 누적 워터마크 점수 | 관측 블록과 재구성한 난수 점수의 통계적 관계 |
| 분포 | 보너스로 원래 분포 변경 | 설정과 조건에 따라 난수에 대해 평균낸 분포 보존 | 주변분포 제약으로 난수에 대해 평균낸 다음 토큰 분포 보존 |
| 핵심 조절 대상 | Green 비율 γ, 보너스 δ | 토너먼트 층 수와 점수·반복 문맥 설정 | 블록 결합과 평균 엔트로피 손실 예산 |

내가 읽은 연결점은 KGW가 보여준 ‘샘플링에 패턴을 남기면 나중에 통계적으로 검출할 수 있다’는 원리다. SynthID-Text는 이를 후보 토너먼트로 구현하고 분포 보존과 대규모 운영을 살핀다. textGrain은 결합분포를 직접 설계하며 고정된 키에서 남는 샘플링 자유도까지 수치로 조절한다. 이 비교는 설명을 위한 순서이며, KGW에서 두 방식으로 단일하게 발전했다는 계보를 주장하는 것은 아니다. [SynthID-Text의 Evaluation](https://www.nature.com/articles/s41586-024-08025-4), [textGrain의 Sections 2–3 및 Appendix C](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf)

## 논문의 품질 실험은 무엇을 보여줄까?

![synthid evaluation](/assets/images/notion/9dc20ab41d934721bd65/08-synthid-evaluation.png)

그림 5. SynthID-Text 논문의 Fig. 3. 왼쪽은 길이에 따른 검출률, 가운데는 선택적 검출의 보류 비율, 오른쪽은 분포를 바꾸는 설정의 검출률·log perplexity 관계다. 출처: [Dathathri et al., 2024](https://www.nature.com/articles/s41586-024-08025-4).

왼쪽 그래프의 세로축은 TPR@FPR=1%다. 워터마크가 없는 글을 잘못 양성으로 판정하는 비율을 1%로 맞췄을 때, 워터마크가 있는 글을 얼마나 찾아내는지 보여준다. ‘검출기의 전체 정확도가 99%’라는 뜻이 아니다. 텍스트가 길수록 모을 수 있는 증거가 늘어난다. 가운데의 abstention도 오검출률과 다른 값으로, 판단을 보류하는 비율이다.

오른쪽 그래프는 distortionary 설정을 비교한다. 워터마크의 존재 자체와 강한 검출 신호를 위해 분포를 바꾸는 설정을 구분해서 읽어야 한다. perplexity는 생성 글이 모델 아래에서 얼마나 그럴듯한지 보는 대리 지표다. PRD의 요구사항이 정확한지, 팀에서 자연스럽게 읽히는지는 별도로 평가해야 한다.

SynthID-Text 논문은 비왜곡 설정에서 사람의 품질 비교와 약 2천만 건의 Gemini 응답 피드백 등을 살폈고, 유의한 품질 차이를 관찰하지 않았다고 보고한다. 이 결과는 해당 모델·설정·평가에서의 근거다. Claude Fable이 만든 한국어 PRD를 직접 비교한 실험 결과는 아니다.

Anthropic도 자체 평가에서 내용·창의성·가독성에 영향을 관찰하지 않았다고 설명한다. OpenAI 역시 Astra 벤치마크와 제품 지표에서 의미 있는 품질 차이를 보지 못했다고 발표했다. 회사가 제시한 평균 평가 결과와 특정 팀의 체감 품질은 측정하는 범위가 다르다. [Anthropic 평가 설명](https://www.anthropic.com/news/claude-text-watermark), [OpenAI 평가 설명](https://openai.com/index/eu-text-provenance/)

임커밋 영상의 [6:09 이후](https://www.youtube.com/watch?v=8CV9GHclD0c&t=369s)에서는 품질과 무관한 기준이 선택에 개입할 때 생길 수 있는 성능 문제를 설명한다. 그 문제의식은 내가 들은 PRD 사례와 연결된다. 다만 일반적인 LLM 샘플링도 항상 1위 후보를 고르는 것은 아니므로, 그 설명을 ‘워터마크는 반드시 최선의 단어를 버리고 평균 품질을 떨어뜨린다’는 증명으로 읽지는 않았다.

## 다시 PRD Generator Agent의 페인포인트로

처음에는 어떻게 표시를 숨기는지가 궁금했는데, 논문을 읽고 나니 모델 선택 이후에 어떤 평가를 붙여야 하는지가 더 구체적으로 보였다. PM이 불편해한 것은 워터마크의 검출 가능성보다 실제로 읽고 쓸 문장의 품질이었다.

예를 들어 ‘사용자는 결제 전에 총액을 확인할 수 있어야 한다’에서 ‘있어야 한다’를 ‘있을 수 있다’로 바꾸면, 문법적으로는 문장이지만 필수 요구사항의 강도가 달라진다. ‘저장한다’와 ‘보관한다’도 문맥에 따라 책임과 기간이 다르게 읽힐 수 있다. 이 문장들은 당시 출력의 인용이 아니라, PRD에서 어휘 선택을 평가할 이유를 설명하려고 만든 예시다.

그래서 이 Agent를 평가한다면 다음처럼 나눠서 보겠다. 아래는 당시 수행한 실험이 아니라 논문을 읽고 정리한 평가안이다.

| 볼 항목 | PRD에서 확인할 내용 |
| --- | --- |
| 단어의 자연스러움 | PM이 수정한 어색한 표현을 문장 단위로 기록하고, 문서 길이를 고려해 비교 |
| 요구사항 의미 | 필수·권장·선택의 강도, 주체, 조건, 수치가 원래 입력과 일치하는지 확인 |
| 용어 일관성 | 동일한 기능·화면·상태를 문서 전체에서 같은 이름으로 부르는지 확인 |
| 실제 수정 부담 | 완성까지의 수정 시간과 모델을 모르는 상태에서의 PM 선호도를 비교 |
| 응답 다양성 | 같은 요청을 반복했을 때 문구만 반복하는지, 다른 유효한 접근도 나오는지 확인 |

원인을 분리하려면 같은 모델 버전·프롬프트·문맥·샘플링 설정을 유지한 채 워터마크 유무만 비교할 수 있어야 한다. 공급자가 그런 조건을 제공하지 않는다면, Fable과 GPT Sol의 비교는 사용성을 판단하는 데 도움이 되지만 워터마크만의 효과를 측정하는 대조 실험은 아니다. 모델 자체의 차이도 함께 들어 있기 때문이다.

그렇다고 PM들이 겪은 불편함을 없던 일로 볼 이유는 없다. 당시에는 Fable의 단어 선택이 어색해졌고, 문서를 사용하는 PM들이 GPT Sol로 전환했다는 구체적인 경험이 있었다. 팀이 쓸 모델을 고르는 결정에는 그 사용성이 중요하다. 다만 원인을 설명하는 글에서는 관찰한 문제와 따로 검증한 인과관계의 범위를 밝혀두는 편이 맞다고 생각한다.

이번에 배운 것은 텍스트 워터마크가 샘플링의 자유도를 이용해 남기는 통계적 신호라는 점이다. 그 신호를 강하게 남기는 일, 원래 분포를 보존하는 일, 같은 프롬프트에서 다양한 응답을 얻는 일은 함께 따져야 한다. PRD Generator Agent에서는 여기에 요구사항의 의미와 PM의 수정 부담까지 붙는다. ‘품질에 영향이 없다’는 설명을 읽을 때도, 그 품질이 내가 만들고 있는 제품에서 무엇을 뜻하는지부터 확인해야겠다.

## 참고 자료

- [Dathathri et al. — SynthID-Text, Nature, 2024](https://www.nature.com/articles/s41586-024-08025-4)

- [Li et al. — textGrain 기술보고서, 2026](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf)

- [Kirchenbauer et al. — A Watermark for Large Language Models, 2023](https://arxiv.org/abs/2301.10226)

- [Anthropic — How Claude’s text watermark works, 2026.08.14](https://www.anthropic.com/news/claude-text-watermark)

- [Claude 모델별 워터마크 적용 범위](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)

- [OpenAI — Our approach to EU text provenance rules, 2026.10.05](https://openai.com/index/eu-text-provenance/)

- [임커밋 — 텍스트 워터마크..? 어떻게?, 2026.09.10](https://www.youtube.com/watch?v=8CV9GHclD0c)

본문의 데이터 흐름 애니메이션 3개는 3Blue1Brown의 오픈소스 [Manim](https://github.com/3b1b/manim)에서 출발한 [Manim Community](https://www.manim.community/)로 만들었다.
