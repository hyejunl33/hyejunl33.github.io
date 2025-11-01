---
layout: single
title: "[모델최적화]TAPT"
date: 2025-11-01
tags:
  - Domain_Common_Project
  - study
  - ModelOptimization
  - TAPT
excerpt: "[모델최적화]TAPT"
math: true
---

# [모델최적화]TAPT

TAPT는 Task Adaptive Pre-Training의 약자로 내가 풀고자 하는 task에 적응시키기 위해 Pretraining 시키는 과정을 말한다. 기존의 BERT모델을 내가 풀고자 하는 영화리뷰 감성분류에 적응시키기 위해 TAPT를 사용할 수 있다.

일반적인 BERT,KoBERT,KLUE-BERT등은 이미 뉴스나 위키피디아, 댓글등 거대한 데이터에 대해 이미 학습된 상태이다. TAPT는 이러한 사전학습된 모델을 바탕으로 Fine-Tuning하는 과정이다.

# TAPT를 왜 하는가?

- 도메인 불일치: 모델이 사전 학습한 범용 데이터와, 실제 해결해야할 task의 데이터는 불일치한다. 따라서 Fine-Tuning이 필요하다.
- TAPT의 목적: 도메인 사이의 격차를 줄여주는 것이 목적이다. 범용 모델을 원하는 task에 미리 적응시켜서 모델이 task의 도메인의 언어패턴을 더 잘 이해하게 만드는것이 핵심이다. 다만 Overfitting이 되지 않도록 유의해야한다.

# TAPT는 어떻게 하는가?

- 데이터: 학습데이터에 대해서 라벨이 없는 학습데이터를 사용한다. 이 데이터를 가지고 MLM방식을 사용해서 마스크된 단어를 예측하는 방식으로 학습한다.

## MLM: Masked Language Model

주변단어로 마스크된 중심단어를 맞추면서 학습한다. 따라서 문장 전체의 문맥을 반영하여 벡터값을 예측한다. 이 MLM을 이용해서 기존의 BERT모델을 Pretrain하면 모델이 훨씬더 데이터셋을 잘 이해할 수 있게 된다.

**작동방식**

1. 입력문장을 가져온 후에, 일부 토큰을 랜덤하게 [MASK]토큰으로 바꾼다.
2. 모델이 문맥을 보고 [MASK]토큰에 들어갈 단어를 맞추도록 학습한다.
    1. BERT의 기본 인코더 층 위에 MLM을 위한 HEAD를 추가로 붙인다.
    2. [MASK]토큰의 최정 백터(`H_MASK`)만 MLM헤드로 전달된다.
    3. 이 헤드의 최종 단계는 어휘의 사전 크기만큼의 차원으로 확장하는 하나의 Linear계층이다.
    4. 계산된 Logits을 이용해서  BackPropagation을 진행한다. → 가중치를 업데이트 한다

이를 통해서 모델은 주변 문맥을 통해 빈칸을 추론하는 방법을 배운다.

# Task에 맞는 데이터셋 준비

```python
# TAPT.ipynb (Cell 9)
# 1. 내가 풀려는 과제의 데이터를 불러옵니다.
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# 2. 'review' 텍스트만 모두 합칩니다.
all_text_data = pd.concat([train_df['review'], test_df['review']])

# 3. TAPT 학습용 텍스트 파일로 저장합니다.
output_filename = "tapt_corpus.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    for line in all_text_data:
        # ... (생략) ...
```

내가 풀고자하는 Task의 데이터를 불러와서 TAPT에 맞는 데이터셋으로 준비한다. 이때 TAPT과정에서 label은 필요없고, review데이터(텍스트)만을 이용한다.