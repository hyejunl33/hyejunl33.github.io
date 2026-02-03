---
layout: single
title: "[이미지기반 카페추천 프로젝트] CLIP으로 이미지 전처리"
date: 2026-01-30
tags:
  - 이미지기반 카페추천 프로젝트
  - CLIP으로 이미지 전처리
excerpt: "CLIP으로 이미지 전처리"
math: true
---

# CLIP으로 이미지 전처리

## 문제정의

크롤링된 가게 이미지에 노이즈가 너무 많다. 서비스 특성상 가게의 분위기를 나타내는 사진이 DB에 있어야 하고, DB에 있는 사진과 인풋으로 들어온 사진의 임베딩을 코사인유사도를 이용해서 비교한다. 그런데 가게 이미지 50장을 크롤링해온 결과에서 대표사진 5장을 뽑을때, 노이즈를 제거하고, 해당 가게의 ‘분위기’를 나타내는 사진만 선정해야한다. 따라서 ‘분위기’를 어떤메트릭으로 정의하고, 선정할것인지가 문제였다.

### 오늘의집 케이스스터디

[오늘의집에서도 비슷한 문제가 있었다.](https://www.bucketplace.com/post/2023-05-22-%EC%9C%A0%EC%82%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B6%94%EC%B2%9C-%EA%B0%9C%EB%B0%9C-1-%EB%B9%84%EC%8A%B7%ED%95%9C-%EA%B3%B5%EA%B0%84/)

오늘의집에서도 주어진 인테리어와 유사한 다른 이미지 검색을 위해서 VGG16모델을 학습시키는 모델을 개발했는데, Finetuning시키는 데이터셋을 직접 tagging하는 방법을 사용했다. 

우리의 case에서 clip을 이용해서 ‘분위기’를 나타내는 대표사진 5장만 걸리낼 수 있는 메트릭 설정만 잘한다면, finetuning없이 프롬프트만으로 걸러낼 수 있을거라고 생각했다.

## 이미지 전처리 기존방식

팀원이 CLIP을 이용해서 대표 이미지 5장만 뽑는 로직을 만들어서 파이프라인안에 잘 통합되어있었다. 그런데, 어느정도 프로젝트를 완성시키고 MVP결과를 확인해보니, 검색결과에 너무 ‘분위기’와는 관련없는 결과가 많이 등장함을 확인했다. 따라서 기존에 대표이미지 5장을 선정하는 로직을 개선해서, 최대한 ‘분위기’를 나타내는 사진만 대표이미지로 선택하는 목표를 설정하고 이를 해결해나갔다.

![image](/assets/images/2026-02-04-01-05-34.png)

### 기존방식

- 인테리어 vs 음식점수를 프롬프트에 대한 softmax 확률값을 점수로 매김
- 점수가 가장 높은 5개가 대표이미지로 선정됨.
- 후보군선정
    - `interior_score` > `food_score`
        - 예외. 모두 인테리어점수가 음식점수보다 낮다면  `interior_score`가 높은 순서
- ****대표성 점수 계산
    1. 중심벡터 만들기: 선정된 후보군 이미지들의 임베딩 벡터를 모두 평균내어 하나의 '중심 벡터만들기
    2. 유사도 계산: 각 후보 이미지의 중심벡터에 대한 Cosine Similarity를 계산
- **최종 점수 산출**
    - Final Score=(0.6×Interior Score)+(0.4×Similarity to Centroid)

**Top-K를 뽑는 근거**

- 0.6을 Weight로 줘서 인테리어 점수와 가까운가? 판단
- 0.4를 Weight로 줘서 Top5 이미지의 평균과 얼마나 가까운가 판단

## **기존 방식의 맹점**

1. Food 점수와 인테리어 점수가 일관되지 않다. 즉 같은 사진이더라도 점수가 다른경우가 존재하고, Food점수는 실제 음식임을 대변하는 점수가 되지 않는다.(소금빵 확대샷이 카페이미지보다 food점수가 낮음)
    - 맹점 1에대한 생각
        
        Softmax에다가 인테리어점수랑 음식점수를 넣는데, 그 과정에서 결과물의 점수가 일관되지 않은것같음 → 절대적인 비교에 Softmax가 필요한가? → softmax말고, 분위기 점수를 일관되게 뽑고,분위기 점수에 thr을 정해서 분위기가 아닌 사진은 걸러내는 알고리즘 → 분위기점수가 thr이 넘는 사진이 5개가 안되는경우 해당 카페는 아예 json에서 제외 → Thr을 주피터노트북에서 바꿔가면서 실험해봐야함
        
2. 평균과 유사한 이미지를 0.4만큼 반영하는데, 이 로직때문에 비슷한 사진만 5장 뽑힌다 → 분위기를 반영하더라도 클러스터링 되어있지 않은 사진이면 뽑히지 않는다.
    - 맹점 2에 대한 생각
        
        평균과 유사한 이미지를 고르는 로직을 뺴는건 어떤가? → 중심벡터를 구해서 비슷한 이미지를 가중치주는 로직은 삭제해야됨
        

## **해결안**

‘분위기’를 점수로 어떻게 나타낼것인가? 에대한 해답을 프롬프트에서 찾았다.  대표이미지로 선택하고자 하는 이미지들의 특징을 positive의 프롬프트로 넣어주었고, 걸러내야할 노이즈 이미지들의 특징을 Negative 프롬프트로 넣어주었다. 이미지마다 Positive와 Negative점수를 내서 그 차(delta score)를 최종 점수로 삼았다.

### Metric 개선

$Metric = Interior Score - Noise Score$

$Interior Score$와 $Noise Score$는 CLIP으로 프롬프트와 이미지의 코사인 유사도를 구해서 구함

**메트릭 설정 과정**

- 크롤링된 사진의 특징을 보고 눈으로 걸러내야할 사진과 포함해야할 분위기 사진의 특징을 파악한다 → 프롬프트로 사용한다.
- 걸러야 할 사진은 Negative(`Noise Score`)에, 포함해야할 사진은 Positive(`Interior Score` )에 프롬프트로 포함시킨다.
- 프롬프트와의 Cosine 유사도를 구한 후 `Interior Score - Noise Score`를 메트릭으로 사용한다
- 분포를 확인하고, 잘 걸러지는지 확인한다.

```python
PROMPTS = {
    "positive": [
        "an interior photo of a cafe", 
        "cozy cafe atmosphere", 
        "indoor lighting and furniture",    
        "wide angle shot of a room",
        # "architectural design of a building exterior", #건물 외관
        "interior design details and decorations", #소픔이나 내부디자인
        "outdoor terrace seating area of a coffee shop"  #야외 테라스
    ],
    "negative": [
        "close-up photo of food", 
        "dessert or cake", # 디저트 얼빡샷 
        "a photo of a menu board, price list or signage", 
        "text document or receipt", 
        "a selfie of a person",
        "beverage in a plastic cup"#컵사진
        "a close-up photo of a beverage cup or hand holding a drink", #사람이 들고있는 컵사진
        "a graphic logo, illustrative icon, or screenshot" #카페 로고
]}
```

### 결과1

한개 카페에 대한 분포

![image](/assets/images/2026-02-04-01-05-51.png)
인테리어 스코어와 노이즈 스코어의 분포(좌), 두개의 차인 Delta Score의 분포(우)

![image](/assets/images/2026-02-04-01-05-59.png)
Top 10 Selected Images (Highest delta_score):

![image](/assets/images/2026-02-04-01-06-08.png)
Worst 5 Images(Lowest delta_score)

델타 스코어의 분포를 보면 느좋카페와 안 느좋카페를 구분할 수 있다.

느좋카페는 thr이 0보다 큰 사진이 많은 반면, 안 느좋 카페는 thr이 0보다 작은 사진이 더 많다.

위의 오른쪽 분포를 보면 score가 0보다 큰 사진이 많으므로 느좋사진이 많다고 볼 수 있다.

### 느좋카페 예시1

![image](/assets/images/2026-02-04-01-06-19.png)

![image](/assets/images/2026-02-04-01-06-25.png)
Top 10 Selected Images (Highest delta_score):

![image](/assets/images/2026-02-04-01-06-31.png)
Bottom 5 Images (Likely Noise/Food)

### 느좋카페 예시 2

![image](/assets/images/2026-02-04-01-06-37.png)

Delta Score를 보면 0보다 큰 사진이 많으므로 느낌좋은 사진이 많음을 확인할 수 있다.

![image](/assets/images/2026-02-04-01-06-45.png)
Top 10 Selected Images (Highest delta_score):

![image](/assets/images/2026-02-04-01-06-50.png)
Bottom 5 Images (Likely Noise/Food):

### 느 안좋 카페 예시

![image](/assets/images/2026-02-04-01-07-02.png)

분포를 보면 Noise Score(빨간색)이 오른쪽에 치우쳐져 있고, Delta Score는 0을 넘는 사진이 거의 없음을 확인할 수 있다.

![image](/assets/images/2026-02-04-01-07-08.png)
Top 10 Selected Images (Highest delta_score):

![image](/assets/images/2026-02-04-01-07-14.png)
Bottom 5 Images (Likely Noise/Food):

## Threshold정하기

Thr을 정하려면 N이 충분히 큰 사진에 대해서 분포를 살펴보고, Score의 어느 경계에서 분위기를 나타내는 사진이 갈리는지 확인하면 된다.

랜덤으로 동3개를 샘플링하고 이에 대해서 (카페 45개)모든 이미지의 분포를 살펴보고 thr을 정한다.

**구로2동, 가산동, 독산3동**

![image](/assets/images/2026-02-04-01-07-22.png)
대략 thr이 0에서 0.01사이 지점에서 가우시안 분포가 나뉘게 된다.

오른쪽 그래프를 보면 빨간색(positive)와 파란색(negative)점수를 갖는 사진 샘플들이 클러스터링을 하며 잘 뭉쳐져있음을 확인할 수 있다.

![image](/assets/images/2026-02-04-01-07-31.png)

![image](/assets/images/2026-02-04-01-07-36.png)

![image](/assets/images/2026-02-04-01-07-41.png)

![image](/assets/images/2026-02-04-01-07-46.png)

![image](/assets/images/2026-02-04-01-07-50.png)

thr을 0.01로 최종적으로 정함

![image](/assets/images/2026-02-04-01-07-56.png)

이 과정 이후 thr을 넘은 사진이 5개가 안되는 카페 907개는 걸러졌음.

결론적으로 걸러내고자 했던 이미지들의 특징을 Negative 프롬프트로 넣어서, Metric에 반영을 한 결과, 노이즈이미지를 걸러낼 수 있었고, ‘분위기’를 나타내는 사진만 대표이미지로 선택할 수 있었다.