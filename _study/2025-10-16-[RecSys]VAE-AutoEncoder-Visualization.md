---
title: "[RecSys][과제4] 변분추론 기반의 생성모델 학습구현"
date: 2025-10-16
tags:
  - RecSys
  - 과제
  - AutoEncoder
  - Variational Auto Encoder
  - Visualization
excerpt: "[RecSys][과제4] 변분추론 기반의 생성모델 학습구현"
math: true
---

# 과제4_변분추론 기반의 생성모델 학습구현

1. 변분추론 기반의 VAE를 구현
2. 유저사이드로 학습하고 클러스터링 결과를 확인
3. 잠재벡터와 클러스터링을 연관짓고 모델의 학습경향을 시각화

## 데이터셋

과제 2와 마찬가지로 MovieLens Dataset을 사용한다.

출처 :Harper, F. M., & Konstan, J. A. (2015). **The MovieLens Datasets: History and Context.**

ACM Transactions on Interactive Intelligent Systems (TIIS), 5(4), 1-19. [http://dx.doi.org/10.1145/2827872](https://www.google.com/url?q=http%3A%2F%2Fdx.doi.org%2F10.1145%2F2827872)

과제 2와 마찬가지로 평점 데이터는 `ratings_df` 로 불러오고, 영화 정보 데이터는 `movies_df` 로 불러온다.

## 모델

과제3에서 정의했던 VAE모델을 그대로 사용한다.

## 유저벡터 활용 클러스터링

### 유저 잠재 벡터(z) 추출

VAE 모델을 사용해서 유저 잠재 벡터(z)를 추출한다. 잠재벡터는 유저-아이템 행렬을 모델에 입력하여 인코딩 레이어에서 생성된다. → 유저의 잠재적 특징을 나타내는 벡터를 얻을 수 있다.→ 유사한 선호도를 가진 유저들이 같은 클러스터에 속하고, 클러스터 별로 명확하게 구분하고 시각화 할 수 있다.

```python
# 추론
with torch.no_grad():
     # FILL HERE # HOMEWORK(1)의 모델 출력 중 임베딩 생성에 필요한 변수만 사용
   hidden, _, _, _ =  vae(torch.Tensor(df_train.values))
```

`vae` 모델에서 `hidden_vector` z만 받아온다.

### 유저 클러스터링

- 추출된 유저 잠재벡터(z)를 이용해서 클러스터링을 수행하고 2차원으로 시각화 한다.
- K-Means를 클러스터링 기법으로 사용해서 유사한 특징을 가진 유저들을 그룹화 한다.

```python
# 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(hidden)#kmeans.fit_predict(VAE의 latent vector)

# train 데이터의 user_item_matrix에 클러스터 추가
train_user_item_matrix = user_item_matrix.loc[df_train.index].copy()
train_user_item_matrix['cluster'] = clusters

# 결과 출력
print(train_user_item_matrix.head())
```

```
movieId    1    2    3    4    5    6    7    8    9   10  ...  1674  1675  \
userId                                                     ...
245      0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0
83       4.0  4.0  0.0  2.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0
317      0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0
351      0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0
466      0.0  1.0  0.0  3.0  0.0  0.0  4.0  0.0  0.0  0.0  ...   0.0   0.0

movieId  1676  1677  1678  1679  1680  1681  1682  cluster
userId
245       0.0   0.0   0.0   0.0   0.0   0.0   0.0        1
83        0.0   0.0   0.0   0.0   0.0   0.0   0.0        1
317       0.0   0.0   0.0   0.0   0.0   0.0   0.0        0
351       0.0   0.0   0.0   0.0   0.0   0.0   0.0        1
466       0.0   0.0   0.0   0.0   0.0   0.0   0.0        1

[5 rows x 1683 columns]
```

유저가 속한 클러스터(0,1,2)를 2차원 matrix로 시각화 할 수 있다.

```python
train_user_item_matrix.cluster.value_counts()
```

클러스터별로 속하는 사람 숫자 수를 표로 출력할 수 있다.



t-SNE를 이용해서 각 클러스터링에 속하는 사람들을 2차원 공간상에 시각화 할 수 있다.

### 클러스터 별 장르/ 영화 구성 확인

```
Cluster 0 Top Genres: Drama, Comedy, Action
Movies:
Drama: Contact (1997), Fargo (1996), Chasing Amy (1997)
Comedy: Liar Liar (1997), Toy Story (1995), Back to the Future (1985)
Action: Star Wars (1977), Return of the Jedi (1983), Air Force One (1997)

Cluster 1 Top Genres: Drama, Comedy, Action
Movies:
Drama: Fargo (1996), English Patient, The (1996), Contact (1997)
Comedy: Toy Story (1995), Liar Liar (1997), Back to the Future (1985)
Action: Star Wars (1977), Return of the Jedi (1983), Godfather, The (1972)

Cluster 2 Top Genres: Drama, Comedy, Action
Movies:
Drama: Contact (1997), English Patient, The (1996), Fargo (1996)
Comedy: Liar Liar (1997), Toy Story (1995), Willy Wonka and the Chocolate Factory (1971)
Action: Star Wars (1977), Return of the Jedi (1983), Independence Day (ID4) (1996)

```

각 클러스터별로 top Genre를 나눠서 추출할 수 있다.

### 클러스터 별 유저 정보 확인

```
Cluster 0 Top Occupations:
student: 22.18%
other: 12.41%
administrator: 8.65%
educator: 8.65%
engineer: 7.14%

Cluster 1 Top Occupations:
student: 16.15%
other: 11.54%
educator: 11.54%
engineer: 8.46%
administrator: 8.08%

Cluster 2 Top Occupations:
student: 21.49%
other: 10.96%
programmer: 9.21%
educator: 8.77%
administrator: 8.77%
```

클러스터링 된 사용자 그룹 내에서 클러스터가 어떤 직업으로 이루어져 있는지를 확인해 볼 수 있다.