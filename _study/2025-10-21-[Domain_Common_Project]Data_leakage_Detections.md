---
title: "[Domain_Common_Project][과제2] 데이터 누수 탐지 및 방지하기"
date: 2025-10-19
tags:
  - Domain_Common_Project
  - 과제
  - Data_leakage_detection
  - Data_preprocessing
excerpt: "[Domain_Common_Project][과제2] 데이터 누수 탐지 및 방지하기"
math: true
---


# 과제2_데이터 누수 탐지 및 방지하기

- **데이터 누수가 없는 올바른 파이프라인**을 직접 구현
- 중복 데이터, 전처리 순서 오류, 타깃 누수 등 실제 현업에서 발생하는 다양한 데이터 누수 시나리오를 체험
- 상호정보량(Mutual Information)을 활용하여 의심스러운 피처를 탐지하고 제거하는 방법론 학습

## Task

1. 중복된 데이터 필터링: 검증/테스트 데이터에 학습 데이터와 중복된 데이터를 제거해서 올바른 성과 검증
2. 올바른 전처리 사용: 올바른 전처리 순서를 적용한 누수 없는 파이프라인 구축
3. 피처 검증 및 필터링: 상호정보량 분석을 통한 누수 의심 피처 식별 및 제거

데이터 누수는 휴먼 에러로 언제든지 일어날 수 있다. 따라서 데이터 누수를 사전에 감지해서 올바른 학습 파이프라인을 구축할 수 있어야 한다.

데이터 누수가 없는 학습 파이프라인을 구현할 수 있도록 누수가 발생할 수 있는 부분들에 대해 체크함수를 구현해보자.

```python
df = pd.read_csv(파일경로)

df.head()
```

pandas라이브러를 이용해서 `read_csv()` 로 파일을 데이터프레임 df로 불러온다.

# 올바른 전처리 사용

```python
def check_scaler_fitted_on_train_only(X_train, scaler):
    # 훈련 데이터의 평균과 분산이 스케일러의 평균과 분산이 일치하는지 확인하는 함수를 구현하세요. (일치하면 True, 아니면 False 반환)
    if np.allclose(np.mean(X_train), scaler.mean_) and np.allclose(np.std(X_train), scaler.scale_):
      return True
    else:
      return False
    raise NotImplementedError

#d
scaler = StandardScaler()
scaler.fit(num_features)

try:
    if not check_scaler_fitted_on_train_only(num_features_train, scaler):
        print("전처리 중에 누수가 있습니다.")
    else:
        print("전처리 중에 누수가 없습니다.")

except Exception as e:
    print(f"구현 중에 오류가 발생했습니다: {e}")
```

여기서 전체 데이터셋을 포함하는 `num_features` 를 `scaler.fit()` 에다 넣어줘서 데이터 누수가 발생한다. 데이터를 `train, test` 로 나누고 각자 따로 scaler를 돌려줘야 되는데, 한번에 scaler를 돌려버리면, train data에도 test data에 대한 정보가 들어간다. 따라서 데이터 누수가 발생한다.

scaler를 돌릴떄는 전체데이터셋을 넣으면 절때 안된다..

# 중복된 데이터 필터링

```python
def check_overlap_between_splits(df_train, df_test):
    #훈련 데이터와 테스트 데이터 간의 중복 샘플을 탐지하는 함수를 구현
    merged = pd.merge(df_train, df_test, how='inner', indicator=True)
    if merged['_merge'].value_counts()['both'] > 0:
      return True
    else:
      return False
    raise NotImplementedError

def remove_overlap_between_splits(df_train, df_test):
    #테스트 데이터에서 훈련 데이터와 겹치는 샘플들을 제거하는 함수를 구현
    df_test = df_test[~df_test.isin(df_train).all(1)]
    return df_test
    raise NotImplementedError

try:
    if check_overlap_between_splits(X_train, X_test):
        print("중복된 샘플이 존재해 누수가 있습니다.")
    else:
        print("중복된 샘플이 존재하지 않아 누수가 없습니다.")

except Exception as e:
    print(f"구현 중에 오류가 발생했습니다: {e}")
```

`check_overlap_between_splits()` 함수에서는 `pd.merge()` 를 사용해서 `df_train`과 `df_test` 를 `inner` 로 merge한다.

여기서 만약 중복된 샘플이 하나라도 있다면 True를 리턴하고, 중복된 샘플이 없다면 False를 리턴한다.

`remove_overlap_between_splits()` 함수에서는 두 데이터셋 에서 겹치는 샘플들을 제거한다.

# Feature 검증 및 필터링

```python
def check_leaky_features(num_features, threshold=0.2):
    # 누수가 의심되는 피처들을 탐지하는 함수를 구현하세요. (상호정보량이 임계값보다 높은 피처들 반환)
    # 상호정보량은 sklearn의 mutual_info_classif를 활용하며, 임계값이 0.2이상 이면 누수가 의심스러운 피처로 판단할 수 있습니다.
    mutual_info = mutual_info_classif(num_features, y_train)
    leaky_features = num_features.columns[mutual_info > threshold]
    return leaky_features
    raise NotImplementedError

leaky_features = check_leaky_features(num_features_train)

try:
    if len(leaky_features) > 0:
        print(f"누수가 의심되는 피처가 존재합니다: {leaky_features}")
    else:
        print("누수가 의심되는 피처가 존재하지 않습니다.")

except Exception as e:
    print(f"구현 중에 오류가 발생했습니다: {e}")
```

`check_leaky_features()` 함수에서는 상호 정보량이 threshold인 0.2보다 큰 feature들을 리턴한다.

이떄 `sklearn`의`mutual_info_classif()` 함수를 이용해서 정답 레이블과 features간의 상호 정보량을 계산한다.