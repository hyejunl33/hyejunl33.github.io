---
title: "[PyTorch][1주차_기본과제_2] Linear_Regression"
date: 2025-10-06
tags:
  - 과제2_Linear_Regression
  - 1주차 과제
  - PyTorch
  - 과제
excerpt: "과제2_Linear_Regression"
math: true
---

# 과제2_Linear_Regression

PyTorch를 이용해서 Boston House Prices 데이터에 대해 주식 가격을 예측하는 Linear Regression

- 주어진 데이터를 pytorch 학습에 맞게 전처리할 수 있다.
- 선형 회귀 분석에 필요한 pytorch 구성 요소들을 구현할 수 있다.
- 실제 데이터에 대해 선형 회귀 분석을 실시할 수 있다.

사용데이터셋: Boston House Prices

보스턴 시의 범죄율, 재산세율, 본인 소유의 주택 가격의 중앙값등 측정된 지표를 포함한 데이터셋

# Boston House Prices 데이터를 저장하고 전처리 하기

```python
import pandas as pd
data = pd.read_csv("Boston-house-price-data.csv", sep = ',', header = 0)
data.head()
```

`Boston-house-price-data.csv` 파일을 pandas의 `read_csv()` 를 이용해서 불러온다. CSV는 Comma Seperate Values의 약자이므로 `comma`를 sep의 인자로 넘겨주고, header는 0번째 행으로 지정해준다.

`data.head()`를 통해 가장 위의 행 5개를 출력한다.

![image](/assets/images/2025-10-07-12-47-19.png)

```python
X = data.drop(['MEDV'], axis=1)
y = data.MEDV
```

`data.drop()` 을 사용→ `axis = 1` 은 열을 뜻하고, 열에 있는 ‘MEDV’를 drop함

MEDV변수는 종속변수인 y에 저장하고, 나머지 예측변수들을 X에 저장함.

```python
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = shuffle(X_train, y_train, random_state=42)
```

`sklearn.model_selection` 에서 train_test_split을 이용해서 train dataset과 test dataset을 나눈다.

이때 보통 train은 60~80%로 설정하고, 나머지를 test dataset으로 설정한다.

미니배치 SGD → 일단 dataset을 shuffle해주고 배치사이즈만큼 데이터를 불러와서 학습을 진행

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
```

모델에 들어가는 데이터를 정규화해야한다. 따라서 예측변수 X의 test와 train dataset을 `sklearn.preprocessing` 에서 `StandardScaler()`를 이용해서 정규화한다. 

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
X_train_tensor = torch.tensor(X_train.values, dtype = torch.float).to(device)
y_train_tensor = torch.tensor(y_train.values.reshape(-1,1), dtype = torch.float).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype = torch.float).to(device)
y_test_tensor = torch.tensor(y_test.values.reshape(-1,1), dtype = torch.float).to(device)
```

`‘cuda’` 를 사용할 수 있으면 device를 ‘cuda’로 설정하고 `torch.cuda.is_available()` 이 False일때 ‘cpu’로 설정한다.

그리고 `X_train, X_test, y_train, y_test` 를 전에 설정한 device에 올린다.

```python
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

`DataLoader, TensorDataset` 을 이용해서 `train_dataset`을 생성하고, `DataLoader`인스턴스를 `train_loader`로 생성한다. 이때 배치사이즈는 바꿀 수 있고, 여기서는 32로 설정했다.

일반적으로 train에서는 `shuffle`을 True로 설정하고, test에서는 False로 설정한다.

# 선형회귀모델 작성하고 학습하기

- **__init**__함수  내에서 인스턴스 `linear`에 `input_size`에 저장된 차원을 받아서 1차원데이터를 반환하는 선형 layer 정의하기
- 위에서 정의한 `linear()` 이용해서 `forward` 함수 정의하기

```python
import torch.nn as nn
class MultipleLinearRegression(nn.Module):
    def __init__(self, input_size):
        super(MultipleLinearRegression, self).__init__()
        #input_size만큼 입력을 받아서 1차원 데이터를 반환하는 Linear Layer
        self.linear = nn.Linear(input_size,1)

    def forward(self, x):
    # init에서 정의한 linear인스턴스를 사용
        return self.linear(x)
```

`MultipleLinearRegression`클래스는 `nn.Module`을 상속받는다.

```python
import torch.optim as optim
#Lossfinction으로 MeanSquaredError Loss를 사용함
criterion = nn.MSELoss()
# optimizer로는 SGD를 사용
optimizer = optim.SGD(model.parameters(), lr = 0.01))
```

```python
import numpy as np

num_epochs = 2000 # epoch를 2000으로 설정
model.train() # 학습을 위해 모델이 gradient를 저장하도록 설정
for epoch in range(num_epochs):
    epoch_loss = 0.0
    data_num = 0
    for inputs, targets in train_loader: # 각 배치마다 반복
        optimizer.zero_grad() # 옵티마이저의 gradient 초기화
        outputs = model(inputs) # 데이터를 넣었을 때의 모델의 출력값 저장
        loss = criterion(outputs, targets) # MSE 손실 계산
        loss.backward() # gradient descent 수행
        optimizer.step() # SGD 방식의 최적화 진행
        epoch_loss += loss.item()*inputs.shape[0]
        data_num += inputs.shape[0]
    if (epoch+1) % 100 == 0: # 100번의 epoch마다 학습 데이터의 손실함수 출력
        print(f"Epoch {epoch+1}, Loss(RMSE): {np.sqrt(epoch_loss/data_num)}"
```

![image](/assets/images/2025-10-07-12-47-38.png)

epoch이 100단위로 커질때마다 RMSE Loss를 출력한다. 이 학습에서는 epoch이 진행되어도 Loss가 거의 감소하지 않음을 볼 수 있다.

RMSE = Root Mean Squared Error: 평균 제곱근 오차

$$\text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}}$$

```python
b0 = list(model.parameters())[1]
b1 = list(model.parameters())[0]
```

학습한 모델로부터 `model.parameters()` 를 이용해서 모델의 학습가능한 파라미터를 반환할 수 있다. 이때 학습한 모델로부터 intercept(절편) 값을 `b0`에 저장하고, 나머지 변수에 대한 회귀 계수 값을 `b1`(기울기값)에 저장 

# Analytic Solution과 비교하기

`sklearn`에서는 `LinearRegression`객체를 지원한다. → 직접 구현안하고 불러와서 간단하게 사용할 수 있다.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

`sklearn` 에서 MSE랑, LinearRegression 불러와서 사용할 수 있다.

```python
lm = LinearRegression()
lm.fit(X_train, y_train)
```

`Im` 으로 불러온 `LinearRegression()` 의 인스턴스를 생성해주고, (X_train, y_train)학습데이터와 정답 레이블 쌍을 모델에 fit시켜준다.

```python
y_pred = lm.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print(f"Train Loss(RMSE): {rmse}")
```

y_pred를 `predict()` 함수를 이용해서 출력할 수 있고, rmse를 공식으로 정의해서 rmse를 출력한다.

```python
print(f"intercept: {lm.intercept_}, tangent: {lm.coef_}")
```

`Im` 모델의 절편(Intercept)항과 기울기(회귀계수)는 인스턴스변수 `intercept_, coef_` 를 통해 접근할 수 있다.