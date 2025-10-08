---
title: "[AI_Math][2주차_기본과제_1] 행렬기본문제"
date: 2025-10-08
tags:
  - 행렬기본문제
  - 2주차 과제
  - AI_Math
  - 과제
excerpt: "행렬기본문제"
math: true
---

# [AI_Math]기본1_행렬기본문제

행렬연산을 구현하는 함수를 numpy로 구현하면서 함수에 대한 감각을 익히기

- `get_transpose`: 주어진 행렬의 전치행렬을 반환함
- `get_inverse`: 주어진 행렬의 역행렬을 반환함
- `get_eigenval_eigenvec` : 주어진 (정방) 행렬의 고유값과 고유벡터를 구함

```python
import numpy as np

def get_transpose(number_matrix):
    # Matrix를 Transpose하고싶으면 T 속성(attrubute)를 쓰면됨
    transposed_matrix = number_matrix.T

    return transposed_matrix

def get_inverse(number_matrix):
    # 역행렬이 존재할때는 np.linalg.inv(matrix)함수를 쓰면 됨
    # 존재하지 않을때는 np.linalg.pinv(matrix)를 쓰면됨
    # pinv는 SVD분해를 사용해서 rank(matrix)와 관계없이 사용가능
    inverse_matrix = np.linalg.pinv(number_matrix)
    return inverse_matrix

def get_eigenval_eigenvec(number_matrix):
    # eigenvalue, eigenvector는 np.linalg.eig(matrix)로 반환가능
    # 둘중 하나만 받고싶으면 나머지 인자는 _로 버리기
    eigenvalue, eigenvector = np.linalg.eig(number_matrix)

    return eigenvalue, eigenvector
```

numpy에서 `transpose` ,`linalg.inv()` , `linalg.eig()` 를 통해서 matrix에서 전치행렬을 반환할 수 있고, 역행렬을 구할 수 있고, 고유값, 고유벡터를 구할 수 있음.

- `get_matmul()` : 주어진 두 행렬의 행렬곱 반환

ValueError가 발생하지 않기 위해서 행렬곱 연산이 가능하려면 앞 행렬의 열의 수와 뒤 행렬의 행의 수가 같아야 한다. (예시: `ij jk → ik` )

```python
import numpy as np

def get_matmul(number_matrix1, number_matrix2):
    # 출력으로 쓸 multiplicated_matrix를 출력 size에 맞게 0으로 채우기
    multiplicated_matrix = np.zeros((len(number_matrix1),len(number_matrix2[0])))

    # 삼중for문으로 돌면서 출력matrix안에 원소들을 곱한 값을 채워넣기
    for i in range(len(number_matrix1)):
        for j in range(len(number_matrix2[0])):
          for k in range(len(number_matrix1[0])):
            multiplicated_matrix[i][j] += number_matrix1[i][k]*number_matrix2[k][j]
# 행렬곱 연산 원리에 의해 앞행렬의 열과 뒷 행렬의 행은 k로 같아야 한다.
    return multiplicated_matrix
```