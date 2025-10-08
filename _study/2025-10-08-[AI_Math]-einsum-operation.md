---
title: "[AI_Math][2주차_기본과제_2] einsum기본문제"
date: 2025-10-08
tags:
  - einsum기본문제
  - 2주차 과제
  - AI_Math
  - 과제
excerpt: "einsum기본문제"
math: true
---

# [AI_Math]기본2_einsum기본문제

einsum을 활용하여 행렬 및 텐서 연산을 NumPy코드로 구현하기

## 행렬 XY의 trace값 계산하기

```python
import numpy as np

def get_answer1(number_matrix1, number_matrix3):
    # einsum으로 두 행렬의 곱을 연산
    mid1 = np.einsum('ik, kj -> ij',number_matrix1, number_matrix2)
    # einsum으로 trace연산
    answer1 = np.einsum('ii -> ', mid1)

    return answer1
```

`np.einsum(’ik, kj → ij’,matrix1,matrix2)` :matrix1과 matrix2에서 k개의 열과 k개의 행을 내적한다.

`np.einsum(’ii→ ’,mid1)` : mid1에서 trace연산을 한다(ii인덱스를 갖는 벡터를 모두 더한다)

## 텐서 X, Y의 3차원 텐서곱에 대해 첫 번째 축을 따라 합계 계산하기

```python
import numpy as np

def get_answer2(number_tensor1, number_tensor2):
    # einsum으로 3차원 텐서곱을 진행하기
    mid2 = np.einsum('ijk, ikl -> ijl',number_tensor1, number_tensor2)
    # 첫번째 축을 없애고, 나머지를 더해서 합을 계산
    answer2 = np.einsum('ijl -> jl',mid2)

    return answer2
```

mid2는 예시에 맞춰서 차원을 einsum의 인자로 넘겨주었다. → 따로 transpose해서 곱하지 않아도, einsum연산에서는 알아서 인자의 차원을 맞춰서 곱해줌

```
number_tensor1 (array): integer 값으로만 구성된 array shape:(2,3,2)
        ex - [[[1, -3], [-2, 3], [ 3, 4]],
              [[1, -4], [ 2, 1], [-4, 4]]]
number_tensor2 (array): integer 값으로만 구성된 array shape:(2,2,3)
        ex - [[[-3, -3, -2], [-3,  3, -3]],
              [[-1, -2, -1], [-3, -3, -1]]

```

## 텐서 X, Y의 4차원 텐서곱에 대해 2차원으로 reshape한 후에, 주어진 행렬을 곱한 최종 행렬의 대각성분 합계 계산하기

einsum과 einops를 사용해서 구현하자

```python
import numpy as np
import einops

def get_answer3(number_tensor1, number_tensor2, number_matrix3):
    # 예시에 맞게 3차원 텐서끼리 곱해서 4차원 텐서곱 만들기
    mid3_1 =np.einsum('b l n, j n k -> b l j k',number_tensor1, number_tensor2)
    # einops의 rearrange를 이용해서 4차원 행렬은 2차으로 reshape하기
    mid3_2 = einops.rearrange(mid3_1, 'b m n k -> (b m) (n k)')
    # einsum을 이용해서 인자로 받은 행렬을 곱하기
    mid3_3 = np.einsum('ij, jk -> ik',mid3_2, number_matrix3)
    # einsum을 이용해서 trace연산 하기
    answer3 = np.einsum('ii -> ',mid3_3)

    return answer
```

```
주어진 두 텐서의 4차원 텐서곱을 2차원으로 reshape한 행렬의 대각성분의 합을 반환함
    Parameters:
        number_tensor1 (array): integer 값으로만 구성된 array
        ex - [[[1, -3], [-2, 3], [ 3, 4]],
              [[1, -4], [ 2, 1], [-4, 4]]]
        number_tensor2 (array): integer 값으로만 구성된 array
        ex - [[[-3, -3, -2], [-3,  3, -3]],
              [[-1, -2, -1], [-3, -3, -1]]]
        number_matrix3 (array): integer 값으로만 구성된 array
        es - [[-4,  2, -2, -3, -4, -1], [-4,  2, -4,  4, -4,  4],
              [-4,  2, -3,  3,  2,  4], [-1, -4,  3,  0,  0,  1],
              [-2, -1, -1, -3,  2, -2], [ 0,  2,  4,  3,  0, -1]]
    Returns:
        answer3 (number): 주어진 연산을 마친 결과
    Examples:
        >>> number_tensor1 = [[[1, -3], [-2, 3], [ 3, 4]],
                              [[1, -4], [ 2, 1], [-4, 4]]]
        >>> number_tensor2 = [[[-3, -3, -2], [-3,  3, -3]],
                              [[-1, -2, -1], [-3, -3, -1]]]
        >>> number_matrix3 = [[-4,  2, -2, -3, -4, -1], [-4,  2, -4,  4, -4,  4],
                              [-4,  2, -3,  3,  2,  4], [-1, -4,  3,  0,  0,  1],
                              [-2, -1, -1, -3,  2, -2], [ 0,  2,  4,  3,  0, -1]]
        >>> import basic_math as bm
        >>> bm.get_answer3(number_tensor1, number_tensor2, number_array3)
```