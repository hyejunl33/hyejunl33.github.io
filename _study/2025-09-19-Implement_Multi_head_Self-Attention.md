---
title: "[3주차_과제_3] Implement_Multi_head_Self-Attention"
date: 2025-09-19
tags:
  - Multi-head Self-Attention
  - 3주차 과제
  - AI LifeCycle
  - 과제
excerpt: "Implement_Multi_head_Self-Attention"
math: true
---

# 과제3_Implement_Multi_head_Self-Attention

$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

- Multi-head Self-Attention의 개념을 이해하고 코드로 구현하기
- Self-Attention의 수식을 이해하고 코드로 구현하기

```python
class MultiHeadSelfAttention(nn.Module):
      def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_size = int(hidden_size / num_heads)
        self.head_size = self.num_heads * self.attn_head_size

        self.Q = nn.Linear(hidden_size, self.head_size)
        self.K = nn.Linear(hidden_size, self.head_size)
        self.V = nn.Linear(hidden_size, self.head_size)
       
        self.dense = nn.Linear(hidden_size, hidden_size)

      def tp_attn(self, x):
      # x의 원래 크기 예시: [batch_size, seq_len, hidden_size]
	    # 예: [1, 10, 768]
        x_shape = x.size()[:-1] + (self.num_heads, self.attn_head_size)
         # x_shape -> ([1, 10], (12, 64)) -> [1, 10, 12, 64]
        x = x.view(*x_shape)
         # x의 크기 -> [1, 10, 12, 64]
         # 2단계: permute (Transpose) - 계산을 위해 축 순서 바꾸기
        return x.permute(0, 2, 1, 3)
         # x의 최종 크기 -> [1, 12, 10, 64]
					
      def forward(self, hidden_states):
        Q, K, V = self.Q(hidden_states), self.K(hidden_states), self.V(hidden_states)
        Q_layer, K_layer, V_layer = self.tp_attn(Q), self.tp_attn(K), self.tp_attn(V)

       
        attn = torch.matmul(Q_layer, K_layer.transpose(-1, -2)) / math.sqrt(self.attn_head_size)
        attn = nn.Softmax(dim=-1)(attn)
        output = torch.matmul(attn, V_layer)
        
       
        output = output.permute(0, 2, 1, 3).contiguous()
        output_shape = output.size()[:-2] + (self.head_size,)
        output = output.view(*output_shape)

        Z = self.dense(output)
       

```

### 1. `init(self, num_heads, hidden_size)`

다양한  subspace의 contextualization을 위해 Multi Head Attention을 사용해야 한다. init에서는 heads 수와 출력의 dimension인 hidden_size를 매개변수로 입력받는다.

Query, Key, Value벡터를 nn모듈의 Linear메소드를 이용하여 생성한다. 이때 Q, K, V벡터는 각각의 서로다른 가중치 행렬$W_Q, W_K, W_V$를 가져야 하므로 각각 `nn.Lienar` 연산을 해준다.

이때 `nn.Linear()` 의 input size는 `hidden_size` 고, output size는 head_size인데, head_size = self.num_heads *int(hidden_size / num_heads).이므로 hidden_size와 같음을 알 수 있다. 일단 Multi-head Attention에 사용할 모든 Weight를 concat한 형태로 초기화 한 후 나중에 Heads수만큼 Weight를 쪼개서 사용한다.

`self.dense = nn.Linear(hidden_size, hidden_size)`

이코드는 Z를 출력할때, Softmax(QK)V를 nn.Linear연산을 해주는 dense를 정의해준다.

 

### 2. `def tp_attn(self, x):`

Multi-Head Attention 계산을 위해 하나로 합쳐진 Q, K, V텐서를 각 헤드별로 분할하고 계산에 적합한 형태로 재배열 한다. Transpose for Attention이며, 텐서의 차원을 바꾸는 전처리 역할을 한다.

`x_shape = x.size()[:-1]` `+ (self.num_heads, self.atten_head_size)`

`x` 텐서는 `[배치 크기, 시퀀스 길이, hidden_size]` 형태를 가진다. (예: `[1, 10, 768]`)

`x.view(*x_shape)`를 통해 이 텐서의 마지막 `hidden_size` 차원을 `(num_heads, attn_head_size)` 차원으로 쪼갠다. (예: `768` -> `12 x 64`)

**결과**: 텐서는 `[배치 크기, 시퀀스 길이, 헤드 수, 헤드별 차원]` 형태가 된다. (예: `[1, 10, 12, 64]`)

### 3. `def forward(self, hidden_states):`

`self.tp_atten()` 함수를 이용해서 헤드별로 쪼갠 Q, K, V를 Q_layer, V_layer, K_layer로 받는다.

그 후  $\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$연산을 진행한다. 이때 Q와 K의 차원을 맞춰주기 위해 K벡터를 transpose해준다.

이후에 output을 `permute` 를 이용해서 `[배치, 헤드 수, 시퀀스 길이, 헤드차원]` 에서 `[배치, 시퀀스 길이, 헤드, 차원]` 으로 바꾼다. → 이렇게 해서 여러 헤드에서 나온 헤드와 헤드차원을 나란히 붙여서 view로 concat할 수 있게 한다. → 헤드*헤드차원으로 concat하면 텐서의 shape이 output _shape에 맞게 변경된다.