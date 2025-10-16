---
title: "[RecSys][과제2] GPT2 모델을 이용한 영화 추천"
date: 2025-10-16
tags:
  - RecSys
  - 과제
  - GPT2
  - 영화추천
excerpt: "[RecSys][과제2] GPT2 모델을 이용한 영화 추천"
math: true
---

# 과제2_생성모델

1. GPT의 prompt로 사용할 추천시스템 데이터를 전처리하고 만들기
2. GPT를 사용해서 prompt에 따른 추천 결과 생성

추천시스템 데이터는 오픈소스로 풀려있는 영화 선호도 데이터를 이용한다.

활용 데이터셋 : MovieLens Dataset

- 출처 :
    
    Harper, F. M., & Konstan, J. A. (2015). **The MovieLens Datasets: History and Context.** ACM Transactions on Interactive Intelligent Systems (TIIS), 5(4), 1-19. [http://dx.doi.org/10.1145/2827872](https://www.google.com/url?q=http%3A%2F%2Fdx.doi.org%2F10.1145%2F2827872)
    

GPT2를 huggingface 라이브러리에서 불러와 사용할 수 있다. GPT에 데이터를 줄때, prompt의 형식에 따라 출력이 달라지게 된다. 따라서 적절한 prompt를 적용할 수 있는 능력을 갖춰야 한다.

```python
import os
import numpy as np
import pandas as pd
import random
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
%matplotlib inline
```

`transformers`라이브러리에서 GPT2 headmodel과 tokenizer를 가져와서 사용한다.

`transformers` 라이브러리는 huggingface에서 만든 라이브러리로 GPT, BERT 같은 NLP모델들을 가져와서 쓸 수 있다.

# 데이터 전처리

`pd.read_csv()` 를 통해 rating_df와 movies_df를 생성한다. 즉 판다스 라이브러리의 `read_csv()` 함수를 이용해서 추천시스템에 필요한 rating 데이터프레임과 movies 데이터프레임을 생성한다. 이때 `u.data` 에 있던 데이터는 평점데이터이고, `u.item` 에 있던 데이터는 영화정보 데이터이다.

```python
ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']

movies_df.columns = ['movieId', 'movie_title', 'release_date', 'video_release_date',
                     'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']
```

이때 원래 `data_path`에 있던 각 data파일은 csv함수에서 `Header=None` 으로 불려와서 각 열에 header가 없는 상태이다. 따라서 올바른 열의 headcer를 `df.columns` 로 지정해준다.

```python
df = pd.merge(ratings_df, movies_df, on='movieId')
df = df[['userId', 'movieId', 'rating','movie_title']]
df['movie_title'] = df.movie_title.map(process_movielens_name)
df.head()
```

여기서 `rating_df, movies_df` 를 merge한 후에 4개의 열만 가져와서 df를 새로 만든다.

유저가 좋아할만한 영화를 추천하기 위해 df에서 별점이 4점 이상인 영화만 남기자.

```python
high_df = df[df['rating']>=4]
```

# 추천시스템에 GPT사용

GPT(Generative Pre-trained Transformer)는 트렌스포머 아키텍처에 기반한다. GPT의 등장 이후 pre-training과 fine-tuning을 나누는 접근법이 등장하게 됐다. 이를 통해 큰 규모의 언어 모델링을 통해 pre-train된 모델을 만들고, 각 task에 맞는 작은 dataset으로 학습하는 fine-tuning을 통해 다양한 NLP task를 우수한 성능으로 해결할 수 있게 됐다.

기존의 추천시스템에서는 BERT4Rec과 같은 모델을 통해 주변 context를 이해하는 cloze task로 학습했다. 하지만 이와달리 GPT생성모델은 텍스트생성에 직접적으로 활용되어서 학습없이 zero-shot 예측이 가능해진다.

## GPT model, tokenizer불러오기

```python
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token
```

gpt2-medium은 355M개의 parameter를 가진 모델이다.

이때 gpt2-medium모델은 기본 패딩 토큰이 없으므로, 문장의 끝을 알리는 `eos_token` 을 패딩토큰으로 사용하도록 지정해줘야한다.

## 추천결과 생성함수 구현

GPT모델을 이용해서 input에 기반하여 새로운 텍스트 추천을 생성하는 함수를 정의한다.

- input을 tokenizer로 encode하기
- output을 `model.generate()` 을 통해 생성하기
- 생성된 output을 decoding해서 추천결과(recommendations)출력하기

```python
def generate_recommendations(input_text, model, tokenizer):
    max_length = 100

    # 입력 텍스트가 최대 길이(max_length)를 초과하는 경우 잘라냄
    if len(input_text) > max_length:
        input_text = input_text[:max_length]

    # 입력 텍스트를 토큰화
    input_ids = tokenizer(input_text, return_tensors='pt')

    # 모델을 사용하여 텍스트 생성
    output = model.generate(input_ids, max_length=max_length, early_stopping=True,pad_token_id = tokenizer.eos_token_id)

    # 생성된 텍스트를 디코딩하여 output(추천결과) 생성
    recommendations = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    return recommendations
```
이때 max_length에 따라서도 출력이 달라진다고 한다. max_length를 300이나 더 큰 숫자로 주면, 영화 제목을 추천하는게 아니라, 영화와 관련된 문장을 생성한다. 그런데, 100으로 주면 문장을 쓰기에는 짧은 제한이기 때문에, 영화 제목을 추천하는식으로 LLM이 동작한다.


## Prompt 형식 비교

LLM은 입력된 프롬프트의 형태에 따라 성능이 크게 달라질 수 있다. → 같은 데이터에서 다른 프롬프트를 사용해서 결과를 비교해보자.

Zero-Shot Recommendation as Language Modeling논문에서는 아래 2개의 prompt로 구성되었고, 보다 효과적인 Prompt디자인을 탐색해서 LLM활용가능성을 극대화 할 수 있다.

### GPT2모델에 입력으로 사용할 input Prompt 생성하기

```python
def create_k_input(userId, high_df, k):
    #high_df에서 입력으로 받은 userId가 평가한 영화만 저장
    user_rated_movies = high_df[high_df['userId'] == userId]
 #영화가 k보다 적다면 None 반환
    if len(user_rated_movies) < k:
        return None
# 영화 제목을 리스트로 만들어서 movies에 저장
    movies = user_rated_movies['movie_title'].tolist()
# movies리스트에서 입력으로 받은 k개의 영화제목을 슬라이싱 하고, comma로 구분하여 text로 합치기
    created_text = ", ".join(movies[:k])
    return created_text
```

이전에 만들었던 4점 이상의 영화 df인 high_df에서 입력으로 받은 `UserID` 에 맞게 k개의 영화 제목을 str형태로 반환하는 함수이다. 여기서 movies리스트를 comma로 구분하는데, comma뒤에 띄어쓰기를 안하면`(”,”`) GPT2모델이 영화 제목을 구분하지 못하는 문제가 있었다. 디버깅을 할때 comma뒤에 공백 한칸이 포함된 `“, ”` 로 나눠주어야 영화제목을 구분해서 출력을 올바르게 하는것을 디버깅 과정에서 알 수 있었다.

LLM은 공백이나, comma, 입력 형식에 따라 출력양상이 아주 달라지는것을 알 수 있었다.

```
Recommendations:
["If you like The Shawshank Redemption, Singin' in the Rain, The Graduate, The Professional, The Sting, The Social Network, The Social Network 2, The Social Network 3, The Social Network 4, The Social Network 5, The Social Network 6, The Social Network 7, The Social Network 8, The Social Network 9, The Social Network 10, The Social Network 11, The Social Network 12, The Social Network 13, The Social Network 14, The Social Network 15, The"]
```

`created_text = ", ".join(movies[:k])` 로 `“, ”` 공백을 포함해서 input 영화 제목을 넣어주면 정상적으로 추천영화 목록이 출력된다.

```

["If you like The Shawshank Redemption,Singin' in the Rain,The Graduate,The Professional,The Sting, yo.\n\nI'm not a fan of the movie, but I'm a fan of the movie. I'm a fan of the movie. I'm a fan of the movie. I'm a fan of the movie. I'm a fan of the movie. I'm a fan of the movie. I'm a fan of the movie. I'm a fan of"]
```

하지만 `“,”` 로 공백을 포함해서 input을 주지않으면, 구분된 영화 제목이라는것을 모델이 인지하지 않고, 의미없는 문장의 나열을 출력한다. → LLM의 입력 prompt의 형식에 따라 출력이 매우 달라진다.

### Prompt 1) if you like<$$m_1$$…$$m_n$$>, you will like <$$m_i$$>

```python
userId = 16
created_text = create_k_input(userId,high_df,k=5)

# Prompt 1 포맷을 가진 input_text를 생성합니다.
input_text = "If you like "+ created_text + ", you will like "
print(input_text)

recommendations = generate_recommendations(input_text, model, tokenizer)

#출력 결과
print("Recommendations:")
print(recommendations)
```

prompt 1의 결과는 위에서 비교한 결과에서 볼 수 있듯이 “the social network”라는 영화를 추천으로 생성한 것을 볼 수 있다.

### Prompt 2) <$$m_1$$…$$m_n$$>,<$$m_i$$>

```python
userId = 16
created_text = create_k_input(userId, high_df,k=5)

# Prompt 2 포맷을 가진 input_text를 생성합니다.
input_text = created_text + ", "
print(input_text)
```

```python
recommendations = generate_recommendations(input_text, model, tokenizer)

print("Recommendations:")
print(recommendations)
```

prompt 1의 형식과는 다르게, 영화 제목만 `“, ”` 로 구분해서 input으로 넣어주면 “The Sound of Music”을 추천으로 생성한 것을 볼 수 있다.

prompt2가 기존에 좋아했던 영화들의 맥락을 더 잘 파악해서 ‘고전영화’라는 맥락을 포함한 ”The Sound of Music”을 추천한 것을 확인할 수 있다.

그러나 같은 영화명을 반복하는것을 보았을때 개선의 여지가 있음을 볼 수 있다.

### Number of movies 비교

`create_k_input()`함수에서 input의 개수인 k에 따라 결과가 어떻게 달라지는지 비교해보자

- k=2 사용 (영화 2개로 추천)

```python
user_id = 16
created_text = create_k_input(user_id, high_df,k=2)

# Prompt 2 포맷을 가진 input_text를 생성합니다.
input_text = created_text + ", "
recommendations = generate_recommendations(input_text, model, tokenizer)

print("Recommendations:")
print(recommendations)
```

```
Recommendations:
["The Shawshank Redemption, Singin' in the Rain, \xa0and The Big Lebowski are all great examples of films that are not only great, but also have a great story.\nThe story of The Shawshank Redemption is a classic example of a film that is not only great, but also has a great story. The story of The Shawshank Redemption is a classic example of a film that is not only great, but also has a great story.\nThe Shaw"
```

출력된 결과를 보면 영화 2개를 입력으로 주면, 영화제목인지 인식하지 못하고, 영화 제목들과 관련된 엉뚱한 문장들을 생성한 것을 볼 수 있다.

- k=6사용 (영화 6개로 추천)

```python
user_id = 16
created_text = create_k_input(user_id, high_df,k=6)

# Prompt 2 포맷을 가진 input_text를 생성합니다.
input_text = created_text + ", "
print(input_text)

recommendations = generate_recommendations(input_text, model, tokenizer)

print("Recommendations:")
print(recommendations)
```

```
Recommendations:
["The Shawshank Redemption, Singin' in the Rain, The Graduate, The Professional, The Sting, Babe, \xa0The Graduate, The Graduate, The Professional, The Sting, Babe, \xa0The Graduate, The Professional, The Sting, Babe, \xa0The Graduate, The Professional, The Sting, Babe, \xa0The Graduate, The Professional, The Sting, Babe, \xa0The Graduate, The Professional, The Sting, Babe, \xa0The Graduate, The Professional"]
```

`k=6` 일때는 적어도 영화제목들을 `“, ”` 로 구분해서 입력으로 제시하고 있다는 맥락을 파악하고, 영화 제목들을 출력으로 생성하고 있는것을 볼 수 있다.

# 소감

GPT같은 LLM모델은 prompt의 형식에 따라 내용은 같더라도 형태는 다른 input에 따라 생성하는 출력이 매우 달라지게 되는것을 확인할 수 있었다. 

그리고 입력의 개수 k 를 조작함에 따라, 출력이 또한 달라지는것을 확인할 수 있었다.

모델이 더 의미있는 출력을 하려면, 원하는 형식(추천 영화 제목을 `“, ”` 로 구분해서 출력) 할때 강화학습이 진행되도록 할 수도 있고, 더 큰 모델을 사용하거나, 특정한 분야(영화 데이터)에 대해 Fine-tuning을 하는 방법도 있을것이다.

zero-shot Recommendation as Language Modeling 논문을 추가로 읽어보고 리뷰를 작성해보면서, 더 심화공부를 해볼 예정이다.