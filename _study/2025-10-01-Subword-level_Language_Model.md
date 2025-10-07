---
title: "[NLP][5주차_기본과제_2] Subword-level Language Model"
date: 2025-10-01
tags:
  - Subword-level Language Model
  - 5주차 과제
  - NLP
  - 과제
excerpt: "Subword-level Language Model"
math: true
---

# 과제2_Subword-level Language Model

1. 서브토큰화의 필요성
2. BPE(Byte Pair Encoding)의 구현
3. Transformers 라이브러리를 활용한 서브워드 토큰화

**서브워드 토큰화:** 서브워드 단위로 토큰화를 한다. 각 알파벳 단위로 토큰화를 하면 연산량이 너무 많으므로, 많이 등장하는 BytePair를 묶어서 토큰화를 한 후에 Vocab에 저장 → 단어끼리 token화를 할때에 비해서 ‘UNK’토큰문제가 없음

RNN의 모델을 불러와서 확인해보면 RNN의 매개변수 수보다 임베딩 매개변수의 개수가 훨씬 많다 → 임베딩 매개변수 수를 줄이기 위해서 Subword-tokenization을 도입

**Out-of-Vocab문제 해결:** 학습에서 등장하지 않은 단어는 모두 [UNK]토큰으로 처리되는데, 이러한 토큰을 입력에 넣으면 전체 모델의 성능이 저하된다.

```python
import re, collections
def get_stats(vocab):
	pairs = collections.defaultdict(int)
	for word, freq in vocab.items():
		symbols = word.split()
	for i in range(len(symbols)-1):
		pairs[symbols[i],symbols[i+1]] += freq
		return pairs

def merge_vocab(pair, v_in):
	v_out = {}
	bigram = re.escape(' '.join(pair))
	p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
	for word in v_in:
		w_out = p.sub(''.join(pair), word)
		v_out[w_out] = v_in[word]
		return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 10
for i in range(num_merges):
	pairs = get_stats(vocab)
	best = max(pairs, key=pairs.get)
	vocab = merge_vocab(best, vocab)
	print(best)
```

Counter로 Sentence에서 단어의 개수를 세어서 이를 가중치로 활용함 → voca를 만들때 같은 단어에 대해서 한번만 연산을 해서 효율적임

## BPE Vocab 만들기

`low, lower, lowst, newst` 단어가 corpus로 주어졌을때 `max_vocab_size`만큼 voca를 만들고, `id2token`으로 리턴하기

sentence의 각 모든 단어 끝에는 ‘_’를 붙여줌 → 단어의 끝임을 표

```python
from typing import List

import collections, re

# 단어 끝을 나타내는 문자
WORD_END = '_'

def build_bpe(
    corpus: List[str],
    max_vocab_size: int
) -> List[int]:
  
    #단어끝에 언더스코어 붙여주기
    #단어 각 알파벳으로 바꿔주기, 단어의 등장빈도를 Counter로 세어서 가중치로 이용하기
    words_freq = collections.Counter(corpus)
    words = {}
    for word, freq in words_freq.items():
      words[' '.join(list(word)) + ' ' + WORD_END] = freq
     #words에 corpus의 각 단어가 나온 횟수를 Count해서 dict로 저장
    print(words)

    #각 단어에서 나올 수 있는 쌍에 대해서 freq를 가중치로 주는 dict인 pair를 리턴
    def get_stats(voca):
      pairs = collections.defaultdict(int)
      for word, freq in voca.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
        #pairs에 freq만큼 가중치를 더해줌
          pairs[symbols[i],symbols[i+1]] += freq
      return pairs
		
    def merge_vocab(pair, v_in):
      v_out = {}
      #token 두개를 공백으로 연결
      bigram = re.escape(' '.join(pair))
      #정규표현식 패턴을 컴파일해서 p에 저장
      #단어 중간에 있는 경우를 배재하기 위해 단어의 시작과 끝에서만 탐색
      p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
      for word in v_in:
      #v_in의 모든 단어에 대해서 두 token을 합쳐서 w_out에 저장
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
      return v_out

    # id2token: List[str] = None

    id2token = set()
    for word in words:
        id2token.update(word.split())
    id2token = list(id2token)
#len(id2token)이 max_vocab_size보다 작을때까지 반복
    while len(id2token) < max_vocab_size:
      pairs = get_stats(words)
      if not pairs:
        break
      best = max(pairs, key = pairs.get)
      words = merge_vocab(best, words)
      # id2token = list(words.keys())
      id2token.append(''.join(best))
      print(id2token)
    #함수 조건에 맞게 길이가 긴 token순으로 정렬
      id2token.sort(key=len, reverse=True)
    ### END YOUR CODE

    return id2token
```

![image](/assets/images/2025-10-01-14-59-29.png)

테스트를 통과했다.

각 iteration마다 print를 찍어보면 두 token의 pair중 가장 빈도가 많은 순으로 `id2token`에 추가됨을 볼 수 있다.

## BPE 인코딩

만들어진 vocab으로 문장 text를 인코딩하자.

```
Vocab: bcde ab cd bc de a b c d e _
abcde ==> ab cd e _
```

이런식으로 greedy를 이용해서 인코딩을 할 수도 있지만 이 과제에서는 가장 길게 매칭되는 것을 전체 텍스트에 대해 먼저 토큰화하는 방법을 사용

```python
'''Vocab: bcde ab cd bc de a b c d e _
abcde ==> a bcde _ '''
```

이 방법은 첫번째 greedy방법보단 느리지만 텍스트를 더 짧게 인코딩할 수 있다.

```python
def encode(
    sentence: str,
    id2token: List[str]
) -> List[int]:
    '''
    1. 문장에서 공백을 기준으로 단어를 하나씩 나눠서 리스트로 만들기
    2. 리스트에서 for문으로 단어 하나씩 가져오기
    3. id2token에서 인덱스 0부터 값 하나씩  가져와서 있는지 확인
      일치하면 해당 단어에서 voca부분 없애고 3번 반복
      일치하지 않으면 다음 voca에서 찾기
    4. 해당단어 끝났으면 len(id2token)-1인 WORD_END를 token_ids에 넣기
    5.return token_ids
    '''
    #sentence에서 공백을 기준으로 단어 나눠서 리스트로 만들기
    token_ids = []
    words = sentence.split(" ")
    print(words)

    for word in words:
      #word에 알파벳이 모두 없어지고, 모두 id와 comma로 바뀔때까지 while문 돌기
      while any(char.isalpha() for char in word) == True:
        #가장 길이가 긴 토큰부터 검사
        for token in id2token:
          token_length=len(token)
          if token in word:
            word = word.replace(token,',' + str(id2token.index(token)))
            # token_ids.append(id2token.index(token))
            print(word)
            break
          else: #일치하지 않으면 다음 token에서 찾기
            continue
            print(word)
      word = word.split(",")
      word.pop(0)
      word = list(map(int,word))
      word.append(len(id2token)-1)
      print(word)
      token_ids.extend(word) #extend는 append와 다르게 요소를 개별적으로 추가함.
    print(token_ids)
    ### END YOUR CODE

    return token_ids
```

1. 문장에서 공백을 기준으로 단어를 하나씩 나눠서 리스트로 만들기

2. 리스트에서 for문으로 단어 하나씩 가져오기

3. id2token에서 인덱스 0부터 값 하나씩  가져와서 있는지 확인

일치하면 해당 단어에서 voca부분 없애고 3번 반복
 일치하지 않으면 다음 voca에서 찾기

4. 해당단어 끝났으면 len(id2token)-1인 WORD_END를 token_ids에 넣기

5. return token_ids

![image](/assets/images/2025-10-01-14-59-44.png)

test의 iteration마다 print를 찍어보면 가장 긴 token부터 검사하면서 단어안에 token이 있으면 id로 replace해주고, Comma로 단어사이를 구분해줬다.

단어를 모두 id로 바꾼 후 Comma로 `split()` 을 해서 리스트로 저장해서 return해줬다.

## BPE 디코딩

```python
def decode(
    token_ids: List[int],
    id2token: List[str]
) -> str:
    sentence = ""
    # token을 공백을 기준으로 나눠서 리스트로 만들기
    # token_ids = token_ids.split(" ")
    #token(숫자) 하나씩 불러와서 token으로 바꾼 후 sentence에 추가해주기
    for id in token_ids:
      sentence+=id2token[id]
    #언더스코어는 공백으로 바꾸기
    #언더스코어를 기준으로 리스트로 만들고, 마지막 언더스코어는 공백으로 남았기 때문에 pop()으로 빼준 뒤 다시 join으로 합치기
    sentence = sentence.split(WORD_END)
    sentence.pop()
    sentence = ' '.join(sentence)

    print(sentence)
    ### END YOUR CODE
    return sentence
```

인코딩된 id를 token으로 다시 디코딩하기.

id에 해당하는 서브워드로 만든 뒤 합치면 됨

```python
'''[ 196 62 20 6 ] ==> [ I_ li ke_ it_ ]==> "I_like_it_" ==> "I like it " ==> "I like it" '''
```