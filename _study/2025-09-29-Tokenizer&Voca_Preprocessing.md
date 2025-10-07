---
title: "[NLP][5주차_기본과제_1] Data_Preprocessing&Tokenization"
date: 2025-09-29
tags:
  - Data_Preprocessing&Tokenization
  - 5주차 과제
  - NLP
  - 과제
excerpt: "Data_Preprocessing&Tokenization"
math: true
---

# 과제1_Data_Preprocessing&Tokenization

텍스트 토큰화 및 Vocabuarly작성을 통해 Data_Preprocessing과 Tokenization 익혀보기

# Tokenizer 구현

토큰화는 입력데이터를 자연어 처리 모델이 인식할 수 있는 단위로 변환해주는 방법이다.

```python
from typing import List
import re
def tokenize(
    sentence: str
) -> List[str]:
		#소문자로 모두 바꿔주기
    sentence = sentence.lower()
		#정규표현식을 통해 .,!?는 별개의 토큰으로 처리하고,n`t랑 단어는 하나의 토큰으로 split
    tokens = re.split(r'\s+|([.,!?]|n\'t|\'\w+)', sentence)
		#빈문자열이나 None을 제외하고 나머지를 tokens에 저장
    tokens = [token for token in tokens if token != "" and token != None]
    print(tokens)
    return tokens
```

1. n`t는 하나의 토큰으로 처리되야 한다.
2. . , ! ?의 문장부호는 별개의 토큰으로 처리되어야 한다.
3. 나머지는 띄어쓰기를 기준으로 단어는 하나의 토큰으로 처리되어야 한다.

# Vocabulary 만들기

각 토큰을 숫자 형식의 유일한 id로 매핑하기 → Preprocessing과정에서 Vocavulary를 만든다.

```python
from typing import List, Tuple, Dict
from itertools import chain

# [UNK] 토큰
unk_token = "[UNK]"
unk_token_id = 0 # [UNK] 토큰의 id는 0으로 처리합니다.

def build_vocab(
    sentences: List[List[str]],
    min_freq: int
) -> Tuple[List[str], Dict[str, int]]:
		
		#문장을 하나의 리스트로 모두 불러오기
    all_tokens = list(chain(*sentences))

    # 각 토큰의 빈도를 세기
    word_counts = {}
    for word in all_tokens:
        # 단어가 있으면 1로 초기화, 이미 key에 있으면 1더하기
        if word in word_counts.keys():
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    # 최소빈도인 min_freq보다 val이 큰 단어들을 골라 filtered_words에 넣기
    filtered_words = []
    for word, count in word_counts.items():
        if count >= min_freq:
            filtered_words.append(word)

    # id2token 리스트는 모든 토큰이 들어있는 리스트
    id2token = [unk_token] + filtered_words

    # 토큰(word)를 id(idx)로 매핑해서 dict로 만들기
    token2id = {}
    for idx, word in enumerate(id2token):
      token2id[word] = idx

    return id2token, token2id
```

# 인코딩과 디코딩

```python
from typing import Callable, List, Dict

# [UNK] 토큰 ID
unk_token_id = 0

def encode(
    tokenize: Callable[[str], List[str]],
    sentence: str,
    token2id: Dict[str, int]
) -> List[int]:
		#tokenize함수를 이용해서 sentence를 token을 ㅗ바꾸기
    tokens = tokenize(sentence)

    token_ids = []

    # token을 id로 바꾸기, 단어가 없으면 0으로 매핑
    for token in tokens:
       
        token_id = token2id.get(token, unk_token_id)

        # id들을 리스트에 append
        token_ids.append(token_id)

    return token_ids
```

```python
def decode(
    token_ids: List[int],
    id2token: List[str]
) -> str:

    return ' '.join(id2token[token_id] for token_id in token_ids)
```

각 id들을 다시 토큰으로 바꿔서 출력함

토큰화 과정에서 공백 및 대소문자 정보를 모두 잃어버리고 UNK토큰으로 인해 사라진 단어들도 있다.

![image](/assets/images/2025-09-29-22-46-03.png)

# spacy를 이용한 영어 텍스트 토큰화 및 전처리

spacy는 토큰화 외에도 품사 및 단어의 기본형 정보 등 문장에 대한 많은 정보를 제공한다. → 특히 불용어 리스트도 제공한다.

불용어: 자주 등장하지만 큰 의미가 없는 단어를 의미한다.

```python
#불용어들 출력하기
print(spacy.lang.en.stop_words.STOP_WORDS)
```

`{'hereafter', "'m", 'six', 'became', 'must', 'though', 'without', 'are', 'everywhere', 'the', 'thereupon', 'whereafter', 'enough', '’ll', 'yourselves', 'last', 'nor', 'per', 'n’t', 'than', 'front', 'one', 'herein', 'fifteen', 'ca', 'latterly', 'that', 'can', 'along', 'side', 'on', 'among', 'might', 'never', 'ten', 'anyway', 'except', 'both', 'but', 'whence', 'for', 'eleven', 'out', 'if', 'most', 'behind', 'been', 'nowhere', 'show', 'get', '‘ve', 'hereby', 'often', 'thus', 'keep', 'below', 'itself', 'mostly', 'we', 'becomes', 'least', 'which', 'seem', 'many', 'have', 'done', 'she', 'rather', 'who', 'own', 'nothing', 'ever', 'meanwhile', '’m', 'become', 'what', 'within', 'hereupon', 'amount', 'into', 'over', 'against', 'hers', 'third', 'had', 'beyond', '’s', 'across', 'how', 'seeming', 'formerly', 'thereby', 'others', 'three', 'back', 'still', 'only', 'really', 'i', 'toward', 'their', 'seemed', 'together', 'up', "'s", 'her', 'throughout', 'part', 'name', 'were', 'to', 'say', 'thru', 'between', 'therein', '‘m', 'top', 'did', 'whether', 'anyhow', 'may', 'does', 'see', 'his', 'after', 'since', 'just', "n't", 'made', 'somewhere', 'also', 'whose', 'wherein', 'twelve', 'move', 'around', "'ve", 'off', 'be', 'anyone', 'is', 'sometime', 'they', 'my', 'indeed', 'bottom', 'everyone', 'when', 'your', 'our', 'anywhere', 'unless', 'because', 'please', 'nine', 'whereas', 'down', 'well', 'a', '’ve', 'neither', 'amongst', 'some', 'yet', 'its', 'during', 'twenty', 'first', 'hence', 'whom', 'however', 'regarding', 'whole', 'me', 'more', 'thence', 'less', 'several', 'further', 'herself', 'or', 'any', 'has', 'here', 'you', 'seems', 'noone', 'give', 'thereafter', 'nobody', 'used', 'whenever', 'beforehand', 'at', 'myself', '‘s', 'this', 'take', 'no', 're', 'former', 'these', 'either', 'whatever', 'somehow', 'would', 'whereby', 'moreover', 'of', 'five', 'few', 'now', 'quite', 'not', 'through', 'call', 'himself', 'under', 'make', 'someone', 'with', 'whereupon', 'via', 'empty', 'being', 'various', 'alone', '‘ll', 'too', 'do', 'hundred', 'even', "'d", 'anything', 'perhaps', 'from', 'where', 'about', '’re', 'eight', 'becoming', 'none', 'same', 'so', 'ourselves', 'before', 'should', 'sixty', 'then', 'else', 'another', 'full', 'and', 'afterwards', "'ll", 'very', 'until', 'onto', 'serious', 'will', 'mine', 'each', 'it', 'put', 'fifty', 'nevertheless', 'using', 'every', '’d', 'by', 'two', 'him', 'ours', 'elsewhere', 'yours', 'am', 'due', 'all', '‘re', 'sometimes', 'us', 'them', 'everything', 'beside', 'besides', 'he', '‘d', "'re", 'why', 'latter', 'upon', 'other', 'again', 'n‘t', 'themselves', 'otherwise', 'was', 'could', 'cannot', 'yourself', 'whither', 'next', 'an', 'something', 'therefore', 'forty', 'once', 'while', 'there', 'wherever', 'almost', 'although', 'whoever', 'above', 'in', 'those', 'much', 'four', 'already', 'doing', 'as', 'towards', 'always', 'namely', 'go', 'such'}`

## Spacy를 이용한 전처리 및 토큰화

```python
def spacy_tokenize(
    tokenizer: spacy.language.Language,
    sentence: str
) -> List[str]:
		#tokenizer를 통해 token들을 doc에 저장
		doc = tokenizer(sentenc)
    tokens = []

    for token in doc:
    # token이 불용어가 아닐때
        if not token.lower_ in spacy.lang.en.STOP_WORDS:
        #token의 어간(원형)을 tokens에 append하기
            tokens.append(token.lemma_)

    return tokens
```

`token.lower()`를 사용하지 않고 `lower_`를 사용하는 이유는 메서드인 lower를 사용하면 token객체는 lower함수가 없으므로 불가능 → token의 속성인 `.lower_`를 사용해야함

`lemma_` 속성은 token의 어간을 가져오는 속성임

### spacy를 이용한 인코딩과 디코딩 결과

![image](/assets/images/2025-09-29-22-45-55.png)