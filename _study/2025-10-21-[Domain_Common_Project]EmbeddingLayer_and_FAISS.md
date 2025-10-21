---
title: "[Domain_Common_Project][과제1] 임베딩 레이어 구현 및 FAISS를 활용한 유사문장 검색 과제"
date: 2025-10-21
tags:
  - Domain_Common_Project
  - 과제
  - 임베딩 레이어
  - FAISS
  - NLP
  - Tokenization
excerpt: "[Domain_Common_Project][과제1] 임베딩 레이어 구현 및 FAISS를 활용한 유사문장 검색 과제"
math: true
---


# 과제1_임베딩 레이어 구현 및 FAISS를 활용한 유사문장 검색 과제

이 과제는 텍스트 임베딩 레이어를 직접 구현하고, 생성된 임베딩 벡터를 기반으로 FAISS 라이브러리를 사용해서 효율적인 유사 문장 검색 시스템을 구축하는 과제이다.

- 문장에서 단어로, 단어에서 임베딩으로 대체 임베딩은 어떻게 진행되는걸까?
- FAISS를 이용해서 대규모 데이터셋에서 어떻게 유사한 항목을 찾아낼까?

## 주요Task

1. Vocabulary 만들기: 주어진 코퍼스에서 고유한 단어들을 추출해서 VOCA를 만든다. 그리고 인식되지 않는 단어는 `‘UNK’` 로 처리한다.
2. 정수 인덱싱 및 패딩: 문장의 각 단어들을 VOCA의 인덱스 시퀀스로 변환하고 최대 문장 길이에 맞춰서 패딩해준다.
3. PyTorch Dataset 및 DataLoader 구현: Dataset과 DataLoader로 데이터를 로딩한다.
4. SentenceEmbeddingModel 구현

# Vocabulary 만들기

```python
# 예시 문장 데이터 (corpus가 이미 있다고 가정)
corpus = [
    "I study NLP",
    "NLP is fun",
    "I love NLP"
]

# 1. 모든 문장을 단어 단위로 분리하여 하나의 리스트로 만듭니다.
#    ex: ['I', 'study', 'NLP', 'NLP', 'is', 'fun', 'I', 'love', 'NLP']
all_words = [word for sentence in corpus for word in sentence.split()]

# 2. set()을 이용해 중복된 단어를 제거하고, 다시 리스트로 만들어 정렬합니다.
#    set()은 순서가 없기 때문에, 재현성을 위해 정렬(sorted)해주는 것이 좋습니다.
#    ex: ['fun', 'I', 'is', 'love', 'NLP', 'study']
unique_words = sorted(list(set(all_words)))

# 3. 특수 토큰(<pad>, <unk>)을 추가합니다.
vocab = ["<pad>", "<unk>"] + unique_words

# 4. 딕셔너리 컴프리헨션(Dictionary Comprehension)을 사용해 간결하게 어휘집을 생성합니다.
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}

# --- 결과 확인 ---
vocab_size = len(word_to_idx)
print(f"어휘집 크기 (Vocab Size): {vocab_size}")
print(f"어휘집 예시: {list(word_to_idx.items())[:10]}...")
print(f"I의 인덱스: {word_to_idx['I']}")
```

한글은 교착어의 특성을 띠기 때문에 단순히 공백을 기준으로 단어를 나누면 안된다. 하지만 이 과제 실습에서는 외부 라이브러리의 `tokenizer` 를 사용하지 않고, 단순하게 공백으로 `tokenizer` 를 구현한다.

`Corpus` 에 리스트 형태로 저장된 문장의 단어를 하나씩 불러와서 일단 리스트로 만든다.

그 후 `set()` 으로중복되는 단어를 제거하고, `vocab[]` 리스트에 특수 토큰과 함께 저장한다.

그 이후에 `word_to_idx` 에는 word를 key로, idx를 val로 갖는 dict를 만들어주고, `idx_to_word` 에는 idx를 key로, word를 val로 갖는 dict를 만들어주어 Voca를 만들어준다.

# Tokenizing and Indexing

```python
max_seq_len = max(len(s.split()) for s in corpus)
print(f"데이터셋 내 최대 문장 길이: {max_seq_len}")

def tokenize_and_index(
    sentence: str, word_to_idx: dict[str, int], max_len: int
) -> list[int]:
    """문장을 토큰화하고 길이를 조절하는 함수"""

    # 띄어쓰기 기준으로 단어를 분리하고, 각 단어를 해당하는 인덱스로 변환합니다.
    # 어휘집에 없는 단어는 '<unk>' 토큰의 인덱스로 처리합니다. (리스트 컴프리헨션 사용)
    indices = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in sentence.split()]

    # 문장의 길이를 max_len에 맞춥니다.
    # 1) 먼저 최대 길이로 자르고, 2) 그 다음에 부족한 부분을 패딩합니다.
    indices = indices[:max_len]
    indices += [word_to_idx["<pad>"]] * (max_len - len(indices))

    return indices

# 2. 모든 문장을 인덱스 시퀀스로 변환
indexed_corpus = [tokenize_and_pad(s, word_to_idx, max_seq_len) for s in corpus]

print(f"\n원본 문장: '{corpus[0]}'")
print(f"인덱스화 및 패딩된 문장: {indexed_corpus[0]}")

print(f"\n원본 문장: '{corpus[1]}'")
print(f"인덱스화 및 패딩된 문장: {indexed_corpus[1]}")
```

```
>>>데이터셋 내 최대 문장 길이: 4
>>>원본 문장: '나는 사과를 좋아해'
>>>인덱스화된 문장: [2, 3, 4, 0]
```

위에서 만든 voca dict를 이용해서 문장을 토큰화하고, 데이터셋 내 최대 문장길이를 넘으면 슬라이싱하는 함수를 이용해서 문장을 인덱스화 한다.

# Dataset 및 Dataloader 구현

```python
class TextDataset(Dataset):
    def __init__(self, indexed_sentences: list[list[int]]) -> None:
        # 생성자(init)에서 미리 모든 데이터 리스트를 Tensor로 변환하여 저장합니다.
        self.tensor_sentences = [
            torch.tensor(s, dtype=torch.long) for s in indexed_sentences
        ]

    def __len__(self) -> int:
        return len(self.tensor_sentences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 미리 변환해 둔 Tensor를 그대로 반환하기만 합니다.
        return self.tensor_sentences[idx]
      
# 데이터셋 인스턴스 생성
dataset = TextDataset(indexed_corpus)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
      
```

`TextDataset` 클래스를 통해 `indexed_corpus` 를 인자로 주면 텐서를 반환해준다. 그리고 DataLoader는 사전에 파이토치 라이브러리에서 가져온 `DataLoader`를 그대로 사용한다. DataLoader를 이용해서 `batch_size=2` 만큼 데이터를 가져와서 학습할 수 있으며, `shuffle`  을 해주면서 데이터를 load한다.

# SentenceEmbeddingModel 구현

```python
# ruff: noqa
class SentenceEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pad_idx: int) -> None:
        super(SentenceEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    # 인풋을 embedding을 통해 임베딩한다.
    # 결과 dimension: (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(input_ids)
		#input_ids에서 패딩토큰이 아닌 부분(0이 아닌부분)만 True인 마스크를 생성한다.
		#마스크: (batch_size, sequence_length)
        non_pad_mask = (input_ids != 0)
		#위에서 만든 마스크를 임베딩 결과에 곱하여 패딩부분의 값을 0으로 만든다.
		# (batch_size, sequence_length, embedding_dim) * (batch_size, sequence_length, 1)
        masked_embedded = embedded * non_pad_mask.unsqueeze(-1)
		#sum()을 하고 실제 단어의 개수로 나누어 평균을 계산한다.
		#(batch_size, embedding_dim)
        sum_embedded = masked_embedded.sum(dim=1)

        # 각 시퀀스의 실제 단어 수 (패딩 제외)
        num_words = torch.sum(non_pad_mask, dim = 1).unsqueeze(-1)

        # 0으로 나누는 것을 방지
        num_words = num_words + (1e-9)
        sentence_embedding = sum_embedded / num_words

        return sentence_embedding
```

`SentenceEmbeddingModel` 클래스를 정의한다. 

`self.embedding()` 을 통해 input을 임베딩하고 non_pad_mask를 만들어서 패딩토큰이 아닌 부분만 True인 마스크를생성한다.

그리고 각 시퀀스의 실제 단어수로 임베딩 `masked_embedd.sum()` 을 나눠서 sentence_embedding을 리턴한다.

```python
# 모델 파라미터 설정
embedding_dim = 128  # 각 단어 벡터의 차원
pad_token_idx = word_to_idx["<pad>"]

# 모델 인스턴스 생성 및 디바이스로 이동
model = SentenceEmbeddingModel(vocab_size, embedding_dim, pad_token_idx).to(device)
print(f"\n모델 구조:\n{model}")

# 모든 코퍼스 문장에 대한 임베딩 생성
model.eval()  # 모델을 평가 모드로 설정
corpus_embeddings = []
with torch.no_grad():
    # DataLoader로부터 데이터를 배치(batch) 단위로 받아옵니다.
    for batch in tqdm(dataloader, desc="Generating corpus embeddings"):
        # DataLoader가 이미 배치 차원을 만들어주므로 unsqueeze가 필요 없습니다.
        # 배치 전체를 device로 보냅니다.
        batch = batch.to(device)

        # 모델을 통해 배치 전체의 임베딩을 한 번에 계산합니다.
        batch_embeddings = model(batch)

        # 계산된 배치 임베딩을 CPU로 옮겨 NumPy 배열로 변환한 뒤 리스트에 추가합니다.
        # 여러 개의 결과를 한 번에 넣기 위해 append 대신 extend를 사용합니다.
        corpus_embeddings.extend(batch_embeddings.cpu().numpy())

# 리스트를 최종 NumPy 배열로 변환
corpus_embeddings = np.array(corpus_embeddings).astype("float32")
print(f"\n생성된 코퍼스 임베딩 형태: {corpus_embeddings.shape}")
```

모델 구조를 출력한다. → 생성된 코퍼스 임베딩 형태`: (10, 128)` →(10개의 문장, 128 dimension)

```python
# FAISS 인덱스 생성 및 학습
# IndexFlatL2: 가장 기본적인 인덱스. L2 거리(유클리드 거리)를 사용하여 검색합니다.
# IndexFlatIP: 내적(Inner Product)을 사용하여 검색합니다. 코사인 유사도 검색 시 유용합니다.
# (코사인 유사도는 내적과 관련: A.B / ||A|| ||B||. 만약 벡터가 정규화되어 있다면 코사인 유사도는 내적과 동일)
embedding_dim = corpus_embeddings.shape[1]  # 임베딩 벡터의 차원

# L2 거리 기반 인덱스 생성
index = faiss.IndexFlatL2(embedding_dim)

print(f"FAISS 인덱스에 {len(corpus_embeddings)}개 벡터 추가 시작")
index.add(corpus_embeddings)  #인덱스에 벡터 추가
print(f"FAISS 인덱스에 {index.ntotal}개 벡터 추가 완료")
```

```python
query_sentence = "나는 과일을 좋아해"  # 검색할 문장
print(f"\n질의 문장: '{query_sentence}'")
indexed_query = tokenize_and_index(query_sentence, word_to_idx, max_seq_len)
query_tensor = (
    torch.tensor(indexed_query, dtype=torch.long).unsqueeze(0).to(device)
)  # (1, max_seq_len)
model.eval()
with torch.no_grad():
    query_embedding = model(query_tensor).squeeze(0).cpu().numpy().astype("float32")

# FAISS를 이용한 유사 문장 상위 3개 검색
k = 3
(
    distances,
    indices,
) =index.search(np.array([query_embedding], dtype = 'float32'), k) # TODO: query_embedding을 이용하여 인덱스에서 상위 k개 문장 검색
print(f"\n가장 유사한 상위 {k}개 문장:")
for i in range(k):
    # FAISS는 기본적으로 거리를 반환 (L2 거리이므로 작을수록 유사)
    # 거리를 유사도로 변환하고 싶다면 (1 - 정규화된 L2_거리) 또는 코사인 유사도 계산
    retrieved_index = indices[0][i]
    similarity_score = distances[0][i]  # L2 거리
    print(f"{i + 1}위: '{corpus[retrieved_index]}' (L2 거리: {similarity_score:.4f})")
```

```
질의 문장: '나는 과일을 좋아해'

>>>가장 유사한 상위 3개 문장:
>>>1위: '나는 사과를 좋아해' (L2 거리: 31.8506)
>>>2위: '나는 바나나를 좋아해' (L2 거리: 32.2216)
>>>3위: '이 영화 정말 재미있어' (L2 거리: 81.3043)
```

FAISS의 `index.search()` 를 이용해서 임베딩과 가장 유사한 상위 3개의 문장을 검색할 수 있다.

# Living Point

- 이 과제에서는 Voca를 단순하게 공백을 기준으로 단어로 나눈 후 Tokenize해서 만들었지만, 실제로는 한글을 공백으로 나누면 큰일난다..(불용어, 조사…)
    - 꼭 똑똑한 개발자들이 만들어둔 라이브러리의 Tokennizer를 이용해서 tokenize하자..
- 문장의 입력형식을 맞춰주기 위해, 최대 문장길이에 도달못한 문장은 패딩을 해줘야하고, 최대문장길이보다 긴 문장은 슬라이싱으로 잘라줘야 한다.
- `nn.embedding()` 으로 인덱스로 바꾼 각 단어들을 임베딩하는건 쉽게 할 수 있다. 하지만 그 전까지의 voca를 만들고 tokenize하는 과정을 잊지말자…
- 문장간의 유사도를 FAISS로 구하기 위해서 각 문장을 `SentenceEmbeddingModel` 클래스로 임베딩했다.
- FAISS라이브러리의 `index.search()` 를 이용하면 임베딩벡터 사이의 거리(유사도)를 구할 수 있다.