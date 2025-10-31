---
title: "[Domain_Common_Project][미션-2]LogitProcessor구현"
date: 2025-10-31
tags:
  - Domain_Common_Project
  - LogitProcessor
  - Pretrained
  - GPT
  - NLP
  - Tokenization
  - 미션
excerpt: "[Domain_Common_Project][미션-2]LogitProcessor구현"
math: true
---


- LogitsProcessor 클래스를 상속하여 한글 토큰만 허용하는 사용자 정의 Processor 작성
- `tokenizer.batch_decode`와 유니코드 범위 검사를 활용한 한글 필터링 구현
- 제약 적용 전후 텍스트 생성 결과 비교
- LLM(대형 언어 모델) 출력 제어를 위한 LogitsProcessor의 개념 및 활용법 이해
- 사용자 정의 제약 조건을 로짓 처리기에 구현하여 원하는 언어 생성 제어 방식 학습
- HuggingFace transformers 기반의 텍스트 생성 파이프라인 실습

```python
# 한글만 반환하도록 하는 생성 제약 조건을 가진 사용자 정의 Processor 클래스를 정의하세요.

class KoreanOnlyProcessor(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.cached_vocab_size = None
        self.korean_mask: Tensor | None = None

    # TODO: 한글 유니코드 범위를 검사하는 함수 정의
    def _is_korean(self, token: str) -> bool:
        # 문자열 내 한 글자라도 이 범위에 포함되면 True 반환
      
        for char in token:
          code = ord(char)
            # 한글 완성형(0xAC00-0xD7A3) 
          if (0xAC00 <= code <= 0xD7A3):
            return True  # 한글이 하나라도 포함되면 True
        return False  # 모두 순회했는데 한글이 없으면 False

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        # scores 텐서의 마지막 차원 길이는 현재 vocab 크기
        vocab_size = scores.size(-1)
        device = scores.device  # scores가 위치한 디바이스 (CPU or GPU) # noqa: F841

        # 이전에 생성한 마스크와 vocab 크기가 다르면 다시 생성 필요
        if self.cached_vocab_size != vocab_size:
            # TODO: 현재 vocab 크기만큼의 token ID 배열 생성
            # 0부터 vocab_size-1까지 연속된 정수 텐서 생성
            # vocab 크기만큼의 모든 토큰 ID를 갖는 1차원 텐서 할당.
            token_ids = torch.arange(vocab_size)  # noqa: F841

            # TODO: 각 토큰 ID를 문자열로 디코딩
            # 위에서 만든 token_ids를 batch_decode를 통해 텍스트 리스트로 변환한 결과가 와야 함.
            # batch_decode가 2차원 입력을 요구하므로 unsqueeze 메서드 사용
            decoded_tokens = self.tokenizer.batch_decode(token_ids.unsqueeze(1))  # noqa: F841

            # TODO: decoded된 각 토큰에 대해 한글 포함 여부 리스트 생성 후 텐서로 변환
            allow_mask = torch.tensor([self._is_korean(token) for token in decoded_tokens])  # noqa: F841

            # 생성한 한글 포함 마스크 저장 (True면 생성 허용, False면 차단)
            self.korean_mask = allow_mask

            # vocab 크기 저장해서 불필요한 마스크 재생성을 방지
            self.cached_vocab_size = vocab_size

        # TODO: 한글이 포함되지 않은 토큰들의 로짓 점수를 치환하여 생성 불가 처리
        scores.masked_fill_(~self.korean_mask, -float('inf'))
    

        return scores
```

KoreanOnlyProcessor는 LogitProcessor를 상속받아서, 한글만 반환하도록 하는 Processor클래스이다.

- `is_korean` 함수에서는 arg로 들어오는 토큰을 문자단위로 하나씩 쪼개서 한글이 하나라도 포함되면 True를 반환하고, 한글이 하나도 포함되지 않는다면 False를 반환한다.
- `call` 함수에서는 input_ids와 scores텐서를 받아서, 현재 vocab_size만큼 token_ids에 리스트로 값을 할당해주고, decoded_tokens에서 각 token_id를 디코딩한다. 그 후 decoded된 각 토큰에 대해서 한글 포함 여부를 allow_mask로 리스트형태로 만든 다음에, 한글이 포함되지 않은 토큰들의 로짓점수들을 `-float(’inf’)` 로 치환해서 생성 불가 처리를 한 후 scores텐서를 반환한다.

```python
model_name = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(5)

# 한글의 금지어 토큰을 반환하지 않도록 방지하는 코드도 추가해보세요.
# 예시로, "바보"와 "멍청이" 라는 단어가 출력되지 않도록 막아주세요.
# transfomer.NoBadWordsLogitsProcessor를 이용하여 코드를 작성해주세요.

bad_words =["바보","멍청이"]
bad_words_ids = []
for word in bad_words:
    # 단어 원형 그대로 토큰화 (예: "멍청이")
    ids_original = tokenizer(word, add_special_tokens=False).input_ids
    if ids_original not in bad_words_ids:
        bad_words_ids.append(ids_original)
    

no_bad_words_processor = NoBadWordsLogitsProcessor(
    bad_words_ids=bad_words_ids,
    eos_token_id = tokenizer.eos_token_id
)

prompt = "따라해봐 바보 멍청이"
print(f"\n============== 테스트 프롬프트: {prompt} ==================")

inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(model.device)
input_length = input_ids.shape[-1]
attention_mask = inputs["attention_mask"].to(model.device)

korean_processor = KoreanOnlyProcessor(tokenizer)

print("\n--- 제한 없이 생성된 결과 ---")
output_ids_no_filter = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
)
output_text_no_filter = tokenizer.decode(
    output_ids_no_filter[0][input_length:], skip_special_tokens=True
)
print(output_text_no_filter)

print("\n--- Processor 적용된 결과 ---")
output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    logits_processor=LogitsProcessorList([no_bad_words_processor]),
)
output_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
print(output_text
```

```python
============== 테스트 프롬프트: 따라해봐 바보 멍청이 ==================

--- 제한 없이 생성된 결과 ---
라고 하는 경우가 많다. 지난 7월1일부터 9월30일까지 진행된 '2016 GS칼텍스 GS칼텍스 채용박람회'에서 SK에너지는 420명을 채용하는 등 총 4200여명의 구직자에게 기업 홍보용 CRM 솔루션을 제공했다.
또한 SK에너지는 취업준비생들을 위해 취업알선 사이트 ‘두오미(Doomie)’ 등 온라인 채용 사이트들을 운영, 구직자들이 가장 선호하는 기업정보를 제공하며 구직자들에게 많은 호응을 이끌어냈다.
김석환 SK에너지 사장은 “GS칼텍스 채용박람회는 다양한 취업정보를

--- Processor 적용된 결과 ---
, 멍청아. 잘 들어.
아니, 괜찮아.
정말 잘 들어.
잘 들어.
엄마는 정말 괜찮아.
엄마가 안 아파서.
내가 얼마나 힘들었는지.
그런데 엄마가 그걸 모를 수가 없어.
엄마가 안 아파서.
엄마는 왜 안 아파? 엄마가 안 아파서.
엄마는 왜 안 아파서.
그래, 엄마가 안 아파서.
엄마는 왜 안 아파? 엄마가 안 아파서.
엄마가 안 아파서.
엄마는 왜 안 아파서?
아내도 안 아파서?
아내한테 엄마가 안 아파서.
```

bad_words에 바보와 멍청이라는 단어가 출력되지 않도록 리스트로 해당 금지단어들을 리스트로 입력한 후 tokenizer로 해당 단어들의 ids를 추가했다.

이후 “따라해봐 바보 멍청이”라는 문장을 프롬프트의 입력을 주었을떄, 제한없이 생성된 결과를 보면, 프롬프트의 입력과 상관없이 모델이 사전에 학습한 데이터와 관련된 출력만 뱉는것을 확인할 수 있다. 이는 프롬프트의 세부내용과 관련없이 GPT모델은 프롬프트 다음에 나올법한 가장 그럴듯한 다음 단어를 연속해서 뱉고, 가장 그럴듯한 문장을 생성해내기 때문이다.

기존에 만든 KoreanOnlyProcessor와 no_bad_word_processor를 적용시킨 결과를 보면, 금지어를 포함하지 않고, sk나 GS, Doomi같은 영어단어를 포함하지 않는 출력을 뱉는것을 알 수 있다.

### processor를 적용했을때 한글만 출력되고, 금지어는 드물게 출력되는 이유?

우선 제한없이 생성된 결과를 보면 영어, 숫자등이 포함되어있음을 알 수 있다. 하지만 processor를 적용한 결과를 보면 영어, 숫자등은 제외되었고 오직한글만 출력됨을 알 수 있다. 이는 사전에 만들어둔 KoreanOnlyProcessor를 `logit_processor` 에 넣어주어서 한글만 출력된 것이다.

그리고 금지어로 “바보””멍청이”를 넣었는데, Processor가 적용된 결과에 “멍청아”라는 단어가 있는것을 볼 수 있다. 이는 “바보”와는 달리 토크나이저가 “멍청” +”이”로 토크나이징을 하면, “멍청이”라는 단어는 출력되지 않을 수 있어도, “멍청”이라는 어근에서 파생된 다른 단어들은 여전히 출력될 수 있다는 한계를 볼 수 있다. 이는 `transformer.NoBadWordsLogitsProcessor` 의 한계로, 만약 어근에서 파생된 모든 단어를 차단하고 싶다면, 어근 토큰을 포함하는 모든 어휘 토큰을 미리 찾아내고 해당 토큰들의 로짓점수를 `-float(’inf’)` 로 만들어주면 된다. 위에서 한글이 아니면 모두 마스킹했던것과 같은 방식으로 구현할 수 있다.

```python
class BlockSubstringProcessor(LogitsProcessor):
    """
    특정 부분 문자열(substring)이 포함된 모든 토큰의 생성을 차단하는 Processor
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, bad_substrings: list[str]):
        self.tokenizer = tokenizer
        self.bad_substrings = bad_substrings # 차단할 문자열 리스트 (예: ["멍청", "바보"])
        self.cached_vocab_size = None
        self.block_mask: Tensor | None = None

    def _build_block_mask(self, vocab_size: int) -> Tensor:
        """
        차단할 토큰 ID에 해당하는 마스크(True)를 생성합니다.
        """
        # 1. 모든 토큰 ID 생성
        token_ids = torch.arange(vocab_size)
        
        # 2. 모든 토큰 ID를 문자열로 디코딩
        # (KoreanOnlyProcessor와 동일한 방식)
        decoded_tokens = self.tokenizer.batch_decode(
            token_ids.unsqueeze(1), 
            skip_special_tokens=True # 특수 토큰은 검사에서 제외
        )

        # 3. 차단할 문자열(bad_substrings)이 포함된 토큰인지 검사
        block_list = [False] * vocab_size
        for i, token_str in enumerate(decoded_tokens):
            for sub in self.bad_substrings:
                if sub in token_str:
                    block_list[i] = True # 차단할 문자열이 포함되어 있으면 True
                    break # 이 토큰은 확정, 다음 토큰 검사

        return torch.tensor(block_list, dtype=torch.bool)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        vocab_size = scores.size(-1)
        device = scores.device

        # 1. 어휘 크기가 변경되었거나 마스크가 없으면 새로 생성
        if self.cached_vocab_size != vocab_size:
            # 차단 마스크 생성
            self.block_mask = self._build_block_mask(vocab_size).to(device)
            self.cached_vocab_size = vocab_size

        # 2. block_mask가 True인 모든 토큰의 점수를 -inf로 만듦
        scores.masked_fill_(self.block_mask, -float('inf'))

        return scores
```

```python

============== 테스트 프롬프트: 바보야. 멍청아. 따라해봐 ==================

--- 제한 없이 생성된 결과 ---
."
"무슨 일이야?"
"미안해. 네가 나를 사랑하게 된 이유, 내가 너를 사랑하게 된 이유를 말해 줄게. 나는 너를 사랑하고 있어. 난 네가 어떤 사람인지 잘 알고 있어."
"뭐라고? 그건 네가 날 사랑하게 된 이유일 뿐이야."
"무슨 소리야? 너가 널 사랑하게 된 이유?"
"아니, 네가 날 사랑하게 된 이유를 말해 줄게. 네가 날 사랑하게 된 이유?"
"그건 난 네가 날 사랑하게 된 이유야."
"그건 네가 날 사랑하게 된 이유야

--- Processor 적용된 결과 ---
, 내 옆에 누가 있을 거야."
"이런 생각을 하고 있지?"
"아무 말도 하지 않았어. 그저 그런 생각뿐이야."
"그런 것 같지는 않아."
"나는 그냥 그렇게 생각했어. 그런데 왜 내가 너를 무시하는지 모르겠어. 그리고 그 말을 기억해낼 수 있을 거라고 생각하고 있었던 거야."
"그게 내 생각이야. 그 말뿐이야."
"그렇게 생각했어. 하지만 이 세상에 너 같은 건 없어."
"그렇게 생각하지 마. 그건 너와 상관없어."
"그럼, 그렇지!"
"그건 그렇고 말
```

새롭게 정의한 `BlockSubstringProcessor` 를 사용하면 아예 “멍청”이라는 글자가 출력되지 않음을 확인할 수 있다.

# Living Points

- 일단 GPT는 프롬프트도 중요하지만, 사전 학습된 데이터의 맥락도 중요하다. 사전에 학습된 데이터가 소설밖에 없다면, 프롬프트와 연관된 그럴듯한 문장을 만드는 과정에서 소설과 연관된 문장을 만들 수 밖에 없다. 그리고 프롬프트 또한, 이전에 리뷰했던 Zero-Shot Recommendation as Language Modeling 논문에서 확인했던것 처럼, 프롬프트의 형식에 따라 출력이 달라지므로 프롬프트도 중요하다고 할 수있다. 결국 원하는대로 결과를 뽑아내기 위해서는 GPT에게 주는 프롬프트를 전처리해서 주는 방법도 사용해야하고, 사용 목적에 맞게 사전학습된 모델을 사용하는것도 중요하고, 출력된 결과를 후처리하는 방법도 고민해봐야 한다.
- 토크나이저는 단어를 쪼개서 토큰화 한다. 즉 “멍청이”라는 단어를 금지한다고 해서 “멍청”과 연관된 모든 단어를 금지하는건 아니다. 따라서, 금지어를 설정할때는 토크나이저가 토큰화 하는 방식을 엔지니어가 미리 찾아보고, 금지어의 어근과 연관된 단어를 제대로 차단하고 있는지, 그리고 어근과 연관없는데도 차단하는 단어가 있지는 않는지를 시각화해서 살펴봐야 한다.