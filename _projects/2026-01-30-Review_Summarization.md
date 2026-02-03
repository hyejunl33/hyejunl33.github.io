---
layout: single
title: "[이미지기반 카페추천 프로젝트]서울시 카페 리뷰 데이터 정제 및 요약"
date: 2026-01-30
tags:
  - 이미지기반 카페추천 프로젝트
  - 리뷰 데이터 정제 및 요약
excerpt: "리뷰데이터 정제 및 요약"
math: true
---

# 리뷰 데이터 정제 및 요약

## 목차
1. [도입 배경 및 문제 정의](#도입-배경-및-문제-정의)
2. [아키텍처 및 핵심 로직](#아키텍처-및-핵심-로직)
3. [데이터 전처리 전략 (Preprocessing)](#데이터-전처리-전략-preprocessing)
4. [프롬프트 엔지니어링 (Gemma3)](#프롬프트-엔지니어링-gemma3)
5. [최적화 및 성능 관리 (Batch Processing)](#최적화-및-성능-관리-batch-processing)
6. [데이터 스키마 및 저장](#데이터-스키마-및-저장)

---

## 도입 배경 및 문제 정의

### 프로젝트 목표

크롤링된 **Raw 리뷰 데이터**는 노이즈가 많고 비정형적이라 검색이나 추천 시스템에 바로 활용하기 어려움. 이를 Google의 **Gemma-3-4B-IT** 모델을 사용하여 정제하고, 장소의 특징을 나타내는 **핵심 태그**와 **요약문**을 생성하여 서비스 품질을 높임.

### 모델 선정 근거 (Gemma3 Performance Benchmark)

다양한 모델 크기와 포맷을 비교 실험하여 최적의 모델을 선정함.

| 모델 | 실행 시간 (리뷰 50개) | 요약 품질 | 비고 |
| --- | --- | --- | --- |
| **Gemma3-4B-IT** | 약 10초 (GPU/API) | **완벽함** | **최종 선정** (4bit 양자화 활용) |
| Gemma3-1B-IT | 약 7초 (GPU/API) | 문맥이 약간 어색함 | 속도는 빠르나 표현력이 부족하여 제외함 |
| Gemma3-4B-GGUF | 약 10분 (CPU) | 좋음 | 4코어 CPU/8GB RAM에서도 동작하나, 대량 처리에 부적합함 |

1B 모델은 문맥 파악 능력이 부족하고, 4B-GGUF(CPU)는 처리 속도가 너무 느림. 따라서 GPU 가속과 4bit 양자화를 적용한 **Gemma3-4B-IT** 모델이 성능과 속도의 균형 면에서 가장 적합하다고 판단함.

### 기술적 도전 과제 및 해결 방안

| 기술적 문제 (Problem) | 상세 내용 | 해결 방안 (Solution) |
| --- | --- | --- |
| **데이터 노이즈** | 이모지 남발, 무의미한 특수문자, 비한국어 리뷰 등으로 분석 비용 증가 및 정확도 저하 | **Token Optimization**: 정규식 정제로 불필요한 이모지 및 특수문자를 제거하여 입력 토큰을 약 30% 감소시키고 할루시네이션 방지함 |
| **비정형 데이터** | 리뷰 내용이 제각각이라 일관된 검색 메타데이터 추출이 어려움 | **Structured Prompting**: 검색을 위한 태그(#)와 전시를 위한 요약문을 분리 생성함 |
| **GPU 리소스 유휴** | V100 GPU (32GB VRAM) 사용 시, 단일 건 처리로는 메모리가 남아돌아 자원 낭비 발생 | **Batch Inference**: 단위 배치 처리로 VRAM을 최대한 활용함. |

---

## 아키텍처 및 핵심 로직

### 파이프라인 구조

Step 2(VLM 묘사) 완료 데이터를 입력으로 받아, 리뷰 요약 및 태그 생성을 수행한 후 Step 3 데이터를 출력함.

```
[Input: Step2 JSON]
(VLM 묘사가 포함된 카페 데이터)
      ↓
[Preprocessing]
(이모지 제거, 길이 필터링, 한글 비율 체크)
      ↓
[LLM Processing (Gemma3)]
(Batch Size: 8)
   ├── [Task 1] Tag Generation (#태그 #생성)
   └── [Task 2] Summary Generation (한 줄 요약 + 상세 설명)
      ↓
[Output: Step3 JSON]
(summary_for_embedding, summary_for_display 추가)
```

### 전처리 구현 (`clean_review`)

```python
def clean_review(text, min_len=5, max_len=200):
    # 이모지 패턴을 [EMOJI] 플레이스홀더로 치환했다가 제거 (텍스트 깨짐 방지)
    emojis = EMOJI_PATTERN.findall(text)
    text = EMOJI_PATTERN.sub(' [EMOJI] ', text)

    # 한글, 영문, 숫자, 기본 부호 외 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9\s,.!?~()\-]', ' ', text)

    # 한글 비율 체크 (50% 미만인 리뷰는 스킵 - 영문/일문 리뷰 등 제외)
    if total_chars > 0 and korean_chars / total_chars < 0.5:
        return ""
    # ...
```

단순한 특수문자 제거를 넘어, 리뷰의 유용성을 판단하기 위해 ‘최소 길이’와 ’한글 비율’ 조건을 적용함. 퀄리티가 낮은 리뷰는 모델 입력 단계에서 미리 차단하여 토큰 비용과 시간을 아낌.

---

## 데이터 전처리 전략 (Preprocessing)

### 1. 노이즈 제거 및 정규화

크롤링된 리뷰에는 “더보기” 같은 UI 텍스트나 반복적인 이모지가 섞여 있음. 이를 정규식(Regex)으로 제거하고, `~`나 `!` 같은 문자가 과도하게 반복되는 것을 표준화함.

### 2. 고품질 리뷰 선별

LLM의 입력 토큰 한도가 있으므로, 모든 리뷰를 넣는 대신 **유의미한 상위 30~50개 리뷰**만 선별함.
- **길이 제한**: 5자 이상 200자 이하 (너무 짧거나 긴 리뷰 제외)
- **중복 제거**: 내용이 완전히 동일한 리뷰는 `set`을 이용해 제거함.

---

## 프롬프트 엔지니어링 (Gemma3)

검색과 가게설명이라는 두 가지 목적을 달성하기 위해 두 종류의 독립된 프롬프트를 사용함.

### Task 1: 가게 태그 생성

가게 요약에 활용할 핵심 키워드 3개를 추출함.

```python
# 프롬프트 예시
"""다음 카페 리뷰들을 보고... 핵심 태그 3개를 생성해줘.
지시사항:
1. 반드시 #으로 시작하는 태그 형식
2. 태그는 3개만 작성
..."""
```

### Task 2: 가게 설명용 요약 생성

사용자에게 보여줄 매력적인 소개글을 작성함. 개인정보(실명 등)가 포함되지 않도록 주의 사항을 포함함.

```python
# 프롬프트 예시
"""다음 리뷰들을 바탕으로... 매력적인 소개문을 작성해주세요.
리뷰에 포함된 사용자의 개인정보는 요약에 포함하지 말아주세요.
형식:
한 줄 요약: ...
상세 설명: ..."""
```

---

## 최적화 및 성능 관리 (Batch Processing)

### Batch Inference 구현

LLM 모델(Gemma-3-4B) 로드 오버헤드를 줄이고 연산 효율을 높이기 위해 배치 처리를 수행함.

```python
def generate_batch(model, tokenizer, prompts, max_new_tokens=300, batch_size=8):
    # Padding과 Truncation을 적용하여 텐서 크기 맞춤
    model_inputs = tokenizer(
        text_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    ).to(device)

    # 배치 단위 생성
    generated_ids = model.generate(**model_inputs, ...)
```

한 번에 8개(설정 가능)의 프롬프트를 모델에 입력하여 결과를 동시에 받아옴. 이를 통해 1건씩 처리할 때보다 처리 속도가 약 3~4배 향상됨.

### 메모리 관리

배치 처리가 끝날 때마다 명시적으로 GPU 캐시를 비워 OOM을 방지함.

```python
del model_inputs, generated_ids
torch.cuda.empty_cache()
gc.collect()
```

---

## 데이터 스키마 및 저장

Step 3 파이프라인을 거치면 JSON 객체에 두 가지 핵심 필드가 추가되어 저장됨.

| 필드명 | 타입 | 설명 | 출처/생성 |
| --- | --- | --- | --- |
| `summary_for_embedding` | List[str] | 검색 태그 리스트 (예: `['#조용한', '#커피맛집', '#주차가능']`) | Task 1 결과 |
| `summary_for_display` | String | 전시용 요약 텍스트 (한 줄 요약 + 상세 설명) | Task 2 결과 |
| `original_data` | Dict | 이전 단계(Step 2)의 원본 데이터 (유지) | Input |

**저장 파일 예시**: `data/processed/json/step3_summarized/강남구_역삼동_step3.json`

---