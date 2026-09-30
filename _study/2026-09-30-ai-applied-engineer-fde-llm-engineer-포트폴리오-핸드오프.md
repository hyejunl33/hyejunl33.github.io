---
layout: modern-single
title: "AI Applied Engineer · FDE · LLM Engineer 포트폴리오 핸드오프"
date: 2026-09-30
tags:
  - Study
  - AI
  - Career
  - Portfolio
  - FDE
  - LLM
excerpt: "AI Applied Engineer·Forward Deployed Engineer(FDE)·LLM Engineer 지원용 포트폴리오/자기소개서/면접의 단일 기준 문서. 2026-09-30에 확인한 공고를 바탕으로 작성했다."
notion_source_id: "sha256:d3f9f672c6355aed62df"
---

## 📌 이 문서의 용도

AI Applied Engineer·Forward Deployed Engineer(FDE)·LLM Engineer 지원용 포트폴리오/자기소개서/면접의 단일 기준 문서. 2026-09-30에 확인한 공고를 바탕으로 작성했다. 공고는 수시로 바뀌므로 지원 직전 원문을 다시 확인한다.

**핵심 원칙:** 기술 이름을 나열하지 않는다. "어떤 운영 문제를 어떤 제약 아래 정의했고 → 어떤 구조를 선택했으며 → 어떻게 검증·개선했고 → 실제 업무 흐름에 어떤 영향을 주었는지"를 쓴다.

---

## 1. 현재 공고에서 반복되는 핵심 역량 — 우선순위

### [P0] 문제정의와 사업·사용자 영향

- 고객/운영팀의 요청을 그대로 구현하지 않고, 실제 병목·위험·성과 지표를 문제로 재정의한다.
- 기능 산출보다 채택, 사용 흐름, 리스크 감소, 운영 가능한 상태를 결과로 제시한다.
- 키워드: problem framing, use-case selection, business outcome, customer workflow, adoption.

### [P0] 평가·신뢰성·피드백 루프

- 대표 데이터와 실패 사례로 평가 기준을 만들고, 구조화된 출력 검증·재시도·폴백·실패 종료 정책을 둔다.
- 모델 오류가 후속 파이프라인으로 전파되지 않게 한다. 정확도만이 아니라 재현성, 비용, 지연, 안전을 함께 판단한다.
- 키워드: evaluation harness, representative data, structured output, validator, retry policy, fallback, guardrail, failure isolation, observability.

### [P0] End-to-end Applied AI 구현

- 데이터/검색·추론·도구 호출·생성·검수·운영 배포를 하나의 사용자 흐름으로 연결한다.
- 모델 선택 이유와 비모델 결정적 처리(스키마, 규칙, 직렬화, 캐시)를 분리해 설명한다.
- 키워드: agent workflow, RAG/retrieval, tool calling, context engineering, API/data integration, production readiness.

### [P1] 재사용 가능한 모듈과 확장

- 한 서비스의 일회성 자동화가 아니라 설정·PRD·공통 스키마·서비스별 어댑터로 다른 국가/서비스에 확장한다.
- 키워드: reusable architecture, configuration-driven, modular pipeline, reference architecture, localization, multi-tenant/service branching.

### [P1] 안전·거버넌스·데이터 품질

- 콘텐츠·IP·개인정보·규제 위험을 입력 단계부터 정책 축과 증거로 처리한다. 과차단/오탐도 측정하고 기준을 수정한다.
- 키워드: safety, compliance, governance, IP risk, data quality, human-in-the-loop, auditability.

### [P1] 협업과 현장 실행

- 마케터/운영/디자이너의 언어를 요구사항·평가 기준·실행 가능한 인터페이스로 바꾼다.
- 키워드: stakeholder alignment, cross-functional collaboration, technical ownership, iteration with user feedback.

### [P2] 인접 역량 — 경험으로 주장하지 말고 학습 계획으로만

- Docker/Kubernetes, AWS·GCP·Azure, CI/CD, vLLM·SGLang·TensorRT, 대규모 서빙·SLA, 보안 네트워킹. 현 경력에 근거가 없으면 이력서에 '구현했다'고 쓰지 않는다.

---

## 2. JD가 직접 요구한 작성 방식 — 공통 가이드

[JD를 합쳐 얻은 작성 원칙 — 토스 AIOC·Agent·증권, OpenAI, Cohere, 국내외 FDE 공고 공통]

**1) 기술보다 문제와 영향으로 시작한다.**

"LangChain/RAG/에이전트를 사용했다"가 아니라, 현장 업무의 병목·위험·사용자 손실을 한 문장으로 정의하고 어떤 업무 지표/운영 상태가 달라졌는지 쓴다.

**2) 제약을 먼저 드러낸다.**

데이터 결손, 안전·IP 리스크, 국가별 로컬라이징, 비용/지연, 모델 출력 불안정처럼 설계가 필요했던 제약을 공개한다. 토스 AIOC가 요구한 흐름은 '제약 상황 → 문제 정의 → 실험 설계 → 평가 → 개선'이다.

**3) 구조 선택의 이유를 설명한다.**

모델 이름보다 왜 공통 파이프라인·설정 분기·구조화 스키마·결정적 후처리·캐시·폴백을 택했는지 쓴다. 에이전트의 역할, retrieval/tool/data의 연결, 사람이 검토하는 지점을 명확히 한다.

**4) 평가 설계와 실패 사례를 같이 쓴다.**

대표 데이터·수동 라벨·오류 사례·운영 신호·사람 판단으로 무엇을 검증했는지, precision/recall·표본·기준값을 함께 제시한다. 실패 후 어떤 threshold/정책/입력을 바꿨는지도 쓴다.

**5) 실행 신뢰성을 증명한다.**

토스 Agent 공고의 핵심은 실험을 프로덕션으로 옮길 때의 신뢰성과 확장성이다. 구조화 출력 검증, 오류 유형별 bounded retry, fallback, fail-closed/종료 기준, 상태 보존, 비용 대비 효과 판단을 사례로 제시한다.

**6) 운영 채택과 협업을 쓴다.**

누가 어떤 화면/산출물을 실제로 사용했는지, 마케터·운영·디자이너의 요구를 어떻게 PRD·정책·인터페이스로 전환했는지를 쓴다. FDE 공고는 고객/현업 workflow의 연결과 재사용 가능한 템플릿을 강조한다.

**7) 숫자는 범위와 조건을 붙인다.**

'개선'만 쓰지 말고 기준선, 표본, 기간, metric, trade-off를 밝힌다. 인과 근거가 없는 ROI·매출 상승을 주장하지 않고, '어떤 운영 범위에서 사용 가능한 산출물을 만들었다'로 쓴다.

### 가장 좋은 한 문장 구조

> "___라는 제약에서 ___를 문제로 정의하고, ___ 구조를 선택했다. ___ 데이터/실패 사례로 평가해 ___를 조정했으며, 결과적으로 ___ 업무 흐름에서 재사용 가능한 ___를 만들었다."

### 토스식 체크

- 기술 스택보다 문제·구조 선택 이유가 먼저인가?
- 제약 → 문제 → 실험 → 평가 → 개선이 보이는가?
- 장애·품질·비용을 어떻게 탐지/종료/복구했는가?
- 실제 사용자/현업팀에 어떤 방식으로 채택됐는가?

---

## 3. 개별 공고에서 확인한 역할별 신호

### 토스 AI Engineer Agent

기술 스택보다 문제와 구조 선택 이유를 먼저 쓰고, 실험에서 프로덕션까지의 신뢰성과 확장성, 장애·성능 저하·비용 문제의 탐지와 해결 경험을 구체적으로 쓰라고 안내한다. 내부 팀의 니즈를 발굴하고 플랫폼 기능으로 연결한 결과도 중요하다.

### 토스증권 AI Engineer

멀티 에이전트 오케스트레이션, tool/function calling, memory/router, 평가 데이터셋·하네스, 비즈니스 니즈 기반 루프 엔지니어링, 정형·비정형 데이터 검증을 강조한다.

### Cohere FDE, Seoul

보안 우선 엔터프라이즈 AI를 고객 워크플로와 데이터에 연결한다. FDE는 고객과 제품팀 사이에서 통합을 설계·구현하며, 보안·신뢰성·확장성·한국어/영어 커뮤니케이션을 중시한다. 클라우드·네트워킹은 인접 역량이다.

### OpenAI Applied AI Engineer, Seoul

유스케이스 선정부터 아키텍처·프로토타입·평가·출시·확장까지 수행한다. 모델/agent/retrieval/tool/data와 신뢰성·관측성·비용·안전·거버넌스의 trade-off를 설명하고, 대표 데이터·grader·운영 신호·사람 판단을 결합한 평가를 중시한다.

### 국내·해외 FDE 공통

Superb AI, Dfinite, Wonderful, CrewAI, C3 AI New Grad 공고는 고객 업무 흐름·문서·데이터·기존 시스템을 연결해 PoC를 빠르게 검증하고, 성공한 구조를 재사용 가능한 템플릿/레퍼런스 아키텍처로 만드는 역할을 반복해 요구한다.

### LLM Engineer 공통

GAIA-BT 등은 product-level LLM app, RAG/vector DB, prompt·MCP·agent/tool use, 데이터셋/평가, 문서화·협업을 요구한다. Dnotitia처럼 모델 학습 중심 역할은 PyTorch/HuggingFace·대규모 데이터/벤치마크가 중심이므로 Applied/FDE 포트폴리오와는 분리해 지원한다.

---

## 4. 내 경험 → JD 신호 매핑

### A. 뤼튼: 광고 소재 제작·운영 자동화

**문제/맥락:** 광고 집행 가능한 스토리만 선별하고, 각 스토리의 매력 포인트를 이미지·카피·영상·플레이어블 제작까지 일관되게 연결해야 했다.

**근거:** 서비스 데이터에서 상태·공개·차단·성인·2차창작 조건을 먼저 필터링하고, 11축 태그·세계관·댓글을 입력으로 strategy_json(훅/감정/소구점/HyDE 시각 질의/CTA)을 구조화했다. 이미지 선택은 다국어 CLIP 임베딩과_hardcut을 사용했고 실패 URL도 상태로 보존했다.

**어필 키워드:** data eligibility, structured output, prompt pipeline, multimodal retrieval, deterministic processing, cache, auditability, production workflow.

**한 줄:** "콘텐츠 원천 데이터의 적격성 필터, 구조화된 소구점 생성, 멀티모달 자산 선택을 분리해 광고 제작 입력을 재현 가능하게 만들었다."

### B. 뤼튼: 안전성·IP 침해 탐지

**문제/맥락:** 유저 생성 이미지·콘텐츠를 광고에 쓰면 IP/매체 리스크가 발생했고, 단순 역검색은 오탐이 많았다.

**근거:** 역검색·태그·스토리 맥락을 함께 입력으로 하여 정해진 스키마의 LLM 판정 레이어를 구성했다. 100건 수동 라벨링을 기준으로 FP를 줄여 정확도 74%, recall 100%를 기록했다. 변형 이미지 누락은 WD-tagger 전단과 검색 임계값 0.055→0.1 조정으로 보완했다. 3회 앙상블은 120건에서 개선 1건으로 효과가 미미해, JSON 파싱 재시도 폴백만 유지하는 비용-효과 판단을 했다.

**어필 키워드:** evidence-grounded classification, eval set, precision/recall, threshold tuning, error analysis, cost-aware design, guardrail.

**한 줄:** "존재 여부만 보는 역검색을 증거 기반 IP 판정으로 바꾸고, 라벨링·오류 분석·임계값 조정으로 안전성과 운영 손실을 함께 다뤘다."

### C. 뤼튼: 한국 Crack AX → 일본 캬라푸·북미 OOC 확장 / 사내 골프톤 우수상

**문제/맥락:** 한국 서비스에서 동작하던 광고 소재 제작 AX를 국가·장르·언어·자산 차이가 있는 두 글로벌 서비스로 확장해야 했다. 결과물은 이미지 배너뿐 아니라 플레이어블과 영상 배너였다.

**근거:** PRD로 서비스별 스키마 결손·언어 프롬프트·폰트·UI·안전성 정책을 명시하고, 공통 파이프라인과 서비스 분기를 분리했다. 캬라푸는 로맨스 세부 장르와 300개 레퍼런스의 장르 태깅을 바탕으로 i2i 이펙트 템플릿을 설계했다. OOC는 SF/fantasy 특성을 반영했다. 사내 Golfthon에서 Codex GOAL skill로 PRD 기반 반복 실행·검증을 구성한 확장 프로젝트가 우수상을 받았다.

**어필 키워드:** PRD-driven development, agent harnessing, configuration-driven localization, modularization, reusable architecture, multimodal generation, global expansion.

**한 줄:** "공통 생성 흐름과 국가별 정책·UI·프롬프트를 분리한 설정 중심 구조로, 한국 광고 AX를 일본·북미의 이미지·영상·플레이어블 제작 흐름으로 확장했다."

### D. 뤼튼: 신뢰성 설계 사례

플레이어블은 스토리보드 → 구조화 DSL → 검증 → 자체완결 HTML 조립으로 분리했고, 검증 실패는 같은 세션에서 멀티턴 수정했다. 이미지 변형은 전송 장애와 해부학 품질 실패를 분리해 각각 최대 2회 재시도하고, 최종 실패 변형은 제외했다.

**어필 키워드:** reliability engineering, validator, bounded retry, failure termination, self-contained artifact, feedback loop.

**한 줄:** "생성 단계와 검증 단계를 분리하고 오류 유형별 제한 재시도·실패 종료를 설계해, 불완전한 모델 출력이 다음 제작 단계로 전달되지 않게 했다."

### E. 부스트캠프 AI Tech

- **Semantic Segmentation:** 29개 hand-bone 클래스를 다루며 HRNetW48·해상도·정규화 조합을 비교했고 Dice 0.9755를 기록했다.
- **Movie sentiment:** 텍스트 정규화와 BERT/SWA 앙상블로 baseline 대비 +2.6%p를 개선했다.

**포지셔닝:** 모델 경력은 '실험 설계·평가 해석' 근거로 짧게 두고, 최신 포트폴리오의 첫 순위는 뤼튼 Applied AI 경험에 둔다.

---

## 5. 포트폴리오/자소서 작성 템플릿

### 프로젝트 카드 5문장

1. **제약과 실제 문제:** "___ 때문에 ___를 자동화/개선해야 했다."
2. **구조 선택:** "___를 입력 단계에서 분리하고, ___ 스키마·검증·폴백 구조를 택했다."
3. **핵심 구현:** "___ 데이터/모델/규칙을 연결해 ___ 산출물을 만들었다."
4. **평가와 개선:** "___ 표본/실패 사례/운영 지표로 검증해 ___를 조정했다."
5. **결과와 범위:** "___가 가능해졌고, 수치가 있으면 기준·표본과 함께 제시한다."

### 반드시 넣을 표현

- "PRD에 서비스별 변경 지점을 명시하고, 공통 파이프라인과 설정 분기를 분리했다."
- "구조화된 출력 스키마와 검증기를 두고 오류 유형별 재시도·폴백·종료 기준을 설계했다."
- "실패 사례를 라벨링/분석한 뒤 임계값과 전단 게이트를 조정했다."
- "단순 생성이 아니라 실제 운영자가 사용하는 제작·검수 흐름으로 연결했다."

### 피해야 할 표현

- 'LLM을 활용했다', '자동화했다', '성능을 개선했다'만 쓰고 입력·평가·결과가 없는 문장.
- 상관 근거 없이 'ROI/매출을 개선했다'고 단정하는 문장.
- AWS·Kubernetes·MLOps·RAG·배포를 실제 근거 없이 키워드만 넣는 문장.

---

## 6. 면접에서 바로 대비할 질문

**Q. 왜 3회 앙상블을 유지하지 않았나?**

A. 120건 중 개선이 1건이어서 호출 비용과 지연 대비 효과가 작았다. 대신 구조화 JSON 파싱 실패에 한정한 재시도 폴백을 두고, 더 큰 오류 원인인 후보 탐지/증거 품질을 개선했다.

**Q. 글로벌 확장에서 번역만으로 해결하지 않은 이유는?**

A. 데이터 스키마, 장르 컨벤션, 폰트·UI, 안전성 필터 축, 광고 레퍼런스가 서비스마다 달랐다. 그래서 PRD에 변경 지점을 명시하고 공통 흐름은 재사용하되 서비스별 설정·프롬프트·템플릿으로 분기했다.

**Q. 신뢰성의 기준은 무엇이었나?**

A. '생성 성공'이 아니라 후속 단계가 그대로 사용할 수 있는 상태다. 따라서 스키마 검증, 품질 게이트, 오류 유형별 제한 재시도, 최종 실패 제외, 캐시/상태 보존을 기준으로 설계했다.

---

## 7. 지원 직전 체크리스트

- 지원 JD의 문제 유형을 먼저 고른다: Agent/Applied AI/FDE/Model.
- 포트폴리오 첫 두 사례에는 뤼튼의 운영 자동화와 IP 탐지를 배치한다.
- 각 사례에 문제·구조 선택 이유·평가·결과를 모두 넣는다.
- 숫자는 모수/기간/측정 기준이 설명 가능한 것만 쓴다.
- '앞으로 할 수 있는 것'은 학습 계획에, '이미 한 것'은 근거와 함께 쓴다.

---

## 8. 조사 전수 목록 · 직무별 세부 신호 · 원문 링크

### Applied AI / FDE

- **OpenAI Applied AI Engineer, Seoul** — use case 선정, 아키텍처, prototype, evaluation, launch/scale; 모델·agent·retrieval·tool·data와 reliability/observability/latency/cost/safety/governance를 한 흐름으로 다룬다.
- **OpenAI FDE/Applied AI Architect 계열** — 기업 데이터 통합, 보안/프라이버시, 기술 계정 계획, 제품 채택과 확장을 강조한다. 신입 포트폴리오에서는 이를 '재사용 구조와 현업 채택'으로 번역한다.
- **Cohere FDE Infrastructure Specialist, Seoul** — 보안 우선 AI workspace, 고객 워크플로·데이터 통합, product/client bridge, 한국어·영어 커뮤니케이션. Cloud/networking은 우대 신호이지만 현재 경험으로 과장하지 않는다.
- **Superb AI FDE** — 현장 고객 문제를 발견해 비즈니스 가치가 나는 맞춤 AI를 설계·개발한다.
- **Dfinite FDE** — ERP/MES·문서·데이터 연결, RAG/Text-to-SQL/ontology, 고객의 근본 문제를 빠르게 검증하고 자산화한다.
- **Wonderful Korea FDE** — 고객 요청 표면이 아니라 실제 업무 문제를 파악해 production AI agent를 구현하고, enterprise data/cloud와 연결한다.
- **Databricks AI Engineer/FDE** — 고객과 함께 GenAI/LLMOps를 구현하고 사용·운영으로 연결한다.
- **CrewAI FDE** — 고객 아키텍처와 data workflow, PoC→pilot→deployment, success metric, API/observability integration, SLA/security, reusable template를 강조한다.
- **C3 AI New Grad FDE** — 데모·PoC·trial·production enterprise AI app을 고객과 직접 수행한다. 신입도 '프로토타입만'보다 실제 사용 흐름을 보여줘야 한다.
- **Applied Intuition New Grad FDE** — 고객 현장 문제를 제품화할 수 있는 실행력, ownership, 빠른 적응을 신호로 본다.

### LLM / Agent Engineer

- **Toss Brain/AIOC** — prompt, agent configuration, RAG, context engineering, evaluation, tool calling, guardrail과 실제 고객 피드백 루프.
- **Toss Agent** — 문제·구조 선택 이유, 실험→운영의 reliability/scalability, 장애·성능·비용 대응, 내부 팀 수요를 제품 기능으로 연결.
- **Toss Securities AI Engineer** — multi-agent orchestration, function calling, memory/router, evaluation dataset/harness, 정형·비정형 데이터 검증.
- **Toss Bank ML Engineer (LLM)** — vLLM/SGLang/TensorRT, LLM gateway, vector DB platform. 시스템 서빙 트랙의 인접 역량으로만 참고한다.
- **Toss Place Applied AI Engineer** — event/high-availability system과 applied AI를 결합하는 포지션. 현재 포트폴리오에는 '운영 신뢰성'으로 연결한다.
- **GAIA-BT LLM Engineer** — product LLM application, framework, vector DB, prompt/RAG/MCP/multi-agent/tool use, 문서화·협업. 학위 요구 수준은 공고별로 높을 수 있다.
- **Dnotitia LLM Engineer** — crawling/filtering dataset, model training, benchmark, Python/C++/PyTorch/HuggingFace. 모델 학습 중심 트랙이다.
- **HITS LLM Engineer** — agent 및 closed-loop DBTL 연구·개발. Applied/FDE와 달리 연구 역량 비중이 높다.
- **TO INFINITY Full-stack AI** — embedding/RAG, 아이디어→프로덕션→반복, API/cloud architecture를 강조한다.

### 이 조사에서 도출한 지원 전략

1. **1순위 타깃은 Applied AI/FDE/Agent Engineer다.** 뤼튼의 데이터 적격성·생성·검증·운영 흐름과 글로벌 확장이 가장 직접적으로 맞는다.
2. **LLM Engineer라도 product/RAG/agent/평가 중심 JD에 지원하고,** foundation-model 학습·서빙 중심 JD는 실험/학습 근거를 추가 확보한 뒤 지원한다.
3. **포트폴리오에는 '하네싱'이라는 단어만 쓰지 말고** PRD→목표 분해→구조화 산출물→검증/재시도→실패 종료→결과물의 구체 흐름으로 증명한다.
4. **'매출/ROI 개선' 대신** 광고 운영에서 사용 가능한 자동화 산출물·글로벌 서비스 확장·일예산 2,000~3,000만원 규모의 운영 범위를 맥락으로 제시한다. 자동화가 매출을 직접 올렸다고 단정하지 않는다.

### 원문 링크

| 회사 | 직무 | 링크 |
| --- | --- | --- |
| Toss | AI Engineer (Brain/AIOC) | https://toss.im/career/job-detail?job_id=7192548003 |
| Toss | AI Engineer Agent | https://toss.im/career/job-detail?job_id=7192548001 |
| Toss | Securities AI Engineer | https://toss.im/career/job-detail?job_id=7192548004 |
| OpenAI | Applied AI Engineer, Seoul | https://openai.com/careers/applied-ai-engineer-seoul-south-korea/ |
| Cohere | FDE Infrastructure Specialist, Seoul | https://www.cohere.com/careers/forward-deployed-engineer-infrastructure-specialist-seoul |
| Superb AI | FDE | https://kr.linkedin.com/jobs/view/forward-deployed-engineer-at-superb-ai-inc-4460144594 |
| Dfinite | FDE | https://kr.linkedin.com/jobs/view/forward-deployed-engineer-fde-at-%EB%94%94%ED%94%BC%EB%8B%88%ED%8A%B8-4430859641 |
| Wonderful Korea | FDE | https://kr.linkedin.com/jobs/view/wonderful-korea-forward-deployed-engineer-applied-ai-english-korean-at-wonderful-4446031211 |
| GAIA-BT | LLM Engineer | https://kr.linkedin.com/jobs/view/llm-engineer-at-gaia-bt-inc-4301117128 |
| C3 AI | New Grad FDE | https://careers.c3.ai/job/Redwood-City-Forward-Deployed-Engineer-New-Grad-CA-94063/1255710700/ |
