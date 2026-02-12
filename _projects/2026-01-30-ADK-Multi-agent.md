---
layout: single
title: "[이미지기반 카페추천 프로젝트] ADK 기반 에이전트 챗봇 구현"
date: 2026-01-30
tags:
  - 이미지기반 카페추천 프로젝트
  - ADK 기반 에이전트 챗봇 구현
excerpt: "ADK 기반 에이전트 챗봇 구현"
math: true
---

## 목차
1. [필요한 기능](#필요한-기능)
2. [동네 처리 (Location Agent)](#동네-처리--location-agent)
3. [Stateful 채팅 구현](#stateful-채팅-구현)
4. [ADK 마이그레이션](#adk-마이그레이션)
5. [Context 기반 추천 질문 생성 (Suggestion)](#context-기반-추천-질문-생성-suggestion)
6. [실험 및 검증](#실험-및-검증)

---

## 필요한 기능

- 인풋 쿼리에서 동을 따로 받아서 해당하는 동네 내에서만 검색결과 보여주는 기능
    - 인풋 쿼리의 랜드마크나 위치를 받아서 행정동으로 변환하는 에이전트
- Stateful 채팅을 구현해서 이전까지의 Context를 기억해서 다른 카페 찾을 수 있는 기능
- Context에 알맞는 다음 예상 질문을 생성하는 기능

---

## 동네 처리 → Location Agent

**문제정의:** 쿼리에서 자연어로 들어오는 동네의 형태는 매우 다양하다. 홍대, 롯데월드근처(랜드마크)일 수도 있고, 잠실역(역 이름)등일수도 있고, ‘연남동’과 같이 동네 이름 자체일 수도 있다. 유저가 제시한 위치의 근처에 위치한 카페를 검색결과로 보여주기 위해, Location Agent를 사용해서 연관 행정동 리스트를 추출하고, 카페마다 DB에 저장된 `dong_group` 에 해당하는 카페만 검색을 해서 결과로 제시하는 기능이 필요했다.

**핵심 목표**: `"홍대"` → `["서교", "연남", "동교", "상수", "합정"]`과 같이 연관 행정동 리스트를 추출하여 검색 재현율(Recall) 향상

### LocationAgent.run(user_query: str) → dict

**입력**: user_query: "홍대 조용한 카페 추천해줘"

**처리 흐름**:

```python
def run(self, user_query: str) -> dict:
    # 1. LLM 호출
    prompt_result = self._call_llm(user_query)

    # 2. JSON 파싱 (실패 시 {}~{} 패턴 추출)
    try:
        result_json = json.loads(prompt_result)
    except JSONDecodeError:
        start = prompt_result.find('{')
        end = prompt_result.rfind('}') + 1
        result_json = json.loads(prompt_result[start:end])

    # 3. 결과 반환
    return {
        "is_location_query": result_json.get("is_location_query", False),
        "region_names": result_json.get("region_names", []),
        "original_query": user_query
    }
```

사용자 쿼리를 Clova LLM에 전달하여 위치 키워드를 추출한다. LLM 응답이 순수 JSON이 아닐 경우를 대비해 `{`와 `}` 사이의 문자열만 추출하는 fallback을 포함했다. 최종적으로 쿼리에 위치 정보가 있는지 여부(`is_location_query`)와 인근 행정동 리스트(`region_names`)를 반환한다.

- **_call_llm() 핵심 프롬프트**

```
"너는 서울시 행정동 전문가야. 사용자 입력에서 위치 정보를 추출하고,
해당 위치와 인접한 서울시 행정동 이름을 정확히 반환해."

[핵심 규칙]
1. 반드시 행정동 이름만 반환 (예: 성수1가, 자양, 화양)
2. '동', '역', '입구' 같은 접미사는 절대 포함하지 마
3. 랜드마크/역 이름 → 실제 행정동으로 변환
4. 총 5개의 인접 행정동 반환

[변환 예시]
- '서울숲' → 성수1가, 성수2가, 금호, 옥수, 행당 (서울숲 ❌)
- '건대입구역' → 화양, 자양, 군자, 구의 (건대입구역 ❌)
- '홍대' → 서교, 연남, 동교, 상수, 합정
```

이 프롬프트를 통해서 LLM이 랜드마크나 역 이름을 그대로 반환하지 않고 실제 행정동 이름으로 변환하도록 유도한다. ‘동’ 접미사를 제거하는 규칙은 DB의 `dong_group` 컬럼과 일치시키기 위함이다. dong_group에는 잠실 1동, 잠실 2동 형식으로 숫자를 포함한 동의 형태로 저장되어있는데, 검색할때는 ‘잠실’이라는 키워드만 포함시켜서, 키워드기반 넓은 동네를 포함하여 검색결과를 보여준다.

**API 설정**:
- Clova Studio HCX-DASH-002
- temperature: 0.1 (일관된 답변)
- maxTokens: 100

**출력**:

```json
{
  "is_location_query": true,
  "region_names": ["서교", "연남", "동교", "상수", "합정"],
  "original_query": "홍대 조용한 카페 추천해줘"
}
```

### 검색 엔진 연동 (`src/engine/retriever.py`)

- `vibe_search()`에 `location_filter` 파라미터 추가
- `dong_group` 컬럼을 이용한 SQL OR 조건 필터링

```sql
WHERE dong_group IN ('서교', '연남', '동교', '상수', '합정')
```

LocationAgent가 추출한 행정동 리스트를 SQL의 IN 조건으로 변환하여, 해당 지역에 속한 카페만 검색 대상에 포함시킴. 이를 통해 검색 범위를 좁히고 해당 동네에 대해서만 검색결과를다.

---

## Stateful 채팅 구현

**문제정의:** 기존에는 한번만 검색하고 결과를 보여주는 서비스였다. 하지만 유저가 추천결과에 대해서 검색을 심화할수도 있고(”더 조용한 카페 추천”), 혹은 도중에 검색하는동네를 바꾸어서 검색을 할수도있고, 추천된 가게의 메타데이터에 대해서 질문을 할수도 있을거라고 예상했다. 따라서 이전의 context를 Pydantic을 이용해서 state로 정의하고, 그 context를 Multi Agent가 참고하면서 추론할 수 있는 로직이 필요했다.

### 주요 기능

- 이전의 추천 정보를 Context로 기억하고 새로운 카페를 추천
- 추천된 카페에 대한 메타정보 질문 시 DB를 retrieval해서 답변
- 이미 추천한 장소는 재추천하지 않음

### AgentState 데이터 모델 (`backend/agents/state.py`)

```python
class AgentState(BaseModel):
    session_id: str              # 세션 고유 ID
    user_query: str              # 현재 사용자 쿼리

    # === Shared Memory (턴 간 유지) ===
    history_place_ids: List[int]       # 이미 추천된 장소 ID (중복 방지)
    last_recommendations: List[Dict]   # 최근 추천 결과 (QnA 참조용)
    current_location_filter: List[str] # 현재 동네 필터 ["서교", "연남"]
    previous_hyde_query: str           # 직전 HyDE 쿼리 (분위기 맥락 유지)
    file_path: str                     # 업로드된 이미지 경로

    # === 현재 턴 상태 ===
    intent: AgentIntent          # SEARCH | FEEDBACK | QNA
    final_response: Dict         # 최종 응답
```

`AgentState`는 Pydantic 모델로, 대화 세션의 모든 상태를 담고 있다. `history_place_ids`는 이미 추천한 장소를 기억해 중복 추천을 방지하고, `previous_hyde_query`는 이전의 HyDE Query를 저장한것으로 “더 조용한 곳” 같은 피드백 시 이전 분위기 맥락을 유지하는 데 사용한다. 모든 에이전트가 이 상태 객체를 공유한다.

### ContextManager (`backend/agents/context_manager.py`)

DuckDB 기반의 영속적 상태 관리 싱글톤

| 메서드 | 입력 | 동작 | 출력 |
| --- | --- | --- | --- |
| `get_or_create_state(session_id, query)` | session_id, user_query | DB에서 기존 세션 로드 또는 신규 생성 | AgentState |
| `get_state(session_id)` | session_id | 현재 세션 상태 조회 | AgentState |
| `save_state(state)` | AgentState | 변경된 상태를 DuckDB에 UPSERT | None |
| `update_history(state, recs)` | state, 추천 결과 | history_place_ids에 추가 (중복 방지) | None |

**저장 구조 (DuckDB)**:

```sql
CREATE TABLE sessions (
    session_id VARCHAR PRIMARY KEY,
    state_json VARCHAR,           -- AgentState를 JSON으로 직렬화
    updated_at TIMESTAMP
);
```

세션 상태를 DuckDB 테이블에 JSON 문자열로 직렬화하여 저장한다. 서버가 재시작되어도 `session_id`를 기반으로 이전 대화 상태를 복원할 수 있어 영속적인 대화 컨텍스트를 유지할 수 있다.

**상태 저장 로직**:

```python
def save_state(state: AgentState):
    state_json = state.model_dump_json()

    con.execute("""
        INSERT INTO sessions (session_id, state_json, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT (session_id) DO UPDATE SET
            state_json = excluded.state_json,
            updated_at = excluded.updated_at
    """, (state.session_id, state_json))
```

Pydantic의 `model_dump_json()`으로 상태를 JSON 문자열로 변환한 후, DuckDB의 UPSERT 구문(`INSERT ... ON CONFLICT DO UPDATE`)으로 저장한다. 같은 `session_id`가 있으면 업데이트, 없으면 새로 생성된다.

---

## ADK 마이그레이션

### 현재 구현된 Agent

| Agent | 역할 | 입력 | 출력 |
| --- | --- | --- | --- |
| **Router** | 유저 의도 파악 | user_query | AgentIntent (SEARCH/FEEDBACK/QNA) |
| **Location** | 랜드마크 → 행정동 변환 | user_query | region_names 리스트 |
| **Search** | 카페 검색 실행 | state (필터, 히스토리 포함) | recommendations 리스트 |
| **QnA** | 장소 메타정보 답변 | state (last_recommendations 참조) | 답변 메시지 |

### 왜 ADK?

- **Intent-Based Routing**: 사용자 발화 의도를 `SEARCH`, `FEEDBACK`, `QNA`로 분류
- **LoopAgent**: 에이전트들을 순차 실행하며, `escalate=True` 발생 시까지 반복
- **ADK Viewer**: 웹 기반 디버깅 UI 제공

![image](/assets/images/2026-02-12-17-04-18.png)

---

## 파일별 역할 및 상세 동작

![image](/assets/images/2026-02-12-17-04-24.png)

```
backend/
├── main.py                    # FastAPI 앱 설정 & 서버 시작
├── api/
│   └── chat.py                # API 엔드포인트 (Entry Point)
├── workflow.py                # LoopAgent 정의 (에이전트 순서)
├── lib/
│   └── google_adk_agents.py   # ADK Wrapper (BaseAgent 구현)
├── agents/
│   ├── router.py              # 의도 분류 로직
│   ├── location_agent.py      # 위치 파싱 로직
│   ├── qna_agent.py           # Q&A 로직
│   ├── context_manager.py     # 상태 관리
│   └── state.py               # AgentState 모델
└── services/
    ├── search_service.py      # HyDE + Retrieval
    ├── reasoning.py           # 추천 이유 생성
    └── suggestion.py          # 추천 질문 생성
```

백엔드 로직은 **API 계층(`backend/api`)**, **워크플로우 계층(`backend/lib`)**, **핵심 로직 계층(`backend/agents`)**, **서비스 계층(`backend/services`)**으로 나뉜다.

### Entry Point (`backend/api/chat.py`)

```python
@router.post("/", response_model=RecommendedPlacesResponse)
async def create_chat(session_id, user_query, file):
    # 1. 상태 로드/생성
    init_state = context_mgr.get_or_create_state(session_id, user_query)
    init_state.file_path = file_path_disk
    context_mgr.save_state(init_state)

    # 2. ADK Runner 설정 & 실행
    runner = InMemoryRunner(agent=adk_app, app_name="vibe_pick")
    user_msg = types.Content(role="user", parts=[types.Part(text=user_query)])

    async for event in runner.run_async(user_id, session_id, new_message=user_msg):
        pass  # 에이전트 순차 실행

    # 3. 결과 추출
    final_state = context_mgr.get_state(session_id)
    recommendations = final_state.final_response.get("recommendations", [])

    # 4. 추천 질문 생성 (Suggestion)
    suggestions = generate_suggestions(user_query, place_names[:3], hyde_query)

    # 5. 응답 반환
    return RecommendedPlacesResponse(
        recommendations=recommendations,
        suggestions=suggestions,
        ...
    )
```

API의 엔트리포인트다. 먼저 `ContextManager`로 세션 상태를 로드/생성한 후, ADK의 `InMemoryRunner`를 통해 LoopAgent를 실행한다. 에이전트들이 순차적으로 실행되고, 완료되면 `final_state`에서 결과를 추출한다. 마지막으로 `generate_suggestions()`를 호출해 다음 추천 질문을 생성하여 응답에 포함한다.

### Workflow (`backend/workflow.py`)

```python
from google.adk.agents.loop_agent import LoopAgent

router = ADKRouterAgent(name="router")
location = ADKLocationAgent(name="location")
search = ADKSearchAgent(name="search")
qna = ADKQnAAgent(name="qna")

app = LoopAgent(
    name="vibe_pick_agent",
    sub_agents=[router, location, search, qna],
    max_iterations=5  # 무한루프 방지를 위한 안전장치
)
```

ADK의 `LoopAgent`를 사용해 에이전트 실행 순서를 정의한다. 배열 순서대로 Router → Location → Search → QnA가 실행되며, 중간에 `escalate=True` 이벤트가 발생하면 루프가 종료된다.

### ADKRouterAgent (`backend/lib/google_adk_agents.py`)

```python
async def _run_async_impl(self, ctx: InvocationContext):
    session_id = ctx.session.id

    # 1. 텍스트/이미지 추출
    user_query = ""
    image_path = None
    for part in ctx.user_content.parts:
        if getattr(part, "text", None):
            user_query += part.text
        if getattr(part, "inline_data", None):
            # 이미지 저장: temp_images/{uuid}.jpg
            image_path = save_image(part.inline_data)

    # 2. 상태 로드
    state = context_mgr.get_or_create_state(session_id, user_query)
    state.file_path = image_path

    # 3. 라우팅 실행
    intent = self.router_logic.route(state)  # Router.route() 호출
    state.intent = intent  # SEARCH | FEEDBACK | QNA

    # 4. 상태 저장
    context_mgr.save_state(state)

    yield Event(author=self.name, custom_metadata={"type": "routed"})
```

LoopAgent에서 가장 먼저 실행되는 에이전트이다. ADK의 `ctx.user_content`에서 텍스트와 이미지를 추출하고, `Router.route()` 메서드를 호출해 사용자 의도(SEARCH, FEEDBACK, QNA 중 하나)를 판단한다. 판단 결과를 `state.intent`에 저장하여 이후 에이전트들이 참조할 수 있게 한다.

### ADKLocationAgent

```python
async def _run_async_impl(self, ctx: InvocationContext):
    state = context_mgr.get_state(ctx.session.id)

    # Intent가 SEARCH/FEEDBACK일 때만 실행
    if state.intent not in [AgentIntent.SEARCH, AgentIntent.FEEDBACK]:
        return  # 조기 반환

    # 위치 파싱
    loc_result = self.loc_agent.run(state.user_query)
    # {"is_location_query": true, "region_names": ["서교", "연남"]}

    new_filters = loc_result.get("region_names", [])
    if new_filters:
        state.current_location_filter.extend(new_filters)
        state.current_location_filter = list(set(state.current_location_filter))

    context_mgr.save_state(state)
    yield Event(author=self.name, custom_metadata={"type": "location_parsed"})
```

Router가 판단한 `intent`가 SEARCH나 FEEDBACK일 때만 동작하고, QNA면 바로 반환한다(QnA에서는 location의 변화가 없으므로). `LocationAgent.run()`을 호출해 쿼리에서 지역명을 추출하고, 새로 발견된 지역을 `current_location_filter`에 추가한다. `list(set(...))`으로 중복을 제거한다.

### ADKSearchAgent

```python
async def _run_async_impl(self, ctx: InvocationContext):
    state = context_mgr.get_state(ctx.session.id)

    if state.intent not in [AgentIntent.SEARCH, AgentIntent.FEEDBACK]:
        return

    # 이전 HyDE 컨텍스트 참조 (FEEDBACK 시 분위기 누적)
    prev_context = state.previous_hyde_query

    # 검색 실행
    hyde_query, recommendations = await hyde_search(
        state.user_query,           # "더 조용한 곳"
        state.file_path,            # 업로드된 이미지
        state.current_location_filter,  # ["서교", "연남"...]
        state.history_place_ids,    # 이미 추천한 장소 제외
        prev_context                # "홍대 모던한 인테리어..." (이전 HyDE)
    )

    # 상태 업데이트
    state.previous_hyde_query = hyde_query
    context_mgr.update_history(state, recommendations)
    state.final_response = {"hyde_query": hyde_query, "recommendations": recommendations}
    context_mgr.save_state(state)

    # 루프 종료 (escalate=True)
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=output_text)]),
        actions=EventActions(escalate=True)
    )
```

실제 검색을 수행하는 에이전트다. `hyde_search()` 함수에 현재 쿼리, 이미지, 지역 필터, 제외할 장소 ID 목록, 이전 HyDE 쿼리를 전달한다. `history_place_ids`를 통해 이미 추천한 장소를 제외하고, `prev_context`를 통해 이전 분위기 맥락과 결합된 새 HyDE 쿼리를 생성한다. 검색이 완료되면 `escalate=True`로 LoopAgent를 종료시킨다.

### QnAAgent.run(state: AgentState) → dict

```python
def run(self, state: AgentState) -> dict:
    query = state.user_query  # "첫 번째 카페 주차 돼?"

    # 1. 대상 장소 식별 (LLM 사용)
    target_places = self._identify_target_places(query, state.last_recommendations)
    # [{"place_id": 101, "place_name": "블루보틀 연남점"}]

    if not target_places:
        return {"message": "참고할 추천 결과가 없어서 답변드리기 어려워요."}

    # 2. 장소 상세 정보 조회 (DB)
    all_details = []
    for place in target_places:
        details = self._fetch_place_details(place['place_id'])
        # places: address, business_hours, contact
        # amenities: 주차, 예약, 와이파이 등 10개 항목
        # reviews: summary, keywords
        # menus: name, price
        all_details.append(f"=={place['place_name']} ==\n{details}")

    # 3. LLM 답변 생성
    answer = self._generate_answer(query, place_names, combined_details)
    # "네, 주차 가능합니다. 건물 지하주차장 이용 가능해요."

    return {"message": answer, "related_place_id": target_places[0]['place_id']}
```

QnA 에이전트는 추천된 가게의 메타데이터에 대한 질문을 답변하는 에이전트로 3단계로 동작한다. 먼저 `_identify_target_places()`에서 LLM을 사용해 “첫 번째”, “두 번째” 같은 표현이 가리키는 실제 장소를 식별한다. 그 다음 `_fetch_place_details()`로 해당 장소의 주소, 편의시설, 리뷰, 메뉴 정보를 DB에서 조회한다. 마지막으로 `_generate_answer()`에서 조회된 정보를 바탕으로 LLM이 자연스러운 답변을 생성한다.

- **_identify_target_places() 프롬프트**:

```
"너는 사용자의 질문이 '추천 장소 목록' 중 어떤 장소를 지칭하는지 식별하는 분류기야.
아래 목록을 보고, 질문이 가리키는 장소들의 ID를 JSON 리스트로 반환해."

[추천 장소 목록]
1. 블루보틀 연남점 (ID: 101)
2. 앤트러사이트 (ID: 102)
3. 프리베 (ID: 103)

→ 출력: [101]
```

LLM에게 추천 목록과 함께 사용자 질문을 전달하면, “첫 번째” → ID 101, “두 번째와 세 번째” → [102, 103] 처럼 해당하한다. “모두”, “전부” 같은 표현이면 전체 ID를 반환하도록 프롬프트에 명시되어 있다.

---

## Context 기반 추천 질문 생성 (Suggestion)

**문제정의:** 유저가 어떤 검색을 해야할지 잘 모르겠는 상황이 있을거라고 생각했다. 따라서 이전의 유저의 요구를 Context로 이용해서 다음 질문을 예측하는 로직을 추가해서 챗봇에 추가했다.

**기능:** 검색 완료 후, 사용자가 다음에 물어볼 법한 질문 4개를 자동 생성

### generate_suggestions() (`backend/services/suggestion.py`)

**입력(Context)**:

`user_query` =“홍대 조용한 카페” 
`place_names` = [“블루보틀 연남점”, “앤트러사이트”, “프리베”]
`hyde_query` =“홍대 연남동 조용하고 넓은 우드톤 카페…” 

유저의 원본 쿼리와 hyde쿼리를 가져와서 유저의 의도를 파악하도록 했고, 이전에 추천된 가게의 이름을 `place_name`으로 제공해주었다.

**처리 흐름**:

```python
def generate_suggestions(user_query, place_names, hyde_query=None) -> list[str]:
    # 1. Context 포맷팅
    place_str = ', '.join(place_names[:3])
    user_content = f"검색어:{user_query}\n추천 장소:{place_str}\n분위기:{hyde_query or '없음'}"

    # 2. 시스템 프롬프트
    system_prompt = """너는 카페 추천 서비스의 대화 어시스턴트야.
    사용자가 카페 추천을 받았어. 사용자가 다음에 물어볼 법한 질문 4개를 생성해줘.

    [질문 유형 - 반드시 다양하게 섞어서]
    1. 장소 상세 질문: 주차, 메뉴, 영업시간 등
    2. 결과 조정 요청: 더 조용한 곳, 더 넓은 곳, 분위기 다른 곳 등

    [규칙]
    - 각 질문은 30자 이내로 짧게
    - 자연스러운 구어체 ("첫 번째 카페 주차 가능해?", "더 아늑한 곳 없어?")
    - 순수 JSON 배열만 출력: ["질문1", "질문2", "질문3", "질문4"]
    """

    # 3. Clova API 호출 (temperature=0.7 다양성 확보)
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7,
        "maxTokens": 100
    }

    res = requests.post(clova_api_url, headers=headers, json=payload)
    content = res.json()['result']['message']['content']

    # 4. JSON 파싱 (마크다운 코드블록 제거)
    suggestions = extract_json_array(content)

    # 5. Fallback (4개 미만 시 기본 질문으로 채움)
    if len(suggestions) < 4:
        defaults = ["주차 가능해?", "메뉴 추천해줘", "더 조용한 곳은?", "영업시간 알려줘"]
        return (suggestions + defaults)[:4]

    return suggestions[:4]
```

검색 결과의 Context(검색어, 추천 장소 이름, 분위기 키워드)를 조합해 LLM에 전달하고, 사용자가 다음에 물어볼 법한 자연스러운 질문 4개를 생성한다. `temperature=0.7`로 설정해 다양한 질문이 생성되도록 하고, LLM 응답이 4개 미만이면 fallback으로 기본 질문으로 채운다. 질문 유형은 QNA(장소 상세)와 FEEDBACK(결과 조정)이 섞이도록 프롬프트에 명시되어 있어, 사용자가 클릭하면 자연스럽게 다음 대화로 이어진다.

![image.png](image%202.png)

**출력 예시**:["첫 번째 카페 주차 가능해?", "두 번째 카페 메뉴 추천해줘", "더 넓은 곳 없어?", "영업시간 알려줘"]

### 질문 유형 설계 의도

| 생성되는 질문 유형 | Router 분류 | 처리 Agent |
| --- | --- | --- |
| 장소 상세 질문 | QNA | QnAAgent |
| 결과 조정 요청 | FEEDBACK | SearchAgent |

→ 사용자가 추천 질문 클릭 시, Router가 자동 분류하여 적절한 에이전트 실행

### 호출 시점 (`chat.py`)

```python
# 검색 완료 후 suggestions 생성
suggestions = None
if recommendations:
    place_names = [p.place_name for p in recommendations[:3]]
    try:
        suggestions = generate_suggestions(
            user_query=user_query,
            place_names=place_names,
            hyde_query=hyde_query
        )
    except Exception as e:
        suggestions = ["주차 가능해?", "메뉴 추천해줘", "더 조용한 곳은?", "영업시간 알려줘"]

return RecommendedPlacesResponse(
    ...
    suggestions=suggestions
)
```

추천 결과가 있을 때만 `generate_suggestions()`를 호출한다. 생성된 추천 질문은 프론트엔드에서 버튼 형태로 표시된다.

---

## 전체 데이터 흐름도

![image](/assets/images/2026-02-12-17-05-18.png)

전체 흐름을 요약하면 사용자 요청이 들어오면 LoopAgent가 Router → Location → Search/QnA 순서로 에이전트를 실행한다. 각 에이전트는 `state.intent`를 확인해 자신이 처리해야 할 요청인지 판단하고, 해당되지 않으면 조기 반환한다. 최종적으로 검색 결과와 추천 질문을 함께 응답한다.

---

## 실험 및 검증

### Case 1(Feedback): 검색된 카페의 분위기가 마음에 들지 않아서 추가적인 분위기를 제공할때

동작: Location(동네)는 변경하지 않고 이전의 HyDE 쿼리에 추가로 HyDE쿼리를 생성해서 두 HyDE쿼리를 합친 후 다시 검색함

![image](/assets/images/2026-02-12-17-05-24.png)

![image](/assets/images/2026-02-12-17-05-31.png)

> 상황: 성수동 검색 결과가 마음에 안 들어서 "더 개방감 있는 분위기로 찾아줘"라고 했을 때
> 
1. **Router Agent**: "좀 더", "분위기로" 같은 표현과 이전 맥락(Context)을 보고 의도를 으로 분류.
2. **Location Agent**: 의도가 이므로 동작하지만, 새로운 지명이 없으므로 **기존의 `current_location_filter`("성동구 성수동")를 그대로 유지**.
3. **Search Agent**:
    - `FEEDBACK` 의도일 경우, 이전 검색을 위해 생성했던 시각적 묘사(HyDE)를 가져옴
    - 새로운 쿼리의 HyDE를 합쳐서 **더 구체적인 새로운 HyDE 쿼리**를 생성합니다. (예: "모던한... + 밝은 자연광이 들어오는...")
    - **Retriever**: 기존 위치 필터("성수동") 내에서 새로운 HyDE 쿼리와 유사한 이미지를 가진 장소를 재검색

### Case 2 (QnA): 검색된 카페에 대한 메타정보를 물어볼때

![image](/assets/images/2026-02-12-17-05-36.png)

> 상황: 검색된 결과 중 "첫 번째 곳 주차 돼?"라고 물었을 때
> 
1. **Router Agent**: "사람들의 평은 어때?",  같은 의도를 인식하여 의도를 로 분류
2. **Location Agent / Search Agent**: 의도가 이므로 **동작하지 않고 건너뜀.**
3. **QnA Agent**:
    - **Selector (LLM)**: 질문("첫 번째 곳")과 (이전 검색 결과 3개)를 보고 "1번 장소(ID: 101)"를 대상으로 식별
    - **DB Fetch**: 해당 장소의 **amenities**, **places** 테이블을 조회하여 주차 정보 등을 가져옴.
    - **Generator (LLM)**: "이 장소는 주차가 가능합니다"라는 답변을 생성하여 사용자에게 전달.

### Case 3(new Search): 이전의 지역이 아니라 새로운 지역에 대해서 검색하고 싶을때

![image](/assets/images/2026-02-12-17-05-43.png)

> 상황: "강남역 근처 카공하기 좋은 곳 추천해줘"라고 했을 때
> 
1. **Router Agent**:
    - 먼저 **Location Agent**를 호출.
    - LLM이 명확한 지역을 인식하고(”강남역”) Intent를 새로운 지역에 대한 Search로 인식함
2. **Location Agent**:
    - "강남역"을 **["강남구 역삼동", "서초구 서초동"]** 등의 행정동 리스트로 변환하여 를 **덮어씀(Overwrite).**
3. **Search Agent**:
    - 변경된 필터(역삼/서초) 내에서 검색하여 결과를 반환.

---

## 완료된 기능

- [x]  인풋 쿼리에서 동을 파싱하여 해당 동네 내에서만 검색
- [x]  Stateful 채팅으로 이전 Context 기억
- [x]  Context에 알맞는 예상 질문 생성 (Suggestion)
- [x]  QnA Agent로 장소 메타정보 답변
- [x]  이미 추천한 장소 중복 방지 (history_place_ids)
- [x]  분위기 피드백 시 HyDE 누적 (previous_hyde_query)