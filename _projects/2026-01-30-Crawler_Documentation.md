---
layout: single
title: "네이버지도에서 서울시 카페 전부 크롤링"
date: 2026-01-30
tags:
  - 이미지기반 분위기로 카페 찾기 프로젝트
  - Crawling
excerpt: ""
math: true
---

# 네이버 지도 서울 카페 분산 크롤러 구현

## 목차
1. [도입 배경 및 문제 정의](#1-도입-배경-및-문제-정의)
2. [아키텍처 및 핵심 로직](#2-아키텍처-및-핵심-로직)
3. [데이터 추출 전략](#3-데이터-추출-전략)
4. [안정성 및 리소스 관리](#4-안정성-및-리소스-관리)
5. [데이터 스키마 및 저장](#5-데이터-스키마-및-저장)

---

## 도입 배경 및 문제 정의

### 크롤링 목표

서울시 전역 **424개 행정동**에 위치한 카페 데이터를 정밀하게 수집하여 이미지 기반 카페추천 서비스의 기반 데이터를 구축함. 단순한 장소명 수집을 넘어, 메뉴 이미지, 리뷰, 영업시간, 편의시설 정보 등 상세 메타데이터 확보가 필수적임.

### Playwright 도입 배경 (vs Selenium)

기존 Selenium 방식은 React 기반의 동적 페이지(네이버 지도)를 처리하는 데 구조적 한계가 있어, 최신 자동화 도구인 **Playwright**를 도입함.

### 1. Auto-waiting

- **문제**: 네이버 지도는 수시로 DOM을 다시 그리기 때문에, Selenium은 요소를 찾은 직후 ‘요소가 사라짐(Stale Element)’ 에러를 빈번하게 발생시킴.
- **해결**: Playwright는 클릭 직전까지 요소가 화면에 고정되고 클릭 가능한 상태인지 **자동으로 대기(Auto-waiting)**함. 이로 인해 `explicit_wait` 남발 없이도 안정적인 클릭이 가능함.

### 2. 효율적인 Iframe 제어 (Frame Locator)

- **문제**: 검색 결과(`searchIframe`)와 상세 정보(`entryIframe`)가 서로 다른 iframe에 존재하여, Selenium은 `switch_to.frame`으로 컨텍스트를 계속 오가야 함.
- **해결**: Playwright의 **Frame Locator** API를 사용하여 프레임 간 전환 비용 없이 DOM 요소에 즉시 접근함.
    - *Code*: `self.page.frame_locator("#entryIframe").locator(...)`

### 3. 고속 데이터 추출 (Fast Evaluation)

- **문제**: Selenium의 `execute_script`는 대용량 데이터 전송 시 직렬화 병목이 발생함.
- **해결**: Playwright는 브라우저 엔진(V8)과 직접 통신하는 `evaluate()` 메서드를 제공함. 이를 통해 수 MB 크기의 `__APOLLO_STATE__` 객체를 지연 없이 즉시 Python 객체로 변환함.

### 기술적 도전 과제 및 해결 방안

| 기술적 문제 (Problem) | 상세 내용 | 해결 방안 (Solution) |
| --- | --- | --- |
| **대규모 데이터 처리** | 단일 프로세스로 수만 개 페이지 방문 시 실행 시간 과다 소요 | **Multiprocessing**: 동(Dong) 단위로 프로세스를 분할하여 병렬 처리함 |
| **메모리 누수 (OOM)** | Chromium 로직 특성상 장시간 실행 시 메모리 사용량 지속 증가 | **3-Tier Protection**: 프로세스(재생성), 브라우저(재시작), 카페(타임아웃) 단위로 다중 리소스 관리함 |
| **동적 렌더링** | React 기반 SPA로 DOM 구조가 수시로 변경되고 렌더링 시점 불규칙 | **Hybrid Extraction**: `window.__APOLLO_STATE__` 직접 파싱과 DOM 탐색을 병행함 |
| **IP 차단 및 탐지** | 반복적인 자동화 접근 시 CAPTCHA 발생 또는 IP 차단으로 수집 중단 | **Stealth & Auto-Recovery**: User-Agent 위장, 랜덤 딜레이, 3회 실패 시 브라우저 교체함 |

---

## 아키텍처 및 핵심 로직

### Manager-Worker 패턴 구조

전체 시스템은 작업을 할당하는 **Manager**와 실제 수집을 수행하는 **Worker**로 구성됨.

```
[CrawlerManager]
      |
      +--- 파트 파일 로드 (dongs_part_1.json)
      |
      +--- Multiprocessing Pool 생성 (num_workers=3)
               |
               +--- [Worker Process 1] ──(spawn)──> [Chromium] : "역삼동 카페" 수집
               |         (작업 완료 후 프로세스 소멸 및 메모리 반환)
               |
               +--- [Worker Process 2] ──(spawn)──> [Chromium] : "논현동 카페" 수집
               |         (작업 완료 후 프로세스 소멸 및 메모리 반환)
```

### 병렬 처리 구현 (`crawler/manager.py`)

```python
def run_parallel(self, part_id, num_workers=3):
    # ...
    # maxtasksperchild=1: 각 작업(동 하나)이 끝날 때마다 프로세스를 새로 띄워
    # 브라우저 메모리 누적 문제를 원천 차단함.
    try:
        with Pool(processes=num_workers, maxtasksperchild=1) as pool:
            pool.map(_worker_wrapper, tasks)
    except KeyboardInterrupt:
        pool.terminate()
```

Python `multiprocessing.Pool`의 `maxtasksperchild=1` 옵션을 사용하여, 하나의 동 작업이 끝나면 해당 프로세스를 완전히 종료하고 새로 생성함. 이는 Chromium이 점유했던 메모리를 OS 수준에서 강제로 회수함으로써 메모리 문제 해결.

---

## 데이터 추출 전략

### 1. Apollo State 파싱

네이버 지도는 React 앱의 초기 상태를 `window.__APOLLO_STATE__` 변수에 저장함. 이를 직접 파싱 하여 DOM 렌더링을 기다릴 필요 없이 순수한 JSON 데이터를 추출함.

**코드 구현 (`crawler/scraper.py`)**:

```python
def _get_apollo_state(self):
    """페이지 소스에 포함된 Apollo State(JSON) 추출"""
    states_to_check = ["__APOLLO_STATE__", "__PLACE_STATE__", "__INITIAL_STATE__"]
    for state_name in states_to_check:
        try:
            # 브라우저 컨텍스트에서 JS 변수 추출
            state_json = frame.evaluate(f"() => window.{state_name} ? JSON.stringify(window.{state_name}) : null")
            if state_json:
                data = json.loads(state_json)
                merged_state.update(data)
        except: pass
    return merged_state
```

 화면에 표시되는 텍스트를 긁어오는 방식(Scraping) 대신, 데이터 소스 자체를 추출(Extraction)하므로 DOM 클래스 변경에 영향을 받지 않으며 속도가 매우 빠름.

### 2. DOM Fallback (Secondary Strategy)

Apollo State에 없는 최신 정보나 동적으로 로딩되는 데이터(리뷰, 상세 영업시간)는 DOM 탐색으로 보완함.

```python
# 이름 추출 실패 시 DOM 선택자로 재시도
if meta["name"] == "Unknown":
    name_selectors = [
        "span.GHAhO", "span.Fc1rA", "span.YwYLL",
        ".place_name", "._name", "h2.place_name"
    ]
    for selector in name_selectors:
        name_el = frame.locator(selector).first
        if name_el.is_visible():
            meta["name"] = name_el.inner_text().strip()
            break
```

> **설명**: 네이버 지도의 잦은 클래스명 변경에 대응하기 위해 다양한 CSS 선택자 리스트(`name_selectors`)를 순차적으로 시도하는 Fallback 로직을 구현함.
> 

### 3. 레이지 로딩(Lazy Loading) 극복

메뉴 사진이나 리뷰는 스크롤 시점에 로딩되므로, JS를 활용한 공격적인 스크롤링을 수행함.

```python
# 메뉴 사진 로딩을 위한 강제 스크롤
for _ in range(3):
    frame.locator("body").evaluate("window.scrollBy(0, 500)")
    time.sleep(0.3)

# 아이템별 스크롤 이동
for item in items:
    item.scroll_into_view_if_needed()
```

### 4. 시각 정보 우선순위 (Visual Priority)

카페 추천 서비스 특성상 음식 사진보다 **매장 분위기(Vibe)**가 더 중요한 판단 기준임. 무작위로 사진을 수집하는 대신 ‘내부’ 탭을 우선적으로 탐색함.

```python
# 사진 탭 이동 후 '내부' 카테고리 우선 클릭 (crawler/scraper.py)
if category:
    try:
        cat_btn = frame.locator("a, span").filter(has_text=re.compile(f"^{category}$")).first
        if cat_btn.is_visible():
            cat_btn.click()
            time.sleep(1)
    except: pass
```

사진 탭에 진입하면 기본적으로 전체 사진이 보이지만, 코드에서 명시적으로 **‘내부’** 탭을 찾아 클릭함. 이를 통해 매장 인테리어와 분위기를 파악할 수 있는 이미지를 최우선으로 확보함.

---

## 안정성 및 리소스 관리

장시간 크롤링 시 발생할 수 있는 모든 불안정 요소를 방어함.

### 1. 프로세스 라이프사이클 관리

- 앞서 언급한 `maxtasksperchild=1` 설정을 통해 동 단위 작업 완료 시 프로세스를 초기화함.

### 2. 브라우저 정기 재시작 (`crawler/worker.py`)

동일 프로세스 내에서도 브라우저 인스턴스가 오래 유지되면 느려질 수 있으므로 일정 개수마다 재시작함.

```python
BROWSER_RESTART_INTERVAL = 10  # 10개 카페마다 브라우저 재시작

if cafes_since_restart >= BROWSER_RESTART_INTERVAL:
    print(f"[Worker-{self.dong}] 브라우저 재시작 (메모리 관리)...")
    self.scraper.stop()
    time.sleep(2)
    self.scraper = NaverMapScraper(headless=True)
    self.scraper.start()
    cafes_since_restart = 0
```

10개의 카페를 수집할 때마다 브라우저를 닫고 새로 엶. 이는 브라우저 내부 캐시와 메모리 파편화를 정리하여 일정한 수집 속도를 유지하게 해줌.

### 3. 타임아웃 핸들링

특정 페이지에서 무한 로딩이 발생할경우, 전체 프로세스가 멈추지 않도록 제한 시간을 둠.

```python
@contextmanager
def timeout_handler(seconds, cafe_name=""):
    def _timeout_handler(signum, frame):
        raise CafeTimeoutError(f"'{cafe_name}' 처리 시간 초과 ({seconds}초)")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds) 
    try:
        yield
    finally:
        signal.alarm(0)
```

크롤링 로직이 3분(180초)을 초과하면 강제로 예외를 발생시키고 다음 카페로 넘어감.

---

## 데이터 스키마 및 저장

수집된 데이터는 엄격한 필드 순서를 가진 JSON으로 저장됨.

| 필드명 | 타입 | 설명 |
| --- | --- | --- |
| `name` | String | 카페 이름 (정규화됨) |
| `address` | String | 도로명 주소 |
| `description` | String | 업체 소개글 |
| `business_hours` | String | 영업시간 (HH:MM - HH:MM 포맷) |
| `contact` | String | 전화번호 |
| `instagram` | String | 인스타그램 URL |
| `amenities` | List | 편의시설 태그 (주차, 와이파이 등) |
| `structured_data` | Dict | 주차 여부, 세부 편의시설 구조화 데이터 |
| `menus` | List | 메뉴 이름, 가격, 사진 URL, 설명 |
| `photos` | List | 매장 내부/외부 사진 URL 리스트 |
| `reviews` | List | 방문자 영수증 리뷰 텍스트 리스트 |
| `dong_group` | String | 수집 대상 행정동 이름 |

**저장 파일 예시**: `data/crawled/seoul/results_part_1_역삼동.json`

---

## 완료된 기능

- [x]  **하이브리드 데이터 추출**: Apollo State 파싱과 DOM 탐색 결합으로 데이터 완전성 확보함
- [x]  **멀티프로세싱 병렬 처리**: 동 단위 프로세스 분할 및 `maxtasksperchild=1` 적용함
- [x]  **안정적 메모리 관리**: 10개 카페 단위 브라우저 재시작 로직 구현함
- [x]  **예외 처리 및 자동 복구**: 타임아웃(180초) 및 연속 실패 시 자동 재시작함
- [x]  **동적 컨텐츠 수집**: 메뉴 사진, 방문자 리뷰 더보기 자동 클릭 및 스크롤링함
- [x]  **광고 필터링**: 검색 결과 상단의 광고(‘광고’ 뱃지) 자동 식별 및 스킵함