---
layout: modern-single
title: "카피 A/B 테스트와 밈으로 광고 성과를 끌어올리기"
date: 2026-10-12
tags:
  - Project
excerpt: "광고 배너를 자동으로 만들 수 있게 된 뒤에는 어떤 카피를 얹을지가 다음 과제였다. 같은 스토리의 소구점도 행동을 권하는 문장으로 쓸 수 있고, 캐릭터의 대사나 상황을 보여주는 문장으로 쓸 수 있었다. 서비스별로 마케터분들이 선호하는 톤도 달랐다."
project_group: wrtn-internal-agent-builder
notion_source_id: sha256:965ecbf938973af9b474
---

광고 배너를 자동으로 만들 수 있게 된 뒤에는 어떤 카피를 얹을지가 다음 과제였다. 같은 스토리의 소구점도 행동을 권하는 문장으로 쓸 수 있고, 캐릭터의 대사나 상황을 보여주는 문장으로 쓸 수 있었다. 서비스별로 마케터분들이 선호하는 톤도 달랐다.

카피의 톤을 바꿔 비교할 수 있는 생성 경로를 만들고, 스토리에 어울리는 밈을 적용하는 테스트도 했다. 카피 A/B 테스트와 밈 테스트는 실제 운영에서 광고 성과를 끌어올린 방법이었다. 카피를 한 방향으로 고정하는 대신 여러 표현을 만들어 집행하고, 반응을 보며 다음 소재에 반영하는 작업으로 이어졌다.

## 현지 마케터의 의견을 테스트할 수 있는 형태로

OOC에서는 구어체 카피, 캬라푸에서는 행동유도형 카피가 좋을 것이라는 마케터분들의 의견이 있었다. 특정 방향으로 고정하기보다 여러 톤을 만들어 테스트해 보자는 방향으로 정리됐다.

같은 스토리 소구점에서 세 가지 형식을 만들었다.

- 행동유도형: 헤드라인이 유저에게 행동을 요구하거나 권한다.

- 캐릭터 대사형: 그림 속 캐릭터가 그 장면에서 할 법한 구어체 한마디다.

- 상황형: 소구점의 상황을 경악·깨달음·곤경으로 표현한다.

![슬라임 스토리의 캐릭터 대사형 카피](/assets/images/notion/965ecbf938973af9b474/01-ad-performance-experiments.png)

<p class="caption">슬라임 스토리의 캐릭터 대사형 카피</p>

![슬라임 스토리의 행동유도형 카피](/assets/images/notion/965ecbf938973af9b474/02-ad-performance-experiments.png)

<p class="caption">슬라임 스토리의 행동유도형 카피</p>

![슬라임 스토리의 상황형 카피](/assets/images/notion/965ecbf938973af9b474/03-ad-performance-experiments.png)

<p class="caption">슬라임 스토리의 상황형 카피</p>

![좀비 스토리의 캐릭터 대사형 카피](/assets/images/notion/965ecbf938973af9b474/04-ad-performance-experiments.png)

<p class="caption">좀비 스토리의 캐릭터 대사형 카피</p>

![좀비 스토리의 행동유도형 카피](/assets/images/notion/965ecbf938973af9b474/05-ad-performance-experiments.png)

<p class="caption">좀비 스토리의 행동유도형 카피</p>

![좀비 스토리의 상황형 카피](/assets/images/notion/965ecbf938973af9b474/06-ad-performance-experiments.png)

<p class="caption">좀비 스토리의 상황형 카피</p>

같은 소재에서도 헤드라인과 CTA가 어떻게 달라지는지 볼 수 있었다. 생성 경로는 공통 입력에서 유형별 카피를 만들고 렌더링하는 형태로 구성했다.

![카피 유형별 생성과 렌더링 흐름](/assets/images/notion/965ecbf938973af9b474/07-ad-performance-experiments.png)

<p class="caption">카피 유형별 생성과 렌더링 흐름</p>

초기 작업 기록에는 7월 15일부터 라이브하고 성과를 모니터링할 예정이라고 남아 있다. 카피 유형을 나눠 생성하는 구현에서 출발해, 실제 운영에서는 이 변형들을 집행하고 비교하며 성과를 개선하는 데 활용했다.

## 밈을 카피에 넣고 실제로 테스트했다

D0 ROAS를 개선할 가설 중 하나로 최신 밈을 카피에 넣어보는 작업을 했다. 클릭을 유도할 만한 표현이어도 스토리와 맞지 않으면 억지스럽고, 위험한 맥락이 있으면 광고에 쓸 수 없었다.

밈 카탈로그에는 의미, 사용 맥락, 톤, 어울리는 캐릭터 유형, 사용 패턴과 템플릿 슬롯, 안전성, 신선도 등을 정리했다. 같은 유행어를 모든 캐릭터에 붙이기보다 어떤 관계와 상황에서 자연스러운지 구분하려는 정보다. 안전 가이드와 사람 검수를 거치고, 재미없거나 맥락에 맞지 않는 결과는 걸러야 했다.

```yaml
- id: ssyagal
  phrase: 쌰갈
  meaning: 한중 국제커플 유튜버 '여단오'의 멤버 여루가 기분 나쁠 때 '씨발'을 돌려 말하는 표현으로, 짜증이나 황당함이 날 때 널리 쓰입니다.
  usage_context:
  - 짜증
  - 황당함
  - 분노
  - 코믹한 분노
  - 스트레스
  - 일상
  tone:
  - 코믹한
  - 짜증난
  - 유쾌한
  - 캐주얼한
  archetype:
  - 다혈질 소꿉친구
  - 스트레스 받는 직장인
  - 개그 캐릭터
  - 허당 악당
  - 츤데레
  usage_pattern: append
  template_slots: '{짜증나는 상황}... 아 쌰갈!'
  source_type: person
  applicability: 3
  safety: yellow
  safety_note: 욕설(씨발)을 돌려 말하는 유행어
  freshness: 5
  era: 2023년경부터 입소문을 타기 시작해 숏폼 및 AI 음성 밈으로 널리 쓰이는 현역 밈
  origin_country: 한국

- id: byeongi_e_naeryeo
  phrase: ○○를 변기에 넣고서 내려
  meaning: 상대를 변기에 넣고 물 내리듯 압도적으로 짓밟겠다는 강한 디스·도발 펀치라인
  usage_context:
  - 디스
  - 도발
  - 라이벌
  - 압도
  - 기싸움
  - 선전포고
  tone:
  - 도발
  - 공격적
  - 힙합
  - 자신감
  - 살벌
  archetype:
  - 라이벌 캐릭터
  - 디스전/래퍼
  - 도발적 빌런
  - 기싸움 연애물
  - 먼치킨
  template_slots: '{상대}를 변기에 넣고서 내려'
  source_type: song
  applicability: 3
  safety: yellow
  safety_note: 실명 비방 원문(빅나티/BIG Naughty) 제거하고 ○○ 슬롯으로 일반화
  freshness: 5
  era: 2025~ 현역(스윙스 디스전 밈)
  origin_country: 한국
- id: bamti-ugly-fashion
  phrase: 밤티
  meaning: 라인플레이의 '밤티'라는 유저의 기괴한 코스튬에 다른 유저가 심한 돌직구를 날린 장면에서 유래한 표현으로, 기괴하거나 촌스러운 외모·코디를
    뜻하는 대명사로 쓰입니다.
  usage_context:
  - 패션테러리스트
  - 팩트폭력
  - 촌스러움
  - 기상천외한코디
  - 무덤덤한반응
  tone:
  - 돌직구
  - 황당함
  - 시니컬
  - 유쾌함
  archetype:
  - 패션테러리스트
  - 마이웨이캐릭터
  - 팩폭러
  - 눈치없는캐릭터
  usage_pattern: append
  template_slots: 밤티님 죄송한데 진짜 개 좆같이 생기 셨네요... 소리가 절로 나오는 상황
  source_type: meme-native
  applicability: 3
  safety: yellow
  safety_note: 실존 유저의 닉네임에서 유래한 외모 비하/촌스러움의 대명사로 쓰여 실제 인물에 대한 외모 비하로 사용하지 않도록 주의해야 함
  freshness: 4
  era: 2020~2021년 라인플레이 커뮤니티에서 시작되어 2024~2025년 X(트위터) 등 SNS에서 다시 확산된 밈
  origin_country: 한국
```

![밈 카탈로그를 활용하는 생성 흐름](/assets/images/notion/965ecbf938973af9b474/08-ad-performance-experiments.png)

<p class="caption">밈 카탈로그를 활용하는 생성 흐름</p>

![밈을 활용한 광고 소재 예시](/assets/images/notion/965ecbf938973af9b474/09-ad-performance-experiments.png)

<p class="caption">밈을 활용한 광고 소재 예시</p>

![스토리에 밈을 적용한 카피 예시](/assets/images/notion/965ecbf938973af9b474/10-ad-performance-experiments.png)

<p class="caption">스토리에 밈을 적용한 카피 예시</p>

![같은 스토리의 다른 밈 카피](/assets/images/notion/965ecbf938973af9b474/11-ad-performance-experiments.png)

<p class="caption">같은 스토리의 다른 밈 카피</p>

밈 테스트도 실제 광고 성과를 끌어올리는 방법이 됐다. 다만 카탈로그에서 유행어 하나를 골라 붙이는 것으로 제작이 끝나는 것은 아니었다. 스토리의 관계와 장면에 맞는지, 광고 문구로 써도 되는지 확인하는 과정이 함께 있어야 했다.

## 성과를 보고 다음 카피를 고르는 작업

카피 A/B 테스트는 같은 소구점을 여러 톤으로 표현하는 경로였고, 밈 테스트는 그 표현에 익숙한 유행어와 사용 맥락을 더하는 경로였다. 둘 다 배너를 생성하는 기존 흐름에 붙여 실제 소재를 내보낼 수 있게 만들었다.

자동 생성의 역할도 카피 한 개를 정답처럼 내놓는 것에서, 비교할 수 있는 후보를 만드는 쪽으로 넓어졌다. 마케터의 의견을 카피 유형으로 만들고, 밈의 의미와 사용 패턴을 카탈로그로 남긴 이유도 같은 가설을 다음 소재에서 다시 테스트할 수 있게 하기 위해서였다.
