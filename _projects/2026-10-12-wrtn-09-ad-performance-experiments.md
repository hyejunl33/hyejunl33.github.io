---
layout: modern-single
title: "광고를 집행하기 전에 성과를 예측할 수 있을까?"
date: 2026-10-12
tags:
  - Project
excerpt: "어떤 광고가 잘될지는 결국 집행 비용을 써봐야 알 수 있었다. 소재를 많이 만들 수 있게 된 다음에는, 같은 스토리에서 어떤 카피를 내보낼지와 집행 전에 후보를 걸러낼 수 있을지가 궁금해졌다."
project_group: wrtn-internal-agent-builder
notion_source_id: sha256:965ecbf938973af9b474
---

어떤 광고가 잘될지는 결국 집행 비용을 써봐야 알 수 있었다. 소재를 많이 만들 수 있게 된 다음에는, 같은 스토리에서 어떤 카피를 내보낼지와 집행 전에 후보를 걸러낼 수 있을지가 궁금해졌다.

카피의 톤을 바꿔 비교할 수 있는 생성 경로를 만들고, 합성 페르소나로 배너의 선호도를 예측하는 실험도 했다. 구현한 것과 성능을 확인한 것은 구분해서 기록하려고 한다. 특히 페르소나 실험은 아직 광고 성과 예측을 입증했다고 말할 수 없는 결과였다.

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

작업 기록에는 7월 15일부터 라이브하고 성과를 모니터링할 예정이라고 남아 있다. 세 유형의 산출물은 있지만, 어느 유형의 CTR이나 ROAS가 더 좋았는지 확인한 결과는 이 기록에 없다. 지금 말할 수 있는 것은 톤을 고정하지 않고 비교할 수 있게 만든 설계와 구현까지다.

## 밈도 가설로 넣어봤다

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

예시는 밈을 반영한 카피와 제작 흐름이다. 밈이 들어가면 CTR이 올라갈 것이라는 기대는 가설이다. 유행어를 적용한 산출물만으로 성과가 개선됐다고 말할 수는 없었다.

## 페르소나가 배너에 반응하게 만들면 어떨까

다음 실험은 NVIDIA Nemotron Persona-Korea 데이터셋을 활용했다. 페르소나에 배너와 카피를 보여주고 선호도를 물으면, 집행 전에 좋은 후보를 찾을 수 있지 않을까 하는 생각이었다. 충분히 검증된다면 일정 점수 이상의 배너만 집행하는 방식으로 연결할 수 있었다.

![페르소나 데이터의 인구통계 분포](/assets/images/notion/965ecbf938973af9b474/12-ad-performance-experiments.png)

<p class="caption">페르소나 데이터의 인구통계 분포</p>

처음에는 연령, 성별, 교육, 직업, 지역을 층화해서 표본을 만들었다. 다음에는 타깃을 20~39세로 좁히고 남성향·여성향에 성별을 맞췄다. 더 나아가 서비스에 관심을 가질 만한 비슷한 프로필 집단을 임베딩과 벡터 검색으로 고르는 방식도 살펴봤다.

![배너와 카피에 대한 선호도 질문](/assets/images/notion/965ecbf938973af9b474/13-ad-performance-experiments.png)

<p class="caption">배너와 카피에 대한 선호도 질문</p>

![페르소나별 선호도 평가 결과](/assets/images/notion/965ecbf938973af9b474/14-ad-performance-experiments.png)

<p class="caption">페르소나별 선호도 평가 결과</p>

![페르소나 임베딩의 군집 분포](/assets/images/notion/965ecbf938973af9b474/15-ad-performance-experiments.png)

<p class="caption">페르소나 임베딩의 군집 분포</p>

![서비스 타겟과 유사한 페르소나 검색 결과](/assets/images/notion/965ecbf938973af9b474/16-ad-performance-experiments.png)

<p class="caption">서비스 타겟과 유사한 페르소나 검색 결과</p>

어떤 사람을 대표하게 만들 것인지가 실험 입력의 큰 부분이었다. 전연령을 균등하게 나누는 것과, 실제 서비스 타겟에 가까운 집단을 고르는 것은 다른 평가다. 같은 질문을 넣어도 페르소나 구성에 따라 점수 분포가 달라질 수 있었다.

## 높은 점수와 실제 CTR이 같은 방향으로 움직였나

검증에서 보고 싶었던 것은 실제 CTR과 시뮬레이션 점수의 상관관계였다. 당시에는 correlation 0.5 이상이면 활용을 검토할 수 있겠다는 기준을 뒀다. 이 값은 실험에서 정한 판단 기준이지 모든 광고 예측에 적용되는 공통 기준은 아니다.

![전연령 층화 표본의 시뮬레이션 점수와 실제 CTR](/assets/images/notion/965ecbf938973af9b474/17-ad-performance-experiments.png)

<p class="caption">전연령 층화 표본의 시뮬레이션 점수와 실제 CTR</p>

![20~39세 표본의 시뮬레이션 점수와 실제 CTR](/assets/images/notion/965ecbf938973af9b474/18-ad-performance-experiments.png)

<p class="caption">20~39세 표본의 시뮬레이션 점수와 실제 CTR</p>

![페르소나 표본 조건별 예측 점수 비교](/assets/images/notion/965ecbf938973af9b474/19-ad-performance-experiments.png)

<p class="caption">페르소나 표본 조건별 예측 점수 비교</p>

|  | 실험1 | 실험2 | 실험3 |
| --- | --- | --- | --- |
| 타겟연령 | 전연령 Stratified | 20-39 | 20-39 |
| 성별 | 성별 Stratified | 남성향은 남성 여성향은 여성 | 남성향은 남성 여성향은 여성 |
| 특징 | 교육, 직업, 지역등 Stratified하게 나눔 | 20~30대 남성, 여성 전부 | 20~30대 남성, 여성중 크랙서비스를 좋아할만한 Cluster선정 |
| Corr | -0.219 | -0.048 |  |

정리표에는 실험 1의 corr가 -0.219, 실험 2가 -0.048로 남아 있다. 이미지 중 두 번째 산점도에는 약 -0.079가 표시돼 있어 기록된 결과를 한 숫자로 통일하기 어렵다. 두 번째 산점도도 양의 상관을 보인 결과는 아니다. 서비스 친화 클러스터를 고르는 세 번째 실험은 표에 corr가 비어 있다.

20~30대로 좁힌 평가에서는 전반적으로 긍정적인 점수를 주는 편향도 있었다. 높은 점수를 받는다는 사실만으로 배너 사이의 실제 성과 차이를 잘 구분하는 것은 아니었다. 점수가 모두 비슷하게 높으면 집행 후보를 가르는 데 쓸 수 없다.

시간상 충분한 표본을 확보하지 못했고, 작은 표본의 결과만으로 예측력을 판단하기 어려웠다. 표본을 늘리면 양의 상관이 생길 가능성을 생각했지만, 현재 기록의 결과가 그것을 뒷받침하지는 않는다. 표본 확대는 검증을 이어갈 방법이지 좋은 결과를 보장하는 방법은 아니다.

## 생성한 카피와 예측한 점수를 다루는 방식

카피 유형과 밈은 실제로 내보내 비교할 수 있는 소재를 만들었다. 페르소나는 그 비교를 집행 전에 일부 대신해보려는 실험이었다. 두 작업 모두 제작 파이프라인에 연결할 수 있었지만, 연결 가능하다는 이유만으로 자동 필터로 사용하기에는 근거가 부족했다.

계속한다면 실제 집행 소재와 페르소나 평가가 정확히 대응하는지부터 맞추고, 어느 기간의 CTR을 비교할지와 표본 구성을 함께 정해야 한다. 그 후에 같은 조건에서 후보 순위를 얼마나 잘 맞히는지 확인할 수 있을 것이다.

당시 실험에서 남긴 것은 선호도 질문, 페르소나 그룹 구성, 점수와 실제 CTR을 비교하는 흐름이었다. 아직 돈을 쓰지 않고 광고 성과를 알 수 있게 된 것은 아니었다. 잘된 사례뿐 아니라, 기대했던 예측 신호가 나오지 않았던 기록도 남겨두려고 한다.
