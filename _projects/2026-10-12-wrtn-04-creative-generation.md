---
layout: modern-single
title: "하나의 스토리에서 이미지·영상·플레이어블 광고 만들기"
date: 2026-10-12
tags:
  - Project
excerpt: "스토리의 소구점과 이미지 자산을 정리한 뒤에는 표현할 수 있는 형식을 늘렸다. 정적 배너에 카피를 얹는 것부터 시작했지만, 같은 스토리를 짧은 모션 영상으로 보여줄 수도 있고, 읽히는 롱폼이나 직접 선택하는 플레이어블로 만들 수도 있었다."
project_group: wrtn-internal-agent-builder
notion_source_id: sha256:1bcbb67d0be297ea92ee
---

스토리의 소구점과 이미지 자산을 정리한 뒤에는 표현할 수 있는 형식을 늘렸다. 정적 배너에 카피를 얹는 것부터 시작했지만, 같은 스토리를 짧은 모션 영상으로 보여줄 수도 있고, 읽히는 롱폼이나 직접 선택하는 플레이어블로 만들 수도 있었다.

형식이 늘어날수록 모든 것을 생성 모델에 한 번에 맡기기는 어려웠다. 어떤 단계는 모델이 기획하고, 어떤 단계는 코드가 결정적으로 조립하며, 비용이 큰 생성 앞에는 사람이 확인하는 지점을 두는 식으로 경로를 나눴다.

## 공통 기획과 형식별 제작

공통 입력은 스토리 정보, `strategy_json`, 이미지 자산이었다. 이 중 훅 포인트와 타깃 감정, 소구점, CTA 방향은 형식이 달라져도 유지한다. 정적 배너의 입력은 아래처럼 구성했다.

<table>
<tbody>
<tr><td>strategy_json</td><td>소구점 N개 · 훅 포인트 · HyDE 비주얼 쿼리 · 이상적 폰트 샘플 문장.</td></tr>
<tr><td>image_assets</td><td>CLIP 통과 이미지 + 임베딩. 이미지 선택에 LLM은 쓰지 않는다 — Stage 2가 만든 HyDE 쿼리를 CLIP으로 임베딩해 이미지 벡터와 비교하고 가장 유사한 이미지를 고른다 (= CLIP top-1). 소구점마다 다른 그림이 뽑히도록 이미 쓴 이미지를 누적 제외한다.</td></tr>
<tr><td>폰트 카탈로그</td><td>장르·소구점 임베딩으로 고른다. </td></tr>
<tr><td>카피 폼</td><td>대사형 · 액션형 · 상황형 3폼. 언어별로 프롬프트 원본 파일이 따로 있고, 그 파일 하나가 유일한 기준이다.</td></tr>
</tbody>
</table>

```yaml
AGENTS/APPEAL_EMBEDDER.PY · STRATEGY_SYSTEM_INSTRUCTION (크랙 원문)
너는 모바일 앱 퍼포먼스 마케팅 전략가야. 크랙 앱의 캐릭터챗 콘텐츠를 타겟(여성향/남성향, 장르별 서브컬처)에 맞춰 광고 소재화하는 전문가야.

소구점은 스토리의 세계관·줄거리·장르 태그(11축)·유저 댓글을 종합해서 도출해 — 어느 한 신호에도 과의존하지 마. 특히 댓글은 '있으면 참고'하는 보조 신호일 뿐이야:
댓글이 없거나, 수가 적거나, 뻘글(무의미한 반응)만 있는 스토리가 많기 때문에, 댓글이 빈약하면 세계관·장르 컨벤션·캐릭터 설정·태그만으로 충분히 소구점을 도출하고 뻘글에 휘둘리지 마.

추상어·일반론 금지. 구체적인 캐릭터성·관계성·장면이 드러나야 해. CTR과 전환율을 높이는 것이 최우선 목표야.
```

```yaml
🚨주의: 철벽남 소꿉친구의 무서운 집착!🚨 평생 친구라며 선 긋던 그가 내가 다른 남자와 웃자 눈빛이 달라집니다. 밀당으로 그의 독점욕을 폭발시키세요.

#BL #소꿉친구 #크랙
```

이미지 배너에서는 검색으로 그림을 선택하고 카피와 폰트를 얹는다. 배너 렌더러가 규격별 결과물을 만드는 과정은 앞 글에서 정리했다. 모션이나 롱폼으로 확장할 때는 이 입력을 그대로 복사하는 게 아니라, 해당 형식에서 무엇을 보여줘야 할지 다시 계획하게 했다.

## 7초 모션: 먼저 그림에 무엇이 있는지 읽기

모션화는 배너 그림 한 장을 스토리 정보에 맞춰 움직이는 방식이었다. 처음부터 인물 동작을 지어내게 하기 전에, 비전 LLM이 `scene_read`로 이미지에 실제로 보이는 것을 적게 했다. 그 내용을 바탕으로 0~1.5초, 1.5~3.5초, 3.5~5초, 5~7초의 구간별 스토리보드를 만든다.

![7초 모션 영상 제작 흐름](/assets/images/notion/1bcbb67d0be297ea92ee/01-creative-generation.png)

<p class="caption">7초 모션 영상 제작 흐름</p>

구간에는 인물 동작뿐 아니라 신체 역학, 지속되는 움직임, 표정, 카메라 워크, 효과음이 들어갔다. 이 구조화된 결과를 i2v 프롬프트로 펼치는 단계는 LLM을 다시 호출하지 않고 코드로 처리했다. 기획에서 이미 정해진 구간을 재해석할 필요가 없어서였다.

```yaml
정규화 장르 (13슬러그): {슬러그}
캐릭터 속성: {tag_character_attribute}
대표 대사: {simple_description}
줄거리: {content_summary}
[소구점] 훅포인트 / 타깃 감정 / 어필 포인트 / CTA 방향
세계관(발췌, 1500字)
```

```yaml
scene_read: "Akari stands angled in a low-angle perspective against a blood moon in a red sky, wearing black lacquered armor plates over a dark outfit."
scene_concept: "Under the blood moon, Akari shifts her grip on her katana with a cautious, resolute gaze down toward the viewer."
audio_plan: "Howling high-altitude wind, distant rhythmic taiko drums, and the slide of a katana habaki locking inside its wooden scabbard."
physics_notes: "The long black hair strands and thin crimson ribbon whip in the crosswinds while the rigid black armor plates stay firmly anchored."
segments:
  - time_range: "0-1.5s"
    character_action: "Akari pivots her shoulders slightly toward the camera, thumb clicking the katana guard forward out of its scabbard by an inch."
    body_mechanics: "Starting in a three-quarters back stance, her core rotates 45 degrees toward the viewer while her right thumb levers the gold tsuba forward and her other hand steadies the scabbard."
```

```plaintext
[0-1.5s] {동작}; body mechanics: {신체 역학}; moving continuously through the entire segment; expression: {표정}; camera: {카메라}; sfx: {효과음}.
[1.5-3.5s] ...
[3.5-5s] ...
[5-7s] ...

Global camera direction: {전역 카메라 지시}
One continuous take. Keep the character moving throughout the shot. No on-screen text, captions, subtitles, or watermarks.
```

영상 입력에는 오버레이 제거본을 썼다. 예전에는 21:9로 확장한 이미지를 넣었는데, 세로 영상으로 cover-crop되면서 캐릭터가 잘려서 그 경로를 폐기했다. 9:16 영상은 한 번만 만들고 가로·정사각은 좌우를 채워 파생했다.

정책 차단이 발생하면 같은 프롬프트를 계속 고치기보다 다른 후보 이미지로 최대 네 장까지 시도했다. 오디오를 켰을 때 의도하지 않은 사람 음성이 나오면 발화 가드로 처리했다. 생성 모델의 결과가 요청한 조건과 완전히 같다고 가정할 수 없어서 후처리가 필요했다.

## 롱폼: 대본을 HTML로 조판하고 녹화하기

롱폼은 스토리를 읽히는 형식으로 재구성해 HTML 페이지를 만들고, 스크롤되는 화면을 녹화했다. 대본 모델에 세계관과 인물 정보를 주고 참고 이미지 여러 장도 함께 넣었다. 이미지는 중립적인 파일명으로 표시하고, 어떤 장면에 어떤 그림을 쓸지는 대본 안의 마커로 선택하게 했다.

![롱폼 대본 생성과 조판·녹화 흐름](/assets/images/notion/1bcbb67d0be297ea92ee/02-creative-generation.png)

<p class="caption">롱폼 대본 생성과 조판·녹화 흐름</p>

```yaml
[S] 장면 블록 - 상황 서술
[U] 유저(시청자) 발화 블록
N  나레이션    D  대사    H  하이라이트    P  사진 배치

예: P img_3.jpg  ← 참고 이미지를 그 장면에 배치

· 첫 [S] 블록 = 후킹 정점 장면이고, 여기엔 P 이미지를 반드시 1장 넣는다
· 참고 이미지가 0장이면 P 마커를 절대 쓰지 않는다
· 라이브 채팅용 출력 포맷(**이름**|내용, 상태창)은 쓰지 않는다
```

첫 블록에는 후킹할 정점 장면과 이미지 한 장, 짧은 문장을 두었다. 릴스에서 시작하자마자 스크롤이 올라가는데 글이 길면 읽히지 않았다. 첫 블록 위치도 0.8에서 시작했을 때는 초반이 비었고, 0.0에서는 이미지와 글이 바로 잘려서 0.5로 정착했다.

대본에 '유저'라는 호칭이 남는 것도 어색했다. 첫 줄에 이름 메타 정보를 선언하게 하고, 남은 호칭은 후처리로 치환했다. 금칙어는 해당 줄만 다시 수선하고, 출력 언어가 다른 언어로 흐르면 실패로 처리했다.

조판과 녹화는 1080×1920에서 한 번 하고 다른 규격은 scale과 pad로 만들었다. 규격 하나가 실패했다고 모두 버리지 않고, 저장 가능한 결과는 남겼다. 긴 영상은 상한을 넘으면 트림하도록 했다.

## 플레이어블: 시나리오를 실행 가능한 조합으로

플레이어블은 클릭할 수 있는 미니게임 형태의 광고다. 스토리를 자체완결 HTML 하나로 만들기 위해, 스토리보드와 실제 시나리오 조합을 분리했다.

![플레이어블의 기획·검증·HTML 조립 흐름](/assets/images/notion/1bcbb67d0be297ea92ee/03-creative-generation.png)

<p class="caption">플레이어블의 기획·검증·HTML 조립 흐름</p>

대표 이미지는 CLIP 상위 후보 다섯 장을 비전 LLM에 보여주고 하나를 고르게 했다. 캐릭터가 선명한지, 장르와 무드가 맞는지 보게 했고, 파싱 실패나 범위를 벗어난 선택에는 CLIP 1위로 돌아가는 대체 경로를 뒀다.

Stage A에서 장면 흐름과 연출 방향을 만든 뒤 Stage B에서 조합 DSL로 바꾼다. 검증에 걸리면 같은 세션에서 멀티턴으로 수정했다. 채팅·선택·게이지·공개·CTA 등 준비된 요소를 조합하는 방식이라, 결과를 검증하고 렌더러에 넘길 수 있었다.

```yaml
{ theme, gauges[], steps[], branches{} }

steps 프리미티브: chat · choice · burst · cta                    (v1)
card_pick · gauge · stat_panel · reveal                    (확장)
scratch · qte_tap · swipepick

게이지 스타일: danger · heart · qi · bar · suspicion
화면 형태(layout): chat · battle · raise · harem · investigate
immersion · scratch · swipe
```

최종 HTML에는 레이아웃과 게이지, 이펙트, 서비스 스킨을 함께 넣었다. 외부 자산 요청이 없어도 실행되게 만드는 것은 매체에 전달할 때의 조건이었다.

## UGC 리메이크: 훅을 만드는 방법이 네 번 바뀌었다

OOC에서는 이미 잘된 UGC의 대본과 화면 구성을 유지하면서, 화면 속 인물을 AI로 바꾸는 경로도 만들었다. 원본 영상의 단어별 타임스탬프를 뽑고 컷을 나눈 다음, 훅과 발화 컷을 별도로 생성했다.

![UGC 리메이크의 생성과 합성 흐름](/assets/images/notion/1bcbb67d0be297ea92ee/04-creative-generation.png)

<p class="caption">UGC 리메이크의 생성과 합성 흐름</p>

컷 경계는 문장 경계로 맞췄다. 훅이 첫 문장의 앞토막만 말하면 이후 편집이 잘돼도 어색하기 때문이다. 훅은 원본 클립을 움직임과 구도의 레퍼런스로, 승인된 얼굴 이미지를 인물 레퍼런스로 넣는 r2v 방식으로 만들었다. 이 구간의 소리는 원본을 다시 합성했다. 발화 컷은 앵커와 대사를 넣어 음성이 포함된 영상으로 만들고, 전사 결과를 맞춰 앞뒤를 잘랐다.

인물 매팅 이후에는 얼굴 기준 크롭, 우하단 PIP, 자막, 엔딩 로고를 합성했다. 얼굴 갤러리는 인종·성별별 앵커를 미리 만들어 사람이 승인한 것만 사용했다. 공식 틱톡 계정의 리포스트는 리메이크 대기 큐로 활용했다.

훅 생성 방식은 완전 재현, 프레임 스왑+i2v, 스토리보드 기반 생성, r2v 단일 호출까지 바뀌었다. 실배치에서는 발화가 사라지거나, DB 락이 생기거나, 매팅 결과가 전부 실패하는 문제가 드러났다. 단위테스트가 통과해도 이 형식을 쓸 수 있다고 말하기 어려웠던 이유다. 실제 실패 사례를 확인한 뒤 그 조건이 다음 가드가 됐다.

## Anime teaser: 비싼 생성 전에 플랜을 확인했다

오리지널 스토리에서는 30초 애니메이션 티저를 만드는 흐름도 구현했다. 감독 단계에서 세계관과 소구점, 레퍼런스 이미지를 받아 컷과 샷, 캐스트, 대사를 계획한다. 후보 이미지 풀에서 선택한 이미지를 캐릭터·배경·소품으로 나눠 컷에 연결했다.

![애니메이션 티저의 플랜·생성·합성 흐름](/assets/images/notion/1bcbb67d0be297ea92ee/05-creative-generation.png)

<p class="caption">애니메이션 티저의 플랜·생성·합성 흐름</p>

![애니메이션 티저 생성 화면과 컷 계획](/assets/images/notion/1bcbb67d0be297ea92ee/06-creative-generation.png)

<p class="caption">애니메이션 티저 생성 화면과 컷 계획</p>

사람이 플랜을 확인한 다음 본편 영상 모델을 호출한다. 컷별 영상을 이어 붙이는 대신 컷 플랜과 캐스트를 넣어 본편을 한 번에 생성하는 경로였다. 실제 발화를 다시 전사해 자막 타이밍을 맞추고 서비스별 엔딩과 BGM을 합성했다.

```yaml
shot_size: "medium"
beat: "Chloe scrambles on the floor after spilling game pieces."
motion: "Chloe hurriedly gathers scattered colorful plastic tokens off the floor, rushing back up toward the desk."
action: "Chloe crouches low to scoop up dice from the linoleum floor, then moves back behind the wooden desk."
scene: "Chloe crouches beside the cluster of desks in the sunset-lit classroom."
line:
  speaker: "Chloe"
  from_source: true
  voice: "flustered and breathless, catching her voice in a panicked stammer"
  line: "Aah! S-sorry! My hand slipped! B-but... I will have everything picked up right away!"
advances: "The spilled pieces are collected and placed back onto the playing surface on the desk."
```

|  | 크랙 | OOC |
| --- | --- | --- |
| 대상 | IP 등록 오리지널만 | 공식계정 제작 레인 |
| 프롬프트 | 한국어 별도 벌(번역 아님) | 영어 벌 |
| 인트로 | 빨강 워드마크 + ORIGINAL 배지 | 로고 2.0초 |
| 엔드카드 | 없음 | 섬네일 고정 2.5초 |
| 아웃트로 | 세로 로고 애니 4.0초 | 로고 3.5초 |
| 매체 | 메타 · 틱톡 · X | 틱톡 · X |

한국어와 영어는 발화량 기준도 달랐다. 영어 플랜을 한국어 기준으로 검사하면 발화 과다로 잘못 판단하는 경우가 생겼다. 원작에서 인용할 수 있는 대사가 있는 경우와 없는 경우도 나눴다. 소구점이 없으면 티저 생성은 실패로 처리했다. 무엇을 광고하는지 정해지지 않은 상태에서 영상부터 만드는 건 피하고 싶었다.

자막이나 BGM, 꼬리 부분만 잘못되면 저장된 중간 영상부터 합성 단계를 다시 실행했다. 본편 생성까지 다시 호출할 필요가 없게 만드는 것이 비용 면에서도 중요했다.

형식마다 모델과 검수 방식은 달랐지만, 입력과 산출물의 계약을 정하고 수정 가능한 중간본을 남기는 방향은 같았다. 생성 결과가 한 번에 잘 나오기를 기다리는 것보다, 어느 단계가 잘못됐는지 확인하고 그 부분만 다시 처리할 수 있어야 운영할 수 있었다.
