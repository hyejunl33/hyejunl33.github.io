---
layout: modern-single
title: "한국에서 만든 광고 파이프라인을 북미·일본으로 확장하기"
date: 2026-10-12
tags:
  - Project
excerpt: "광고 소재 제작 파이프라인은 크랙에서 먼저 만들었다. 이후 북미 OOC와 일본 캬라푸로 확장했다. 이미 돌아가는 생성 흐름이 있으니 언어만 바꾸면 될 것 같지만, 실제로는 입력 데이터와 UI, 폰트, 디자인 레퍼런스가 달랐다."
project_group: wrtn-internal-agent-builder
notion_source_id: sha256:a64cb1cb140d8e54856c
---

광고 소재 제작 파이프라인은 크랙에서 먼저 만들었다. 이후 북미 OOC와 일본 캬라푸로 확장했다. 이미 돌아가는 생성 흐름이 있으니 언어만 바꾸면 될 것 같지만, 실제로는 입력 데이터와 UI, 폰트, 디자인 레퍼런스가 달랐다.

공통으로 쓸 수 있는 제작 단계는 유지하고, 서비스마다 다른 조건을 어디에서 분기할지 정하는 작업이었다.

## 입력 스키마부터 달랐다

크랙에서 쓰던 `ending`, `play_guide`, `taste_score`, 추천용 11축 태그가 OOC·캬라푸 테이블에는 없었다. 같은 필드를 당연히 읽는 코드로는 확장할 수 없었다. 없는 값은 입력에서 빼고, 각 서비스가 제공하는 세계관과 시작 상황을 중심으로 기획하도록 구성했다.

소구점과 카피 프롬프트도 언어별로 분기했다. 한국어의 장르 표현을 그대로 번역하기보다는, 해당 언어에서 실제 광고로 쓸 수 있는 문장이 필요했다. 생성 결과의 폴더와 파일명에도 서비스가 드러나게 해서, 마케터분들이 결과를 구분할 수 있도록 했다.

![크랙·OOC·캬라푸 서비스별 제작 분기](/assets/images/notion/a64cb1cb140d8e54856c/01-global-expansion.png)

<p class="caption">크랙·OOC·캬라푸 서비스별 제작 분기</p>

같은 스토리→소구점→소재 흐름을 유지하되, 플레이어블 UI는 각 서비스에 맞춰 바꿨다. 이미지 배너와 플레이어블, 모션 등 기본 제작 경로를 먼저 확장하고, 현지 레퍼런스에 맞추는 작업을 이어서 했다.

## 폰트와 디자인도 서비스의 일부였다

기존 폰트가 일본어를 지원하지 않아 폰트 라이브러리를 새로 구성했다. 카피를 일본어로 잘 만들더라도 글자가 표시되지 않으면 배너로 쓸 수 없었다. 언어 지원은 스타일 이전에 확인해야 하는 조건이었다.

![일본어를 지원하는 폰트 라이브러리](/assets/images/notion/a64cb1cb140d8e54856c/02-global-expansion.png)

<p class="caption">일본어를 지원하는 폰트 라이브러리</p>

OOC에서는 기존 배너에 폰트 색상과 이펙트를 추가하는 식으로 시작할 수 있었다. 캬라푸의 레퍼런스는 더 다양했다. 일본 마케터분들과 싱크를 맞추고, 어떤 표현을 생성 규칙으로 옮길지 정리했다.

![캬라푸에서 생성한 이미지 배너](/assets/images/notion/a64cb1cb140d8e54856c/03-global-expansion.png)

<p class="caption">캬라푸에서 생성한 이미지 배너</p>

![OOC에서 생성한 소재 예시](/assets/images/notion/a64cb1cb140d8e54856c/04-global-expansion.png)

<p class="caption">OOC에서 생성한 소재 예시</p>

현지인이 아닌 내가 어울리는 효과를 혼자 정하기 어려워서, 레퍼런스를 기반으로 역설계하는 방식을 택했다. Figma에 정리된 일본향 배너 약 300장을 장르별로 보고, 자주 쓰이는 효과와 분위기를 텍스트 템플릿으로 만들었다.

## 장르 템플릿만으로는 모든 그림이 비슷해졌다

장르별 `mood_template`에는 색상, 조명, 꽃잎·입자·후광 같은 효과 방향을 넣었다. 로맨스와 판타지, 코미디, 미스터리는 레퍼런스에서 사용하는 표현이 달랐다.

![효과 설계에 활용한 일본향 배너 레퍼런스](/assets/images/notion/a64cb1cb140d8e54856c/05-global-expansion.png)

<p class="caption">효과 설계에 활용한 일본향 배너 레퍼런스</p>

![레퍼런스에서 정리한 효과 구성 요소](/assets/images/notion/a64cb1cb140d8e54856c/06-global-expansion.png)

<p class="caption">레퍼런스에서 정리한 효과 구성 요소</p>

<table>
<tbody>
<tr><td><strong>romance-f</strong></td><td>sweet otome romance — warm pink/crimson/gold, soft dreamy soft-focus, layered 玉ボケbokeh, rose &amp; sakura 花吹雪 petals, キラキラ sparkle, a tender glowing aura, floating hearts</td></tr>
<tr><td><strong>romance-m</strong></td><td>warm school romance (male lead) — pink &amp; warm tones, bright rim light, soft bokeh, sakura petals, hearts and キラキラ, cheerful bright grade</td></tr>
<tr><td><strong>bl</strong></td><td>dramatic romance — cool blue/violet or moody warm gold, strong rim light &amp; dramatic glow, rose petals, halftone, refined bokeh, tasteful sparkle</td></tr>
<tr><td><strong>fantasy</strong></td><td>epic &amp; magical — luminous god-rays/light-pillars, arcane 魔法陣 magic circle &amp; glyphs, glowing aura, prism dispersion, aurora, floating orbs &amp; light particles</td></tr>
<tr><td><strong>humor</strong></td><td>bright poppy comedy (少女漫画 pop) — vivid CMYK pop colours, comic halftone burst, exclamation sparkles, hearts &amp; confetti, high-key cheerful</td></tr>
<tr><td><strong>mystery</strong></td><td>cool suspense / neon-noir — moody cyan/magenta, dramatic spotlight &amp; deep shadow, faint glitch &amp; chromatic aberration, cold sparkle, tense contrast</td></tr>
<tr><td><strong>underworld</strong></td><td>dark &amp; dangerous — grunge ink splatter, red/black high contrast, diagonal caution stripes, drifting embers &amp; smoke, gold glints, broken-heart accents, deep shadow</td></tr>
<tr><td><strong>daily</strong></td><td>soft wholesome slice-of-life — warm sunlight 玉ボケ, gentle pastel bokeh, sakura petals, small hearts and sparkles, dreamy soft-focus</td></tr>
<tr><td><strong>sf</strong></td><td>neon cyber — electric cyan/magenta, digital particles &amp; data motes, non-text HUD glints, 稲妻 lightning, lens flare, glowing grid lines</td></tr>
<tr><td><strong>period</strong></td><td>elegant traditional — gold accents, 桜 sakura &amp; 花吹雪, ink-wash (水墨) haze, seigaiha wave motifs, refined warm glow, stained-glass light</td></tr>
<tr><td><strong>sports</strong></td><td>dynamic &amp; energetic — bold speed lines &amp; 集中線, impact bursts, motion streaks, vivid punchy complementary contrast, sweat-drop energy</td></tr>
<tr><td>etc · (기본)</td><td>vivid &amp; eye-catching — bold mood grade, layered bokeh, sparkle-dust, confetti, dramatic rim light</td></tr>
</tbody>
</table>

그런데 같은 장르에 같은 템플릿만 넣으면 서로 다른 캐릭터에도 비슷한 효과가 붙었다. 그래서 이미지마다 따로 읽은 palette를 추가했다. 장르의 기본 분위기와 해당 캐릭터 그림의 색·표정·구도를 함께 쓰는 방식이다.

효과는 인물 주변과 뒤쪽에 배치하고 얼굴이나 몸을 가리지 않게 했다. 템플릿, 이미지별 효과 방향, 원본 이미지를 i2i 모델에 넣었다. 효과가 많아지는 것 자체보다 원본의 인물과 카피가 살아 있는지를 봐야 했다.

![코미디 장르의 이미지 효과 적용 예시](/assets/images/notion/a64cb1cb140d8e54856c/07-global-expansion.png)

<p class="caption">코미디 장르의 이미지 효과 적용 예시</p>

![판타지 장르의 이미지 효과 적용 예시](/assets/images/notion/a64cb1cb140d8e54856c/08-global-expansion.png)

<p class="caption">판타지 장르의 이미지 효과 적용 예시</p>

예시는 장르별 분위기를 반영한 산출물이다. 배너가 레퍼런스와 비슷한 형태가 됐다는 것과 실제 광고 성과가 좋다는 것은 별개라, 디자인 변화만으로 성과 개선을 주장하지는 않으려고 한다.

## 원본 썸네일을 규격에 맞추는 별도 경로

캬라푸에서는 앱 썸네일 원본 자체를 네 가지 광고 사이즈로 변형하는 경로도 만들었다. 여기는 카피를 새로 얹는 배너와 조건이 달랐다. 원본에 들어 있는 타이포를 유지하면서 크레딧과 로고만 붙여야 했다.

![앱 썸네일을 광고 규격으로 바꾸는 흐름](/assets/images/notion/a64cb1cb140d8e54856c/09-global-expansion.png)

<p class="caption">앱 썸네일을 광고 규격으로 바꾸는 흐름</p>

모델을 부르기 전에 생성 크기부터 계산했다. 두 변은 16의 배수, 최종 결과와 같은 비율, 면적 하한과 최종 크기 이상이라는 조건을 맞췄다. 원본과 목표 비율의 상대오차가 2% 이내면 생성 모델을 호출하지 않고 리사이즈만 했다. 불필요하게 다시 그리게 하면 모델이 원본 문구를 바꾸는 문제가 있었기 때문이다.

원본의 글자는 하나의 고정된 이미지 자산으로 취급했다. 글자 블록을 통째로 옮기거나 줄일 수는 있어도, 새로운 문구를 쓰거나 글자 자체를 바꾸지 않게 했다. 원본에 글자가 없으면 결과에도 없어야 한다는 조건도 넣었다.

안전 영역을 알려주는 것만으로는 잘림 문제가 충분히 줄지 않았다. 글자 블록이 차지할 수 있는 높이를 숫자로 명시한 뒤에야 축소하는 결과가 나왔다. 상단에는 규격별로 로고·크레딧 공간을 예약했다.

```yaml
AGENTS/KYARAPU_FX/MINIMAL_META.PY · REFRAME_PROMPT_TMPL (구조)

① 글자 불가침 선언 — 맨 앞에.
원본에 박힌 타이포는 "고정 자산"이며 새로 만들거나 바꾸지 않는다.

② ANATOMY 절.
가려진 신체를 예측해서 그리지 말 것. (손가락 4개/6개의 근원)

③ 상단 예약대 (reserved_top_frac).
9:16 = 11% · 1:1 = 18% · 4:5 = 14% — 크레딧·로고가 들어갈 자리.

④ 가로 SAFE AREA
가로 규격은 잘림 위험이 커서 안전 영역을 따로 선언한다.

⑤ 무자막 역방향 조항.
원본에 글자가 없으면 없는 채로 둔다 — 창작 금지.
```

전송 장애 재시도와 해부학 오류 재생성은 따로 처리했다. 손가락이나 팔다리 형태가 계속 잘못 나오면 해당 베리에이션을 버렸다. 이 경로에서는 앱 썸네일을 광고하는 것이므로, 실패했다고 전혀 다른 그림으로 교체할 수는 없었다.

## 현지화는 계속 피드백을 받는 작업이었다

데이터 분기와 기본 제작 기능을 구현한 뒤에도 현지 고도화는 남았다. 언어가 맞고 파일이 생성된다는 사실만으로 현지 광고의 톤이 맞는 것은 아니었다. 마케터분들이 보는 카피 방향과 디자이너분들의 레퍼런스를 받아서, 반복해서 적용할 수 있는 조건으로 바꿔야 했다.

공통 파이프라인을 유지하면 새 형식과 수정사항을 여러 서비스에 전달하기 쉽다. 반면 서비스의 차이를 공통화하려고 억지로 없애면 입력도 디자인도 어색해질 수 있다. 같은 흐름을 쓰되, 없는 필드와 언어별 문장, 폰트와 규격 제약은 분명하게 나누는 쪽으로 확장했다.
