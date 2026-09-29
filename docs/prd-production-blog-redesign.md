# PRD — Archive for AI Study 운영 블로그 개편

## 0. 문서 정보

| 항목 | 내용 |
| --- | --- |
| 문서 목적 | 승인된 데모를 실제 `hyejunl33.github.io` 운영 블로그에 적용하기 위한 제품·디자인·기술 요구사항 정의 |
| 대상 저장소 | `github.com/hyejunl33/hyejunl33.github.io` |
| 배포 대상 | `https://hyejunl33.github.io` |
| 기준 브랜치 | `master` |
| 구현 기준 | 현재 Jekyll 콘텐츠와 URL을 보존한 점진적 UI 교체 |
| 선행 결과물 | `demo/index.html`, `demo/archive.html`, `demo/article.html` |
| 문서 상태 | 구현 착수 가능 |

## 1. 제품 정의

### 1.1 한 줄 정의

**Archive for AI Study는 AI 학습 기록, 알고리즘 풀이, 프로젝트 회고와 경력을 한곳에서 탐색할 수 있는 인터랙티브 기술 블로그이자 개인 포트폴리오다.**

### 1.2 핵심 콘셉트

> 차분한 연구 아카이브 위에, 커서에 반응하는 하나의 궤도와 빛을 둔 포트폴리오.

승인된 데모의 두 요소를 운영 사이트의 디자인 언어로 확정한다.

- **Orbital Archive:** 홈 Hero의 원형 궤도와 부드러운 포인터 반응
- **Aurora Index:** 대표 프로젝트 카드의 제한된 홀로그램/오로라 표면
- **Content First:** 글 목록과 본문에서는 효과를 줄이고 타이포그래피·여백·탐색성을 우선

### 1.3 레퍼런스 원칙

- OpenAI 웹사이트에서 넓은 여백, 강한 제목 위계, 절제된 메뉴와 콘텐츠 중심 섹션 구성을 참고한다.
- [SEED React](https://seed-design.io/react)의 composition, responsive design, interaction states처럼 반복 UI를 일관된 토큰과 조합 가능한 구성 요소로 만든다.
- [Toss Apps React Native 문서](https://developers-apps-in-toss.toss.im/documentation/react-native)의 화면 단위 정보 위계와 예측 가능한 내비게이션 흐름을 참고한다.
- [Pokémon Cards CSS Holographic Effect](https://poke-holo.simey.me/)의 pointer-driven gradient, blend mode, CSS 3D transform을 대표 카드에만 제한적으로 적용한다.
- 레퍼런스의 시각물을 복제하지 않는다. 현재 데모의 색상·궤도·카드 표현을 사이트 고유의 디자인으로 발전시킨다.

## 2. 문제 정의와 목표

### 2.1 현재 문제

1. 기존 Academic Pages/Minimal Mistakes 기반 UI가 작성자의 AI 엔지니어링 정체성을 충분히 전달하지 못한다.
2. 홈이 포트폴리오보다 테마 기본 페이지에 가깝고, 대표 프로젝트와 최신 기록이 명확히 드러나지 않는다.
3. 컬렉션별 목록과 글 본문의 정보 위계가 약하고, 긴 기술 문서·코드·표의 가독성이 부족하다.
4. 컬렉션은 분리되어 있지만 한 화면에서 전체 학습 기록을 탐색하기 어렵다.
5. `_data/cv.json`은 템플릿 샘플 데이터가 남아 있고, 실제 경력은 `_pages/cv.md`에 관리된다.

### 2.2 제품 목표

1. 첫 화면 10초 안에 “AI 학습 아카이브 + 프로젝트 포트폴리오”임을 전달한다.
2. 기존 Markdown 원본 71개와 CV, 이미지, URL을 보존한다.
3. `Project`, `Study`, `Algorithm`, `WeeklyReview`, `CV`, `ETC`를 전 화면에서 일관되게 탐색한다.
4. 코드·표·수식·이미지가 많은 기술 글을 모바일과 데스크톱에서 편안하게 읽게 한다.
5. 인터랙션을 추가하되 정적 사이트의 속도, SEO, 접근성을 유지한다.
6. 채용담당자가 MCP 지원 LLM에서 프로젝트·협업·문제 해결의 근거를 원문 URL과 함께 조회할 수 있게 한다.

### 2.3 비목표

- CMS, 데이터베이스, 로그인, 댓글 백엔드 개발(공개·읽기 전용 Recruiter MCP Worker는 예외)
- Three.js/R3F 또는 GLB 모델을 사용하는 전체 화면 3D 경험
- 기존 글의 문장 교정·내용 재작성
- 컬렉션 URL 변경 또는 Markdown을 다른 포맷으로 일괄 변환
- 이번 단계에서 Astro/Next.js로 프레임워크 마이그레이션

## 3. 현행 시스템과 기술 결정

### 3.1 콘텐츠 현황

| 콘텐츠 | 원본 | 현재 수량 | 운영 역할 |
| --- | --- | ---: | --- |
| Project | `_projects/*.md` | 17 | 프로젝트 과정·실험·결과 |
| Algorithm | `_algorithm/*.md` | 9 | 알고리즘 풀이 |
| WeeklyReview | `_weeklyreview/*.md` | 10 | 주차별 학습 및 프로젝트 회고 |
| ETC | `_etc/*.md` | 5 | 커리어·생각·개인 기록 |
| Study | `_study/*.md` | 30 | AI 이론·논문·구현 학습 기록이자 전체 기록 허브 |
| CV | `_pages/cv.md` | 1 | 실제 경력·교육 정보의 단일 원천 |

### 3.2 프레임워크 결정: Jekyll 유지

운영 개편은 **Jekyll 4.3.x를 유지하고 테마의 표현 계층만 교체**한다.

근거:

- 컬렉션과 permalink가 이미 운영되고 있어 콘텐츠 이전이 필요 없다.
- GitHub Actions의 Jekyll 빌드·Pages 배포가 정상 구성되어 있다.
- Kramdown, Rouge, MathJax, sitemap/feed/redirect 플러그인을 재사용할 수 있다.
- 승인된 모션은 CSS와 작은 vanilla JavaScript로 충분히 구현 가능하다.
- 프레임워크 마이그레이션 없이 기존 URL과 SEO 위험을 최소화한다.

### 3.3 유지할 계약

- 컬렉션 permalink 형식 `/:collection/:path/`를 변경하지 않는다.
- 기존 frontmatter 키를 삭제하거나 일괄 변경하지 않는다.
- `/projects/`, `/study/`, `/algorithm/`, `/weeklyreview/`, `/cv/`, `/etc/`를 유지한다.
- `master` push → GitHub Actions → GitHub Pages 배포 흐름을 유지한다.
- `jekyll-feed`, `jekyll-sitemap`, `jekyll-redirect-from`, `jemoji`, `jekyll-gist`를 유지한다.

## 4. 정보 구조와 URL

### 4.1 전역 메뉴

좌측 브랜드는 모든 화면에서 정확히 **Archive for AI Study**로 표시한다.

메뉴 순서:

1. Project → `/projects/`
2. Study → `/study/`
3. Algorithm → `/algorithm/`
4. WeeklyReview → `/weeklyreview/`
5. CV → `/cv/`
6. ETC → `/etc/`

데스크톱에서는 한 줄 메뉴, 모바일에서는 가로 스크롤 메뉴 또는 접근 가능한 메뉴 버튼을 사용한다. 항목을 숨겨 탐색을 막지 않는다.

### 4.2 페이지 구조

```text
/
├── Hero + Orbital Field
├── Featured Project + Aurora Card
├── Recent Notes
├── Archive Categories
└── Footer

/{collection}/
├── Collection Header
├── Category Navigation
├── Sort/Filter (필요한 최소 범위)
└── Post Cards

/{collection}/{slug}/
├── Breadcrumb
├── Article Header
├── Sticky TOC
├── Markdown Body
├── Taxonomy/Share
└── Previous/Next

/cv/
├── Career
├── Education
├── Activities/Skills (데이터가 있을 때)
└── PDF/외부 링크 (실제 파일이 존재할 때만)
```

### 4.3 Study 정의

`/study/`는 Study 원본 30개와 다음 컬렉션을 최신순으로 합치는 **통합 학습 허브**로 구현한다.

- `projects`
- `algorithm`
- `weeklyreview`
- `etc`

새 `_study/*.md`도 자동으로 같은 목록에 포함한다. 중복 페이지 `_pages/study.md`와 루트 `study.md` 중 실제 라우트 소유자를 하나로 통합해 빌드 충돌을 제거한다.

## 5. 화면별 요구사항

### 5.1 홈 `/`

#### Header

- 브랜드와 여섯 메뉴를 제공한다.
- 74px 데스크톱, 64px 모바일 높이를 기준으로 한다.
- 스크롤 시 반투명 surface와 blur를 유지하되 본문 대비를 해치지 않는다.
- 현재 메뉴에는 시각적 상태와 `aria-current="page"`를 함께 제공한다.

#### Hero

- Eyebrow: `Archive for AI Study`
- 제목은 AI 학습·응용 정체성을 전달하는 2~3줄 문장으로 구성한다.
- CTA: `Read the archive`, `View projects`
- 우측에 중심 원과 2~3개의 궤도를 표시한다.
- 포인터에 따라 중심 원 최대 8px, 궤도 최대 4도만 반응한다.
- 모바일·coarse pointer에서는 정적인 원으로 표시한다.

#### Featured Project

- 최신 또는 frontmatter `featured: true`인 프로젝트 1건을 표시한다.
- `featured: true`가 없으면 `site.projects | sort: 'date' | reverse | first`를 사용한다.
- Aurora 효과는 이 카드에만 적용한다.
- 제목·요약·태그·날짜·상세 링크는 효과 없이도 항상 읽혀야 한다.

#### Recent Notes

- 전체 컬렉션을 합쳐 최신 3건을 표시한다.
- 각 카드에는 컬렉션, 제목, 날짜를 필수 표시한다.
- excerpt가 있으면 두 줄 이내로 표시하고, 없으면 본문에서 자동 생성한다.

#### Archive Categories

- Project, Algorithm, WeeklyReview, ETC의 실제 글 수를 Liquid로 계산한다.
- 숫자를 하드코딩하지 않는다.
- Study와 CV는 상단 메뉴에서 접근하되, 카테고리 영역 포함 여부는 구현 단계에서 화면 밀도로 결정한다.

### 5.2 컬렉션 목록

- 동일한 `archive` 레이아웃을 Project/Algorithm/WeeklyReview/ETC에 재사용한다.
- 카드 데이터: collection label, date, title, excerpt, tags.
- 최신순 정렬을 기본으로 한다.
- 카드 전체를 클릭할 수 있게 하되 내부 링크 중첩은 피한다.
- 긴 한국어 제목은 줄임표보다 자연스러운 2~3줄 wrapping을 우선한다.
- 빈 컬렉션은 빈 화면 대신 설명과 다른 컬렉션 CTA를 제공한다.

### 5.3 글 상세

#### 상단

- Home / Collection / Article breadcrumb
- collection badge
- 제목, excerpt, 게시일, 예상 읽기 시간
- H1은 페이지당 하나만 사용한다.

#### 목차

- Markdown H2/H3에서 생성한다.
- 데스크톱에서는 본문 왼쪽 sticky TOC, 모바일에서는 본문 위 가로 스크롤 TOC로 표시한다.
- 현재 읽는 섹션 표시를 선택적으로 지원하되 JavaScript 실패 시 기본 anchor 목록은 동작해야 한다.

#### 본문

- 읽기 폭 680~760px, 본문 16~18px, line-height 1.7~1.85.
- 문단, 제목, 목록, 이미지, 표, 인용문, callout, 수식 사이의 수직 간격을 토큰화한다.
- 한국어는 `word-break: keep-all`을 기본으로 하고 긴 URL·코드는 안전하게 줄바꿈한다.
- 이미지에는 `max-width: 100%`, 명시적 크기/비율 또는 안정적인 placeholder를 적용한다.
- 이미지 아래의 단독 문단은 caption으로 오인하지 않는다. caption이 필요하면 명시적 markup/class를 사용한다.

#### 읽기 진행 표시

- 헤더 아래 2px 진행 바로 제공한다.
- passive scroll listener와 `requestAnimationFrame`을 사용하거나 CSS scroll timeline 지원 시 progressive enhancement로 구현한다.
- JavaScript가 없어도 읽기 기능에는 영향이 없어야 한다.

### 5.4 CV

- 실제 콘텐츠 원천은 현재 `_pages/cv.md`로 정한다.
- `_data/cv.json`의 샘플 이름·이메일·학교 정보는 운영 UI에 노출하지 않는다.
- JSON 기반 CV를 사용할 경우 먼저 실제 정보로 교체한 뒤 전환한다.
- 존재하지 않는 `/files/cv.pdf` 다운로드 버튼은 표시하지 않는다.
- 저장소의 실제 PDF를 노출할 경우 파일명과 공개 범위를 확인한 뒤 명시적으로 연결한다.

### 5.5 404

- 전역 header/footer를 유지한다.
- “Archive로 돌아가기”, “최근 글 보기” CTA를 제공한다.
- 장식은 정적인 작은 orbital motif만 사용한다.

### 5.6 Recruiter Mode `/recruiter/`

목적은 LLM이 지원자를 대신 평가하게 만드는 것이 아니라, 블로그에 실제로 작성된 근거를 빠르게 찾고 원문으로 검증하게 하는 것이다.

- 홈 하단에 `Recruiter mode · MCP ready` 진입 카드를 둔다.
- 페이지 상단에는 MCP endpoint 상태, 복사 버튼과 데이터 공개 원칙을 표시한다.
- MCP 미연결 상태에서도 같은 데이터셋을 검색하는 `Evidence explorer`를 제공한다.
- 추천 질문은 multi-agent 협업, MLOps 파이프라인, 모델 최적화, 팀 협업을 기본으로 제공하되 검색어는 자유 입력 가능하다.
- `llms.txt`, 구조화 JSON, MCP 서버 소스 링크를 공개한다.
- LLM 답변은 반환된 원문 URL을 근거로 사용하고, 사실과 추론을 구분하도록 서버 instruction과 prompt에 명시한다.
- 숫자형 채용 점수, 합격/불합격 자동 판정, 비공개 개인정보 추론은 제공하지 않는다.

## 6. 디자인 시스템

### 6.1 원칙

- 시각 효과보다 콘텐츠 위계를 우선한다.
- 같은 역할의 요소는 같은 토큰, 간격, 상태를 사용한다.
- hover만으로 의미를 전달하지 않는다.
- 한 화면의 강조 색상과 고강도 모션을 제한한다.
- 컴포넌트는 Liquid include와 BEM에 가까운 명시적 class로 조합한다.

### 6.2 컬러 토큰

| 토큰 | Light 역할 | Dark 역할 |
| --- | --- | --- |
| `--color-bg` | warm off-white | blue-green black |
| `--color-surface` | white | elevated dark green |
| `--color-text` | dark charcoal | near white |
| `--color-text-muted` | gray green | desaturated light green |
| `--color-border` | 14% text alpha | 16% light alpha |
| `--color-mint` | orbital highlight | orbital highlight |
| `--color-violet` | aurora secondary | aurora secondary |
| `--color-peach` | aurora accent | aurora accent |

구체적인 색 값은 데모를 출발점으로 하되 WCAG 대비 검사를 통과하도록 조정한다.

### 6.3 타이포그래피

- 시스템 sans-serif 우선. 외부 폰트를 사용한다면 로컬 호스팅·subset·`font-display: swap`을 적용한다.
- Display H1: `clamp()` 기반 48~108px, line-height 0.9~1.0.
- Article H1: `clamp()` 기반 42~82px. 긴 한국어 제목이 4줄 이상 되면 상한을 낮춘다.
- H2/H3: 본문과 명확한 간격·크기 차이를 둔다.
- 본문: 16~18px, 1.7~1.85 line-height.
- Meta/label: 11~13px. 필수 정보는 12px 미만으로 만들지 않는다.

### 6.4 간격·모서리·모션

- 4px 기반 spacing scale을 정의한다: 4, 8, 12, 16, 24, 32, 48, 72, 96, 128.
- 일반 카드 radius 12~18px, featured 카드 24~30px.
- hover/press 전환 150~250ms.
- 콘텐츠 reveal은 최초 1회, 12~20px 이내로 제한한다.
- `prefers-reduced-motion: reduce`에서는 reveal, tilt, 궤도 자동 움직임을 제거한다.

## 7. Markdown 렌더링 요구사항

### 7.1 기본 파이프라인

- 기존 Kramdown GFM 설정을 유지한다.
- 코드 하이라이트는 Jekyll의 Rouge를 단일 원천으로 사용한다.
- fenced code block에 언어가 없으면 기본 Python 스타일을 적용하되, 가능하면 기존 글에 언어 표기를 점진적으로 추가한다.
- 코드 하이라이트를 위한 무거운 client-side 라이브러리는 추가하지 않는다.

### 7.2 코드 블록

- 배경·본문·주석·키워드·함수·문자열·숫자에 구분 가능한 색을 제공한다.
- 언어 라벨을 상단에 표시한다. 언어 정보가 없으면 `Python`을 기본값으로 표시한다.
- 가로 스크롤을 허용하고 페이지 전체 폭은 늘리지 않는다.
- 최소 13px monospace, line-height 1.6 이상.
- focus 가능한 스크롤 영역 또는 keyboard 접근 가능한 복사 버튼을 선택적으로 제공한다.
- 줄 번호는 기본 비활성화한다. 긴 디버깅 글에서 frontmatter로 켤 수 있게 확장 가능하다.

### 7.3 표

- `table`을 `.table-wrapper`로 자동 감싸거나 Kramdown 출력 주변에 CSS `overflow-x: auto`를 적용한다.
- `thead`, `tbody`, row divider, cell padding을 명확히 구분한다.
- 모바일에서 열을 억지로 축소하지 않고 표 컨테이너만 가로 스크롤한다.
- 표만으로 의미를 전달하지 않도록 문맥 설명을 본문에 둔다.

### 7.4 기타 Markdown 요소

- Blockquote: 본문보다 강한 왼쪽 표시와 중립 surface.
- Inline code: 주변 문장과 구분되지만 과한 채도는 피한다.
- MathJax: overflow 처리와 모바일 수식 스크롤 제공.
- Mermaid/Plotly: 기존 지원을 유지하되 해당 스크립트는 필요한 페이지에서만 로드한다.
- Footnote, heading anchor, task list, horizontal rule의 focus/hover 상태를 정의한다.

## 8. 콘텐츠·데이터 계약

### 8.1 필수 frontmatter

신규 글은 다음 키를 권장한다.

```yaml
---
title: "글 제목"
date: 2026-01-30
excerpt: "목록과 SEO에 사용할 한두 문장 요약"
tags:
  - tag
toc: true
featured: false
---
```

- 기존 글에 키가 없더라도 빌드가 실패하면 안 된다.
- `layout`은 collection defaults가 제공하므로 신규 글에서 생략 가능하게 한다.
- `categories`와 `tags`의 중첩 배열 등 기존 변형을 허용하고 Liquid에서 방어적으로 처리한다.

### 8.2 실제 콘텐츠 보존

- `_projects`, `_algorithm`, `_weeklyreview`, `_etc`의 Markdown 본문은 UI 작업 중 수정하지 않는다.
- `/assets/images/**` 경로를 유지한다.
- 깨진 이미지·잘못된 Markdown은 별도 콘텐츠 정리 PR로 분리한다.
- 날짜, slug, permalink를 바꾸는 경우 반드시 `redirect_from`을 추가한다.

## 9. 구현 아키텍처

### 9.1 권장 파일 구조

```text
_layouts/
  home.html                 # 운영 홈
  archive-modern.html       # 컬렉션 목록
  single-modern.html        # 글 상세

_includes/
  site-header-modern.html
  orbital-field.html
  featured-project.html
  post-card-modern.html
  article-header.html
  article-toc.html
  site-footer-modern.html

_sass/
  modern/
    _tokens.scss
    _base.scss
    _header.scss
    _home.scss
    _archive.scss
    _article.scss
    _markdown.scss
    _motion.scss

assets/js/
  modern-ui.js              # theme, orbit, aurora, progress, TOC enhancement

docs/
  prd-production-blog-redesign.md
```

기존 파일을 한 번에 삭제하지 않는다. 새 레이아웃을 추가하고 페이지/collection defaults를 순차적으로 전환한 뒤, 사용되지 않는 구형 스타일을 별도 정리한다.

### 9.2 Liquid 데이터 처리

- 홈 최신 글은 `site.posts`, `site.projects`, `site.study`, `site.algorithm`, `site.weeklyreview`, `site.etc`를 합쳐 정렬한다.
- Jekyll Liquid가 빈 배열을 안전하게 처리하도록 `default: empty` 또는 조건문을 사용한다.
- 컬렉션 label/URL 매핑은 `_data/collections.yml` 같은 단일 데이터 파일로 둔다.
- 카드 include에 page 객체와 variant만 전달한다. 텍스트·글 수를 하드코딩하지 않는다.

### 9.3 JavaScript 원칙

- 모든 핵심 콘텐츠와 링크는 서버 생성 HTML에 존재해야 한다.
- JS는 theme, visual motion, progress, active TOC에만 사용한다.
- pointer 이벤트는 `requestAnimationFrame`으로 조절하고 passive listener를 사용한다.
- localStorage 접근은 예외 처리한다.
- JS 오류가 발생해도 메뉴·글 목록·글 본문은 완전히 사용할 수 있어야 한다.

### 9.4 데모 처리

- `demo/`는 디자인 기준 자료이며 운영 콘텐츠의 source of truth가 아니다.
- 운영 전환이 끝나면 다음 중 하나를 택한다.
  - `_config.yml` `exclude`에 `demo`를 추가해 Pages 배포에서 제외
  - 별도 디자인 브랜치/문서로 이동 후 운영 브랜치에서 제거
- 운영 페이지에서 demo의 하드코딩 글 수·제목·본문 데이터를 사용하지 않는다.

### 9.5 Recruiter MCP와 공개 데이터 파이프라인

GitHub Pages는 서버 실행이 불가능하므로 다음처럼 정적 사이트와 원격 MCP를 분리한다.

1. `scripts/build-recruiter-data.mjs`가 CV와 다섯 컬렉션의 frontmatter/본문을 읽는다.
2. 빌드마다 `assets/data/recruiter-portfolio.json`, `llms.txt`, `llms-full.txt`를 생성한다.
3. GitHub Pages는 생성된 데이터와 `/recruiter/` UI를 정적으로 제공한다.
4. `mcp-server/`의 Cloudflare Worker는 공개 JSON을 읽어 Streamable HTTP `/mcp` endpoint로 제공한다.
5. MCP는 stateless, public, read-only로 운영하며 recruiter query를 저장하지 않는다.

| 종류 | 이름 | 역할 |
| --- | --- | --- |
| Tool | `get_candidate_snapshot` | 공개 CV, 콘텐츠 수, 주요 태그, canonical link 반환 |
| Tool | `search_portfolio` | 제목·태그·본문 근거 검색 및 원문 URL 반환 |
| Tool | `get_project_case_study` | 가장 가까운 프로젝트 기록의 긴 authored excerpt 반환 |
| Tool | `get_role_evidence` | 직무 요건과 관련된 작성 근거를 모으되 평가 점수는 만들지 않음 |
| Resource | `portfolio://candidate/profile` | 지원자 공개 프로필 JSON |
| Resource | `portfolio://evidence/index` | 전체 근거 인덱스 JSON |
| Prompt | `evaluate_candidate_with_evidence` | 강점·근거·누락 정보·면접 질문 순서의 검증형 리뷰 |

- 원격 transport는 MCP 공식 권장인 Streamable HTTP를 사용한다.
- Worker endpoint는 `/mcp`, 상태 확인은 `/health`로 제한한다.
- Origin allowlist, 입력 길이 제한, 최대 반환 개수 제한, 5분 public-data cache를 적용한다.
- Worker 배포 후 `_config.yml`의 `mcp_endpoint`에 실제 URL을 기록해야 사이트 복사 버튼이 활성화된다.
- 공개 데이터만 다루므로 초기 버전은 인증 없이 제공한다. 향후 비공개 자료가 추가되면 OAuth 전환 전까지 MCP에 포함하지 않는다.

## 10. SEO·접근성·성능

### 10.1 SEO

- 기존 canonical URL을 유지한다.
- 각 페이지에 title, description/excerpt, canonical, Open Graph, Twitter Card를 제공한다.
- `jekyll-seo-tag`를 Gemfile과 plugins 설정에 일관되게 등록하거나 기존 custom SEO include와 역할 중복을 정리한다.
- sitemap.xml과 feed.xml 생성 여부를 빌드 검증한다.
- Article JSON-LD와 Person/ProfilePage JSON-LD는 실제 정보만 사용한다.

### 10.2 접근성

- WCAG 2.2 AA를 목표로 한다.
- skip link, landmark, heading hierarchy, focus-visible을 제공한다.
- 모든 상호작용은 keyboard로 가능해야 한다.
- 장식용 orb/gradient는 `aria-hidden="true"` 및 pointer-events none.
- color 하나만으로 상태를 표현하지 않는다.
- 200% zoom 및 320px CSS viewport에서 기능 손실이 없어야 한다.
- reduced motion과 고대비/forced-colors 환경을 확인한다.

### 10.3 성능 예산

| 항목 | 목표 |
| --- | --- |
| Lighthouse Performance | 모바일 85+, 데스크톱 90+ |
| Lighthouse Accessibility/SEO/Best Practices | 각 95+ |
| 초기 커스텀 JS | gzip 35KB 이하 |
| LCP | 2.5초 이하 (중간급 모바일 기준) |
| CLS | 0.1 이하 |
| INP | 200ms 이하 |
| Hero 3D 자산 | 없음 |

## 11. 배포·검증 전략

### 11.1 개발 흐름

1. 별도 기능 브랜치에서 작업한다.
2. 기존 URL 목록과 빌드 결과를 baseline으로 기록한다.
3. 새 레이아웃·토큰·include를 추가한다.
4. 홈 → 컬렉션 → 글 상세 → CV → 404 순서로 전환한다.
5. 로컬 production build를 검증한다.
6. GitHub Actions와 동일한 Ruby 3.1 환경에서 빌드한다.
7. PR preview 또는 로컬 캡처로 데스크톱·모바일을 승인한다.
8. `master` 병합 후 Pages 배포와 핵심 URL을 확인한다.

### 11.2 필수 자동 검증

```bash
bundle exec jekyll build
```

추가 검증 도구는 구현 시 선택하되 다음을 검사해야 한다.

- 내부 링크·이미지 경로 404
- 중복 permalink
- HTML landmark/heading 오류
- 홈 및 대표 글의 Lighthouse
- JavaScript console error

### 11.3 수동 테스트 매트릭스

| 화면 | 320/375px | 768px | 1440px | Reduced motion | Keyboard |
| --- | --- | --- | --- | --- | --- |
| 홈 | 필수 | 필수 | 필수 | 필수 | 필수 |
| 각 컬렉션 | 필수 | 대표 1회 | 필수 | 해당 없음 | 필수 |
| 코드 포함 글 | 필수 | 필수 | 필수 | 해당 없음 | 필수 |
| 표/수식 포함 글 | 필수 | 필수 | 필수 | 해당 없음 | 필수 |
| CV/404 | 필수 | 선택 | 필수 | 해당 없음 | 필수 |

## 12. 단계별 구현 계획

### Phase 1 — Foundation

- 디자인 토큰, base typography, header/footer, theme
- 새 레이아웃과 include 추가
- 기존 URL 및 콘텐츠 snapshot 확보

완료 조건: 스타일을 적용하지 않은 콘텐츠 fallback과 새 전역 shell이 모두 빌드된다.

### Phase 2 — Home & Archive

- 운영 홈의 실시간 LLM 처리 그래프, SEED 원칙 기반 MCP 안내, featured project, latest notes
- 프로젝트별 이미지형 갤러리와 URL 공유 가능한 글 필터
- 컬렉션 공통 목록 및 Study 통합 허브
- 모바일 전역 메뉴

완료 조건: 데모의 정보 구조가 실제 Liquid 데이터로 렌더링되고 하드코딩 데이터가 없다.

### Phase 3 — Article & Markdown

- article header gradient art, 현재 구간을 추적하는 좌측 TOC, progress, previous/next
- 하단 최근 글 3개의 `더 읽어보기` 탐색 UI
- Rouge Python syntax theme
- table, quote, inline code, MathJax, image 스타일

완료 조건: ADK 글, N-DP 글, WrapupReport를 대표 fixture로 사용해 코드·표·긴 글을 검증한다.

### Phase 4 — CV, SEO, Accessibility

- CV 실제 데이터 연결
- 404, metadata, feed/sitemap
- 키보드, contrast, reduced motion, mobile 테스트

완료 조건: 존재하지 않는 CV/다운로드 링크가 없고 Lighthouse 목표를 충족한다.

### Phase 5 — Recruiter MCP

- recruiter 공개 데이터 생성기와 Evidence explorer
- Cloudflare Worker MCP tool/resource/prompt 구현 및 Inspector 검증
- 실제 Worker URL을 사이트에 연결

완료 조건: 로컬 MCP typecheck와 Inspector tool 호출이 성공하고, 반환되는 모든 근거에 원문 URL이 포함된다.

### Phase 6 — Release

- demo 배포 제외
- dead CSS/JS 정리
- GitHub Actions build/deploy 확인
- 배포 후 핵심 URL smoke test

완료 조건: 기존 URL과 원본 콘텐츠가 유지되고 운영 사이트가 새 UI로 제공된다.

## 13. 운영 전환 수용 기준

### 콘텐츠와 URL

- [ ] 기존 Project 17, Study 30, Algorithm 9, WeeklyReview 10, ETC 5 글이 모두 빌드된다.
- [ ] 기존 collection permalink가 유지된다.
- [ ] Markdown 본문과 이미지 원본이 의도치 않게 변경되지 않는다.
- [ ] `/projects/`, `/study/`, `/algorithm/`, `/weeklyreview/`, `/cv/`, `/etc/`가 200을 반환한다.
- [ ] 중복 `/study/` source를 하나로 정리한다.

### UI

- [ ] 좌측 상단에 `Archive for AI Study`가 모든 화면에서 표시된다.
- [ ] 여섯 개 전역 메뉴가 데스크톱과 모바일에서 접근 가능하다.
- [ ] 홈의 실시간 LLM 그래프와 featured aurora 카드가 승인 데모의 절제된 디자인 언어를 유지한다.
- [ ] 홈 정보 순서는 LLM 시각화 → MCP mode → Featured Project → Latest Notes이며 Collections를 노출하지 않는다.
- [ ] 모든 글 카드와 상세 헤더에 제목 기반 gradient art가 있고 포인터에 반응한다.
- [ ] Project 갤러리는 카페 추천·감성 분류·EduTech 단위로 글을 필터링하고 선택 상태를 URL에 보존한다.
- [ ] 컬렉션 목록과 글 상세가 실제 Liquid 데이터로 렌더링된다.
- [ ] 글 상세의 breadcrumb, 현재 구간 목차, 메타, 진행 표시, 최근 글 3개가 동작한다.
- [ ] CV는 Hyejun, Career, Education 정보만 본문에 노출하며 프로필 사진과 연락 바로가기를 제거한다.

### Markdown

- [ ] Python 코드의 keyword/function/string/number/comment가 구분된다.
- [ ] 언어 없는 code fence도 읽기 가능한 기본 스타일을 갖는다.
- [ ] 표가 모바일에서 페이지 가로 폭을 깨지 않는다.
- [ ] 수식, 이미지, 인용문, 목록, inline code가 대표 글에서 정상 표시된다.

### 품질

- [ ] JavaScript 비활성 상태에서도 모든 콘텐츠와 링크를 사용할 수 있다.
- [ ] reduced motion에서 tilt/reveal/자동 궤도 모션이 제거된다.
- [ ] 320px~1440px에서 전역 가로 스크롤이 없다. 코드/표 내부 스크롤은 허용한다.
- [ ] 브라우저 console error가 없다.
- [ ] Jekyll production build와 GitHub Actions가 성공한다.
- [ ] Lighthouse와 Core Web Vitals 목표를 만족한다.

### Recruiter MCP

- [ ] 블로그 빌드 시 71개 문서와 CV가 구조화 데이터에 반영된다.
- [ ] `/recruiter/`, `/llms.txt`, `/llms-full.txt`, 공개 JSON이 200을 반환한다.
- [ ] Evidence explorer가 LLM/API key 없이도 관련 원문을 검색한다.
- [ ] MCP가 snapshot/search/case-study/role-evidence 네 tool과 두 resource, 한 prompt를 노출한다.
- [ ] MCP 응답의 포트폴리오 근거에는 canonical 원문 URL이 포함된다.
- [ ] 서버는 읽기 전용이며 채용담당자의 query와 판단 결과를 저장하지 않는다.
- [ ] 미배포 endpoint를 LIVE로 표시하거나 동작하는 것처럼 오인시키지 않는다.

## 14. 구현 모델에 전달할 작업 지침

1. 이 PRD와 `demo/`를 디자인 기준으로 사용한다.
2. 작업 전 기존 콘텐츠·URL·dirty worktree를 확인한다.
3. 기존 Markdown과 이미지에는 손대지 않는다.
4. Jekyll/Liquid/SCSS/vanilla JS로 구현한다.
5. 화면 데이터는 실제 collection에서 읽고 데모의 하드코딩 값을 복사하지 않는다.
6. 새 레이아웃을 추가한 뒤 단계적으로 defaults를 전환한다.
7. 각 Phase 끝에 build, 모바일, 접근성, 핵심 링크를 검증한다.
8. 범위를 벗어난 콘텐츠 정리나 프레임워크 마이그레이션은 별도 제안으로 남긴다.

## 15. 구현 전 확인된 리스크

| 리스크 | 영향 | 대응 |
| --- | --- | --- |
| 루트 `study.md`와 `_pages/study.md`가 같은 `/study/` 사용 | 중복 출력/빌드 불확실성 | 하나만 canonical source로 유지 |
| `_data/cv.json`에 샘플 데이터 존재 | 잘못된 개인정보 노출 | 운영 UI는 `_pages/cv.md` 우선, JSON은 실제화 전 비활성 |
| `cv-json.md`가 없는 `/files/cv.pdf` 링크 가능 | 404 | 파일 존재 여부 검사 후 링크 조건부 출력 |
| 기존 이미지 수와 용량이 큼 | LCP·배포 시간 | 본문 lazy loading, 크기 지정, 홈 이미지 최소화 |
| frontmatter 형식 불균일 | 카드/태그 오류 | Liquid default와 normalize include로 방어 |
| 기존 테마 CSS와 새 CSS 충돌 | 레이아웃 회귀 | modern namespace/cascade layer 후 구형 CSS 점진 제거 |
| `demo/`가 Jekyll 결과에 포함될 수 있음 | 중복 공개 페이지 | 릴리스 전에 `_config.yml` exclude 또는 제거 |
| GitHub Pages에서 MCP 프로세스를 실행할 수 없음 | endpoint 부재 | Cloudflare Worker로 분리하고 정적 JSON을 단일 공개 데이터 원천으로 사용 |
| LLM이 작성하지 않은 역량을 과장할 수 있음 | 채용 신뢰도 저하 | 원문 URL 의무화, 사실/추론 분리 instruction, 숫자형 평가 금지 |
| 공개 MCP 남용 또는 과도한 응답 | 비용·가용성 | 읽기 전용, 입력/limit 제한, 캐시, 필요 시 Cloudflare rate limit 추가 |
