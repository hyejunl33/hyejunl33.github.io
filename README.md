# Archive for AI Study

[hyejunl33.github.io](https://hyejunl33.github.io)는 AI 프로젝트, 학습 기록, 알고리즘 풀이와 주간 회고를 쌓아가는 개인 아카이브입니다.

초기 기반은 Academic Pages/Jekyll 계열이었지만, 현재의 화면·콘텐츠 구조·상호작용은 이 저장소에서 별도로 구현하고 운영합니다. 따라서 이 저장소를 Academic Pages 포크로 취급하거나 upstream 템플릿을 동기화 대상으로 삼지 않습니다.

## 구성

- **Modern UI**: 홈, 아카이브, 글 상세, 목차, 읽기 진행도, 링크 복사, 추천 글을 포함한 반응형 인터페이스
- **콘텐츠 컬렉션**: Project, Study, Algorithm, Weekly Review, ETC
- **정적 배포**: GitHub Actions가 `master`의 변경을 Jekyll로 빌드해 GitHub Pages에 배포
- **LLM 탐색**: `llms.txt`와 `llms-full.txt`로 공개 글의 인덱스와 전체 원문 제공

주요 구현 위치는 다음과 같습니다.

| 영역 | 위치 |
| --- | --- |
| 홈 화면 | `_layouts/modern-home.html` |
| 글 상세 화면 | `_layouts/modern-single.html` |
| 공통 셸과 헤더·푸터 | `_layouts/modern-base.html`, `_includes/modern/` |
| Modern UI 스타일 | `_sass/modern/` |
| 브라우저 상호작용 | `assets/js/modern-ui.js` |
| 사이트·컬렉션 설정 | `_config.yml`, `_data/navigation.yml` |

## 콘텐츠 작성

컬렉션별 Markdown 파일을 추가하면 됩니다.

| 메뉴 | 디렉터리 | URL |
| --- | --- | --- |
| Project | `_projects/` | `/projects/` |
| Study | `_study/` | `/study/` |
| Algorithm | `_algorithm/` | `/algorithm/` |
| Weekly Review | `_weeklyreview/` | `/weeklyreview/` |
| ETC | `_etc/` | `/etc/` |

글은 아래처럼 작성합니다.

```markdown
---
layout: modern-single
title: "글 제목"
date: 2026-10-01
tags:
  - Study
excerpt: "목록과 검색 결과에 표시할 짧은 설명"
---

본문을 작성합니다.
```

Notion에서 가져온 글은 `notion_source_id`를 유지하고, 이미지 등 첨부 자산은 `assets/images/notion/<source-hash>/`에 저장합니다. 만료되는 Notion 서명 URL을 본문에 직접 넣지 않습니다.

## 로컬 실행

Ruby, Bundler, Node.js가 필요합니다.

```bash
bundle install
npm install
npm run build:blog-data
bundle exec jekyll serve --livereload
```

브라우저에서 `http://localhost:4000`을 열어 확인합니다. 배포 전 정적 빌드만 확인하려면 다음을 실행합니다.

```bash
npm run build:blog-data
bundle exec jekyll build
git diff --check
```

## 배포

`master`에 푸시하면 `.github/workflows/jekyll.yml`이 사이트를 빌드하고 GitHub Pages에 배포합니다. 공개 반영은 Actions 실행이 성공한 뒤 [사이트](https://hyejunl33.github.io)에서 확인합니다.

## 기반과 라이선스

이 저장소는 과거 Academic Pages/Minimal Mistakes 계열을 출발점으로 사용했습니다. 현재 제품 코드와 UI는 별도 운영되며, 템플릿 upstream과 자동 동기화하지 않습니다. 기반 프로젝트의 라이선스 및 고지는 [LICENSE](LICENSE)에서 확인할 수 있습니다.
