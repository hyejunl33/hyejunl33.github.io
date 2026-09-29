(function () {
  'use strict';

  var root = document.documentElement;
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function initTheme() {
    var button = document.querySelector('[data-theme-toggle]');
    if (!button) return;

    function syncButton() {
      var isDark = root.dataset.theme === 'dark';
      button.setAttribute('aria-pressed', String(isDark));
      button.setAttribute('aria-label', isDark ? '라이트 테마로 전환' : '다크 테마로 전환');
    }

    syncButton();
    button.addEventListener('click', function () {
      root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('archive-theme', root.dataset.theme);
      syncButton();
    });
  }

  function initReveal() {
    var elements = Array.from(document.querySelectorAll('.modern-reveal'));
    if (!elements.length) return;
    if (reduceMotion || !('IntersectionObserver' in window)) {
      elements.forEach(function (element) { element.classList.add('is-visible'); });
      return;
    }
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.14 });
    elements.forEach(function (element) { observer.observe(element); });
  }

  function initOrbitalField() {
    var field = document.querySelector('[data-orbital-field]');
    if (!field || reduceMotion || !window.matchMedia('(pointer: fine)').matches) return;
    var frame = 0;
    field.addEventListener('pointermove', function (event) {
      if (frame) return;
      frame = window.requestAnimationFrame(function () {
        var rect = field.getBoundingClientRect();
        var x = ((event.clientX - rect.left) / rect.width - .5) * 18;
        var y = ((event.clientY - rect.top) / rect.height - .5) * 18;
        field.style.setProperty('--orb-x', x.toFixed(2) + 'px');
        field.style.setProperty('--orb-y', y.toFixed(2) + 'px');
        frame = 0;
      });
    });
    field.addEventListener('pointerleave', function () {
      field.style.setProperty('--orb-x', '0px');
      field.style.setProperty('--orb-y', '0px');
    });
  }

  function initAuroraCard() {
    document.querySelectorAll('[data-aurora-card]').forEach(function (card) {
      if (reduceMotion || !window.matchMedia('(pointer: fine)').matches) return;
      var frame = 0;
      card.addEventListener('pointermove', function (event) {
        if (frame) return;
        frame = window.requestAnimationFrame(function () {
          var rect = card.getBoundingClientRect();
          card.style.setProperty('--aurora-x', (((event.clientX - rect.left) / rect.width) * 100).toFixed(1) + '%');
          card.style.setProperty('--aurora-y', (((event.clientY - rect.top) / rect.height) * 100).toFixed(1) + '%');
          frame = 0;
        });
      });
    });
  }

  function stringHue(value) {
    var hash = 0;
    for (var index = 0; index < value.length; index += 1) hash = ((hash << 5) - hash + value.charCodeAt(index)) | 0;
    return 105 + Math.abs(hash % 190);
  }

  function initGradientCards() {
    document.querySelectorAll('[data-gradient-art]').forEach(function (art) {
      art.style.setProperty('--art-hue', String(stringHue(art.dataset.artKey || 'archive')));
      if (reduceMotion || !window.matchMedia('(pointer: fine)').matches) return;
      var card = art.closest('[data-gradient-card]') || art;
      var frame = 0;
      card.addEventListener('pointermove', function (event) {
        if (frame) return;
        frame = window.requestAnimationFrame(function () {
          var rect = art.getBoundingClientRect();
          var x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
          var y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
          art.style.setProperty('--art-x', (x * 100).toFixed(1) + '%');
          art.style.setProperty('--art-y', (y * 100).toFixed(1) + '%');
          if (card.classList.contains('modern-post-card')) {
            card.style.setProperty('--card-rx', ((.5 - y) * 2.4).toFixed(2) + 'deg');
            card.style.setProperty('--card-ry', ((x - .5) * 3.2).toFixed(2) + 'deg');
          }
          frame = 0;
        });
      });
      card.addEventListener('pointerleave', function () {
        card.style.setProperty('--card-rx', '0deg');
        card.style.setProperty('--card-ry', '0deg');
        art.style.setProperty('--art-x', '68%');
        art.style.setProperty('--art-y', '24%');
      });
    });
  }

  function initLlmArchitecture() {
    var stage = document.querySelector('[data-llm-visual]');
    var canvas = stage && stage.querySelector('[data-llm-canvas]');
    if (!stage || !canvas) return;
    var context = canvas.getContext('2d');
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    var width = 0;
    var height = 0;
    var pointer = { x: 0, y: 0, tx: 0, ty: 0 };
    var layers = [6, 9, 11, 9, 7, 4];
    var nodes = [];
    var edges = [];
    var particles = [];
    var start = performance.now();
    var lastStat = 0;

    layers.forEach(function (count, layer) {
      for (var row = 0; row < count; row += 1) {
        nodes.push({ layer: layer, row: row, count: count, x: (layer - 2.5) * 1.18, y: (row - (count - 1) / 2) * .48, z: Math.sin((row + layer) * 1.7) * .42 });
      }
    });
    for (var a = 0; a < nodes.length; a += 1) {
      for (var b = 0; b < nodes.length; b += 1) {
        if (nodes[b].layer !== nodes[a].layer + 1) continue;
        if ((a * 7 + b * 11) % 9 < 2) edges.push({ from: nodes[a], to: nodes[b] });
      }
    }
    for (var particle = 0; particle < 34; particle += 1) particles.push({ edge: edges[particle % edges.length], offset: (particle * .137) % 1, speed: .00008 + (particle % 7) * .000012 });

    function resize() {
      var rect = stage.getBoundingClientRect();
      width = Math.max(1, Math.round(rect.width));
      height = Math.max(1, Math.round(rect.height));
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function project(node) {
      var cy = Math.cos(pointer.y); var sy = Math.sin(pointer.y);
      var cx = Math.cos(pointer.x); var sx = Math.sin(pointer.x);
      var rx = node.x * cy - node.z * sy;
      var rz = node.x * sy + node.z * cy;
      var ry = node.y * cx - rz * sx;
      rz = node.y * sx + rz * cx;
      var perspective = 1 / (1 + (rz + 5.5) * .045);
      var scale = Math.min(width, height) * .125;
      return { x: width * (width < 720 ? .58 : .67) + rx * scale * perspective, y: height * .43 + ry * scale * perspective, z: rz, p: perspective };
    }

    function draw(now) {
      pointer.x += (pointer.tx - pointer.x) * .045;
      pointer.y += (pointer.ty - pointer.y) * .045;
      context.clearRect(0, 0, width, height);
      var projected = new Map();
      nodes.forEach(function (node) { projected.set(node, project(node)); });

      edges.forEach(function (edge) {
        var from = projected.get(edge.from); var to = projected.get(edge.to);
        var fade = .07 + Math.max(0, (from.z + to.z + 2) * .012);
        context.beginPath(); context.moveTo(from.x, from.y); context.lineTo(to.x, to.y);
        context.strokeStyle = 'rgba(164, 219, 196, ' + fade.toFixed(3) + ')'; context.lineWidth = .7; context.stroke();
      });

      nodes.slice().sort(function (one, two) { return projected.get(one).z - projected.get(two).z; }).forEach(function (node) {
        var point = projected.get(node); var radius = (node.layer === layers.length - 1 ? 3.4 : 2.3) * point.p;
        context.beginPath(); context.arc(point.x, point.y, radius, 0, Math.PI * 2);
        context.fillStyle = node.layer === 0 ? 'rgba(169,146,255,.86)' : node.layer === layers.length - 1 ? 'rgba(199,242,103,.92)' : 'rgba(213,240,228,.58)'; context.fill();
      });

      particles.forEach(function (moving) {
        var t = reduceMotion ? moving.offset : (moving.offset + (now - start) * moving.speed) % 1;
        var from = projected.get(moving.edge.from); var to = projected.get(moving.edge.to);
        var x = from.x + (to.x - from.x) * t; var y = from.y + (to.y - from.y) * t;
        var glow = context.createRadialGradient(x, y, 0, x, y, 8);
        glow.addColorStop(0, 'rgba(199,242,103,.95)'); glow.addColorStop(1, 'rgba(199,242,103,0)');
        context.fillStyle = glow; context.beginPath(); context.arc(x, y, 8, 0, Math.PI * 2); context.fill();
      });

      if (now - lastStat > 850) {
        var token = stage.querySelector('[data-llm-tokens]'); var path = stage.querySelector('[data-llm-paths]'); var latency = stage.querySelector('[data-llm-latency]');
        if (token) token.textContent = String(128 + Math.floor((now / 850) % 47));
        if (path) path.textContent = String(22 + Math.floor((now / 1200) % 9));
        if (latency) latency.textContent = String(16 + Math.floor((now / 980) % 7));
        lastStat = now;
      }
      if (!reduceMotion && !document.hidden) window.requestAnimationFrame(draw);
    }

    resize();
    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(stage);
    window.addEventListener('resize', resize, { passive: true });
    if (!reduceMotion) {
      stage.addEventListener('pointermove', function (event) {
        var rect = stage.getBoundingClientRect();
        pointer.ty = ((event.clientX - rect.left) / rect.width - .5) * .3;
        pointer.tx = (.5 - (event.clientY - rect.top) / rect.height) * .2;
      });
      stage.addEventListener('pointerleave', function () { pointer.tx = 0; pointer.ty = 0; });
      document.addEventListener('visibilitychange', function () {
        if (!document.hidden) window.requestAnimationFrame(draw);
      });
    }
    window.requestAnimationFrame(draw);
  }

  function initArchiveSearch() {
    var input = document.querySelector('[data-archive-search]');
    if (!input) return;
    var items = Array.from(document.querySelectorAll('[data-search-item]'));
    var empty = document.querySelector('[data-search-empty]');
    var filters = Array.from(document.querySelectorAll('[data-project-filter]'));
    var count = document.querySelector('[data-project-result-count]');
    var label = document.querySelector('[data-project-result-label]');
    var allowedFilters = filters.map(function (filter) { return filter.dataset.projectFilter; });
    var requested = new URLSearchParams(window.location.search).get('project');
    var activeProject = allowedFilters.includes(requested) ? requested : 'all';

    function updateItems() {
      var query = input.value.trim().toLocaleLowerCase();
      var visible = 0;
      items.forEach(function (item) {
        var matchesQuery = !query || item.textContent.toLocaleLowerCase().includes(query);
        var matchesProject = activeProject === 'all' || item.dataset.projectGroup === activeProject;
        var matched = matchesQuery && matchesProject;
        item.hidden = !matched;
        if (matched) visible += 1;
      });
      if (empty) empty.hidden = visible !== 0;
      if (count) count.textContent = String(visible);
      if (label) label.textContent = activeProject === 'all' ? '' : ' · 선택한 프로젝트';
      filters.forEach(function (filter) {
        var active = filter.dataset.projectFilter === activeProject;
        filter.classList.toggle('is-active', active);
        filter.setAttribute('aria-pressed', String(active));
      });
    }

    input.addEventListener('input', updateItems);
    filters.forEach(function (filter) {
      filter.addEventListener('click', function () {
        activeProject = filter.dataset.projectFilter;
        var url = new URL(window.location.href);
        if (activeProject === 'all') url.searchParams.delete('project'); else url.searchParams.set('project', activeProject);
        window.history.replaceState({}, '', url.pathname + url.search + '#project-notes');
        updateItems();
        var notes = document.getElementById('project-notes');
        if (notes) notes.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
      });
    });
    updateItems();
  }

  function slugify(text, index) {
    var slug = text.trim().toLocaleLowerCase()
      .replace(/[^\p{Letter}\p{Number}\s-]/gu, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-');
    return slug || 'section-' + index;
  }

  function initArticle() {
    var body = document.querySelector('[data-article-body]');
    if (!body) return;

    body.querySelectorAll('table').forEach(function (table) {
      if (table.parentElement.classList.contains('modern-table-scroll')) return;
      var wrapper = document.createElement('div');
      wrapper.className = 'modern-table-scroll';
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });

    body.querySelectorAll('img').forEach(function (image) {
      image.loading = 'lazy';
      image.decoding = 'async';
    });

    body.querySelectorAll('div.highlighter-rouge, figure.highlight').forEach(function (block) {
      var classSource = block.className + ' ' + (block.querySelector('code') || {}).className;
      var match = classSource.match(/language-([\w+-]+)/);
      block.dataset.language = match ? match[1] : 'code';
    });

    var headings = Array.from(body.querySelectorAll('h2, h3'));
    var toc = document.querySelector('[data-toc]');
    var tocList = document.querySelector('[data-toc-list]');
    var usedIds = new Set();
    var links = [];
    headings.forEach(function (heading, index) {
      var baseId = heading.id || slugify(heading.textContent, index);
      var id = baseId;
      var suffix = 2;
      while (usedIds.has(id)) { id = baseId + '-' + suffix++; }
      usedIds.add(id);
      heading.id = id;
      if (!tocList) return;
      var link = document.createElement('a');
      link.href = '#' + encodeURIComponent(id);
      link.textContent = heading.textContent;
      link.dataset.level = heading.tagName.slice(1);
      tocList.appendChild(link);
      links.push(link);
    });
    if (toc && headings.length < 2) toc.hidden = true;

    if (links.length) {
      var tocFrame = 0;
      var currentIndex = -1;
      function updateToc() {
        var nextIndex = 0;
        headings.forEach(function (heading, index) {
          if (heading.getBoundingClientRect().top <= 170) nextIndex = index;
        });
        if (nextIndex !== currentIndex) {
          currentIndex = nextIndex;
          links.forEach(function (link, index) { link.classList.toggle('is-active', index === currentIndex); });
          var activeLink = links[currentIndex];
          if (activeLink && toc && (activeLink.offsetTop < toc.scrollTop || activeLink.offsetTop > toc.scrollTop + toc.clientHeight - 48)) {
            toc.scrollTo({ top: Math.max(0, activeLink.offsetTop - 48), behavior: reduceMotion ? 'auto' : 'smooth' });
          }
        }
        tocFrame = 0;
      }
      window.addEventListener('scroll', function () {
        if (!tocFrame) tocFrame = window.requestAnimationFrame(updateToc);
      }, { passive: true });
      updateToc();
    }

    var progress = document.querySelector('[data-reading-progress]');
    if (progress) {
      var progressFrame = 0;
      function updateProgress() {
        var rect = body.getBoundingClientRect();
        var start = window.scrollY + rect.top - window.innerHeight * .2;
        var distance = Math.max(1, body.offsetHeight - window.innerHeight * .65);
        var value = Math.min(1, Math.max(0, (window.scrollY - start) / distance));
        progress.style.width = (value * 100).toFixed(2) + '%';
        progressFrame = 0;
      }
      window.addEventListener('scroll', function () {
        if (!progressFrame) progressFrame = window.requestAnimationFrame(updateProgress);
      }, { passive: true });
      updateProgress();
    }
  }

  function initCopyLink() {
    var button = document.querySelector('[data-copy-link]');
    if (!button) return;
    button.addEventListener('click', function () {
      var promise = navigator.clipboard && navigator.clipboard.writeText
        ? navigator.clipboard.writeText(window.location.href)
        : Promise.reject();
      promise.then(function () {
        button.textContent = '복사됨';
        window.setTimeout(function () { button.textContent = '링크 복사'; }, 1600);
      }).catch(function () {
        window.prompt('이 주소를 복사하세요.', window.location.href);
      });
    });
  }

  function initRecruiterMode() {
    var status = document.querySelector('[data-mcp-endpoint]');
    var endpointButton = document.querySelector('[data-mcp-copy]');
    if (status && endpointButton && status.dataset.mcpEndpoint) {
      endpointButton.addEventListener('click', function () {
        navigator.clipboard.writeText(status.dataset.mcpEndpoint).then(function () {
          endpointButton.textContent = '복사됨';
          window.setTimeout(function () { endpointButton.textContent = 'Endpoint 복사'; }, 1600);
        });
      });
    }

    var lab = document.querySelector('[data-portfolio-source]');
    var form = document.querySelector('[data-evidence-form]');
    var input = document.querySelector('[data-evidence-input]');
    var results = document.querySelector('[data-evidence-results]');
    if (!lab || !form || !input || !results) return;
    var portfolioPromise;

    function loadPortfolio() {
      if (!portfolioPromise) {
        portfolioPromise = fetch(lab.dataset.portfolioSource, { headers: { accept: 'application/json' } }).then(function (response) {
          if (!response.ok) throw new Error('Portfolio data unavailable');
          return response.json();
        });
      }
      return portfolioPromise;
    }

    function searchDocuments(documents, query) {
      var terms = query.toLocaleLowerCase().split(/[^\p{Letter}\p{Number}+#.-]+/u).filter(function (term) { return term.length > 1; });
      return documents.map(function (document) {
        var title = document.title.toLocaleLowerCase();
        var tags = document.tags.join(' ').toLocaleLowerCase();
        var body = (document.excerpt + ' ' + document.content).toLocaleLowerCase();
        var score = terms.reduce(function (total, term) {
          return total + (title.includes(term) ? 8 : 0) + (tags.includes(term) ? 5 : 0) + (body.includes(term) ? 2 : 0);
        }, 0);
        return { document: document, score: score };
      }).filter(function (item) { return item.score > 0; }).sort(function (a, b) { return b.score - a.score; }).slice(0, 6);
    }

    function render(items, query) {
      results.replaceChildren();
      if (!items.length) {
        var empty = document.createElement('p');
        empty.textContent = '“' + query + '”와 직접 연결되는 작성 근거를 찾지 못했습니다.';
        results.appendChild(empty);
        return;
      }
      items.forEach(function (item) {
        var document = item.document;
        var link = documentNode('a', 'modern-evidence-result');
        link.href = document.url;
        var meta = documentNode('span');
        meta.textContent = document.collectionLabel + (document.date ? ' · ' + document.date : '');
        var title = documentNode('h3');
        title.textContent = document.title;
        var excerpt = documentNode('p');
        excerpt.textContent = document.excerpt;
        link.append(meta, title, excerpt);
        results.appendChild(link);
      });
    }

    function documentNode(tag, className) {
      var element = document.createElement(tag);
      if (className) element.className = className;
      return element;
    }

    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var query = input.value.trim();
      if (query.length < 2) return;
      results.replaceChildren();
      var loading = documentNode('p');
      loading.textContent = '공개 기록에서 근거를 찾는 중…';
      results.appendChild(loading);
      loadPortfolio().then(function (data) {
        render(searchDocuments(data.documents || [], query), query);
      }).catch(function () {
        results.replaceChildren();
        var error = documentNode('p');
        error.textContent = '지금은 데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.';
        results.appendChild(error);
      });
    });

    document.querySelectorAll('[data-evidence-prompt]').forEach(function (button) {
      button.addEventListener('click', function () {
        input.value = button.dataset.evidencePrompt;
        form.requestSubmit();
      });
    });

    var sharedQuery = new URLSearchParams(window.location.search).get('q');
    if (sharedQuery && sharedQuery.trim().length >= 2) {
      input.value = sharedQuery.trim();
      form.requestSubmit();
    }
  }

  initTheme();
  initReveal();
  initOrbitalField();
  initAuroraCard();
  initGradientCards();
  initLlmArchitecture();
  initArchiveSearch();
  initArticle();
  initCopyLink();
  initRecruiterMode();
}());
