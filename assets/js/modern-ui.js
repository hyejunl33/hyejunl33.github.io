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
    var surfaceCanvas = stage && stage.querySelector('[data-loss-surface]');
    if (!stage || !canvas || !surfaceCanvas) return;
    var context = canvas.getContext('2d');
    var gl = surfaceCanvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: true });
    var glProgram;
    var glPosition;
    var glColor;
    var glBuffer;
    var glPitch;
    var glYaw;
    var glScale;
    var surfaceLayer = document.createElement('canvas');
    var surfaceContext = surfaceLayer.getContext('2d');
    var surfaceDirty = true;
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    var width = 0;
    var height = 0;
    var camera = { pitch: -.62, yaw: -.62, targetPitch: -.62, targetYaw: -.62 };
    var gridSize = 45;
    var surface = [];
    var cells = [];
    var trajectories = {};
    var optimizerStyles = {
      sgd: { color: '#ff806f', rgb: '255,128,111' },
      momentum: { color: '#c7f267', rgb: '199,242,103' },
      adagrad: { color: '#87ebcf', rgb: '135,235,207' },
      rmsprop: { color: '#77bff5', rgb: '119,191,245' },
      adam: { color: '#c1a8ff', rgb: '193,168,255' },
      adamw: { color: '#ff9ad5', rgb: '255,154,213' }
    };
    var activeOptimizer = 'all';
    var drag = { active: false, x: 0, y: 0 };
    var start = performance.now();

    if (gl) {
      var vertexSource = 'attribute vec3 a_position; attribute vec4 a_color; uniform float u_pitch; uniform float u_yaw; uniform vec2 u_scale; varying vec4 v_color; void main(){ float cy=cos(u_yaw); float sy=sin(u_yaw); float cx=cos(u_pitch); float sx=sin(u_pitch); float rx=a_position.x*cy-a_position.z*sy; float rz=a_position.x*sy+a_position.z*cy; float centeredY=a_position.y-.75; float ry=centeredY*cx-rz*sx; float depth=centeredY*sx+rz*cx; float perspective=1.0/(1.0+(depth+4.7)*.035); gl_Position=vec4(rx*u_scale.x*perspective, -.04+ry*u_scale.y*perspective, -depth/8.0, 1.0); v_color=a_color; }';
      var fragmentSource = 'precision mediump float; varying vec4 v_color; void main(){ gl_FragColor=v_color; }';
      function compileShader(type, source) {
        var shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
      }
      var vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
      var fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
      if (vertexShader && fragmentShader) {
        glProgram = gl.createProgram();
        gl.attachShader(glProgram, vertexShader);
        gl.attachShader(glProgram, fragmentShader);
        gl.linkProgram(glProgram);
        if (!gl.getProgramParameter(glProgram, gl.LINK_STATUS)) glProgram = null;
      }
      if (glProgram) {
        glPosition = gl.getAttribLocation(glProgram, 'a_position');
        glColor = gl.getAttribLocation(glProgram, 'a_color');
        glPitch = gl.getUniformLocation(glProgram, 'u_pitch');
        glYaw = gl.getUniformLocation(glProgram, 'u_yaw');
        glScale = gl.getUniformLocation(glProgram, 'u_scale');
        glBuffer = gl.createBuffer();
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      } else {
        gl = null;
      }
    }

    function loss(x, z) {
      return .17 * (x * x + z * z) + .34 * Math.sin(1.3 * x) * Math.cos(1.15 * z) + .1 * Math.sin(2.2 * x + .7 * z) + .38;
    }

    function gradient(x, z) {
      return {
        x: .34 * x + .442 * Math.cos(1.3 * x) * Math.cos(1.15 * z) + .22 * Math.cos(2.2 * x + .7 * z),
        z: .34 * z - .391 * Math.sin(1.3 * x) * Math.sin(1.15 * z) + .07 * Math.cos(2.2 * x + .7 * z)
      };
    }

    for (var row = 0; row < gridSize; row += 1) {
      var line = [];
      for (var column = 0; column < gridSize; column += 1) {
        var x = -3.15 + column / (gridSize - 1) * 6.3;
        var z = -3.15 + row / (gridSize - 1) * 6.3;
        line.push({ x: x, y: loss(x, z), z: z });
      }
      surface.push(line);
    }
    for (var gridRow = 0; gridRow < gridSize - 1; gridRow += 1) {
      for (var gridColumn = 0; gridColumn < gridSize - 1; gridColumn += 1) {
        cells.push([surface[gridRow][gridColumn], surface[gridRow][gridColumn + 1], surface[gridRow + 1][gridColumn + 1], surface[gridRow + 1][gridColumn]]);
      }
    }

    if (gl) {
      var surfaceVertices = [];
      cells.forEach(function (cell) {
        [0, 1, 2, 0, 2, 3].forEach(function (index) {
          var vertex = cell[index];
          var color = surfaceChannels(vertex.y);
          surfaceVertices.push(vertex.x, vertex.y, vertex.z, color[0] / 255, color[1] / 255, color[2] / 255, .98);
        });
      });
      gl.bindBuffer(gl.ARRAY_BUFFER, glBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(surfaceVertices), gl.STATIC_DRAW);
    }

    function makeTrajectory(type) {
      var points = [];
      var position = { x: 2.75, z: -2.55 };
      var velocity = { x: 0, z: 0 };
      var firstMoment = { x: 0, z: 0 };
      var secondMoment = { x: 0, z: 0 };
      for (var step = 1; step <= 112; step += 1) {
        points.push({ x: position.x, y: loss(position.x, position.z) + .045, z: position.z });
        var slope = gradient(position.x, position.z);
        if (type === 'sgd') {
          position.x -= slope.x * .075;
          position.z -= slope.z * .075;
        } else if (type === 'momentum') {
          velocity.x = .84 * velocity.x + slope.x;
          velocity.z = .84 * velocity.z + slope.z;
          position.x -= velocity.x * .027;
          position.z -= velocity.z * .027;
        } else if (type === 'adagrad') {
          secondMoment.x += slope.x * slope.x;
          secondMoment.z += slope.z * slope.z;
          position.x -= .2 * slope.x / (Math.sqrt(secondMoment.x) + .06);
          position.z -= .2 * slope.z / (Math.sqrt(secondMoment.z) + .06);
        } else if (type === 'rmsprop') {
          secondMoment.x = .9 * secondMoment.x + .1 * slope.x * slope.x;
          secondMoment.z = .9 * secondMoment.z + .1 * slope.z * slope.z;
          position.x -= .052 * slope.x / (Math.sqrt(secondMoment.x) + .09);
          position.z -= .052 * slope.z / (Math.sqrt(secondMoment.z) + .09);
        } else {
          firstMoment.x = .9 * firstMoment.x + .1 * slope.x;
          firstMoment.z = .9 * firstMoment.z + .1 * slope.z;
          secondMoment.x = .99 * secondMoment.x + .01 * slope.x * slope.x;
          secondMoment.z = .99 * secondMoment.z + .01 * slope.z * slope.z;
          var correctedX = firstMoment.x / (1 - Math.pow(.9, step));
          var correctedZ = firstMoment.z / (1 - Math.pow(.9, step));
          var varianceX = secondMoment.x / (1 - Math.pow(.99, step));
          var varianceZ = secondMoment.z / (1 - Math.pow(.99, step));
          var weightDecay = type === 'adamw' ? .022 : 0;
          position.x -= .058 * (correctedX / (Math.sqrt(varianceX) + .1) + weightDecay * position.x);
          position.z -= .058 * (correctedZ / (Math.sqrt(varianceZ) + .1) + weightDecay * position.z);
        }
      }
      return points;
    }

    Object.keys(optimizerStyles).forEach(function (type) { trajectories[type] = makeTrajectory(type); });

    function resize() {
      var rect = stage.getBoundingClientRect();
      width = Math.max(1, Math.round(rect.width));
      height = Math.max(1, Math.round(rect.height));
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      surfaceCanvas.width = Math.round(width * dpr);
      surfaceCanvas.height = Math.round(height * dpr);
      if (gl) gl.viewport(0, 0, surfaceCanvas.width, surfaceCanvas.height);
      surfaceLayer.width = Math.round(width * dpr);
      surfaceLayer.height = Math.round(height * dpr);
      surfaceContext.setTransform(dpr, 0, 0, dpr, 0, 0);
      surfaceDirty = true;
    }

    function project(node) {
      var cy = Math.cos(camera.yaw); var sy = Math.sin(camera.yaw);
      var cx = Math.cos(camera.pitch); var sx = Math.sin(camera.pitch);
      var rx = node.x * cy - node.z * sy;
      var rz = node.x * sy + node.z * cy;
      var centeredY = node.y - .75;
      var ry = centeredY * cx - rz * sx;
      var depth = centeredY * sx + rz * cx;
      var perspective = 1 / (1 + (depth + 4.7) * .035);
      var scale = Math.min(width, height) * (width < 640 ? .15 : .17);
      return { x: width * .5 + rx * scale * perspective, y: height * .52 - ry * scale * perspective, depth: depth, p: perspective };
    }

    function surfaceChannels(value) {
      var palette = [
        [64, 34, 125], [76, 67, 184], [47, 116, 207], [43, 190, 196],
        [75, 195, 128], [190, 221, 75], [240, 170, 75], [235, 99, 96]
      ];
      var normalized = Math.max(0, Math.min(1, (value - .04) / 2.45));
      var scaled = normalized * (palette.length - 1);
      var lower = Math.floor(scaled);
      var upper = Math.min(palette.length - 1, lower + 1);
      var mix = scaled - lower;
      return palette[lower].map(function (channel, index) { return channel + (palette[upper][index] - channel) * mix; });
    }

    function surfaceColor(value, alpha) {
      var color = surfaceChannels(value).map(function (channel) { return Math.round(channel); });
      return 'rgba(' + color.join(',') + ',' + alpha + ')';
    }

    function renderSurface() {
      if (gl) {
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(glProgram);
        gl.bindBuffer(gl.ARRAY_BUFFER, glBuffer);
        gl.enableVertexAttribArray(glPosition);
        gl.vertexAttribPointer(glPosition, 3, gl.FLOAT, false, 28, 0);
        gl.enableVertexAttribArray(glColor);
        gl.vertexAttribPointer(glColor, 4, gl.FLOAT, false, 28, 12);
        var scale = Math.min(width, height) * (width < 640 ? .15 : .17);
        gl.uniform1f(glPitch, camera.pitch);
        gl.uniform1f(glYaw, camera.yaw);
        gl.uniform2f(glScale, 2 * scale / width, 2 * scale / height);
        gl.drawArrays(gl.TRIANGLES, 0, cells.length * 6);
        surfaceDirty = false;
        return;
      }
      surfaceContext.clearRect(0, 0, width, height);
      var projectedCells = cells.map(function (cell) {
        var points = cell.map(project);
        return { cell: cell, points: points, depth: points.reduce(function (sum, point) { return sum + point.depth; }, 0) / 4 };
      }).sort(function (one, two) { return one.depth - two.depth; });

      projectedCells.forEach(function (entry) {
        var samples = entry.cell.map(function (point, index) { return { loss: point.y, point: entry.points[index] }; })
          .sort(function (one, two) { return one.loss - two.loss; });
        var low = samples[0];
        var high = samples[samples.length - 1];
        var fill = surfaceContext.createLinearGradient(low.point.x, low.point.y, high.point.x, high.point.y);
        fill.addColorStop(0, surfaceColor(low.loss, .95));
        fill.addColorStop(.5, surfaceColor((low.loss + high.loss) / 2, .95));
        fill.addColorStop(1, surfaceColor(high.loss, .95));
        surfaceContext.beginPath();
        surfaceContext.moveTo(entry.points[0].x, entry.points[0].y);
        for (var pointIndex = 1; pointIndex < entry.points.length; pointIndex += 1) surfaceContext.lineTo(entry.points[pointIndex].x, entry.points[pointIndex].y);
        surfaceContext.closePath();
        surfaceContext.fillStyle = fill;
        surfaceContext.fill();
        surfaceContext.strokeStyle = fill;
        surfaceContext.lineWidth = 1.4;
        surfaceContext.lineJoin = 'round';
        surfaceContext.stroke();
      });
      surfaceDirty = false;
    }

    function drawTrajectory(type, cycle) {
      var trajectory = trajectories[type];
      var style = optimizerStyles[type];
      var projectedTrajectory = trajectory.slice(0, Math.min(cycle + 1, trajectory.length)).map(project);
      if (!projectedTrajectory.length) return;
      context.beginPath();
      context.moveTo(projectedTrajectory[0].x, projectedTrajectory[0].y);
      projectedTrajectory.forEach(function (point) { context.lineTo(point.x, point.y); });
      context.strokeStyle = style.color;
      context.globalAlpha = activeOptimizer === 'all' ? .76 : .98;
      context.lineWidth = activeOptimizer === 'all' ? 1.8 : 2.8;
      context.stroke();
      context.globalAlpha = 1;
      var head = projectedTrajectory[projectedTrajectory.length - 1];
      var radius = activeOptimizer === 'all' ? 17 : 25;
      var glow = context.createRadialGradient(head.x, head.y, 0, head.x, head.y, radius);
      glow.addColorStop(0, 'rgba(' + style.rgb + ',1)');
      glow.addColorStop(.22, 'rgba(' + style.rgb + ',.9)');
      glow.addColorStop(1, 'rgba(' + style.rgb + ',0)');
      context.fillStyle = glow;
      context.beginPath(); context.arc(head.x, head.y, radius, 0, Math.PI * 2); context.fill();
      context.fillStyle = '#ffffff';
      context.beginPath(); context.arc(head.x, head.y, activeOptimizer === 'all' ? 3 : 4.2, 0, Math.PI * 2); context.fill();
    }

    function draw(now) {
      var previousPitch = camera.pitch;
      var previousYaw = camera.yaw;
      camera.pitch += (camera.targetPitch - camera.pitch) * .045;
      camera.yaw += (camera.targetYaw - camera.yaw) * .045;
      if (Math.abs(camera.pitch - previousPitch) > .00005 || Math.abs(camera.yaw - previousYaw) > .00005) surfaceDirty = true;
      context.clearRect(0, 0, width, height);
      if (surfaceDirty) renderSurface();
      if (!gl) context.drawImage(surfaceLayer, 0, 0, width, height);

      var cycle = reduceMotion ? 111 : Math.min(111, Math.floor(((now - start) % 11000) / 72));
      var visibleTypes = activeOptimizer === 'all' ? Object.keys(trajectories) : [activeOptimizer];
      visibleTypes.forEach(function (type) { drawTrajectory(type, cycle); });

      if (!reduceMotion && !document.hidden) window.requestAnimationFrame(draw);
    }

    resize();
    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(stage);
    window.addEventListener('resize', resize, { passive: true });
    if (!reduceMotion) {
      stage.addEventListener('pointerdown', function (event) {
        drag.active = true;
        drag.x = event.clientX;
        drag.y = event.clientY;
        stage.setPointerCapture(event.pointerId);
      });
      stage.addEventListener('pointermove', function (event) {
        if (!drag.active) return;
        camera.targetYaw += (event.clientX - drag.x) * .009;
        camera.targetPitch = Math.max(-1.32, Math.min(-.12, camera.targetPitch + (event.clientY - drag.y) * .007));
        drag.x = event.clientX;
        drag.y = event.clientY;
      });
      function stopDragging(event) {
        drag.active = false;
        if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
      }
      stage.addEventListener('pointerup', stopDragging);
      stage.addEventListener('pointercancel', stopDragging);
      document.addEventListener('visibilitychange', function () {
        if (!document.hidden) window.requestAnimationFrame(draw);
      });
    }
    document.querySelectorAll('[data-optimizer]').forEach(function (button) {
      button.addEventListener('click', function () {
        activeOptimizer = button.dataset.optimizer;
        start = performance.now();
        document.querySelectorAll('[data-optimizer]').forEach(function (candidate) {
          var selected = candidate.dataset.optimizer === activeOptimizer;
          candidate.classList.toggle('is-active', selected);
          candidate.setAttribute('aria-pressed', String(selected));
        });
      });
    });
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
