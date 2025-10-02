---
layout: archive
title: "Algorithm"
collection: algorithm
permalink: /algorithm/
---

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in site.algorithm | sort: 'date' | reverse %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>