---
layout: archive
title: "Algorithm"
collection: algorithm
permalink: /algorithm/
---

{% assign sorted_posts = site.algorithm | sort: 'date' | reverse %}

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in sorted_posts %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>