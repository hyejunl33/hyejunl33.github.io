---
layout: archive
title: "Study"
collection: study
permalink: /study/
---

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in site.study %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>