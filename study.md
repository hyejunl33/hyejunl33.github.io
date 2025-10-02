---
layout: archive
title: "Study"
collection: study
permalink: /study/
---

{% assign sorted_posts = site.study | sort: 'date' | reverse %}

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in sorted_posts %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>