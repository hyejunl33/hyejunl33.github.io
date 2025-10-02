---
layout: archive
title: "Projects"
collection: projects
permalink: /projects/
---

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in site.study | sort: 'date' | reverse %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>