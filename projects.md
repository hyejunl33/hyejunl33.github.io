---
layout: archive
title: "Projects"
collection: projects
permalink: /projects/
---

{% assign sorted_posts = site.projects | sort: 'date' | reverse %}

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in sorted_posts %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>