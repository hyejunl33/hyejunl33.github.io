---
layout: page
title: "Projects"
permalink: /projects/
---

<div class="entries-{{ page.entries_layout | default: 'list' }}">
{% for post in site.projects %}
    {% include archive-single.html type="grid" %}
{% endfor %}
</div>