---
layout: archive
title: "Weekly Review"
collection: weeklyreview
permalink: /weeklyreview/
---

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in site.weeklyreview | sort: 'date' | reverse %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>