---
layout: archive
title: "Weekly Review"
collection: weeklyreview
permalink: /weeklyreview/
---

{% assign sorted_posts = site.weeklyreview | sort: 'date' | reverse %}

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in sorted_posts %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>