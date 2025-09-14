---
layout: archive
title: "Recent Posts"
---

{% comment %}{% endcomment %}
{% assign all_posts = site.posts | concat: site.projects | concat: site.study | concat: site.algorithm | concat: site.weeklyreview %}

{% comment %}{% endcomment %}
{% assign sorted_posts = all_posts | sort: 'date' | reverse %}

<div class="entries-{{ page.entries_layout | default: 'list' }}">
  {% for post in sorted_posts limit:10 %}
    {% include archive-single.html type=page.entries_layout %}
  {% endfor %}
</div>