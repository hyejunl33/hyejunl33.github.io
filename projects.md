---
layout: page
title: "Projects List"
permalink: /projects/
---

## My Projects

<hr>

<ul>
  {% for project in site.projects %}
    <li style="margin-bottom: 2em;">
      <h3><a href="{{ site.baseurl }}{{ project.url }}">{{ project.title }}</a></h3>
      <p><strong>Published on:</strong> {{ project.date | date: "%B %d, %Y" }}</p>
      <p>{{ project.excerpt }}</p>
    </li>
  {% endfor %}
</ul>