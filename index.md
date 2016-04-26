---
layout: default
title: Lihuanghai's blog
---
<div class="posts">
{% for post in site.posts %}
  <div class="post">
  <h1 class="post-title">
  <a href="{{post.url}}">
  {{post.title}}
  </a>
  </h1>
  <span class="post-date">{{ post.date | date_to_string }}</span>
  {{post.content || split:'!-- more -->' | first }}
  <h7><a href="{{ post.url title='阅读全文'}}">阅读全文</a></h7>
  </div>
  {% endfor %}
  </div>
