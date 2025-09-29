---
layout: single
title: "[Programmers/Python] Level 3: 양과 늑대"
categories:
  - algorithm
tags:
  - [Python, Programmers, DFS, graph]
toc: true
toc_sticky: true
---
# 문제: 양과 늑대

![image](/assets/images/2025-09-29-22-20-02.png)
![image](/assets/images/2025-09-29-22-20-08.png)
![image](/assets/images/2025-09-29-22-20-11.png)

## 풀이

```python
def solution(info, edges):
    answer = 0
    '''
    늑대>=양 -> 양 몰살
    최대한 많은 양 모아서 다시 루트노드로 돌아오기
    info: 그래프
    edges: 노드간의 연결관계
    
    1. edge리스트 graph로 만들기 -> info에서 각 node 확인해서 count
    2. DFS로 늑대, 양 개수 count하면서 경우의 수 탐색
    3. 양 최댓값 return
    '''
    # 인접 리스트로 그래프 생성
    graph = [[] for _ in range(len(info))]
    for i in range(len(edges)):
        graph[edges[i][0]].append(edges[i][1])
    # print(graph)
    max_ = 0
    
    def DFS(sheep, wolf, node):
        nonlocal max_
        
        #max 갱신
        if sheep > max_:
            max_ = sheep
        
        for n in node:
            next_node = [k for k in node if k != n] + graph[n]
            # 다음 노드가 양일때
            if info[n] == 0:
                DFS(sheep + 1, wolf, next_node)
            else: #다음노드가 늑대일때
                if sheep > wolf +1: #늑대 1추가해도 양보다 더 적을때만 탐색
                    DFS(sheep, wolf +1, next_node)
    DFS(1,0,graph[0])
    return max_
               
```

## 배울점

1. 인접리스트로 그래프 만든다음에 DFS로 탐색하는 문제
2. 전형적인 그래프 탐색문제긴 했는데, 양숫자랑, 늑대 숫자를 DFS로 넘겨줘서 제한조건을 걸어줘야 하는 문제였음
3. `next_node`로 다음에 탐색해야 하는 `node` 는 자신을 제외한 이번 DFS의 node와 자신과 연결된 node인 `graph[n]` 임.