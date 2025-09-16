---
layout: single
title: "[백준/Python] 16234번: 인구 이동"
categories:
  - algorithm
tags:
  - [Python, BOJ, Gold5, BFS, Simulation]
toc: true
toc_sticky: true
---
# 문제: 인구이동

N×N크기의 땅이 있고, 땅은 1×1개의 칸으로 나누어져 있다. 각각의 땅에는 나라가 하나씩 존재하며, r행 c열에 있는 나라에는 A[r][c]명이 살고 있다. 인접한 나라 사이에는 국경선이 존재한다. 모든 나라는 1×1 크기이기 때문에, 모든 
국경선은 정사각형 형태이다.

오늘부터 인구 이동이 시작되는 날이다.

인구 이동은 하루 동안 다음과 같이 진행되고, 더 이상 아래 방법에 의해 인구 이동이 없을 때까지 지속된다.

- 국경선을 공유하는 두 나라의 인구 차이가 L명 이상, R명 이하라면, 두 나라가 공유하는 국경선을오늘 하루 동안 연다.
- 위의 조건에 의해 열어야하는 국경선이 모두 열렸다면, 인구 이동을 시작한다.
- 국경선이 열려있어 인접한 칸만을 이용해 이동할 수 있으면, 그 나라를 오늘 하루 동안은 연합이라고 한다.
- 연합을 이루고 있는 각 칸의 인구수는 (연합의 인구수) / (연합을 이루고 있는 칸의 개수)가 된다. 편의상 소수점은 버린다.
- 연합을 해체하고, 모든 국경선을 닫는다.

각 나라의 인구수가 주어졌을 때, 인구 이동이 며칠 동안 발생하는지 구하는 프로그램을 작성하시오.

## 입력

첫째 줄에 N, L, R이 주어진다. (1 ≤ N ≤ 50, 1 ≤ L ≤ R ≤ 100)

둘째 줄부터 N개의 줄에 각 나라의 인구수가 주어진다. r행 c열에 주어지는 정수는 A[r][c]의 값이다. (0 ≤ A[r][c] ≤ 100)

인구 이동이 발생하는 일수가 2,000번 보다 작거나 같은 입력만 주어진다.

## 출력

인구 이동이 며칠 동안 발생하는지 첫째 줄에 출력한다.

## 내 풀이
```python
import sys
input = sys.stdin.readline
from collections import deque

A = []
N, L, R=map(int, input().split())

#A 리스트 값 불러오기
for _ in range(N):
    A.append(list(map(int, input().split())))

'''
L이상 R이하 사이의 인구차이라면 국경열기

1. BFS로 A리스트 돌면서 상하좌우에 L이상 R이하인지 확인 -> 연합 찾기
2. 연합 다 찾았으면(큐가 비면) 인구 재분배
3. 모든칸에 대해 반복
4. 더이상 인구이동이 없다면 종료 -> 반복한 횟수 출력
'''
dx = [0,0,-1,1]
dy = [1,-1,0,0]
day = 0

while 1:
    chk = 0
    visited = [[False] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            if not visited[r][c]:
                people_sum = 0
                open_country = []
                
                queue = deque([(r,c)])
                visited[r][c] = True
                people_sum+=A[r][c]
                open_country.append((r,c))
                
                while queue:
                    now_r, now_c = queue.popleft()
                    
                    for k in range(4):
                        x, y = now_r + dx[k], now_c + dy[k]
                    
                        #좌표가 유효하고 방문하지 않았다면
                        if 0<=x<N and 0<=y<N and not visited[x][y]:
                            delta = abs(A[now_r][now_c] - A[x][y])
                            #인구 차가 L이상 R이하라면
                            if L <= delta <= R:
                                chk = 1
                                visited[x][y] = True
                                queue.append((x,y))
                                open_country.append((x,y))
                                people_sum += A[x][y]
              
                if len(open_country)>1:
                    mean_people = people_sum // len(open_country)
                    for m,n in open_country:
                        A[m][n] = mean_people
                    
    if chk == 0:
        break
    #더이상 연합이 없었다면 -> chk = 0 -> while문 break
    day += 1
    
print(day)
```

## 배울점

1. visited리스트가 있으니깐, 굳이 list에 현재상태 저장해두고 한번에 업데이트 할필요 없었다.
2. pypy3로 제출하면 시간초과가 안뜬다..
3. 전형적인 BFS문제 → BFS로 검사해서 연합할 수 있는애들 리스트에 모아서 한번에 연합해주고 계산해주기 → 각 날마다 visited는 초기화해야됨을 유의해야됨 → 각BFS마다 visited는 새로생성
4. `input = sys.stdin.readline` 를 인풋으로 써야 더 빠름