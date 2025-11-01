'''
1. 단계를 하나씩 불러와서 2**단계 만큼 격자를 만듬
2. 격자를 90도 회전
3. 상하좌우 검사해서 얼음양 1 줄어들기

4. matrix에 남아있는 얼음의 합 구하기
5. 남아있는 얼음중 가장 큰 덩어리 칸 개수 구하기
    DFS로 구하기 -> 메모리초과
    BFS사용
''' 
import sys
sys.setrecursionlimit(10**6) 
from collections import deque

N, Q = map(int,input().split())
matrix = []
for _ in range(2**N):
    matrix.append(list(map(int,input().split())))
Level = list(map(int,input().split()))
# print(matrix)
# print(Level)
dirs = [(-1,0),(1,0),(0,-1),(0,1)]

#Level에서 하나씩 가져와서 격자마다 90도 회전
for L in Level:
    # for i in range(0,2**N,2**L):
    #     for j in range(0,2**N,2**L):
    #         temp = []
    #         for k in range(2**L):
    #             row = matrix[i+k][j:j+2**L]
    #             temp.append(row)
    #         rotated_temp = [list(r) for r in zip(*temp[::-1])]
    #         for r in range(2**L):
    #             for c in range(2**L):
    #                 matrix[i+r][j+c] = rotated_temp[r][c]
        # 1. 회전 (메모리 최적화)
    # 임시 격자를 한 번만 생성하여 회전 결과를 저장
    sub_size = 2**L
    grid_size = 2**N
    new_matrix = [[0] * grid_size for _ in range(grid_size)]
    for i in range(0, grid_size, sub_size):
        for j in range(0, grid_size, sub_size):
            for r in range(sub_size):
                for c in range(sub_size):
                    # 90도 회전: (r, c) -> (c, sub_size - 1 - r)
                    new_matrix[i + c][j + sub_size - 1 - r] = matrix[i + r][j + c]
    matrix = new_matrix
    
    matrix_copy = [row[:] for row in matrix]
    #인접한 4칸중 얼음과 인접해있는지 검사
    # melt_memory = []
    for i in range(2**N):
        for j in range(2**N):
            cnt = 0
            for dir in dirs:
                #좌표 유효성 검사
                if 0<=i+dir[0]<2**N and 0<=j+dir[1]<2**N:
                    if matrix_copy[i+dir[0]][j+dir[1]]>0: #얼음이 있으면
                        cnt+=1
                else: #좌표가 유효하지 않으면 pass
                    pass
            if cnt>=3: #인접해있는 얼음이 3이상이면 pass
                pass
            else: #인접해있는 얼음이 3미만이면 값 1 빼기
                if matrix[i][j]>0:
                    matrix[i][j] -= 1
                else:
                    pass
    #             melt_memory.append((i,j))
    # for melt in melt_memory:
    #     if matrix[melt[0]][melt[1]]>0:
    #         matrix[melt[0]][melt[1]] -= 1
         
#matrix의 얼음양 다 더하기, 가장 큰 덩어리 dfs로 찾기
visited = [[False for _ in range(2**N)]for _ in range(2**N)]
# def dfs(r,c):
#     #이미 방문했거나, 좌표가 유효하지 않거나,얼음이 없으면 0 리턴하기
#     if not (0<=r<2**N and 0<=c<2**N) or visited[r][c] or matrix[r][c]==0:
#         return 0
#     visited[r][c] = True
#     count = 1
#     for dir in dirs:
#         count+=dfs(r+dir[0],c+dir[1])
#     return count

def bfs(r,c):
    queue = deque([(r,c)])
    visited[r][c]=True
    count = 1
    
    while queue:
        now = queue.popleft()
        for dir in dirs:
            if not (0<=now[0]+dir[0]<2**N and 0<=now[1]+dir[1]<2**N) or visited[now[0]+dir[0]][now[1]+dir[1]] or not matrix[now[0]+dir[0]][now[1]+dir[1]]>0:
                continue
            visited[now[0]+dir[0]][now[1]+dir[1]] = True
            queue.append((now[0]+dir[0],now[1]+dir[1]))
            count += 1
    return count

max_size = 0
now_size = 0 
sum = 0
for i in range(2**N):
    for j in range(2**N):
        sum+=matrix[i][j]
        if matrix[i][j]>0 and not visited[i][j]:
            now_size = bfs(i,j)
            if max_size<now_size: #max값 업데이트
                max_size = now_size
print(sum)
print(max_size)