---
layout: single
title: "[백준/Python] 22862: 가장 긴 짝수 연속한 부분 수열"
categories:
  - algorithm
tags:
  - [Python, 백준, two pointer]
toc: true
toc_sticky: true
---
# 문제: 가장 긴 짝수 연속한 부분 수열

![](/assets/images/2025-09-18-11-22-58.png)
![](/assets/images/2025-09-18-11-23-14.png)

## 풀이

```python
import sys
input = sys.stdin.readline

# N: 수열의 길이
N, K=map(int, input().split())

S = list(map(int,input().split()))
#max는 최대 수열길이
max = 0
'''
N까지의 수 중에서 K번 삭제한 후 

만약 N//2 <= K
N=6, K=3 ,2,4,6
N=6, K=4  2  4  6 -> 답은 N//2
N=7, K=4 ,2,,4,,6,

N=8, K=4 ,2,,4,,6,8

N//2 > K
N = 10, k = 4
1,2,,4,,6,,8,,10 -> 답은 k+1

S가 정렬되어있지않을 수도  -> 근데 정렬은 상관없느듯
짝수 뭉탱이가 중간에 숨어있을 수 있음 -> 찾아야됨

홀수 K개 지울 수 있음 -> 투포인터로 홀수 찾으면서 K개 될때까지

두개의 포인터 안에 홀수가 K보다 작을때
    end포인터 +1
두개의 포인터 안에 홀수가 K보다 클떄
    st포인터 +1
두개의 포인터 안에 홀수의 수 = K일때
    end 포인터, st +1
    if end - st > max:
        max = end-st
'''
#end랑 st는 투포인터
end = 0
st = 0
# print(N//2 if N//2<=K else K+1)
length=len(S)
odd = 0

#매번 odd를 세니깐 시간초과
# odd를 저장해두고,포인터 이동에따라 1씩 더하거나 빼기
while end < length:
    # for i in S[st:end+1]:
    #     if i%2 == 1:
    #         odd +=1
    
    #홀수라면 odd +=1
    if S[end]%2 == 1:
        odd +=1

    # if odd < K:
    #     if end - st + 1 - odd > max:
    #         max = end-st+1 - odd
    #     end +=1
    while odd > K:
        if S[st] % 2 == 1:
            odd-=1
        st += 1
    #odd <=K일때 max길이 검사
    if end - st +1 - odd > max:
        max = end-st+1 - odd
    end+=1

# print(S)
print(max)                    
```

## 배울점

1. 어렴풋이 알고는 있었는데, 당연히 while문 안에서 매번 수열 S를 슬라이싱해서 홀수개수를 세면 시간초과뜸 → 이럴경우에 정수로 개수를 센다음에 while 루프마다 정수 +-1씩 업데이트 해가면서 개수를 세는게 맞음 → 약간 DP table만들어서 이전 정보 활용하는 느낌 → 매번계산x
2. odd = K일때 max길이 검사해야 하는것은 알았는데, odd<K에도 max길이 검사해야하는것을 놓쳤음 → 홀수 수는 K보다 크지만 않으면 상관없음.