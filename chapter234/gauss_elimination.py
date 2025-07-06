### 가우스 소거법
# 가우스 소거법은 다음 2단계로 진행된다.
# 1단계 전방 소거법(Forward Elimination)
# 2단계 후방 대입법(Back substitution)

# 알고리즘
# 1. 첫번째 행을 기준으로 시작으로 하여 현재 행의 아래에 있는 모든 행의 피벗을 0으로 만든다.
# 계산을 시작하기 전에 첫번째 원소가 0인 행은 아래쪽으로 가도록 한다.
# (0에 뭘 곱하든 0이기 때문에 다른 행의 피벗을 0으로 만들 수 없기 때문이다.)
# 첫번째 행을 시작으로 하여 맨 마지막 행의 바로 직전 행까지 반복을 수행한다.
# 현재 행과 다음 행의 피벗의 절대값 비교하여 그 다음 행의 피벗이 더 크면 두 행의 순서를 변경 (swap)
# 이 과정을 통해 피벗을 순서대로 나열하진 않아도 피벗이 0인 행은 아래쪽에 모여있게 된다.

# ---> 이 방법의 문제점은 피벗값이 최대가 아닌 행의 피벗으로 나누게 되어 정석 알고리즘보다 정확도가 떨어진다는 점이다.
# 정석 알고리즘은 피벗값이 최대인 행이 가장 위로 올라가도록 한다.

# 미지수의 개수와 방정식의 개수가 같다고 가정 (계수 행렬이 정방행렬)
# 유일한 해가 존재 한다고 가정 (0x = 0 이거나 1x = 0 과 같은 상황 x)

import numpy as np

# 계수 행렬 A
A = np.array(
    [[0, 1, 1, 2, 1],
    [4, -6, 0, 3, 4],
    [-2, 7, 2, -2, 1],
    [2, 4, 5, 2, 1],
    [-1, 2, 0, 3, 2]
    ], dtype=float
)

# 상수 행렬 b
b = np.array(
    [[5],
    [-2],
    [9],
    [-1],
    [3]
    ], dtype=float
)

# x 변수 행렬 x
size = len(A)
x = np.zeros((size, 1), dtype=float)

# numpy의 선형방정식의 해를 구하는 함수
print(f'\nnp.linalg.solve(A, b) = \n{np.linalg.solve(A, b)}\n')

# 1. 계수 행렬 A를 순회하며 상삼각행렬 꼴로 바꾼다. (전방 소거법))
for i in range(size):
    # 행렬 A의 주대각원소와 그 아래에 위치한 원소들만 처리
    for j in range(i, size - 1):
        # 현재 행의 피벗과 그 다음 행의 피벗을 비교
        pivot = abs(A[j][i])
        next_pivot = abs(A[j+1][i])

        # 다음 행의 피벗이 더 크다면 행의 위치를 서로 바꿈 (상수 행렬도 바꾼다)
        if pivot < next_pivot:
            temp = A[j].copy()
            A[j] = A[j+1]
            A[j+1] = temp

            temp2 = b[j].copy()
            b[j] = b[j+1]
            b[j+1] = temp2
    
    # 정렬된 행렬의 (i,i) 값으로 나머지 행(i+1, size)의 첫번쨰 원소를 0으로 만든다.
    for k in range(i+1, size):
        factor = (A[k][i] / A[i][i])
        A[k][i:] -= factor * A[i][i:]
        b[k] -= factor * b[i]

# 상삼각행렬의 꼴로 변환된 계수행렬과 상수행렬 출력
print('convert A into REF')
print(f'A = \n{A}')
print(f'b = \n{b}')

# 변환된 계수 행렬의 마지막 행을 이용해 x행렬의 마지막 행의 값을 얻는다 x[size-1]
x[size - 1] = b[size - 1] / A[size - 1][size - 1]

# 이 값을 이용해서 역으로 대입하며 나머지 x값들을 찾는다.

# 2. 마지막 행의 직전 행부터 첫번째 행까지 역순으로 반복 (역 대입법)
for i in range(size - 2, -1, -1):
    # 현재 구하려는 x값의 다음 행의 x값은 이미 구했기 때문에 이를 이용해서 현재 x값을 구한다.
    sum_ax = 0
    for j in range(i+1, size):
        sum_ax += A[i, j] * x[j]
    x[i] = (b[i] - sum_ax) / A[i][i]

print(f'\n\n <calculated by my function>')
print(f'x = \n{x}')