### 여인수(cofactor)를 이용하여 행렬식(determinant)을 구하는 코드

# 실제로 컴퓨터에서 행렬식을 구할 때는 여인수를 이용하지 않고 LU분해를 이용해서 구한다.
# 그 이유는 여인수 전개를 통한 행렬식을 구하는 과정의 계산량이 LU분해를 이용할 때보다
# 훨씬 많기 때문이다. (여인수전개 → O(n!), LU분해 → O(n^3) 가우스 소거법과 계산량이 같다.)
# 그리고 LU분해는 행렬식도 얻을 수 있지만, 선형시스템의 해를 구할 때도 사용할 수 있어 더 효율적이다. (재사용 가능)

# A = [[a,b], [c,d]] 라고 했을 때, 여인수 전개를 이용해서 행렬식을 얻으려면 ad - bc 2번의 곱셈 계산이 필요하다.
# 크기가 3x3 일때는 3x2 (2x2 크기의 행렬식 계산을 3번해야함),
# 4x4 일때는 4x3x2 (3x3 크기의 행렬식 계산을 4번) ··· → 계산량이 팩토리얼 크기로 증가한다.

import numpy as np

# 여인수 전개를 통해 행렬식을 구하는 함수
def det_cofactor_expansion(np_array):
    if (np_array.shape[0] != np_array.shape[1]):
        print("정방행렬만 가능")
        return
    size = np_array.shape[0]
    if (size == 2):
        return ((np_array[0][0] * np_array[1][1]) - (np_array[0][1] * np_array[1][0]))
    elif (size == 3):
        C1 = np_array[0][0] * (np_array[1][1]*np_array[2][2] - np_array[1][2]*np_array[2][1])
        C2 = -np_array[0][1] * (np_array[1][0]*np_array[2][2] - np_array[1][2]*np_array[2][0])
        C3 = np_array[0][2] * (np_array[1][0]*np_array[2][1] - np_array[1][1]*np_array[2][0])

        return (C1 + C2 + C3)
    else:
        print("행렬의 크기는 2 또는 3")
        return
    
# 행렬을 LU분해하는 함수
def LU_decomposition(np_array):
    if (np_array.shape[0] != np_array.shape[1]):
        print("정방행렬만 가능")
        return
    n = np_array.shape[0]

    # U -> A에 가우스소거법을 적용해 상삼각행렬(U)을 만들어 저장할 행렬
    U = np_array.astype(np.float64).copy()
    # A → U로 바뀌는 연산을 이용하여 하삼각행렬(L)로 만들 단위행렬
    L = np.eye(n, dtype=np.float64)

    # 각 열을 순회하며 피벗팅 및 소거 진행 → 가우스 소거법과 같은 방식
    for j in range(n - 1):  # 열(column) 기준
        for i in range(j + 1, n):  # 해당 열의 아래 행들 (피벗 밑의 행들)
            # 승수(multiplier) 계산
            # U[i, j]를 0으로 만들기 위한 계수
            multiplier = U[i, j] / U[j, j]

            # L 행렬에 승수 저장
            L[i, j] = multiplier

            # U 행렬의 행 업데이트 (가우스 소거)
            U[i, :] -= multiplier * U[j, :]
    return L, U

if __name__ == '__main__':
    A = np.array([
        [1,2,-1],
        [-2,0,7],
        [3,0,7]
    ])

    det_A = det_cofactor_expansion(A)

    print(f"A =\n{A}\n")

    print(f'### Det using cofactor expansion\n')

    print(f'det of A -> {det_A}')
    print()
    print(f'np.linalg.det(A) = {np.linalg.det(A)}')

    print('-' * 30)
    print('### Det using LU decomposition\n')

    L, U = LU_decomposition(A)
    print(f'L = \n{L}\nU = \n{U}')
    print(f'\nL@U = \n{L@U}\n')
    detL = np.prod(np.diag(L))
    detU = np.prod(np.diag(U))

    print(f'Det(L) = {detL}')
    print(f'Det(U) = {detU}\n')
    # Det(L)Det(U) = Det(LU) = Det(A)
    print(f'Det(A) = Det(L)Det(U) = {detL * detU}')