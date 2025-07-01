### 전치행렬을 구하는 코드

# 알고리즘
# 1. 행렬의 크기 파악 (run only when, rows >= 2 and cols >= 2)
#    row size → len(input), col size → len(input[0])
#    혹은 numpy의 shape attribute 이용하기 → input.shape (튜플 형태)
# 2. 입력 받은 행렬의 크기를 이용해 결과를 저장할 행렬 만들기 (input_ij → output_ji)
# 3. for 반복문을 이용해 입력 받은 행렬의 0~(i-1) , 0~(j-1) 까지 순회하며
#    결과 행렬의 (j, i) 위치에 저장하기

import numpy as np

def getTranspose(np_array):
    origin_row = len(np_array)
    origin_col = len(np_array[0])

    if (origin_row < 2 or origin_col < 2):
        print(f"minimun matrix size is 2x2")
        return -1
    else:
        T = np.empty((origin_col, origin_row), dtype=np_array.dtype)
        for row in range(origin_row):
            for col in range(origin_col):
                T[col][row] = np_array[row][col]
        
        return T
    

input_M = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

print(f"input matrix shape is {input_M.shape}")
print("input matrix is ...")
print(input_M)
print()

output = getTranspose(input_M)

print(f"output matrix shape is {output.shape}")
print("output matrix is ...")
print(output)
print()