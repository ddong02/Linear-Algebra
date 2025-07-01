### 행렬곱을 수행하는 코드

import numpy as np

def multiplyMatrix(Mat1, Mat2):
    mat1_row, mat1_col = Mat1.shape
    mat2_row, mat2_col = Mat2.shape

    if (mat1_col != mat2_row):
        print("행렬 크기 오류")
        return -1
    else:
        output = np.empty((mat1_row, mat2_col), dtype=Mat1.dtype)

        for i in range(mat1_row):
            for j in range(mat2_col):
                for k in range(mat1_col):
                    output[i][j] += Mat1[i][k] * Mat2[k][j]

        return output

mat1 = np.array([[1,2], [3,4], [5,6], [7,8]])
mat2 = np.array([[1,2,3,4], [5,6,7,8]])

output = multiplyMatrix(mat1, mat2)

print(f'mat1 size = {mat1.shape}, mat2 size = {mat2.shape}')

print("\n### multiply Matrix with numpy")
print('np.matmul(mat1, mat2)=')
print(np.matmul(mat1, mat2))
print()
print("### multiply Matrix with my function")
print('multiplyMatrix(mat1, mat2)=')
print(output)