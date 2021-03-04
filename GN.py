import numpy as np
import math
from scipy.linalg import ldl
np.random.seed(1)

# 定义真实参数值
a = 1.0
b = 2.0
c = 1.0
# 任意选择的初始参数值
ae = 2.0
be = -1.0
ce = 5.0


# 建立正态分布，生成100个随机数点，这里的方差设置为3，横坐标的取值范围为[1, 100]
def get_random_point(num=100, cov=1):
    sample_list = {}
    real_list = {}
    for i in range(num):
        # 原函数表达式
        x = i / 100
        real_val = math.exp(a * x * x + b * x + c)
        # 将real_val作为均值，从一元正态分布中采样
        sample_val = np.random.normal(loc=real_val, scale=cov, size=1)[0]
        sample_list[str(x)] = sample_val
        real_list[str(x)] = real_val
    return sample_list, real_list


# 设置高斯牛顿函数，求解非线性最小二乘问题
def GA_model(samples, matrix_x):
    matrix_delta_x = np.zeros(shape=(3, 1))
    J = np.zeros(shape=(len(samples), 3))
    e = np.zeros(shape=(len(samples), 1))
    ax = matrix_x[0, 0]
    bx = matrix_x[1, 0]
    cx = matrix_x[2, 0]
    for i in range(len(samples)):

        # 设计雅可比矩阵，求解da，db，dc放入该m*3的矩阵中
        x = i / 100
        da = - x * x * math.exp(ax * x * x + bx * x + cx)  # 对a进行求导
        db = - x * math.exp(ax * x * x + bx * x + cx)  # 对b进行求导
        dc = - math.exp(ax * x * x + bx * x + cx)  # 对c进行求导
        J[i, 0] = da
        J[i, 1] = db
        J[i, 2] = dc
        # 计算误差
        error = samples[str(x)] - math.exp(ax * x * x + bx * x + cx)
        # 记录真实值与测量值之间的误差
        e[i, 0] = error

    cost = np.sum(e)
    print(cost)
    # 计算更新值矩阵，为一个3*1的矩阵
    H = np.dot(J.T, J)
    b = -np.dot(J.T, e)

    # 也可以使用cholesy分解来求解deltax
    l = np.linalg.cholesky(H)
    y = np.linalg.inv(l).dot(b)
    result = np.linalg.inv(l.T).dot(y)

    update = np.dot(np.linalg.inv(H), b)
    return update
  

# 设置模型
def model(iterations=100):
    matrix_x = np.array([[ae], [be], [ce]])
    sample_list, real_list = get_random_point()
    for i in range(iterations):
        update = GA_model(sample_list, matrix_x)
        matrix_x += update
    return matrix_x

k = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
L = np.linalg.cholesky(k)
print(L)
print(L.dot(L.T))
lu, d, perm = ldl(k, lower=False)
print(lu)
print(lu.dot(d).dot(lu.T))

matrix_x = model()
print(matrix_x)
