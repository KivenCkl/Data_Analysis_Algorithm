"""
逻辑回归算法(LR)
---------------
对外两个方法：

1. estimate_beta(x, y, method=None)
利用梯度下降法寻找使似然函数最大的最优 beta

2. predict(new_x, beta, prob=0.5)
预测数据属于 0 或 1 类别
"""
import math
import random
from data_basic_func import dot, vector_add
from functools import reduce, partial
from gradient_descent import GradientDescent


class LR:
    def __init__(self):
        gd = GradientDescent()
        self.maximize_stochastic = gd.maximize_stochastic
        self.maximize_batch = gd.maximize_batch

    def _logistic(self, x):
        """逻辑函数(sigmoid)"""
        return 1.0 / (1 + math.exp(-x))

    def _logistic_prime(self, x):
        """逻辑函数的导数"""
        return self._logistic(x) * (1 - self._logistic(x))

    def _logistic_log_likelihood_i(self, x_i, y_i, beta):
        """逻辑函数的对数似然函数"""
        if y_i == 1:
            return math.log(self._logistic(dot(x_i, beta)))
        else:
            return math.log(1 - self._logistic(dot(x_i, beta)))

    def _logistic_log_likelihood(self, x, y, beta):
        """数据集整体的对数似然"""
        return sum(
            self._logistic_log_likelihood_i(x_i, y_i, beta)
            for x_i, y_i in zip(x, y))

    def _logistic_log_partial_ij(self, x_i, y_i, beta, j):
        """对数似然的梯度分量"""
        return (y_i - self._logistic(dot(x_i, beta))) * x_i[j]

    def _logistic_log_gradient_i(self, x_i, y_i, beta):
        """对数似然的梯度"""
        return [
            self._logistic_log_partial_ij(x_i, y_i, beta, j)
            for j, _ in enumerate(beta)
        ]

    def _logistic_log_gradient(self, x, y, beta):
        """数据集整体的对数似然梯度"""
        return reduce(vector_add, [
            self._logistic_log_gradient_i(x_i, j_i, beta)
            for x_i, j_i in zip(x, y)
        ])

    def estimate_beta(self, x, y, method=None):
        """利用梯度下降法寻找使似然函数最大的最优 beta

        Params
        ------
        x: 输入形状为 (m x n+1) 的 x 列表
        y: 输入长度为 m 的 y 列表
        method: 计算方法，默认为完全梯度下降，'sgd' 为随机梯度下降

        Return
        ------
        beta: 返回逻辑回归系数 beta 列表，长度为 n + 1
        """
        beta_initial = [random.random() for _ in x[0]]
        if method is None:
            return self.maximize_batch(
                partial(self._logistic_log_likelihood, x, y),
                partial(self._logistic_log_gradient, x, y), beta_initial)
        elif method == 'sgd':
            return self.maximize_stochastic(self._logistic_log_likelihood_i,
                                            self._logistic_log_gradient_i, x,
                                            y, beta_initial)
        else:
            print("无效的方法！请输入 'sgd' 或不输入！")

    def predict(self, new_x, beta, prob=0.5):
        """预测数据属于 0 或 1 类别

        Params
        ------
        new_x: 新数据集，形状为 (m x n+1)
        beta: 逻辑回归系数 beta 列表，长度为 n + 1
        prob: 临界值，默认为 0.5，当大于等于 prob 时属于类 1，反之属于类 0

        Return
        ------
        y: 长度为 m 的列表，其中元素为 0 或 1
        """
        y = []
        for new_x_i in new_x:
            prediction = self._logistic(dot(new_x_i, beta))
            if prediction >= prob:
                y.append(1)
            else:
                y.append(0)

        return y
