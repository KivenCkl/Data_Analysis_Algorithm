"""
多元线性回归算法(MLR)
--------------------
对外四个方法：

1. estimate_beta(x, y)
利用随机梯度下降法寻找最优的 beta

2. multiple_r_squared(x, y, beta)
评价指标，确定系数(R-square)

3. estimate_beta_ridge(x, y, alpha)
利用随机梯度下降寻找岭回归(ridge regression)的最优回归系数 beta

4. predict(new_x, beta)
预测新数据集
"""
import random
from data_basic_func import dot, de_mean, sum_of_squares, vector_add
from gradient_descent import GradientDescent
from functools import partial


class MLR:
    def __init__(self):
        gd = GradientDescent()
        self.minimize_stochastic = gd.minimize_stochastic

    def _predict(self, x_i, beta):
        """多元线性回归预测，x 为第一个元素为 1 长度为 n(变元数量) + 1 的向量"""
        return dot(x_i, beta)

    def _error(self, x_i, y_i, beta):
        """多元线性回归的误差"""
        return y_i - self._predict(x_i, beta)

    def _squared_error(self, x_i, y_i, beta):
        """误差平方"""
        return self._error(x_i, y_i, beta)**2

    def _squared_error_gradient(self, x_i, y_i, beta):
        """误差平方项对应关于 beta 的梯度"""
        return [-2 * x_ij * self._error(x_i, y_i, beta) for x_ij in x_i]

    def estimate_beta(self, x, y):
        """利用随机梯度下降法寻找最优的 beta

        Params
        ------
        x: 输入形状为 (m x n+1) 的 x 列表
        y: 输入长度为 m 的 y 列表

        Return
        ------
        beta: 返回多元线性回归系数 beta 列表，长度为 n + 1
        """
        beta_initial = [random.random() for _ in x[0]]
        return self.minimize_stochastic(self._squared_error,
                                        self._squared_error_gradient, x, y,
                                        beta_initial, 0.001)

    def _total_sum_of_squares(self, y):
        """数据 y 与均值偏差的平方和"""
        return sum_of_squares(de_mean(y))

    def multiple_r_squared(self, x, y, beta):
        """评价指标，确定系数(R-square)

        Params
        ------
        x: 输入形状为 (m x n+1) 的 x 列表
        y: 输入长度为 m 的 y 列表
        beta: 输入长度为 n + 1 的回归系数

        Return
        ------
        r_squared: float
        """
        sum_of_squared_errors = sum(
            self._error(x_i, y_i, beta)**2 for x_i, y_i in zip(x, y))
        return 1.0 - sum_of_squared_errors / self._total_sum_of_squares(y)

    def _ridge_penalty(self, beta, alpha):
        """惩罚项，beta 中第一项为常数，不惩罚，alpha 为惩罚系数"""
        return alpha * sum_of_squares(beta[1:])

    def _squared_error_ridge(self, x_i, y_i, beta, alpha):
        """误差平方加上惩罚项"""
        return self._error(x_i, y_i, beta)**2 + self._ridge_penalty(
            beta, alpha)

    def _ridge_penalty_gradient(self, beta, alpha):
        """惩罚项的梯度"""
        return [0] + [2 * alpha * beta_j for beta_j in beta[1:]]

    def _squared_error_ridge_gradient(self, x_i, y_i, beta, alpha):
        """包含惩罚项的误差平方的对应梯度"""
        return vector_add(
            self._squared_error_gradient(x_i, y_i, beta),
            self._ridge_penalty_gradient(beta, alpha))

    def estimate_beta_ridge(self, x, y, alpha):
        """利用随机梯度下降寻找岭回归(ridge regression)的最优回归系数 beta

        Params
        ------
        x: 输入形状为 (m x n+1) 的 x 列表
        y: 输入长度为 m 的 y 列表
        alpha: 惩罚系数，当为 0 时，等效于 estimate_beta 方法

        Return
        ------
        beta: 返回多元线性回归系数 beta 列表，长度为 n + 1
        """
        beta_initial = [random.random() for _ in x[0]]
        return self.minimize_stochastic(
            partial(self._squared_error_ridge, alpha=alpha),
            partial(self._squared_error_ridge_gradient, alpha=alpha), x, y,
            beta_initial, 0.001)

    def predict(self, new_x, beta):
        """预测新数据集

        Params
        ------
        new_x: 新数据集，形状为 (m x n+1)
        beta: 多元线性回归系数 beta 列表，长度为 n + 1

        Return
        ------
        y: 长度为 m 的列表
        """
        return [self._predict(new_x_i, beta) for new_x_i in new_x]
