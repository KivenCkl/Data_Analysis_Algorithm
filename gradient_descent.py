"""
梯度下降算法
-----------
对外四个方法：

1. minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001)
利用完全梯度下降寻找使得目标函数最小的 theta

2. maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001)
利用完全梯度下降寻找使得目标函数最大的 theta

3. minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01)
利用随机梯度下降寻找使得目标函数最小的 theta

4. maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01)
利用随机梯度下降寻找使得目标函数最大的 theta
"""
import random
from data_basic_func import vector_subtract, scalar_multiply


class GradientDescent:
    def _step(self, v, direction, step_size):
        """v 沿着 direction 方向移动 step_size"""
        return [
            v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)
        ]

    def _safe(self, f):
        """防止某些步长可能导致函数的输入无效"""

        def safe_f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except:
                return float('inf')  # Python 中的“无限大”

        return safe_f

    def minimize_batch(self,
                       target_fn,
                       gradient_fn,
                       theta_0,
                       tolerance=0.000001):
        """利用梯度下降来寻找使得目标函数最小的 theta

        Params
        ------
        target_fn: 目标函数
        gradient_fn: 计算梯度函数
        theta_0: 初始 theta 值
        tolerance: 设定误差值

        Return
        ------
        theta: float
        """

        step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        theta = theta_0  # 设定 theta 为初始值
        target_fn = self._safe(target_fn)
        value = target_fn(theta)

        while True:
            gradient = gradient_fn(theta)
            next_thetas = [
                self._step(theta, gradient, -step_size)
                for step_size in step_sizes
            ]

            # 选择一个使残差函数最小的值
            next_theta = min(next_thetas, key=target_fn)
            next_value = target_fn(next_theta)

            # 当收敛时停止
            if abs(value - next_value) < tolerance:
                return theta
            else:
                theta, value = next_theta, next_value

    def _negate(self, f):
        """返回一个函数，使得对于任一一个输入 x 都有 -f(x)"""
        return lambda *args, **kwargs: -f(*args, **kwargs)

    def _negate_all(self, f):
        """当 f 返回多个值时，都取为相反数"""
        return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

    def maximize_batch(self,
                       target_fn,
                       gradient_fn,
                       theta_0,
                       tolerance=0.000001):
        """利用梯度下降来寻找使得目标函数最大的 theta

        Params
        ------
        target_fn: 目标函数
        gradient_fn: 计算梯度函数
        theta_0: 初始 theta 值
        tolerance: 设定误差值

        Return
        ------
        theta: float
        """
        return self.minimize_batch(
            self._negate(target_fn), self._negate_all(gradient_fn), theta_0,
            tolerance)

    def _in_random_order(self, data):
        """生成随机序列数据"""
        indexes = [i for i, _ in enumerate(data)]  # 生成索引列表
        random.shuffle(indexes)  # 随机打乱数据
        for i in indexes:
            yield data[i]

    def minimize_stochastic(self,
                            target_fn,
                            gradient_fn,
                            x,
                            y,
                            theta_0,
                            alpha_0=0.01):
        """利用随机梯度下降来寻找使得目标函数最小的 theta

        Params
        ------
        target_fn: 目标函数
        gradient_fn: 计算梯度函数
        x: 输入 x 值
        y: 输入 y 值
        theta_0: 初始 theta 值
        alpha_0: 初始步长

        Return
        ------
        theta: float
        """
        data = zip(x, y)
        theta = theta_0
        alpha = alpha_0  # 初始步长
        min_theta, min_value = None, float('inf')  # 当前最小值
        iterations_with_no_improvement = 0

        # 如果循环超过 100 次仍无改进，则停止
        while iterations_with_no_improvement < 100:
            value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

            if value < min_value:
                # 如果找到新的最小值，则记录下来
                # 并返回到最初的步长
                min_theta, min_value = theta, value
                iterations_with_no_improvement = 0
                alpha = alpha_0
            else:
                # 尝试缩小步长，否则没有改进
                iterations_with_no_improvement += 1
                alpha *= 0.9

            # 在每一个数据点上向梯度方向前进一步
            for x_i, y_i in self._in_random_order(data):
                gradient_i = gradient_fn(x_i, y_i, theta)
                theta = vector_subtract(theta,
                                        scalar_multiply(alpha, gradient_i))

        return min_theta

    def maximize_stochastic(self,
                            target_fn,
                            gradient_fn,
                            x,
                            y,
                            theta_0,
                            alpha_0=0.01):
        """利用随机梯度下降来寻找使得目标函数最大的 theta

        Params
        ------
        target_fn: 目标函数
        gradient_fn: 计算梯度函数
        x: 输入 x 值
        y: 输入 y 值
        theta_0: 初始 theta 值
        alpha_0: 初始步长

        Return
        ------
        theta: float
        """
        return self.minimize_stochastic(
            self._negate(target_fn), self._negate_all(gradient_fn), x, y,
            theta_0, alpha_0)
