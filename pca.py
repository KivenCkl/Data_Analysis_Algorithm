"""
主成分分析算法 (PCA)
-------------------
从数据中提取出一个或多个维度，以捕获数据中尽可能多的变差。

对外两个方法：

1. principal_component_analysis(X, num_components, method=None)
返回数据 X 中任意数目的主成分

2. transform(X, components)
将原数据转换为由主成分生成的低维空间的点

参考：《数据科学入门》
"""
from functools import partial
from data_basic_func import direction, directional_variance, directional_variance_gradient, directional_variance_i, directional_variance_gradient_i, project, vector_subtract, dot
from gradient_descent import GradientDescent


class PCA:
    def __init__(self):
        gd = GradientDescent()
        self.maximize_batch = gd.maximize_batch
        self.maximize_stochastic = gd.maximize_stochastic

    def _first_principal_component(self, X):
        """利用完全梯度下降法提取数据 X 的第一主成分"""
        guess = [1 for _ in X[0]]
        unscaled_maximizer = self.maximize_batch(
            partial(directional_variance, X),
            partial(directional_variance_gradient, X), guess)
        return direction(unscaled_maximizer)

    def _first_principal_component_sgd(self, X):
        """利用随机梯度下降法提取数据 X 的第一主成分"""
        guess = [1 for _ in X[0]]
        unscaled_maximizer = self.maximize_stochastic(
            lambda x, _, w: directional_variance_i(x, w),
            lambda x, _, w: directional_variance_gradient_i(x, w), X,
            [None for _ in X], guess)
        return direction(unscaled_maximizer)

    def _remove_projection_from_vector(self, v, w):
        """从数据 v 中移除 v 在方向向量 w 上的投影"""
        return vector_subtract(v, project(v, w))

    def _remove_projection(self, X, w):
        """对向量集中每一个向量都移除在 w 方向上的投影"""
        return [self._remove_projection_from_vector(x_i, w) for x_i in X]

    def principal_component_analysis(self, X, num_components, method=None):
        """返回数据 X 中任意数目的主成分

        Params
        ------
        X: 输入数据集
        num_components: 所需提取主成分的数目
        method: 计算方法，默认为完全梯度下降，'sgd' 为随机梯度下降

        Return
        ------
        components: 主成分列表
        """
        components = []
        method_map = {
            None: self._first_principal_component,
            'sgd': self._first_principal_component_sgd
        }
        try:
            first_principal_component = method_map[method]
        except:
            print("无效的方法！请输入 'sgd' 或不输入！")
        for _ in range(num_components):
            component = first_principal_component(X)
            components.append(component)
            X = self._remove_projection(X, component)

        return components

    def _transform_vector(self, v, components):
        return [dot(v, w) for w in components]

    def transform(self, X, components):
        """将原数据转换为由主成分生成的低维空间的点

        Params
        ------
        X: 原数据集
        components: 主成分生成的空间

        Return
        ------
        list: 转换后的数据
        """
        return [self._transform_vector(x_i, components) for x_i in X]
