"""
数据分析预处理的一些基本函数
参考：《数据科学入门》
"""
import math
import random
from functools import reduce, partial
from collections import Counter, defaultdict


def vector_add(v, w):
    """向量 v 和 w 相加"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v, w):
    """向量 v 和 w 相减"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]


# 或者
# vector_sum = partial(reduce, vector_add)
def vector_sum(vectors):
    """一系列向量相加"""
    return reduce(vector_add, vectors)


def scalar_multiply(c, v):
    """c 是一个数，v 是一个向量"""
    return [c * v_i for v_i in v]


def vector_mean(vectors):
    """得到一个向量，其中第 i 个值是所有输入向量第 i 个元素的均值"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def dot(v, w):
    """向量 v 和 w 的点乘"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    """向量 v 的平方和"""
    return dot(v, v)


def magnitude(v):
    """计算向量的长度"""
    return math.sqrt(sum_of_squares(v))


def distance(v, w):
    """计算两个向量间的距离"""
    return magnitude(vector_subtract(v, w))


def squared_distance(v, w):
    """两个向量间的距离的平方"""
    return sum_of_squares(vector_subtract(v, w))


def shape(A):
    """返回矩阵 A 的形状"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_row(A, i):
    """返回矩阵 A 的第 i 行"""
    return A[i]


def get_column(A, j):
    """返回矩阵 A 的第 j 列"""
    return [A_i[j] for A_i in A]


def make_matrix(num_rows, num_cols, entry_fn):
    """构造一个第 [i, j] 个元素是 entry_fn(i, j) 的 num_rows * num_cols 矩阵"""
    return [
        [
            entry_fn(i, j)  # 根据 i 创建一个列表
            for j in range(num_cols)
        ]  # [entry_fn(i, 0), ... ]
        for i in range(num_rows)
    ]  # 为每一个 i 创建一个列表


def mean(x):
    """返回数据 x 的均值"""
    return sum(x) / len(x)


def median(x):
    """返回数据 x 的中位数"""
    n = len(x)
    sorted_x = sorted(x)
    midpoint = n // 2

    if n % 2 == 0:
        # 如果是奇数，返回中间值
        return sorted_x[midpoint]
    else:
        # 如果是偶数，返回中间两个值的均值
        lo = midpoint - 1
        hi = midpoint
        return (sorted_x[lo] + sorted_x[hi]) / 2


def quantile(x, p):
    """返回数据 x 的 p 分位数"""
    p_index = int(p * len(x))
    return sorted(x)[p_index]


def mode(x):
    """返回数据 x 的众数"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


def data_range(x):
    """返回数据 x 的极差，即最大元素与最小元素的差"""
    return max(x) - min(x)


def de_mean(x):
    """重构均值为 0 的数据"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]


def variance(x):
    """返回数据 x 的方差"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(x):
    """返回数据 x 的标准差"""
    return math.sqrt(variance(x))


def covariance(x, y):
    """返回数据 x, y 的协方差"""
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)


def correlation(x, y):
    """返回数据 x, y 的相关系数"""
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0


def normal_pdf(x, mu=0, sigma=1):
    """正态分布的概率密度函数"""
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x - mu)**2 / 2 / sigma**2) / sqrt_two_pi * sigma)


def normal_cdf(x, mu=0, sigma=1):
    """正态分布的累积分布函数"""
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """利用二分查找近似求取正态分布中特定概率下的相应值"""
    # 如果非标准型，先调整单位使之服从标准型
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    lo_z = -10.0  # normal_cdf(-10) 是(非常接近) 0
    hi_z = 10.0  # normal_cdf(10) 是(非常接近) 1
    while hi_z - lo_z > tolerance:
        mid_z = (lo_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            # midpoint 太低，搜索比它大的值
            lo_z = mid_z
        elif mid_p > p:
            # midpoint 太高，搜索比它小的值
            hi_z = mid_z
        else:
            break

    return mid_z


def difference_quotient(f, x, h):
    """定义单变量函数 f 在 x+h 点与 x 点的差商"""
    return (f(x + h) - f(x)) / h


def estimate_derivative(f, x, h=0.00001):
    """利用 h 趋于 0 的差商估算单变量函数 f 在 x 处的导数"""
    return difference_quotient(f, x, h)


def partial_difference_quotient(f, v, i, h):
    """计算多变量函数 f 在第 i 个变量 v_i 处的偏差商"""
    w = [
        v_j + (h if j == i else 0)  # 只对 v 的第 i 个元素增加 h
        for j, v_j in enumerate(v)
    ]

    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):
    """利用 h 趋于 0 的差商估算多变量函数 f 的梯度"""
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]


def correlation_matrix(data):
    """返回 num_columns x num_columns 的矩阵，其中第 [i, j] 元素为第 i 列和第 j 列的相关系数"""

    _, num_columns = shape(data)

    def matrix_entry(i, j):
        return correlation(get_column(data, i), get_column(data, j))

    return make_matrix(num_columns, num_columns, matrix_entry)


def group_by(grouper, rows, value_transform=None):
    """利用 grouper 函数对 rows 进行分组，并选择性地使用 value_transform 函数"""
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)

    if value_transform is None:
        return grouped
    else:
        return {key: value_transform(rows) for key, rows in grouped.items()}


def scale(data_matrix):
    """返回每列数据的均值和标准差"""
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix, j)) for j in range(num_cols)]
    stdevs = [
        standard_deviation(get_column(data_matrix, j)) for j in range(num_cols)
    ]
    return means, stdevs


def rescale(data_matrix):
    """调整原矩阵为每一列数据均值为 0 标准差为 1 的数据矩阵"""
    means, stdevs = scale(data_matrix)

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]

    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)


def de_mean_matrix(A):
    """重构每列数据均值均为 0 的数据矩阵"""
    nr, nc = shape(A)
    column_means, _ = scale(A)
    return make_matrix(nr, nc, lambda i, j: A[i][j] - column_means[j])


def direction(w):
    """返回单位方向向量"""
    mag = magnitude(w)
    return [w_i / mag for w_i in w]


def directional_variance_i(x_i, w):
    """向量 x_i 在非零向量 w 方向上的方差"""
    return dot(x_i, direction(w))**2


def directional_variance(X, w):
    """向量集 X 在非零向量 w 方向上的方差"""
    return sum(directional_variance_i(x_i, w) for x_i in X)


def directional_variance_gradient_i(x_i, w):
    """向量 x_i 在 w 方向向量上方差的梯度分量"""
    projection_length = dot(x_i, direction(w))
    return [2 * projection_length * x_ij for x_ij in x_i]


def directional_variance_gradient(X, w):
    """向量集 X 在 w 方向向量上方差的梯度"""
    return vector_sum(directional_variance_gradient_i(x_i, w) for x_i in X)


def project(v, w):
    """向量 v 在 w 方向向量上的投影"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)


def split_data(data, prob):
    """将数据划分为两部分，分别占 [prob, 1 - prob] 比例"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results


def train_test_split(x, y, test_pct):
    """将数据成对划分成训练集和测试集，其中测试集所占比例为 test_pct"""
    data = zip(x, y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test


def accuracy(tp, fp, fn, tn):
    """准确率"""
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


def precision(tp, fp, fn, tn):
    """查准率"""
    return tp / (tp + fp)


def recall(tp, fp, fn, tn):
    """查全率"""
    return tp / (tp + fn)


def f1_score(tp, fp, fn, tn):
    """F1 score"""
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)


def matrxi_product_entry(A, B, i, j):
    """矩阵乘法第 [i, j] 个元素的计算表达式"""
    return dot(get_row(A, i), get_column(B, j))


def matrix_multiply(A, B):
    """矩阵乘法"""
    n1, k1 = shape(A)
    n2, k2 = shape(B)
    if k1 != n2:
        return ArithmeticError("incompatible shapes!")

    return make_matrix(n1, k2, partial(matrxi_product_entry, A, B))
