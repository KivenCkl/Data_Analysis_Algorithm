"""
聚类(cluster)
-------------
两个类：

1. KMeans
k 均值算法(k-means)

    对外两个方法：

    1. classify(input)
    返回聚类输入最近的簇

    2. train(inputs)
    训练算法

2. HierarchicalClustering
自下而上的分层聚类

    对外两个方法：

    1. bottom_up_cluster(inputs, distance_agg=min)
    创建自下而上聚类算法

    2. generate_clusters(base_cluster, num_clusters)
    通过执行适当次数的分拆动作来产生任意数量的聚类
"""
import random
from data_basic_func import squared_distance, vector_mean, distance


class KMeans:
    def __init__(self, k):
        self.k = k  # 聚类的数目
        self.means = None  # 聚类的均值

    def classify(self, input):
        """返回聚类输入最近的簇

        Params
        ------
        input: 欲分类的数据

        Return
        ------
        int
        """
        return min(
            range(self.k),
            key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        """训练算法

        Params
        ------
        inputs: 输入数据
        """
        # 选择 k 个随机点作为初始的均值
        self.means = random.sample(inputs, self.k)
        assignments = None
        while True:
            # 查找新分配
            new_assignments = map(self.classify, inputs)

            # 如果所有数据点都不再被重新分配，那么就停止
            if assignments == new_assignments:
                return

            # 否则就重新分配
            assignments = new_assignments

            # 并基于新的分配计算新的均值
            for i in range(self.k):
                # 查找分配给聚类 i 的所有的点
                i_points = [p for p, a in zip(inputs, assignments) if a == i]

                # 确保 i_points 不是空的
                if i_points:
                    self.means[i] = vector_mean(i_points)


def squared_clustering_errors(inputs, k):
    """计算总误差"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)

    return sum(
        squared_distance(input, means[cluster])
        for input, cluster in zip(inputs, assignments))


class HierarchicalClustering:
    def _is_leaf(self, cluster):
        """当该类长度为 1 时即为叶"""
        return len(cluster) == 1

    def _get_children(self, cluster):
        """当该类是一个合并的类，就返回其两个子节点，当该类为叶时，报错"""
        if self._is_leaf(cluster):
            raise TypeError("a leaf cluster has no children")
        else:
            return cluster[1]

    def _get_values(self, cluster):
        """如果该类为叶，则返回其值，否则，返回其下所有叶的值"""
        if self._is_leaf(cluster):
            return cluster  # 已经是一个包含值的一元组
        else:
            return [
                value for child in self._get_children(cluster)
                for value in self._get_values(child)
            ]

    def _cluster_distance(self, cluster1, cluster2, distance_agg=min):
        """计算类 1 和类 2 的距离，并对结果应用 distance_agg，
        当其为 min 时，易产生巨大的链式聚类，聚类间挨得不是很紧，
        当其为 max 时，利用得到紧凑的球状聚类"""
        return distance_agg([
            distance(input1, input2) for input1 in self._get_values(cluster1)
            for input2 in self._get_values(cluster2)
        ])

    def _get_merge_order(self, cluster):
        """合并次序，该数字越小，表示合并的次序越靠后"""
        if self._is_leaf(cluster):
            return float('inf')
        else:
            return cluster[0]  # merge_order 是二元组中的第一个元素

    def bottom_up_cluster(self, inputs, distance_agg=min):
        """创建自下而上聚类算法

        Params
        ------
        inputs: 输入数据，列表
        distance_agg: 合并标准

        Return
        ------
        base_cluster: 包含所有类
        """
        # 最开始每个输入都是一个叶聚类/一元组
        clusters = [(input, ) for input in inputs]

        # 只要剩余一个以上的聚类
        while len(clusters) > 1:
            # 就找出最近的两个聚类
            c1, c2 = min(
                [(cluster1, cluster2) for i, cluster1 in enumerate(clusters)
                 for cluster2 in clusters[:i]],
                key=lambda x: self._cluster_distance(x[0], x[1], distance_agg))

            # 从聚类列表中将它们移除
            clusters = [c for c in clusters if c != c1 and c != c2]

            # 使用 merge_order = 剩余聚类的数目来合并它们
            merged_cluster = (len(clusters), [c1, c2])

            # 并添加它们的合并
            clusters.append(merged_cluster)

        # 当只剩一个聚类时，返回它
        return clusters[0]

    def generate_clusters(self, base_cluster, num_clusters):
        """通过执行适当次数的分拆动作来产生任意数量的聚类

        Params
        ------
        base_cluster: 由 bottom_up_cluster 函数产生的类
        num_clusters: 欲生成类的数目

        Return
        ------
        clusters: 满足数目的分类
        """
        # 开始的列表只有基本聚类
        clusters = [base_cluster]

        while len(clusters) < num_clusters:
            # 选择上一个合并的聚类
            next_cluster = min(clusters, key=self._get_merge_order)
            # 将它从列表中移除
            clusters = [c for c in clusters if c != next_cluster]
            # 并将它的子聚类添加到列表中（即拆分它）
            clusters.extend(self._get_children(next_cluster))

        return clusters
