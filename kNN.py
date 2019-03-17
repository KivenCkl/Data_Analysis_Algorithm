"""
k 最近邻算法 (kNN)
-----------------
对外一个方法：

1. knn_classify(k, labeled_points, new_point)
kNN 分类器
"""
from collections import Counter
from data_basic_func import distance


class kNN:
    def _majority_vote(self, labels):
        """减少 k 值直到找到唯一的获胜者，假设 labels 按距离从近到远排列"""
        vote_counts = Counter(labels)
        winner, winner_count = vote_counts.most_common(1)[0]
        num_winners = len(
            [count for count in vote_counts.values() if count == winner_count])
        if num_winners == 1:
            return winner  # 唯一的获胜者，返回它的值
        else:
            return self._majority_vote(labels[:-1])  # 去掉最远元素，再次尝试

    def knn_classify(self, k, labeled_points, new_point):
        """kNN 分类器，每个 labeled_points 都应该为 (point, label) 数据对形式

        Params
        ------
        k: 选择 k 个最近点
        labeled_points: 已知数据集，每个元素均为 (point, label) 形式
        new_point: 需要预测的数据

        Return
        ------
        label: 预测数据的类别
        """

        # 把标记好的点按从最近到最远的顺序排序
        by_distance = sorted(
            labeled_points, key=lambda data: distance(data[0], new_point))

        # 寻找 k 个最近邻的标签
        k_nearest_labels = [label for _, label in by_distance[:k]]

        # 然后让它们投票
        return self._majority_vote(k_nearest_labels)
