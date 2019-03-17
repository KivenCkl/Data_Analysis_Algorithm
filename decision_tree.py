"""
决策树(decision tree)
---------------------
对外三个方法：

1. entropy(class_probabilities)
给定类别所占比，计算该信息熵

2. build_tree_id3(_inputs, split_candidates=None)
利用训练数据建立决策数的具体表示形式

3. classify(tree, _input)
应用给定决策树对输入进行分类
"""
import math
from collections import Counter, defaultdict
from functools import partial


class DecisionTree:
    def entropy(self, class_probabilities):
        """计算信息熵

        Params
        ------
        class_probabilities: 给定类别数据所占比例

        Return
        ------
        float
        """
        return sum(-p * math.log(p, 2) for p in class_probabilities if p)

    def _class_probabilities(self, labels):
        """计算各类别的概率"""
        total_count = len(labels)
        return [count / total_count for count in Counter(labels).values()]

    def _data_entropy(self, labeled_data):
        """计算数据信息熵"""
        labels = [label for _, label in labeled_data]
        probabilities = self._class_probabilities(labels)
        return self.entropy(probabilities)

    def _partition_entropy(self, subsets):
        """将数据划分后的熵"""
        total_count = sum(len(subset) for subset in subsets)
        return sum(
            self._data_entropy(subset) * len(subset) / total_count
            for subset in subsets)

    def _partition_by(self, _inputs, attribute):
        """将输入按属性分组，输入为 (attribute_dict, label)，
        返回字典：attribute_value -> _inputs"""
        groups = defaultdict(list)
        for _input in _inputs:
            key = _input[0][attribute]  # 得到特定属性的值
            groups[key].append(_input)  # 然后把这个输入加到正确的列表中
        return groups

    def _partition_entropy_by(self, _inputs, attribute):
        """计算对应划分的熵"""
        partitions = self._partition_by(_inputs, attribute)
        return self._partition_entropy(partitions.values())

    def classify(self, tree, _input):
        """应用给定决策树对输入进行分类

        Params
        ------
        tree: 决策树，为元组 (attribute, subtree_dict) 形式
        _input: 需要预测的输入数据，为 attributes_dict

        Return
        ------
        label: True or False
        """

        # 如果这是一个叶节点，则返回其值
        if tree in [True, False]:
            return tree

        # 否则这个树就包含一个需要划分的属性和一个字典，
        # 字典的键是那个属性的值，值是下一步需要考虑的子树
        attribute, subtree_dict = tree

        subtree_key = _input.get(attribute)  # 如果输入的是缺失的属性，则返回 None

        if subtree_key not in subtree_dict:  # 如果键没有子树
            subtree_key = None  # 则需要用到 None 子树

        subtree = subtree_dict[subtree_key]  # 选择恰当的子树
        return self.classify(subtree, _input)  # 并用它来对输入分类

    def build_tree_id3(self, _inputs, split_candidates=None):
        """利用训练数据建立决策数的具体表示形式

        Params
        ------
        _inputs: 输入数据列表，每个列表元素形式为 (attributes_dict, label)
        split_candidates: 默认为 None

        Return
        ------
        tree: 决策树
        """

        # 如果这是第一步，第一次输入的所有的键就都是 split_candidates
        if split_candidates is None:
            split_candidates = _inputs[0][0].keys()

        # 对输入里的 True 和 False 计数
        num_inputs = len(_inputs)
        num_trues = len([label for _, label in _inputs if label])
        num_falses = num_inputs - num_trues

        if num_trues == 0:
            return False  # 若没有 True，则返回一个 "False" 叶节点
        if num_falses == 0:
            return True  # 若没有 False，则返回一个 "True" 叶节点

        if not split_candidates:  # 若不再有 split_candidates
            return num_trues >= num_falses  # 则返回多数叶节点

        # 否则在最好的属性上进行划分
        best_attribute = min(
            split_candidates, key=partial(self._partition_entropy_by, _inputs))

        partitions = self._partition_by(_inputs, best_attribute)
        new_candidates = [a for a in split_candidates if a != best_attribute]

        # 递归地创建子树
        subtrees = {
            attribute_value: self.build_tree_id3(subset, new_candidates)
            for attribute_value, subset in partitions.items()
        }

        subtrees[None] = num_trues > num_falses  # 默认情况

        return (best_attribute, subtrees)
