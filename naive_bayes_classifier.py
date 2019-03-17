"""
朴素贝叶斯分类器
---------------
以垃圾邮件过滤器举例说明该算法思想

对外两个方法：

1. train(training_set)
训练朴素贝叶斯分类器

2. classify(message)
利用训练好的数据对 message 进行预测
"""
import re
import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def _tokenize(self, message):
        """解析邮件中的单词"""
        message = message.lower()  # 转换为小写
        all_words = re.findall("[a-z0-9']+", message)  # 提取单词
        return set(all_words)  # 移除副本

    def _count_words(self, training_set):
        """统计单词出现在已做标记的邮件训练集中的次数，
        training_set 包含 (message, is_spam) 数据对"""
        counts = defaultdict(lambda: [0, 0])
        for message, is_spam in training_set:
            for word in self._tokenize(message):
                counts[word][0 if is_spam else 1] += 1
        return counts

    def _word_probabilities(self, counts, total_spams, total_non_spams, k=0.5):
        """利用平滑技术将这些技术转换为估计概率，返回一个列表，列表元素包含
        w, p(w | spam), p(w | ~spam)"""
        return [(w, (spam + k) / (total_spams + 2 * k),
                 (non_spam + k) / (total_non_spams + 2 * k))
                for w, (spam, non_spam) in counts.items()]

    def _spam_probability(self, word_probs, message):
        """给邮件赋予概率"""
        message_words = self._tokenize(message)
        log_prob_if_spam = log_prob_if_not_spam = 0.0

        # 迭代词汇表中的每一个单词
        for word, prob_if_spam, prob_if_not_spam in word_probs:

            # 如果 *word* 出现在了邮件中，则增加看到它的对数概率
            if word in message_words:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_not_spam += math.log(prob_if_not_spam)

            # 如果 *word* 没有出现在邮件中，则增加看不到它的对数概率
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_not_spam = math.exp(log_prob_if_not_spam)
        return prob_if_spam / (prob_if_spam + prob_if_not_spam)

    def train(self, training_set):
        """训练朴素贝叶斯分类器"""

        # 对垃圾邮件和非垃圾邮件计数
        num_spams = len([is_spam for _, is_spam in training_set if is_spam])
        num_non_spams = len(training_set) - num_spams

        # 运行训练数据
        word_counts = self._count_words(training_set)
        self.word_probs = self._word_probabilities(word_counts, num_spams,
                                                   num_non_spams, self.k)

    def classify(self, message):
        """朴素贝叶斯分类器

        Params
        ------
        message: 输入邮件

        Return
        ------
        prob: 返回该邮件是垃圾邮件的概率
        """
        return self._spam_probability(self.word_probs, message)
