"""
神经网络(Neural Network)
------------------------
仅有一层隐藏层

对外两个方法：

1. feed_forward(neural_network, input_vector)
前馈神经网络输出

2. backpropagate(network, input_vector, targets)
反向传播训练神经网络
"""
import math
from data_basic_func import dot


class NN:
    def _sigmoid(self, z):
        """sigmoid 函数"""
        return 1 / (1 + math.exp(-z))

    def _neuron_output(self, weights, inputs):
        """神经元的输出"""
        return self._sigmoid(dot(weights, inputs))

    def feed_forward(self, neural_network, input_vector):
        """前馈神经网络输出

        Params
        ------
        neural_network: 包含每一层神经元的权值列表
        input_vector: 输入数据，不需要添加首元素为 1

        Return
        ------
        outputs: list
        """
        outputs = []

        # 每次处理一层
        for layer in neural_network:
            input_with_bias = input_vector + [1]  # 增加一个偏倚输入
            output = [
                self._neuron_output(neuron, input_with_bias)
                for neuron in layer
            ]
            outputs.append(output)

            # 然后下一层的输入就是这一层的输出
            input_vector = output

        return outputs

    def backpropagate(self, network, input_vector, targets):
        """反向传播训练神经网络

        Params
        ------
        network: 包含每一层神经元的权值列表
        input_vector: 输入数据，不需要添加首元素为 1
        targets: 目标值
        """
        hidden_outputs, outputs = self.feed_forward(network, input_vector)

        # output * (1 - output) 是 sigmoid 函数的导数
        output_deltas = [
            output * (1 - output) * (output - target)
            for output, target in zip(outputs, targets)
        ]

        # 对输出层神经元权值进行调整
        for i, output_neuron in enumerate(network[-1]):
            # 第 i 个神经元
            for j, hidden_output in enumerate(hidden_outputs + [1]):
                # 基于该神经元的残差和第 j 个输入调整第 j 个权值
                output_neuron[j] -= output_deltas[i] * hidden_output

        # 向隐藏层反向传递误差
        hidden_deltas = [
            hidden_output * (1 - hidden_output) * dot(
                output_deltas, [n[i] for n in network[-1]])
            for i, hidden_output in enumerate(hidden_outputs)
        ]

        # 调整隐藏层的神经元权值
        for i, hidden_neuron in enumerate(network[0]):
            for j, input in enumerate(input_vector + [1]):
                hidden_neuron[j] -= hidden_deltas[i] * input
