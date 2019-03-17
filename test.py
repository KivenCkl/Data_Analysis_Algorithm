from data_basic_func import inverse_normal_cdf, make_matrix, de_mean_matrix, shape
import random
import matplotlib.pyplot as plt
from pca import PCA
from kNN import kNN
from multiple_linear_regression import MLR
from logistic_regression import LR
from decision_tree import DecisionTree
from cluster import HierarchicalClustering
from not_quite_a_base import Table


def random_normal():
    return inverse_normal_cdf(random.random())


xs = [random_normal() for _ in range(1000)]
ys = [x + random_normal() / 2 for x in xs]

# plt.scatter(xs, ys)
# plt.show()

A = make_matrix(len(xs), 2, lambda i, j: xs[i] if j == 0 else ys[i])
A = de_mean_matrix(A)
print(shape(A))
# print(A)
pca = PCA()
components = pca.principal_component_analysis(A, 1)
print(components)
A_t = pca.transform(A, components)
a = pca._remove_projection(A, components[0])
plt.scatter([x[0] for x in a], [x[1] for x in a])
plt.show()

knn = kNN()
x = [-100 * random.random() for _ in range(100)]
y = [40 * random.random() for _ in range(100)]
lang = ['python', 'java', 'r', 'c']
z = [random.choice(lang) for _ in range(100)]
data = [([x_i, y_i], z_i) for x_i, y_i, z_i in zip(x, y, z)]
res = knn.knn_classify(3, data, [100, 40])
print(res)

x = [[1, 2, 5], [1, 3, 7], [1, 6, 12], [1, -5, -11]]
y = [7, 9, 16, -15]
mlr = MLR()
beta = mlr.estimate_beta(x, y)
r_square = mlr.multiple_r_squared(x, y, beta)
print(beta)
print(r_square)
beta = mlr.estimate_beta_ridge(x, y, 0)
r_square = mlr.multiple_r_squared(x, y, beta)
print(beta)
print(r_square)
new_x = [[1, 2, -2]]
print(mlr.predict(new_x, beta))

y = [1, 1, 1, 0]
lr = LR()
beta = lr.estimate_beta(x, y)
print(beta)
new_x = [[1, 2, -2]]
print(lr.predict(new_x, beta))

inputs = [({
    'level': 'Senior',
    'lang': 'Java',
    'tweets': 'no',
    'phd': 'no'
}, False),
          ({
              'level': 'Senior',
              'lang': 'Java',
              'tweets': 'no',
              'phd': 'yes'
          }, False),
          ({
              'level': 'Mid',
              'lang': 'Python',
              'tweets': 'no',
              'phd': 'no'
          }, True),
          ({
              'level': 'Junior',
              'lang': 'Python',
              'tweets': 'no',
              'phd': 'no'
          }, True),
          ({
              'level': 'Junior',
              'lang': 'R',
              'tweets': 'yes',
              'phd': 'no'
          }, True),
          ({
              'level': 'Junior',
              'lang': 'R',
              'tweets': 'yes',
              'phd': 'yes'
          }, False),
          ({
              'level': 'Mid',
              'lang': 'R',
              'tweets': 'yes',
              'phd': 'yes'
          }, True),
          ({
              'level': 'Senior',
              'lang': 'Python',
              'tweets': 'no',
              'phd': 'no'
          }, False),
          ({
              'level': 'Senior',
              'lang': 'R',
              'tweets': 'yes',
              'phd': 'no'
          }, True),
          ({
              'level': 'Junior',
              'lang': 'Python',
              'tweets': 'yes',
              'phd': 'no'
          }, True),
          ({
              'level': 'Senior',
              'lang': 'Python',
              'tweets': 'yes',
              'phd': 'yes'
          }, True),
          ({
              'level': 'Mid',
              'lang': 'Python',
              'tweets': 'no',
              'phd': 'yes'
          }, True),
          ({
              'level': 'Mid',
              'lang': 'Java',
              'tweets': 'yes',
              'phd': 'no'
          }, True),
          ({
              'level': 'Junior',
              'lang': 'Python',
              'tweets': 'no',
              'phd': 'yes'
          }, False)]
dt = DecisionTree()
tree = dt.build_tree_id3(inputs)
print(tree)

new_input = {'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}
label = dt.classify(tree, new_input)
print(label)

inputs = [[19, 28], [21, 27], [20, 23], [28, 13], [11, 15], [13, 13], [-49, 0],
          [-46, 5], [-41, 8], [-49, 15], [-34, -1], [-22, -16], [-19, -11],
          [-25, -9], [-11, -6], [-12, -8], [-14, -5], [-18, -3], [-13, -19],
          [-9, -16]]
hcl = HierarchicalClustering()
base_cluster = hcl.bottom_up_cluster(inputs)
print(base_cluster)
clusters = hcl.generate_clusters(base_cluster, 3)
print(clusters)

users = Table(["user_id", "name", "num_friends"])
users.insert([0, "Hero", 0])
users.insert([1, "Dunn", 2])
users.insert([2, "Sue", 3])
users.insert([3, "Chi", 3])
users.insert([4, "Thor", 3])
users.insert([5, "Clive", 2])
users.insert([6, "Hicks", 3])
users.insert([7, "Devin", 2])
users.insert([8, "Kate", 2])
users.insert([9, "Klein", 3])
users.insert([10, "Jen", 1])
print(users)
users.update(
    {
        'num_friends': 3
    },  # 设定 num_friends = 3
    lambda row: row['user_id'] == 1)  # 在user_id == 1的行中
print(users)
users.delete(lambda row: row['user_id'] == 1)
print(users)
new_users = users.limit(2)
print(new_users)
new_users = users.select(keep_columns=["user_id"])
print(new_users)
new_users = users.where(lambda row: row["name"] == "Dunn").select(
    keep_columns=["user_id"])
print(new_users)
new_users = users.select(
    additional_columns={"name_length": lambda row: len(row['name'])})
print(new_users)


def min_user_id(rows):
    return min(row["user_id"] for row in rows)


new_users = stats_by_length = users.select(additional_columns={
    "name_length": lambda row: len(row['name'])
}).group_by(
    group_by_columns=["name_length"],
    aggregates={
        "min_user_id": min_user_id,
        "num_users": len
    })
print(new_users)
