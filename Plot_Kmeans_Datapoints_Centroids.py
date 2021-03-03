import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
from kmeans_updated_first import kmeans

features, true_labels = make_moons(n_samples = 250, noise = 0.05, random_state = 42)
features, true_labels = torch.from_numpy(features), torch.from_numpy(true_labels)
print(features.shape)
print(true_labels.shape)
a ,b = kmeans(features, 2)

fig, ax = plt.subplots(figsize=(9, 7))
plt.scatter(features[:, 0], features[:, 1], c=a, cmap='cool', marker='o')  # a is the label getting from kmeans
#
plt.scatter(b[:, 0], b[:, 1], marker='x')  # b is the centroid getting from kmeans
plt.show()
