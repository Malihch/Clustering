import torch
import numpy as np
from kmeans_pytorch import kmeans
from Sphere_Data import Sphere, ToTensor
import matplotlib.pyplot as plt
# data
# train_dataset = Sphere([100,150,200],[1, 2, 3], transform=ToTensor())

data_size, dims, num_clusters = 1000, 2, 3

x = np.random.randn(data_size, dims) / 6

x = torch.from_numpy(x)
# KMEANS = kmeans(x, 3)



# kmeans
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean')


fig, ax = plt.subplots(figsize=(9, 7))
ax.set_title('Encoded Data', fontsize=18, fontweight='demi')
ax.scatter(x[:, 0], x[:, 1], c = cluster_ids_x,s=None, cmap=None)
plt.show()