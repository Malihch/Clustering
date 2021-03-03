import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import torch
import numpy as np
from matplotlib import pyplot as plt
#
#
class Sphere2(TensorDataset):
    def __init__(self, num1, num2, radius, transform):  # we consider num1, num2, and radius as a list
        super(Sphere2, self).__init__()
        products = [a * b for a, b in zip(num1, num2)]
        N = sum(products)     # length of dataset

        self.data = np.zeros((N, 3))   # pre-allocated / N,3 should be in tuple
        self.label = np.zeros((N,))
        self.transform = transform

        rad2num = dict(zip(radius, np.arange(len(radius)))) #convert radius into number(index)
        start = 0

        for id, (n1, n2, n, r) in enumerate(zip(num1, num2, products, radius)):   #iterate over two lists in parallel/id: to get the index by using enumerate

            phi = np.linspace(0, np.pi, n1)
            theta = np.linspace(0, 2 * np.pi, n2)

            self.data[start: start+n] = np.concatenate([np.reshape(r * np.outer(np.sin(theta), np.cos(phi)), (n1*n2, 1)),
                                                     np.reshape(r * np.outer(np.sin(theta), np.sin(phi)), (n1*n2, 1)),
                                                      np.reshape(r * np.outer(np.cos(theta), np.ones_like(phi)), (n1*n2, 1))],
                                                axis=1)

            self.label[start: start+n] = rad2num[r]

            start += n

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, label   ###  (data, label) shows that it returns a tuple

    def __len__(self):

        return len(self.data)

class ToTensor(object):
                def __call__(self, x):
                    x = torch.from_numpy(np.float32(x))   ### torch.Tensor(np.float32(x))
                    return x

# SF = Sphere2([10,20,30],[30,20,10], [1, 2, 3], transform=ToTensor())
# Dat = SF.data
# print(Dat.shape)
# # print(Dat.shape)
# Lab = SF.label
# print(Lab)
# # print(Lab.shape)
# #
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Dat[:, 0], Dat[:, 1], Dat[:, 2], cmap=None, c=Lab)
# plt.show()
# #
#
# #
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Dat[:, 0], Dat[:, 1], Dat[:, 2], cmap=None)
# plt.show()
# n,r = 10,2
# phi = np.linspace(0, np.pi, n)
# theta = np.linspace(0, 2 * np.pi, n)
#
# dataa = np.concatenate([np.reshape(r * np.outer(np.sin(theta), np.cos(phi)), (n, 1)),
#                                                      np.reshape(r * np.outer(np.sin(theta), np.sin(phi)), (n, 1)),
#                                                       np.reshape(r * np.outer(np.cos(theta), np.ones_like(phi)), (n, 1))],
#                                                 axis=1)
