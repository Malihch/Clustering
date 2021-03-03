
from torch.utils.data import TensorDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Sphere(TensorDataset):
    def __init__(self, num_samples, radius, transform):  # we consider num_samples and radius as a list
        super(Sphere, self).__init__()

        N = sum(num_samples)     # length of dataset

        self.data = np.zeros((N, 3))   # pre-allocated / N,3 should be in tuple
        self.label = np.zeros((N,))
        self.transform = transform

        rad2num = dict(zip(radius, np.arange(len(radius))))

        start = 0

        for id, (n, r) in enumerate(zip(num_samples, radius)):   #iterate over two lists in parallel/id: to get the index by using enumerate

            theta = np.linspace(0, 2 * np.pi, n, endpoint=False) # shape of theta=n
            phi = np.linspace(0, np.pi, n, endpoint=False)

            self.data[start: start+n] = np.concatenate([(r * np.sin(phi) * np.cos(theta))[:, None],
                                                        (r * np.sin(phi) * np.sin(theta))[:, None],
                                                        (r * np.cos(phi)) [:, None]],
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


#
# AAA = Sphere([100], [1], transform=ToTensor())
# print(AAA.data.shape)
# dataset = Sphere([100, 150, 200], [1, 2, 3], transform=ToTensor())
#
# Dat = Sphere([100, 150, 200], [1, 2, 3], transform=ToTensor()).data
# print(Dat)
# Lab =Sphere([100, 150, 200], [1, 2, 3], transform=ToTensor()).label
# print(Lab)
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Dat[:, 0], Dat[:, 1], Dat[:, 2], cmap=None, c=Lab)
# plt.show()

