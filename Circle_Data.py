
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch
import numpy as np



class Circle(TensorDataset):
    def __init__(self, num_samples, radius, transform):  # we consider num_samples and radius as a list
        super(Circle, self).__init__()
        N = sum(num_samples)  # length of dataset
        self.data = np.zeros((N, 2))  # pre-allocated / N*2 should be in tuple / type of data: , dtype=np.uint8
        self.label = np.zeros((N,))  # better to define data type/google it/ , dtype=np.float32
        self.transform = transform

        rad2num = dict(zip(radius, np.arange(len(radius))))

        start = 0
        for id, (n, r) in enumerate(zip(num_samples,
                                        radius)):  # iterate over two lists in parallel/start: to get the index by using enumerate

            theta = np.linspace(0, 2 * np.pi, n, endpoint=False)  # shape of theta=n
            self.data[start: start + n] = np.concatenate([(r * np.cos(theta))[:, None], (r * np.sin(theta))[:, None]],
                                                         axis=1)
            self.transform = transform
            self.label[start: start + n] = rad2num[r] # it is correct.

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
        x = torch.Tensor()  ####### np.float32(x)
        return x
#
# B = Circle([10,15,20],[1,2,3],transform=ToTensor).data
# print(B)


