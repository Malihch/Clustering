import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
# from Circle_Data import Circle, ToTensor
# radius = ["Neha","Timo","Avani"]

# num2rad= dict(zip(np.arange(len(radius)),radius))

 #is a dictionary
class Circle(TensorDataset):
    def __init__(self, num_samples, radius, transform=None):  # we consider num_samples and radius as a list
        super(Circle, self).__init__()
        N = sum(num_samples)     # length of dataset
        self.data = np.zeros((N, 2))   # pre-allocated / N*2 should be in tuple/ data type: , dtype=np.uint8
        self.label = np.zeros((N, ))   # better to define data type/google it/ data type:, dtype=np.float32
        self.transform = transform

        rad2num = dict(zip(radius, np.arange(len(radius))))

        start = 0
        for id, (n, r) in enumerate(zip(num_samples, radius)):   #iterate over two lists in parallel/start: to get the index by using enumerate

            theta1 = np.linspace(0, 2 * np.pi, n, endpoint=False) # shape of theta=n
            self.data[start: start+n] = np.concatenate([r * np.cos(theta1)[:, None], r * np.sin(theta1)[:, None]], axis=1)

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
        x = torch.Tensor()
        return x


class Autoencoder(nn.Module):
    def __init__(self, n1=2, n2=4, n3=8, n4=2):
        super(Autoencoder, self).__init__()
        self.nl= nn.Tanhshrink()
        # encoder
        self.enc1 = nn.Linear(n1, n2)
        self.enc2 = nn.Linear(n2, n3)
        self.enc3 = nn.Linear(n3, n4)

        # decoder
        self.dec1 = nn.Linear(n4, n3)
        self.dec2 = nn.Linear(n3, n2)
        self.dec3 = nn.Linear(n2, n1)

    def forward(self, x):
        x = self.nl(self.enc1(x))
        x = self.nl(self.enc2(x))
        y = self.nl(self.enc3(x))

        x = self.nl(self.dec1(y))
        x = self.nl(self.dec2(x))
        x = self.nl(self.dec3(x))

        return y, x



def train(model, device, train_loader, optimizer, epoch, log_interval,loss_fn):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        _, dec = model(data)
        loss = loss_fn(data, dec)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
#
#
def eval(model, device, train_loader):
    model.eval()
    p = np.array(())  ##??????

    with torch.no_grad():
        for batch_idx, (data,label) in enumerate(train_loader):
            data = data.to(device)
            # label = label.to(device)
            enc, dec = model(data)

    return enc, dec
#
#
# INSTANTIATE OPTIMIZER CLASS
def main():
    # Training settings
    model = Autoencoder().double()  ### I changed this part
    loss_fn = nn.MSELoss()
    batch_size = 450
    num_epochs = 2000
    learning_rate = 0.001
    log_interval = 2
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = ToTensor()

    # train_dataset = Circle([100, 150, 200], [1, 2, 3], transform=transform)
    train_dataset = Circle([100,150,200],[1, 2, 3], transform=None)
    print(train_dataset.data)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    # model = Net()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=0.01, gamma=0.1)
    for epoch in range(num_epochs):
        # (model, device, train_loader, optimizer, epoch, num_clusters)
        train(model, device, train_loader, optimizer, epoch, log_interval, loss_fn)

        scheduler.step()
    encoding, decoding = eval(model, device, val_loader)

    #### Plot

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.set_title('Original Dataset', fontsize=18, fontweight='demi')
    ax.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], c=train_dataset.label, s=None, cmap=None)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title('Encoded Data', fontsize=18, fontweight='demi')
    ax.scatter(encoding[:, 0], encoding[:, 1], c=train_dataset.label, s=None, cmap=None)


    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title('Decoded Data', fontsize=18, fontweight='demi')
    ax.scatter(decoding[:, 0], decoding[:, 1], c=train_dataset.label, s=None, cmap=None)
    plt.show()


if __name__ == "__main__":
    main()