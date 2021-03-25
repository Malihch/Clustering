import numpy as np
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from Sphere_Data import ToTensor
from CustomLossFunction import custom_loss
from kmeans_updated_one_iteration import kmeans_mod
from Autoencoder_Sphere import Autoencoder
from matplotlib import pyplot as PLT

### DATASET #####
class Twoclusters(TensorDataset):
    def __init__(self ,m ,n ,transform):  # we consider num_samples and radius as a list
        super(Twoclusters, self).__init__()
        A = torch.randint(1, 5, (m, 10)) #data(positive)
        B = torch.randint(-5, -1, (n, 10)) #data(negative)
        self.data = torch.cat((A, B), 0)

        C = torch.randint(0, 1, (m, 1)) #label(0)
        D = torch.randint(1, 2, (n, 1))#label(1)
        self.label = torch.cat((C, D), 0)

        self.transform = transform


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


class Autoencoder(nn.Module):
    def __init__(self, n1=10, n2=8, n3=4, n4=2):
        super(Autoencoder, self).__init__()
        self.nl = nn.Tanhshrink()

        # encoder
        self.enc1 = nn.Linear(n1, n2)
        self.enc2 = nn.Linear(n2, n3)
        self.enc3 = nn.Linear(n3, n4)
        # self.enc4 = nn.Linear(n4, n5)

        # decoder
        # self.dec1 = nn.Linear(n5, n4)
        self.dec1 = nn.Linear(n4, n3)
        self.dec2 = nn.Linear(n3, n2)
        self.dec3 = nn.Linear(n2, n1)

    def forward(self, x):
        x = self.nl(self.enc1(x))
        x = self.nl(self.enc2(x))
        # x = self.nl(self.enc3(x))
        y = self.nl(self.enc3(x))

        x = self.nl(self.dec1(y))
        x = self.nl(self.dec2(x))
        x = self.nl(self.dec3(x))
        # x = self.nl(self.dec4(x))
        return y, x


def train(model, device, train_loader, optimizer, epoch, log_interval, loss_fn):  #here we define loos function as an input
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
    return loss


def val(model, device, val_loader,optimizer, epoch, log_interval, loss_fn):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            target = model(data)
            val_loss += loss_fn(target,data, reduction='sum').item()  # sum up batch loss
            pred = target.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(data.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return val_loss

# def eval(model, device, train_loader):
#     model.eval()
#
#
#     with torch.no_grad():
#         for batch_idx, (data,label) in enumerate(train_loader):
#             data = data.to(device)
#             # label = label.to(device)
#             enc, dec = model(data)
#
#     return enc, dec


# INSTANTIATE OPTIMIZER CLASS
# INSTANTIATE OPTIMIZER CLASS
def main():
    # Training settings
    model = Autoencoder()
    loss_fn = nn.MSELoss()
    batch_size = 13000
    num_epochs = 10
    learning_rate = 0.01
    log_interval = 2
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = Twoclusters(7000,6000, transform=ToTensor())

    train_set, val_set = torch.utils.data.random_split(train_dataset, [round(len(train_dataset) * 0.8), round(len(train_dataset) * 0.2)])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    model.to(device) # load the neural network on to the device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_value_train = np.zeros((num_epochs))
    loss_value_val = np.zeros((num_epochs))
    for epoch in range(num_epochs):

       training_loss = train(model, device, train_loader, optimizer, epoch, log_interval, loss_fn)
       validation_loss = train(model, device, val_loader, optimizer, epoch, log_interval, loss_fn)

       loss_value_train[epoch] = training_loss
       loss_value_val[epoch] = validation_loss

    plt.figure(figsize=(10, 7))
    plt.plot(list(range(len(loss_value_train))), loss_value_train, label='Training loss')
    plt.plot(list(range(len(loss_value_val))), loss_value_val, label='Validation loss')
    plt.legend()
    plt.show()


    fig = PLT.figure()
    ax1 = fig.add_subplot(221, projection='3d')  ## (221), (222), (223), and (224), to create four plots on a page at 10, 2, 8, and 4 o'clock, respectively and in this order.
    ax1.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], cmap=None,
               c=train_dataset.label)
    plt.show()


if __name__ == "__main__":
    main()