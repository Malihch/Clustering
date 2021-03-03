
import torch.optim as optim
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from Sphere_Data import Sphere, ToTensor
from Sphere2_Data import Sphere2



class Autoencoder(nn.Module):
    def __init__(self, n1=3, n2=4, n3=8, n4=2):
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
#
#
def eval(model, device, train_loader):
    model.eval()


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
    model = Autoencoder()
    loss_fn = nn.MSELoss()
    batch_size = 2800
    num_epochs = 1000
    learning_rate = 0.01
    log_interval = 2
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # train_dataset = Sphere([200,200,300],[1, 2, 3], transform=ToTensor())
    train_dataset = Sphere2([10,20,30],[20,40,60],[1, 2, 3], transform=ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False) #### TRue to false
    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    # model = Net()
    model.to(device) # load the neural network on to the device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = StepLR(optimizer, step_size= 0.01, gamma=0.1) # I changed step_size to get decreasing loss values
    for epoch in range(num_epochs):

        train(model, device, train_loader, optimizer, epoch, log_interval, loss_fn)

        # scheduler.step()
    torch.save({'model_state_dict': model.state_dict()}, 'Mali.pt') # "mali.pt" is a path /save weight

    encoding, decoding = eval(model, device, val_loader)

    # fig = PLT.figure()
    #
    # ax1 = fig.add_subplot(221,
    #                       projection='3d')  ## (221), (222), (223), and (224), to create four plots on a page at 10, 2, 8, and 4 o'clock, respectively and in this order.
    # ax1.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], train_dataset.data[:, 2], cmap=None,
    #             c=train_dataset.label)
    #
    # ax2 = fig.add_subplot(222)
    # ax2.set_title('Encoded Data', fontsize=18, fontweight='demi')
    # ax2.scatter(encoding[:, 0], encoding[:, 1], c=train_dataset.label, s=None, cmap=None)
    #
    # ax3 = fig.add_subplot(223, projection='3d')
    # ax3.set_title('Encoded Data', fontsize=18, fontweight='demi')
    # ax3.scatter(decoding[:, 0], decoding[:, 1], decoding[:, 2], cmap=None, c=train_dataset.label)
    #
    # PLT.show()

    # plot of the original data set
    #
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], train_dataset.data[:, 2], cmap=None,
               c=train_dataset.label)
    #
    # # 2D plot when the latent layer = 2
    #
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title('encoded part', fontsize=18, fontweight='demi')
    ax.scatter(encoding[:, 0], encoding[:, 1], c=train_dataset.label, s=None, cmap=None)
    #
    # # 3D plot when the latent layer = 3
    #
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(decoding[:, 0], decoding[:, 1], decoding[:, 2], cmap=None, c = train_dataset.label)
    #
    #
    plt.show()

if __name__ == "__main__":
    main()