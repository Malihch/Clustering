# pip install kmeans-pytorch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans


# Getting Laplacian Matrix(PYTORCH)
def laplacian(x):
    differences = x.unsqueeze(1) - x.unsqueeze(0)
    distances = torch.einsum("ijk, ijk -> ij", differences, differences)
    AdjMat = torch.exp((-10) * distances)
    DigMat = torch.diag(torch.sum(AdjMat, 1))
    LapMat = DigMat - AdjMat

    return LapMat


# DATA PREPARING

class Circle(TensorDataset):
    def __init__(self, num_samples, radius=1, transform=None):
        super(Circle, self).__init__()
        theta1 = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        self.data = np.concatenate([radius * np.cos(theta1)[:, None], radius * np.sin(theta1)[:, None]], axis=1)
        self.transform = transform

    def __getitem__(self, item):
        datapoint = self.data[item]
        if self.transform is not None:
            datapoint = self.transform(datapoint)
        return datapoint

    def __len__(self, ):
        return len(self.data)


class ToTensor(object):
    def __call__(self, x):
        x = torch.Tensor(np.float32(x))
        return x


# CREATE MODEL CLASS(NETWORK)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(2, 4)
        # Output layer, 2 units
        self.fc2 = nn.Linear(4, 2)
        self.reset_parameters()

    def reset_parameters(self, ):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        L = laplacian(x)
        # y = L1.cpu().detach().numpy()
        # print(y)
        # plt.imshow(y, cmap='hot')
        # plt.show()
        _, e, _ = torch.svd(L, compute_uv=True)
        # print("output", output)
        return e, L


def train(model, device, train_loader, optimizer, epoch, num_clusters, log_interval):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        n = len(output)
        loss = output[n - num_clusters] / output[n - num_clusters - 1]
        # print("loss", loss)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, device, train_loader):
    model.eval()
    p = np.array(())

    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            _, output = model(data)
            #             z = output.data.cpu().numpy()
            #             print(z.shape)
            # print("z", '\n', z.shape)
            # print(batch_idx)
            # print(p.shape)
            # p = np.append(p, z)
            # print("p",'\n', p.shape)
            # M = np.reshape(p, (20, 20))
            # print("M", '\n', M.shape)
            return output


# INSTANTIATE OPTIMIZER CLASS
def main():
    # Training settings

    batch_size = 10
    num_epochs = 1

    learning_rate = 0.1
    log_interval = 2
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = ToTensor()
    Circles = [Circle(n, r, transform=ToTensor()) for n, r in zip((2, 3, 5), (1, 2, 3))]
    # ds = [Circle(1000, 2, transform=transform), Circle(1200, 6, transform=transform)]
    k = len(Circles)
    train_dataset = ConcatDataset(Circles)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # ds = [Circle(1000, 2, transform=transform), Circle(1200, 6, transform=transform)]
    # train_dataset = ConcatDataset(ds)
    # test_dataset = ConcatDataset([Circle(32, 3, transform=transform), Circle(1, 0, transform=transform)])

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = Net()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    for epoch in range(num_epochs):
        # (model, device, train_loader, optimizer, epoch, num_clusters)
        train(model, device, train_loader, optimizer, epoch, k, log_interval)
        # test(model, test_loader)
        scheduler.step()
    eval(model, device, train_loader)
    N = eval(model, device, train_loader)
    L = laplacian(N)
    # print(L.shape)

    eigval, eigvec = torch.symeig(L, eigenvectors=True)
    EigMat = eigvec[:, 1:3]

    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    #     device = torch.device('cpu')

    cluster_ids_EigMat_new, cluster_centers = kmeans(EigMat, num_clusters=3, distance='euclidean', device=device)
    # plot
    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(EigMat[:, 0], EigMat[:, 1], c=cluster_ids_EigMat_new, cmap='cool')
    #     plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    plt.axis([-1, 1, -1, 1])
    plt.tight_layout()
    plt.show()


# cluster IDs and cluster centers
#     print(cluster_ids_x)
#     print(cluster_centers)
#     fig, ax = plt.subplots(figsize=(9, 7))
#     ax.set_title('Data after spectral clustering from scratch', fontsize=18, fontweight='demi')
#     ax.scatter(N[:, 0], N[:, 1], c=kmeans.labels_, s=None, cmap=None)
#     plt.show()
# if args.save_model:
#     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()


