
import torch.optim as optim
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from Sphere_Data import Sphere, ToTensor
from CustomLossFunction import custom_loss
from kmeans_updated_one_iteration import kmeans_mod
from Autoencoder_Sphere import Autoencoder


def train(model, device, train_loader, optimizer, epoch, log_interval, loss_fn, num_cluster,  centroids):  #here we define loos function as an input
    model.train()

    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        enc, dec = model(data)

        cluster_ids_x, centroids = kmeans_mod(X=enc, num_clusters=num_cluster, centroid=centroids)

        loss1 = loss_fn(data, dec) # MSE/Autoencoder
        loss2 = custom_loss(enc, cluster_ids_x, centroids) # Clustering Part
        loss = loss1+loss2
        loss.backward()  ####???
        optimizer.step()
######################################
        # losses = []
        # for epoch in range(num_epochs):
        #     running_loss = 0.0
        #     for data in dataLoader:
        #         images, labels = data
        #
        #         outputs = model(images)
        #         loss = criterion_label(outputs, labels)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #         running_loss += loss.item() * images.size(0)
        #
        #     epoch_loss = running_loss / len(dataloaders['train'])
        #     losses.append(epoch_loss)
 #######################################################################


        running_loss +=loss.item()
        if batch_idx % log_interval == 0:

            print(f'epoch loss1 {loss1:.6f}  loss2 {loss2:.6f}   loss{loss:.6f}')

    epoch_loss = running_loss
    return centroids, cluster_ids_x, epoch_loss
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
    num_epochs = 15
    learning_rate = 0.001
    log_interval = 2
    num_cluster = 3
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = Sphere([200,800,1800],[1, 2, 3], transform=ToTensor())   ### why transform should be None?
    # train_dataset = Sphere2([10, 20, 30], [20, 40, 60], [1, 2, 3], transform=ToTensor())
    # train_dataset = torch.from_numpy(train_dataset_numpy)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    # model = Autoencoder()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint = torch.load('Mali.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) # load the neural network on to the device  @@@@@@


    # scheduler = StepLR(optimizer, step_size=0.01, gamma=0.1) # I changed step_size to get decreasing loss values
    loss_value = []
    for epoch in range(num_epochs):

        if epoch == 0:
            c, clusters, loss= train(model, device, train_loader, optimizer, epoch, log_interval, loss_fn, num_cluster, centroids=None)

        else:
            c, clusters, loss= train(model, device, train_loader, optimizer, epoch, log_interval, loss_fn, num_cluster , centroids=c) ##updated c in each epoch
            # when the batch size is not the same size of the whole data set, we need to do it in every iteration
        loss_value.append(loss)
        print(len(loss_value))
        if epoch % 10 == 0:
            plt.figure(figsize=(10, 7))
            plt.plot(list(range(len(loss_value))), loss_value)
            plt.show()
        # ax = ax.set_title('Encoded Data', fontsize=18, fontweight='demi')
        # ax.scatter(loss, cmap=None)




    encoding, decoding = eval(model, device, val_loader)

    # fig = PLT.figure()
    # ax1 = fig.add_subplot(221, projection='3d')  ## (221), (222), (223), and (224), to create four plots on a page at 10, 2, 8, and 4 o'clock, respectively and in this order.
    # ax1.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], train_dataset.data[:, 2], cmap=None,
    #            c=train_dataset.label)
    #
    # ax2 = fig.add_subplot(222)
    # ax2.set_title('Encoded Data', fontsize=18, fontweight='demi')
    # ax2.scatter(encoding[:, 0], encoding[:, 1], c=clusters, s=None, cmap=None)
    #
    # ax3 = fig.add_subplot(223, projection='3d')
    # ax3.set_title('Encoded Data', fontsize=18, fontweight='demi')
    # ax3.scatter(decoding[:, 0], decoding[:, 1], decoding[:, 2], cmap=None, c = train_dataset.label)
    #
    # PLT.show()


    # plot of the original data set

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], train_dataset.data[:, 2], cmap=None,
               c=train_dataset.label)
    #
    # # 2D plot when the latent layer = 2
    #
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title('Encoded Data', fontsize=18, fontweight='demi')
    ax.scatter(encoding[:, 0], encoding[:, 1], c=clusters, s=None, cmap=None)

    # # 3D plot when the latent layer = 3
    #
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(decoding[:, 0], decoding[:, 1], decoding[:, 2], cmap=None, c = train_dataset.label)

    # plt.show()





if __name__ == "__main__":
    main()