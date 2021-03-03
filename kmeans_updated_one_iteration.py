from Sphere_Data import Sphere, ToTensor
import numpy as np
import torch
from tqdm import tqdm


def initialize(X, num_clusters):   ### we use this function only for the first epoch. Because initialization will be doing randomly.
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)###
    initial_state = X[indices]
    return initial_state

# print("initial",'\n', initialize(data,2))

def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis

def kmeans_mod(X,
               num_clusters,
               device=torch.device('cpu'),
               distance = 'euclidean',
               centroid=None
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')

    # getting distance between data points and initial centroids
    # dis = pairwise_distance(X, initial_state)

    dis = torch.cdist(X, initial_state)
    # print("dis",'\n', dis)

    choice_cluster = torch.argmin(dis, dim=1)# getting the index of min value in each row
    # print("choice_cluster",'\n', choice_cluster)

    # This function is differentiable, so gradients will flow back from the result of this operation to input
    initial_state_pre = initial_state.clone()
    # print("initial state pre",'\n',initial_state_pre)

    for index in range(num_clusters):
        # getting the index where num_cluster = 0, 1, and 2 in each iteration
         selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
         # print("selected1", selected)

         selected = torch.index_select(X, 0, selected)
         # print("selected2",'\n',selected)

         initial_state[index] = selected.mean(dim=0)
         # print("initial_state[index]",'\n', initial_state[index])

    tqdm_meter.set_postfix(
         iteration=f'{iteration}' )
    tqdm_meter.update()

    return choice_cluster.cpu(), initial_state.cpu()
