import numpy as np
import torch
from kmeans_updated import kmeans_mod
from tqdm import tqdm

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

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

def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        device=torch.device('cpu'),
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
    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if centroid == None:
        initial_state = initialize(X, num_clusters)

    else:
        initial_state = centroid


    # tqdm_meter = tqdm(desc='[running kmeans]')

    dis = pairwise_distance(X, initial_state)

    choice_cluster = torch.argmin(dis, dim=1)

    initial_state_pre = initial_state.clone()

    for index in range(num_clusters):
        selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

        selected = torch.index_select(X, 0, selected)
        initial_state[index] = selected.mean(dim=0)

    center_shift = torch.sum(
        torch.sqrt(
            torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
        ))


    return choice_cluster.cpu(), initial_state.cpu()


