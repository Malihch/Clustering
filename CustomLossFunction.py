import torch

def custom_loss(data, id, centers):

    # applying k-means on encoded part(data)
    # M = torch.zeros(len(data),)
    sum = 0
    # print(id)
    for i in range(0, len(data)):

        sum += torch.dist(data[i], centers[id[i]])**2


    return  torch.sqrt(torch.tensor(sum))





# data = torch.tensor([[1.00,2.00],[2.00,4.00],[0.00,1.00],[7.00,8.00],[8.00,9.00],[1.00,4.00],[5.00,8.00]])
# data= torch.tensor([[1.00,2.00],[2.00,4.00],[0.00,1.00]])
# N = custom_loss(data, 2)
# print("size", data.size())
# cluster_ids_x, cluster_centers = kmeans(X=data, num_clusters=3, distance='euclidean')
# print(custom_loss(data, cluster_ids_x, cluster_centers))
# print("idx",cluster_ids_x)
# print("center",cluster_centers)
# # print(cluster_centers.size())
# print(data[torch.where(cluster_ids_x==0)])
# print("IN",cluster_centers[0,:].shape)
# print(torch.cdist(data[torch.where(cluster_ids_x==0)],cluster_centers))
#
# A= torch.tensor([[1.00,2.00],[3.00,4.00]])
# B = torch.tensor([[1.00,4.00],[5.00,8.00]])
# print(torch.dist(A,B))


