import torch

from sklearn.cluster import KMeans
# from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from tqdm import tqdm
from torch import nn

class Nodrop(nn.Module):
    def __init__(self):
        super(Nodrop, self).__init__()

    def forward(self, input):
        return input

class DropCluster4D(nn.Module):
    def __init__(self, drop_prob):
        super(DropCluster4D, self).__init__()
        self.drop_prob = drop_prob
        self.transform_matrices = None


    # def Silhouette_search(self,x,n_max=50,n_min=30,alg='Kmeans'):
    #     print('Starting Silhouette_search...')
    #     # reshape data
    #     n_x, n_y = x.shape[2], x.shape[3]
    #     x = x.reshape(x.shape[0], x.shape[1], n_x * n_y)
    #     connectivity = grid_to_graph(n_x=n_x, n_y=n_y)
    #     x = np.transpose(x, (1, 0, 2))
    #     # x = channel_size x batch_size x n_pixels
    #     Silhouette_max =[]
    #     num_of_clusters = []
    #     transform_matrices = []
    #
    #     for i in range(x.shape[0]):
    #         X = x[i]
    #         n_cluster_max = min(n_max, x.shape[-1])
    #         range_n_clusters = range(n_min, n_max + 1)
    #         current_silhouette = 0
    #         current_clusters = 1
    #         current_label = [0]*n_x*n_y
    #         for n_clusters in range_n_clusters:
    #             if alg == 'Kmeans':
    #                 print('Computing n={}...'.format(n_clusters))
    #                 clf = KMeans(n_clusters=n_clusters, random_state=i)
    #                 cluster_labels = clf.fit_predict(X.T)
    #
    #             label_set = set(cluster_labels.flatten().tolist())
    #             if len(label_set) == 1:
    #                 break
    #             else:
    #                 score = silhouette_score(X.T, cluster_labels)
    #                 if score > current_silhouette:
    #                     current_silhouette = score
    #                     current_clusters = n_clusters
    #                     current_label = cluster_labels
    #         Silhouette_max.append(current_silhouette)
    #         num_of_clusters.append(current_clusters)
    #         transform_mask = np.zeros((current_clusters, n_x * n_y))
    #         for i in range(current_clusters):
    #             temp = transform_mask[i]
    #             temp[current_label==i] = 1
    #             transform_mask[i] += temp
    #         transform_matrices.append(transform_mask)
    #     return Silhouette_max, num_of_clusters, transform_matrices

    def mask_drop_cluster(self, transform_matrix):

         # transform_matrics = 1*n_x*n_y

        '''dropcluster at structured channel'''

        num_cluster_total = transform_matrix.shape[0]  # this is equal to cluster size
        random_numbers = torch.rand(num_cluster_total)
        all_indices = torch.arange(num_cluster_total)
        drop_index = all_indices[random_numbers < self.drop_prob]
        mask = transform_matrix[drop_index,:].sum(axis=0)
        mask[mask > 0] = 1

        return mask

    def forward(self, input):
        assert input.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, disparity, height, width)"

        if not self.training or self.drop_prob == 0:
            return input
        n_d,n_h,n_w = input.shape[-3],input.shape[-2],input.shape[-1]
        # x = input.cpu().detach().numpy()
        mask = torch.zeros(input[0].shape)
        for i in range(input.shape[1]):
            # more than 1 cluster, do dropcluster
            if self.transform_matrices[i].shape[0]>1:
                transform_matrix = self.transform_matrices[i]
                current_mask = self.mask_drop_cluster(transform_matrix)
                mask[i] += current_mask.reshape(n_d,n_h,n_w)
            else:
                # only 1 cluster, do dropout
                random_numbers = torch.rand(n_d*n_h*n_w)
                current_mask = torch.zeros(n_d*n_h*n_w)
                current_mask[random_numbers<self.drop_prob]=1
                mask[i] += current_mask.reshape(n_d,n_h,n_w)
        mask = mask.to(input.device)
        mask = 1 - mask
        n = mask.shape[-3]*mask.shape[-2] * mask.shape[-1]
        d = mask.sum(axis=1).sum(axis=1).sum(axis=1)
        d[d == 0] = 1
        d = d.reshape(mask.shape[0], 1, 1, 1)
        d = d.repeat(1, mask.shape[-3],mask.shape[-2], mask.shape[-1])
        mask = mask * n / d
        mask = mask.view(1, mask.shape[0], mask.shape[1], mask.shape[2],mask.shape[3])  # 1 x 64 x 56 x 56
        mask = mask.repeat(input.shape[0], 1, 1, 1,1)  # batch_size x 64 x 56 x 56

        # scale output
        out = input * mask

        return out


############################TODO################################
    def compute_transform_matrices_kMeans(self,x,n_clusters):
        # reshape data
        n_d, n_h, n_w = x.size(-3), x.size(-2), x.size(-1)
        x = x.view(x.size(0), x.size(1), n_d * n_h * n_w)
        x = x.permute(1, 0, 2)
        # x = channel_size x batch_size x n_pixels
        transform_matrices = []
        num_of_clusters = []

        print('Starting computing clusters...')

        for i in tqdm(range(x.size(0))):
            X = x[i]
            current_label = [0] * n_d*n_h*n_w
            current_clusters = 1
            clf = KMeans(n_clusters=n_clusters, random_state=i)
            cluster_labels = clf.fit_predict(X.T.numpy())
            label_set = set(cluster_labels)
            if len(label_set) > 1:
                current_label = cluster_labels
                current_clusters = n_clusters

            transform_mask = torch.zeros((current_clusters, n_d * n_h * n_w))
            for j in range(current_clusters):
                temp = transform_mask[j]
                temp[current_label == j] = 1
                transform_mask[j] += temp
            transform_matrices.append(transform_mask)
            print('channel[{}] n_clusters:{}'.format(i, current_clusters))
        return num_of_clusters, transform_matrices

    def Silhouette_search(self, x, n_max=50, n_min=30, alg='Kmeans'): #skip computing scores
        print('Starting Silhouette_search...')
        # reshape data
        n_d,n_h,n_w = x.size(-3),x.size(-2),x.size(-1)
        x = x.view(x.size(0), x.size(1), n_d*n_h*n_w)
        x = x.permute(1, 0, 2)
        # x = channel_size x batch_size x n_pixels
        Silhouette_max = []
        num_of_clusters = []
        transform_matrices = []

        for i in tqdm(range(x.size(0))):
            X = x[i]
            n_cluster_max = min(n_max, x.shape[-1])
            corr_max = int(x[i].max())

            c_min = min(n_min,corr_max)
            c_max = min(n_max,corr_max)

            step=1
            # if c_max-c_min>20:
            #     step = 5


            # range_n_clusters = range(c_min, c_max+1,step)
            range_n_clusters = range(c_max, c_max + 1, step)
            current_silhouette = 0
            current_clusters = 1
            current_label = [0] * n_d*n_h*n_w
            for n_clusters in tqdm(range_n_clusters):
                if alg == 'Kmeans':
                    clf = KMeans(n_clusters=n_clusters, random_state=i)
                    cluster_labels = clf.fit_predict(X.T.numpy())
                label_set = set(cluster_labels)
                if len(label_set) == 1:
                    break
                else:
                    # score = silhouette_score(X.T.numpy(), cluster_labels)
                    # if score > current_silhouette:
                    #     current_silhouette = score
                    #     current_clusters = n_clusters
                    #     current_label = cluster_labels
                    current_clusters = n_clusters
                    current_label = cluster_labels
            # Silhouette_max.append(current_silhouette)
            num_of_clusters.append(current_clusters)
            transform_mask = torch.zeros((current_clusters, n_d*n_h*n_w))
            for j in range(current_clusters):
                temp = transform_mask[j]
                temp[current_label == j] = 1
                transform_mask[j] += temp
            transform_matrices.append(transform_mask)
            print('channel[{}] best_score:{:.6f} n_clusters:{}'.format(i, current_silhouette, current_clusters))
        return Silhouette_max, num_of_clusters, transform_matrices


