'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def _init_centers(self, points, K, **kwargs):  # [2 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        
#         points.shape = (4000, 2)
#         K = 4
        
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[:K]

    def _kmpp_init(self, points, K, **kwargs): # [3 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """
        
        raise NotImplementedError

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        # update the membership of each point based on the closest center
        eculd = pairwise_dist(centers, points)
        minval = np.argmin(eculd, axis=0)
        
        return minval

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        # Since cluster memberships have changed, cluster centers should also be updated.
        ans = []
        # map each cluster to index that corresponds to cluster assignment
        centerDict = KMeans._get_centers_mapping(self, points, cluster_idx, old_centers)
        # loop through the keys (indices) and find the average
        for key in centerDict: 
            ans.append(np.average(centerDict[key], axis=0))
        
        # change list to numpy array
        return np.array(ans)
        

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """
        #  The loss will be defined as the sum of the squared distances between each point and it's 
        # respective center. 
        # The iteration is implemented for you in the __call__ method.
        dist, loss = pairwise_dist(points, centers), 0.0
        N = points.shape[0]
        for i in range(N):
            loss += np.square(dist[i][cluster_idx[i]])
        
        return loss

    def _get_centers_mapping(self, points, cluster_idx, centers):
        # This function has been implemented for you, no change needed.
        # create dict mapping each cluster to index to numpy array of points in the cluster
        centers_mapping = {key : [] for key in [i for i in range(centers.shape[0])]}
        for (p, i) in zip(points, cluster_idx):
            centers_mapping[i].append(p)
        for center_idx in centers_mapping:
            centers_mapping[center_idx] = np.array(centers_mapping[center_idx])
        self.centers_mapping = centers_mapping
        return centers_mapping

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, center_mapping=False, **kwargs):
        """
        This function has been implemented for you, no change needed.

        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        if center_mapping:
            return cluster_idx, centers, loss, self._get_centers_mapping(points, cluster_idx, centers)
        return cluster_idx, centers, loss


def pairwise_dist(x, y):  # [5 pts]
    # just seed random numbers to make calculation
    np.random.seed(1)
    """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
    """
    if x.ndim == 1:
        x = x[np.newaxis,:]

    xydiff = x[:,:,np.newaxis] - y[:,:,np.newaxis].T
    eucld = (xydiff ** 2).sum(1) 
    return np.sqrt(eucld)
#     return np.sqrt(abs(xx[:, np.newaxis] + yy - 2 * xy))

def silhouette_coefficient(points, cluster_idx, centers, centers_mapping): # [10pts]
    """
    Args:
        points: N x D numpy array
        cluster_idx: N x 1 numpy array
        centers: K x D numpy array, the centers
        centers_mapping: dict with K keys (cluster indicies) each mapping to a C_i x D 
        numpy array with C_i corresponding to the number of points in cluster i
    Return:
        silhouette_coefficient: final coefficient value as a float 
        mu_ins: N x 1 numpy array of mu_ins (one mu_in for each data point)
        mu_outs: N x 1 numpy array of mu_outs (one mu_out for each data point)
    """
    # HINT: use centers_mapping as a parameter in pairwise_distance()
    mu_out = np.zeros(points.shape[0])
    mu_in = np.zeros(points.shape[0])
    si = np.zeros(points.shape[0])
    
    # centers mapping shape; (2913, 3), (2032, 3), (3491, 3)
    # x_i, x_j: points from NxD array
    # y: points from the cluster
    
    
    # mu in: centers, centers_mapping, cluster_idx
    # mu out: deal with all clusters except yours -> use cluster_idx
    N, D, K = points.shape[0], points.shape[1], centers.shape[0]
    for i in range(N):
        point = points[i]
        curr_cluster = centers_mapping[cluster_idx[i]]
        cluster_dist = pairwise_dist(point, curr_cluster)
        
        
        mu_in[i] = np.sum(cluster_dist) / (len(curr_cluster) - 1)
        
        minval = 1000000000000
        for j in range(K):
            # Skip the part where you are in your own cluster
            if cluster_idx[i] == j:
                continue
            intra_cluster = centers_mapping[j]
            cluster_dist = pairwise_dist(point, intra_cluster)
            
            temp = np.sum(cluster_dist) / (len(intra_cluster))
            minval = min(temp, minval)
        
        # set mu_out with the minimum value
        mu_out[i] = minval
        
        
        # calculate s_i here
        si[i] = (mu_out[i] - mu_in[i]) / max(mu_out[i], mu_in[i])
    # average out
    coefficient = np.average(si)
    
    return coefficient, mu_in, mu_out