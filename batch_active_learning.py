# Some functions for doing batch active learning and coreset selection
# Authors: James, Bohan, and Zheng
import graphlearning.active_learning as al
import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import time
import timeit
import os, glob

import torch
import utils
import random

#SKLearn imports
import sklearn.metrics as metrics
import sklearn.datasets as datasets
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.cluster._kmeans import _kmeans_plusplus
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.datasets import make_blobs

#Scipy imports
from scipy.linalg import sqrtm
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigs
import scipy.sparse as sparse
from scipy.optimize import fsolve
from scipy import spatial

##############################################################################################################
### coreset functions
def density_determine_rad(G, x, proportion, r_0=1.0, tol=.02):
    # Determines the radius necessary so that a certain proportion of the data falls in B_r(x)
    # This is a lazy way and more efficient code could be written in c
    # The proportion that we seek is (p-tol, p+tol) where tol is some allowable error and p is the desired proportion
    n = G.num_nodes
    r = r_0
    dists = G.dijkstra(bdy_set=[x], max_dist=r)
    p = np.count_nonzero(dists < r) * 1.0 / n

    iterations = 1
    a = 0
    b = 0
    if p >= proportion - tol and p <= proportion + tol:
        # If within some tolerance of the data, just return
        return p
    elif p > proportion + tol:
        # If radius too big, initialize a, b for bisection
        a = 0
        b = r
    else:
        while p < proportion - tol:
            # If radius too small, try to increase
            r *= 1.5
            dists = G.dijkstra(bdy_set=[x], max_dist=r)
            p = np.count_nonzero(dists < r) * 1.0 / n
        a = .66 * r
        b = r

    # Do bisection method to get answer
    while p < proportion - tol or p > proportion + tol:
        r = (a + b) / 2.0
        p = np.count_nonzero(dists < r) * 1.0 / n

        if p > proportion + tol:
            b = r
        elif p < proportion - tol:
            a = r
        else:
            return r
        iterations += 1
        if (iterations >= 30):
            print("Too many iterations. Density radius did not converge")
            return r
    return r


def coreset_dijkstras(G, rad, DEBUGGING=False, data=None, initial=[], randseed=123, density_info=(False, 0, 1.0),
                      similarity='euclidean', knn_data=None):
    np.random.seed(randseed)
    coreset = initial.copy()
    perim = []

    rad_low = rad / 2.0
    rad_high = rad

    use_density, proportion, r_0 = density_info

    # Once all points have been seen, we end this
    points_seen = np.zeros(G.num_nodes)

    knn_val = G.weight_matrix[0].count_nonzero()

    # This gives actual distances
    if knn_data:
        W_dist = gl.weightmatrix.knn(data, knn_val, similarity=similarity, kernel='distance', knn_data=knn_data)
    else:
        W_dist = gl.weightmatrix.knn(data, knn_val, similarity=similarity, kernel='distance')
    G_dist = gl.graph(W_dist)

    # Construct the perimeter from initial set
    n = len(initial)
    for i in range(n):
        if use_density:
            rad_low = density_determine_rad(G_dist, initial[i], proportion / 2.0, r_0)
            rad_high = density_determine_rad(G_dist, initial[i], proportion, r_0)
        if len(coreset) == 0:
            tmp1 = G_dist.dijkstra(bdy_set=[initial[i]], max_dist=rad_high)
            tmp2 = (tmp1 <= rad_high)
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]
            # Update perim
            perim = list(tmp3)
            # Update points seen
            points_seen[tmp2] = 1
        else:
            # Calculate perimeter from new node
            tmp1 = G_dist.dijkstra(bdy_set=[initial[i]], max_dist=rad_high)
            tmp2 = (tmp1 <= rad_high)
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]
            tmp4 = (tmp1 <= rad_low).nonzero()[0]

            # Get rid of points in perimeter too close to new_node
            for x in tmp4:
                if x in perim:
                    perim.remove(x)

            # Add in points in the perimeter of new_node but unseen by old points
            for x in tmp3:
                if x not in perim and points_seen[x] == 0:
                    perim.append(x)

            points_seen[tmp2] = 1

    # Generate the coreset from the remaining stuff
    iterations = 0

    # while we haven't seen all points or the perimeter is empty
    # Want this to stop when the perimeter is empty
    # But we also want all the points to be seen
    while (np.min(points_seen) == 0 or len(perim) > 0):
        if len(coreset) == 0:
            # Generate coreset
            new_node = np.random.choice(G_dist.num_nodes, size=1).item()
            coreset.append(new_node)
            if use_density:
                rad_low = density_determine_rad(G_dist, new_node, proportion / 2.0, r_0)
                rad_high = density_determine_rad(G_dist, new_node, proportion, r_0)
            # Calculate perimeter
            tmp1 = G_dist.dijkstra(bdy_set=[new_node], max_dist=rad_high)
            tmp2 = (tmp1 <= rad_high)
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]
            # Update perim
            perim = list(tmp3)
            # Update points seen
            points_seen[tmp2] = 1
        elif len(perim) == 0:
            # Make a random choice for a new node
            # This situation is basically a node jump to a new region. It should essentially reduce to situation 1
            avail_nodes = (points_seen == 0).nonzero()[0]
            new_node = np.random.choice(avail_nodes, size=1).item()
            coreset.append(new_node)
            if use_density:
                rad_low = density_determine_rad(G_dist, new_node, proportion / 2.0, r_0)
                rad_high = density_determine_rad(G_dist, new_node, proportion, r_0)
            # Calculate perimeter
            tmp1 = G_dist.dijkstra(bdy_set=[new_node], max_dist=rad_high)
            tmp2 = (tmp1 <= rad_high)
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]

            # Need to make it so that the balls don't overlap
            # Update perim
            perim = list(tmp3)
            # Update points seen
            points_seen[tmp2] = 1
        else:
            # Select a new node from the perimeter
            new_node = np.random.choice(perim, size=1).item()
            coreset.append(new_node)
            if use_density:
                rad_low = density_determine_rad(G_dist, new_node, proportion / 2.0, r_0)
                rad_high = density_determine_rad(G_dist, new_node, proportion, r_0)

            # Calculate perimeter from new node
            tmp1 = G_dist.dijkstra(bdy_set=[new_node], max_dist=rad_high)
            tmp2 = (tmp1 <= rad_high)
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]
            tmp4 = (tmp1 <= rad_low).nonzero()[0]

            # Get rid of points in perimeter too close to new_node
            for x in tmp4:
                if x in perim:
                    perim.remove(x)

            # Add in points in the perimeter of new_node but unseen by old points
            for x in tmp3:
                if x not in perim and points_seen[x] == 0:
                    perim.append(x)

            points_seen[tmp2] = 1

        if (DEBUGGING):
            plt.scatter(data[:, 0], data[:, 1])
            plt.scatter(data[coreset, 0], data[coreset, 1], c='r')
            plt.scatter(data[perim, 0], data[perim, 1], c='y')
            plt.show()

        if iterations >= 1000:
            break
        iterations += 1
    return coreset

def get_poisson_weighting(G, train_ind, tau=1e-8, normalization='combinatorial'):
    n = G.num_nodes
    F = np.zeros(n)
    F[train_ind] = 1
    if normalization == 'combinatorial':
        F -= np.mean(F)
    else:
        F -= np.mean(G.degree_matrix(p=0.5) * F) * G.degree_vector() ** (0.5)

    L = G.laplacian(normalization=normalization)
    if tau > 0.0:
        L += tau * sparse.eye(L.shape[0])

    w = gl.utils.conjgrad(L, F, tol=1e-5)
    w -= np.min(w, axis=0)

    return w

def plusplus_coreset(graph, num_points=10, random_state=None, method='dijkstra', eik_p=1.0, tau=0.1,
                     ofs=0.2, q=1.0, knn_dist=None, kernel='gaussian', plot=False, X=None, initial=None):
    n = graph.num_nodes
    # if want to use 0/1 edge weights, specify kernel = 'uniform'
    if kernel == 'uniform':
        G = gl.graph(graph.adjacency())
    else:
        G = graph

    all_inds = np.arange(G.num_nodes)
    if random_state is None:
        random_state = 0
    random_state = check_random_state(random_state)

    if method == 'peikonal':
        # print("Preparing knn density estimator for p-eikonal")
        if knn_dist is not None:
            alpha = 2.
            d = np.max(knn_dist, axis=1)
            kde = (d / d.max()) ** (-1)
            f = kde ** (-alpha)
        else:
            print("No knn dist info provided, defaulting to just f = 1")
            f = 1.0
    # randomly select initial point for coreset
    if initial:
        indices = initial
        if len(initial) > 1:
            if method == 'dijkstra':
                dists = G.dijkstra(indices[:-1])
            elif method == 'peikonal':
                dists = G.peikonal(indices[:-1], p=eik_p, f=f)
            elif method == 'poisson':
                dists = np.full(n, np.inf)
                for i in range(len(initial)-1):
                    w = get_poisson_weighting(G, [indices[i]], tau=tau)
                    dists_new = 1. / (ofs + w)
                    np.minimum(dists, dists_new, out=dists)
        else:
            dists = np.full(n, np.inf)
        j = 0
    else:
        indices = np.array([random_state.randint(n)])
        dists = np.full(n, np.inf)
        j = 1

    # while still have budget to add to the coreset, propagate dijkstra out
    while j < num_points:
        j += 1
        x = indices[-1]
        if method == 'dijkstra':
            dist_to_x = G.dijkstra([x])
        elif method == 'peikonal':
            dist_to_x = G.peikonal([x], p=eik_p, f=f)
        elif method == 'poisson':
            w = get_poisson_weighting(G, [x], tau=tau)
            dist_to_x = 1. / (ofs + w)
        np.minimum(dists, dist_to_x, out=dists)
        if plot and X is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            p1 = ax1.scatter(X[:, 0], X[:, 1], c=dists)
            ax1.scatter(X[indices, 0], X[indices, 1], c='r', marker='^', s=100)
            ax1.scatter(X[x, 0], X[x, 1], c='pink', marker='^', s=100)
            ax1.set_title("Sampling Probabilities")
            plt.colorbar(p1, ax=ax1)

            p2 = ax2.scatter(X[:, 0], X[:, 1], c=dists ** q)
            ax2.scatter(X[indices, 0], X[indices, 1], c='r', marker='^', s=100)
            ax2.set_title("Updated Distances")
            plt.colorbar(p2, ax=ax2)
            plt.show()

        # sample next point proportionally to the q^th power of the distances
        if q != 1:
            vals = np.cumsum(dists ** q)
        else:
            vals = np.cumsum(dists)
        next_ind = np.searchsorted(vals, vals[-1] * random_state.uniform())
        indices = np.append(indices, next_ind)
    return indices

def plusplus_coreset_sim(graph, num_points=10, random_state=None, method='poisson', tau=0.1,
                         q=1.0, kernel='gaussian', plot=False, X=None):
    n = graph.num_nodes

    # if want to use 0/1 edge weights, specify kernel = 'uniform'
    if kernel == 'uniform':
        G = gl.graph(graph.adjacency())
    else:
        G = graph

    all_inds = np.arange(G.num_nodes)

    # instantiate the random_state object for making random trials consistent
    if random_state is None:
        random_state = 0
    random_state = check_random_state(random_state)

    # randomly select initial point for coreset
    indices = np.array([random_state.randint(n)])

    similarities = np.zeros(n)  # initialize similarities to coreset vector
    # while still have budget to add to the coreset, propagate dijkstra out
    for j in range(1, num_points):
        x = indices[-1]
        if method == 'poisson':
            sim_to_x = get_poisson_weighting(G, [x], tau=tau)
        else:
            raise NotImplementedError()

        np.maximum(similarities, sim_to_x, out=similarities)

        if plot and X is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            p1 = ax1.scatter(X[:, 0], X[:, 1], c=similarities)
            ax1.scatter(X[indices, 0], X[indices, 1], c='r', marker='^', s=100)
            ax1.scatter(X[x, 0], X[x, 1], c='pink', marker='^', s=100)
            ax1.set_title("Updated Similarities")
            plt.colorbar(p1, ax=ax1)

            p2 = ax2.scatter(X[:, 0], X[:, 1], c=np.exp(-similarities ** q))
            ax2.scatter(X[indices, 0], X[indices, 1], c='r', marker='^', s=100)
            ax2.set_title("Sampling Probabilities")
            plt.colorbar(p2, ax=ax2)
            plt.show()

        # sample next point proportionally to the q^th power of the distances
        if q != 1:
            vals = np.cumsum(np.exp(-similarities ** q))
        else:
            vals = np.cumsum(np.exp(-similarities))

        next_ind = np.searchsorted(vals, vals[-1] * random_state.uniform())
        indices = np.append(indices, next_ind)
    return indices

##############################################################################################################
# This is an efficient way to optimize a batch with vopt
# It is inspired by Gauss-Seidel
def vopt_greedy_batch(X, labels, act, acq, model, batch_size=1, given_batch=[], display=False,
                      display_all=False, max_iters=200):
    # Select the initial batch
    batch = []
    if len(given_batch) > 0:
        batch = np.array(given_batch.copy())
        batch_size = len(given_batch)
    else:
        batch = np.random.choice(act.candidate_inds, size=batch_size, replace=False)

    # Optimization parameters
    ind = -1
    iterations = 0
    iters_since_score_update = 0
    score = 0

    # Will go over the batch 5 times
    if max_iters == None:
        max_iters = max(5 * batch_size, 50)

    # Save initial params
    reset_labeled_set = act.current_labeled_set.copy()
    reset_labels = act.current_labels.copy()
    reset_cov_matrix = act.cov_matrix.copy()

    # Plot data
    if display:
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.scatter(X[batch, 0], X[batch, 1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
        plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
        plt.show()

    # Reset the labeled data
    act.current_labeled_set = reset_labeled_set.copy()
    act.current_labels = reset_labels.copy()
    act.cov_matrix = reset_cov_matrix.copy()

    # Add in all points
    act.update_labeled_data(batch, labels[batch])
    u = model.fit(act.current_labeled_set, act.current_labels)

    score_vals = []

    # Repeatedly optimize
    while iterations < max_iters:
        # Pick index for point to optimize
        ind = (ind + 1) % batch_size

        # Remove point: update labels
        act.current_labeled_set = np.delete(act.current_labeled_set, -batch_size + ind)
        act.current_labels = np.delete(act.current_labels, -batch_size + ind)
        # Remove point: update cov_matrix
        vk = act.evecs[batch[ind]]
        Cavk = act.cov_matrix @ vk
        ip = np.inner(vk, Cavk)
        act.cov_matrix += np.outer(Cavk, Cavk) / (act.gamma ** 2. - ip)

        # Select best point now
        act.candidate_inds = np.setdiff1d(act.training_set, act.current_labeled_set)
        u = model.fit(act.current_labeled_set, act.current_labels)
        all_scores = acq.compute_values(act, u)
        query_points = act.select_query_points(acq, u, oracle=None)
        query_labels = labels[query_points]

        # Plot
        if (display_all and display):
            plt.scatter(X[act.candidate_inds, 0], X[act.candidate_inds, 1], c=all_scores)
            plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
            #plt.scatter(X[current_batch, 0], X[current_batch, 1], c='c')
            plt.scatter(X[query_points, 0], X[query_points, 1], c='r', marker='*', s=100)
            plt.colorbar()
            plt.show()

        # Check if batch point updated
        point_updated = (batch[ind] != query_points[0])

        # Add point: update labels and cov_matrix
        act.update_labeled_data(query_points, query_labels)

        # Compute Score
        score = np.trace(act.cov_matrix)

        # Update batch
        batch[ind] = query_points[0]
        # Correct the labeled set (ordering)
        act.current_labeled_set[-batch_size:] = np.array(batch)
        act.current_labels = labels[act.current_labeled_set]

        iterations += 1
        score_vals.append(score)

        if point_updated:
            iters_since_score_update = 0
        else:
            iters_since_score_update += 1

        # End when fully optimized
        if iters_since_score_update >= batch_size:
            break

    # Reset the labeled data
    act.current_labeled_set = reset_labeled_set.copy()
    act.current_labels = reset_labels.copy()
    act.cov_matrix = reset_cov_matrix.copy()

    if display:
        # Plot scores
        plt.plot(np.arange(len(score_vals)), np.array(score_vals))
        plt.show()
        # Plot moons
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.scatter(X[batch, 0], X[batch, 1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
        plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
        plt.show()
        # Original heatmap
        act.candidate_inds = np.setdiff1d(act.training_set, act.current_labeled_set)
        all_scores = acq.compute_values(act, u)
        plt.scatter(X[act.candidate_inds, 0], X[act.candidate_inds, 1], c=all_scores)
        plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
        plt.scatter(X[batch, 0], X[batch, 1], c='c')
        plt.colorbar()
        plt.show()

    # Returns optimized batch, final score, and the scores along the way
    return batch, score, score_vals

##############################################################################################################
## util functions for batch active learning
def local_maxes(W, acq_array):
    # Look at the k nearest neighbors
    # If weights(v) >= weights(u) for all u in neighbors, then v is a local max
    local_maxes = []

    # TODO: acq should be passed in as an array with all the values. Put 0 for labeled set
    n = len(acq_array)
    for i in range(n):
        neighbors = W[i].nonzero()[1]  # Indices for neighbors of 1
        acq_vals = acq_array[neighbors]
        if acq_array[i] >= np.max(acq_vals):
            local_maxes.append(i)

    return local_maxes


def local_maxes_k(knn_ind, acq_array, k, top_cut=None, thresh=None):
    # Look at the k nearest neighbors
    # If weights(v) >= weights(u) for all u in neighbors, then v is a local max
    local_maxes = []
    K = knn_ind.shape[1]
    if k > K:
        k = K

    # TODO: acq should be passed in as an array with all the values. Put 0 for labeled set
    n = len(acq_array)
    for i in range(n):
        neighbors = knn_ind[i, :k]  # Indices for neighbors of 1
        acq_vals = acq_array[neighbors]
        if acq_array[i] >= np.max(acq_vals):
            local_maxes.append(i)

    local_maxes = np.array(local_maxes)
    if top_cut:
        acq_max_vals = acq_array[local_maxes]
        local_maxes = local_maxes[np.argsort(acq_max_vals)[-top_cut:]]
    if thresh:
        acq_max_vals = acq_array[local_maxes]
        local_maxes = local_maxes[acq_max_vals > thresh * np.max(acq_max_vals)]

    return local_maxes


def local_maxes_k_new(knn_ind, acq_array, k, top_num, thresh=0):
    # Look at the k nearest neighbors
    # If weights(v) >= weights(u) for all u in neighbors, then v is a local max
    local_maxes = np.array([])
    K = knn_ind.shape[1]
    if k > K:
        k = K

    sorted_ind = np.argsort(acq_array)[::-1]
    local_maxes = np.append(local_maxes, sorted_ind[0])
    global_max_val = acq_array[sorted_ind[0]]
    neighbors = knn_ind[sorted_ind[0], :k]
    sorted_ind = np.setdiff1d(sorted_ind, neighbors, assume_unique=True)

    while len(local_maxes) < top_num and len(sorted_ind) > 0:
        current_max_ind = sorted_ind[0]
        neighbors = knn_ind[current_max_ind, :k]
        acq_vals = acq_array[neighbors]
        sorted_ind = np.setdiff1d(sorted_ind, neighbors, assume_unique=True)
        if acq_array[current_max_ind] >= np.max(acq_vals):
            if acq_array[current_max_ind] < thresh * global_max_val:
                break
            local_maxes = np.append(local_maxes, current_max_ind)

    return local_maxes.astype(int)

###########################################################################################################
## functions of k-means batch active learning
def random_sample_val(val, sample_num, random_state=None):
    if random_state is None:
        random_state = 0
    random_state = check_random_state(random_state)
    cumval = np.cumsum(val)
    sampled_inds = np.array([]).astype(int)
    ind_list = np.arange(len(val))

    for i in range(sample_num):
        s_ind = np.searchsorted(cumval, cumval[-1] * random_state.uniform())

        sampled_inds = np.append(sampled_inds, ind_list[s_ind])
        cumval[s_ind:] -= val[ind_list[s_ind]]
        cumval = np.delete(cumval, s_ind)
        ind_list = np.delete(ind_list, s_ind)

    return sampled_inds

### functions about K-means betch active learning
def dist_angle(X, y, epsilon=1e-8):
    # y is the current guess for the center
    cos_sim = (X @ y - epsilon) / np.maximum(np.linalg.norm(X, axis=1) * np.linalg.norm(y), epsilon)
    # This epsilon stuff can give out of bounds
    if np.count_nonzero(np.abs(cos_sim) > 1) > 0:
        cos_sim = np.maximum(np.minimum(cos_sim, 1), -1)
    return np.arccos(cos_sim)


def diff_x_angle(X, y, P, epsilon=1e-8):
    theta = dist_angle(X, y)
    y_norm = np.linalg.norm(y)
    y_coeff = np.sum(P * theta / np.tan(theta)) / (y_norm ** 2)
    X_coeffs = (P * theta / np.sin(theta)) / y_norm
    X_normed = X / np.linalg.norm(X, axis=1).reshape((-1, 1))

    return (y_coeff * y - X_normed.T @ X_coeffs) / len(X)

    # return - y / np.sqrt(1 - np.inner(x, y) ** 2 + epsilon) * np.arccos( np.inner(x, y) - epsilon)


def dist_euclidean(X, y):
    return np.linalg.norm(X - y.reshape((1, -1)), axis=1)


def diff_x_euclidean(X, y, P, epsilon=1e-8):
    # P here is the weight matrix we design
    # We just pass in the column corresponding to y
    return (np.sum(P) * y - X.T @ P) / len(X)

## gradient descent
def gradient_descent(diff_fun, x0, args=None, alpha=0.1, max_iter=1000, epsilon=1e-6,
                     normalize=False, descent_alpha=True, descent_step=10, descent_rate=0.9):

    x = x0.copy()
    iteration = 0
    if args:
        grad = diff_fun(x, *args)
    else:
        grad = diff_fun(x)
    while np.linalg.norm(grad) > epsilon and iteration < max_iter:
        iteration += 1

        if args:
            grad = diff_fun(x, *args)
        else:
            grad = diff_fun(x)
        x -= alpha * grad
        if normalize:
            x = x / np.linalg.norm(x)
        if descent_alpha:
            if not iteration % descent_step:
                alpha *= alpha * descent_rate
    return x


def k_means_bal(X, P, batch_size, initial='k-means++', max_iter=50, dist_metric='euclidean',
                randseed=0, solve_mtd='GD', time_info=False, energy_val=False, alpha=0.1,
                combinatorial_weight=None, normalize=False):
    #############################################
    # inputs:
    ### X: input data matrix, size (N,d)
    ### P: weight matrix, size (N,N),
    ###    if the size of P is (N,), it is treated as the vector of acquisition function, use 1-A^TA to calculate P
    ### batch_size: number of centroids
    ### energy_val: bool, output energy information or not
    # outputs: curr_centroids, sorted_points, exact_centroids, E
    ### curr_centroids: indices of centroids
    ### sorted_points: points sorted in clusters
    ### exact_centroids: exact solution of centroids
    ### E: energy values, E[0]: energy on curr_centroids; E[1]: energy on exact centroids
    #############################################

    if normalize:
        if P.ndim == 1:
            P = (P / np.max(P)).copy()

    iteration = 0
    X = X.copy()

    if randseed:
        np.random.seed(randseed)

    if initial == 'k-means++':
        _, initial = _kmeans_plusplus(X, n_clusters=batch_size, random_state=check_random_state(randseed),
                                      x_squared_norms=None, n_local_trials=None)
    elif initial == 'random':
        initial = np.random.choice(len(X), size=batch_size, replace=False)

    curr_centroids = initial
    prev_centroids = None
    all_dists = np.zeros((len(curr_centroids), len(X)))

    if dist_metric == 'angular':
        X /= np.linalg.norm(X, axis=1)[:, None]
        dist_fun = dist_angle
        diff_x_fun = diff_x_angle
        normalize = True
    elif dist_metric == 'euclidean':
        dist_fun = dist_euclidean
        diff_x_fun = diff_x_euclidean
        normalize = False

    # K means algorithm ends iff the current centroids are not changing anymore
    while np.not_equal(curr_centroids, prev_centroids).any() and iteration < max_iter:
        sorted_points = [[] for _ in range(batch_size)]
        for i in range(len(curr_centroids)):
            all_dists[i, :] = dist_fun(X, X[curr_centroids[i]])

        if combinatorial_weight:
            weighted_all_dist = all_dists.copy()
            for i in range(batch_size):
                weighted_all_dist[i] *= combinatorial_weight(P, P[curr_centroids[i]])
            weighted_centroid_idx = np.argmin(weighted_all_dist, axis=0)
            for i in range(batch_size):
                sorted_points.append((weighted_centroid_idx == i).nonzero()[0])
        else:
            centroid_idx = np.argmin(all_dists, axis=0)
            for i in range(batch_size):
                sorted_points[i] = (centroid_idx == i).nonzero()[0]

        ## should use copy here
        prev_centroids = curr_centroids.copy()
        exact_centroids = []

        for i, ind_c in enumerate(curr_centroids):
            # full_diff = lambda x: np.mean(np.array([P[j, ind_c] * diff_x_fun(x, X[j]) for j in sorted_points[i]]), axis=0)
            P_row = np.zeros(len(X))
            if P.ndim == 1:
                if combinatorial_weight:
                    P_row[sorted_points[i]] = combinatorial_weight(P[sorted_points[i]], P[ind_c])
                else:
                    P_row[sorted_points[i]] = P[sorted_points[i]]
            else:
                P_row[sorted_points[i]] = P[ind_c, sorted_points[i]]

            if solve_mtd == 'GD':
                full_diff = lambda y: diff_x_fun(X, y, P_row)
                t_s = timeit.default_timer()
                cent_sol = gradient_descent(full_diff, X[ind_c], args=None, alpha=alpha, normalize=normalize)
                t_solve = timeit.default_timer() - t_s
                if time_info:
                    print('Time Gradient Descent:', t_solve)
                exact_centroids.append(cent_sol)

        for i in range(len(curr_centroids)):
            dists_c = dist_fun(X, exact_centroids[i])
            curr_centroids[i] = np.argmin(dists_c)

        iteration += 1

    if energy_val:
        E = np.zeros(2)
    else:
        E = None

    if energy_val:
        for i in range(batch_size):
            E[0] += np.sum(
                np.array([P[j, ind_c] * (dist_fun(X[curr_centroids[i]], X[j]) ** 2) for j in sorted_points[i]]), axis=0)
            E[1] += np.sum(
                np.array([P[j, ind_c] * (dist_fun(exact_centroids[i], X[j]) ** 2) for j in sorted_points[i]]), axis=0)

    return curr_centroids, sorted_points, exact_centroids, E, iteration


def k_means_bal_randsample(G, X, P, batch_size, sample_rate=0.05, initial='k-means++', max_iter=50, method='dijkstra',
                           eik_p=1.0, tau=0.1, ofs=0.2, randseed=0, time_info=False, knn_dist=None, energy_val=True,
                           combinatorial_weight=None, normalize=False, weighted_dist=True):
    #############################################
    # inputs:
    ### G: graph, the kernel should be 'distance'
    ### X: dataset related to graph G, used to calculate the initial k-means++
    ### P: weight matrix, size (N,N),
    ###    if the size of P is (N,), it is treated as the vector of acquisition function, use 1-A^TA to calculate P
    ### batch_size: number of centroids
    ### energy_val: bool, output energy information or not
    # outputs: curr_centroids, sorted_points, exact_centroids, E
    ### curr_centroids: indices of centroids
    ### sorted_points: points sorted in clusters
    ### exact_centroids: exact solution of centroids
    ### E: energy values, E[0]: energy on curr_centroids; E[1]: energy on exact centroids
    #############################################

    iteration = 0
    if randseed:
        np.random.seed(randseed)
    E = None

    if normalize:
        if P.ndim == 1:
            P = (P / np.max(P)).copy()

    if initial == 'k-means++':
        _, initial = _kmeans_plusplus(X, n_clusters=batch_size, random_state=check_random_state(randseed),
                                      x_squared_norms=None, n_local_trials=None)

    curr_centroids = initial
    prev_centroids = None

    N = G.num_nodes
    W = G.weight_matrix

    if method == 'peikonal':
        # print("Preparing knn density estimator for p-eikonal")
        if knn_dist is not None:
            alpha = 2.
            d = np.max(knn_dist, axis=1)
            kde = (d / d.max()) ** (-1)
            f = kde ** (-alpha)
        else:
            print("No knn dist info provided, defaulting to just f = 1")
            f = 1.0

    # K means algorithm ends iff the current centroids are not changing anymore
    while np.not_equal(curr_centroids, prev_centroids).any() and iteration < max_iter:
        iteration += 1
        sorted_points = []
        if energy_val:
            # total energy
            E = 0

        if method == 'dijkstra':
            if combinatorial_weight:
                all_dists = np.zeros((batch_size, N))
                for i, ind_c in enumerate(curr_centroids):
                    all_dists[i] = G.dijkstra([ind_c])
                centroid_idx = curr_centroids[np.argmin(all_dists, axis=0)]
            else:
                _, centroid_idx = G.dijkstra(curr_centroids, return_cp=True)
        elif method == 'peikonal':
            all_dists = np.zeros((batch_size, N))
            for i, ind_c in enumerate(curr_centroids):
                all_dists[i] = G.peikonal([ind_c], p=eik_p, f=f)
            centroid_idx = curr_centroids[np.argmin(all_dists, axis=0)]
        elif method == 'poisson':
            all_dists = np.zeros((batch_size, N))
            for i, ind_c in enumerate(curr_centroids):
                w = get_poisson_weighting(G, [ind_c], tau=tau)
                all_dists[i] = 1. / (ofs + w)
            centroid_idx = curr_centroids[np.argmin(all_dists, axis=0)]

        if combinatorial_weight and weighted_dist:
            weighted_all_dist = all_dists.copy()
            for i in range(batch_size):
                weighted_all_dist[i] *= combinatorial_weight(P, P[curr_centroids[i]])
            weighted_centroid_idx = np.argmin(weighted_all_dist, axis=0)
            for i in range(batch_size):
                sorted_points.append((weighted_centroid_idx == i).nonzero()[0])
        else:
            for i in range(batch_size):
                sorted_points.append((centroid_idx == curr_centroids[i]).nonzero()[0])


        # for i in range(batch_size):
        #     if method == 'dijkstra':
        #         sorted_points.append((centroid_idx == curr_centroids[i]).nonzero()[0])
        #     elif method == 'peikonal' or method == 'poisson':
        #         sorted_points.append((centroid_idx == i).nonzero()[0])

        prev_centroids = curr_centroids.copy()

        for i, c_ind in enumerate(prev_centroids):
            min_energy = np.inf
            G_sub = gl.graph(W[sorted_points[i]][:, sorted_points[i]])
            num_samples = int(len(sorted_points[i]) * sample_rate)
            sampled_inds = np.random.permutation(len(sorted_points[i]))[:num_samples]

            # add c_ind into sampled_inds
            new_ind_c = (sorted_points[i] == c_ind).nonzero()[0][0]
            sampled_inds = np.insert(sampled_inds, 0, new_ind_c)

            t_s = timeit.default_timer()

            for s_ind in sampled_inds:
                # weights for s_ind
                if P.ndim == 1:
                    if combinatorial_weight:
                        P_row = combinatorial_weight(P[sorted_points[i]], P[s_ind])
                    else:
                        P_row = P[sorted_points[i]]
                else:
                    P_row = P[s_ind, sorted_points[i]]

                # dist between s_ind to other nodes in this cluster
                if method == 'dijkstra':
                    dist_sub = G_sub.dijkstra([s_ind])
                elif method == 'peikonal':
                    dist_sub = G_sub.peikonal([s_ind], p=eik_p, f=f)
                elif method == 'poisson':
                    w_sub = get_poisson_weighting(G_sub, [s_ind], tau=tau)
                    dist_sub = 1. / (ofs + w_sub)

                # energy related s_ind
                curr_energy = np.sum(P_row * dist_sub ** 2)
                if curr_energy < min_energy:
                    min_energy = curr_energy
                    curr_centroids[i] = sorted_points[i][s_ind]

            t_solve = timeit.default_timer() - t_s
            if time_info:
                print('Time Gradient Descent:', t_solve)

            if energy_val:
                E += min_energy

    return curr_centroids, sorted_points, E, iteration

###########################################################################################################
#particle
def greedy_optimization(X, this_batch, acq_vals, dist_coeff):

    dist_fun = lambda A: -.01 * np.log(A)

    batch = this_batch.copy()
    batch_size = len(batch)
    acq_coeff = (batch_size - 1) / 2.0
    num_iters = 6 * batch_size

    energy = np.zeros(num_iters)

    for i in range(num_iters):
        batch_old = batch.copy()
        for j in range(batch_size):
            # batch = np.random.choice(act.candidate_inds, size=batch_size, replace=False)
            # Want to get energy for all choices of this point
            batch_tmp = np.delete(batch, j)
            acq_part = np.sum(acq_vals[batch_tmp]) + acq_vals
            acq_part[batch_tmp] = 0

            # This is all distances from batch_tmp
            # others = np.setdiff1d(np.arange(len(X)), batch_tmp)
            dist_mat = metrics.pairwise_distances(X[batch_tmp], X)
            dist_mat[np.arange(batch_size - 1), batch_tmp] = 1

            # Lets do the stuff for inter-batch_tmp distances
            inter_dist_mat = dist_mat[:, batch_tmp].copy()
            # inter_dist_mat[np.arange(batch_size-1), np.arange(batch_size-1)] = 1
            inter_dist_weight = dist_fun(dist_coeff * inter_dist_mat)
            inter_dist_part = (np.sum(inter_dist_weight) - batch_size) / 2.0

            # Now we do the outer dist part
            outer_dist_mat = dist_fun(dist_coeff * dist_mat).copy()
            outer_dist_mat[np.arange(batch_size - 1), batch_tmp] = np.inf
            outer_dist_part = np.sum(outer_dist_mat, axis=0)

            # combine
            dist_part = inter_dist_part + outer_dist_part

            # Pick the best point
            energy_tmp = acq_coeff * acq_part - dist_part
            batch[j] = np.argmax(energy_tmp)
            energy[i] = energy_tmp[batch[j]]

            # batch_hist[i, :] = batch.copy()
        if len(np.setdiff1d(batch_old, batch)) == 0:
            break
    return energy, batch  # batch_hist


###########################################################################################################
## implement batch active learning function
def coreset_run_experiment(X, labels, W, coreset, num_iter=1, method='Laplace',
                           display=False, use_prior=False, al_mtd='local_max', debug=False,
                           acq_fun='vopt', knn_data=None, mtd_para=None,
                           savefig=False, savefig_folder='../BAL_figures', batchsize=5,
                           dist_metric='euclidean', knn_size=50, q=1, thresholding=0, randseed=0):

    '''
        al_mtd: 'local_max', 'global_max', 'rs_kmeans', 'gd_kmeans', 'acq_sample', 'greedy_batch', 'particle', 'random', 'topn_max'
    '''
    if knn_data:
        knn_ind, knn_dist = knn_data
    else:
        knn_data = gl.weightmatrix.knnsearch(X, knn_size, method='annoy', similarity=dist_metric)
        knn_ind, knn_dist = knn_data

    if al_mtd == 'local_max':
        if mtd_para:
            k, thresh = mtd_para
        else:
            k, thresh = np.inf, 0
    elif al_mtd == 'gd_kmeans':
        if mtd_para:
            alpha, cw, max_iter, normalize = mtd_para
        else:
            alpha, cw, max_iter, normalize = 1, None, 200, False
    elif al_mtd == 'rs_kmeans':
        if mtd_para:
            sample_rate, rs_mtd, eik_p, tau, ofs, cw, max_iter, normalize, weighted_dist = mtd_para
        else:
            sample_rate, rs_mtd, eik_p, tau, ofs, cw, max_iter, normalize, weighted_dist = 0.05, 'dijkstra', 1.0, 0.1, 0.2, None, 200, False, False
    elif al_mtd == 'particle':
        if mtd_para:
            dist_coeff = mtd_para
        else:
            dist_coeff = 1e3

    list_num_labels = []
    list_acc = np.array([]).astype(np.float64)

    train_ind = coreset
    if use_prior:
        class_priors = gl.utils.class_priors(labels)
    else:
        class_priors = None

    if al_mtd == 'rs_kmeans':
        W_rs = gl.weightmatrix.knn(X, knn_size, kernel='distance', knn_data=knn_data)

    if method == 'Laplace':
        model = gl.ssl.laplace(W, class_priors=class_priors)
    elif method == 'rw_Laplace':
        model = gl.ssl.laplace(W, class_priors, reweighting='poisson')
    elif method == 'Poisson':
        model = gl.ssl.poisson(W, class_priors)

    if acq_fun == 'mc':
        acq_f = al.model_change()
    elif acq_fun == 'vopt':
        acq_f = al.v_opt()
    elif acq_fun == 'uc':
        acq_f = al.uncertainty_sampling()
    elif acq_fun == 'mcvopt':
        acq_f = al.model_change_vopt()

    if debug:
        t_al_s = timeit.default_timer()
    act = al.active_learning(W, train_ind, labels[train_ind], eval_cutoff=min(200, len(X) // 2))
    
    #print("DEBUGGING FOR THE AL_THING")
    #print(W)
    #print(train_ind)
    #print(labels[train_ind])
    #print(act)
    #print(model)
    #print(act.current_labeled_set)
    #print(act.current_labels)
    #print(type(act.current_labeled_set))
    #print(type(act.current_labels))

    u = model.fit(act.current_labeled_set, act.current_labels)  # perform classification with GSSL classifier
    if debug:
        t_al_e = timeit.default_timer()
        print('Active learning setup time = ', t_al_e - t_al_s)

    current_label_guesses = model.predict()

    #acc = np.sum(current_label_guesses == labels) / u.shape[0]
    acc = gl.ssl.ssl_accuracy(current_label_guesses, labels, len(act.current_labeled_set))

    if display:
        plt.scatter(X[:, 0], X[:, 1], c=current_label_guesses)
        plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
        if savefig:
            plt.savefig(os.path.join(savefig_folder, 'bal_coreset_.png'))
        plt.show()

        print("Size of coreset = ", len(coreset))
        print("Using ", 100.0 * len(coreset) / len(labels), '%', "of the data")
        print("Current Accuracy is ", acc, '%')

    # record labeled set and accuracy value
    list_num_labels.append(len(act.current_labeled_set))
    list_acc = np.append(list_acc, acc)

    for iteration in range(num_iter):  # todo get rid of printing times

        if debug:
            t_iter_s = timeit.default_timer()

        act.candidate_inds = np.setdiff1d(act.training_set, act.current_labeled_set)
        if acq_fun in ['mc', 'uc', 'mcvopt']:
            acq_vals = acq_f.compute_values(act, u)
        elif acq_fun == 'vopt':
            acq_vals = acq_f.compute_values(act)

        modded_acq_vals = np.zeros(len(X))
        modded_acq_vals[act.candidate_inds] = acq_vals

        if al_mtd == 'local_max':
            if knn_data:
                batch = local_maxes_k_new(knn_ind, modded_acq_vals, k, batchsize, thresh)
                # batch = local_maxes_k(knn_ind, modded_acq_vals, k, top_cut, thresh)
        elif al_mtd == 'global_max':
            batch = act.candidate_inds[np.argmax(acq_vals)]
        elif al_mtd == 'gd_kmeans':
            batch_inds, clusters, _, _, iterations = k_means_bal(X[act.candidate_inds], acq_vals ** q, batchsize,
                                                                 initial='k-means++', max_iter=max_iter,
                                                                 dist_metric=dist_metric,
                                                                 randseed=randseed, solve_mtd='GD', time_info=False,
                                                                 energy_val=False,
                                                                 alpha=alpha, combinatorial_weight=cw, normalize=normalize)
            batch = act.candidate_inds[batch_inds]

        elif al_mtd == 'rs_kmeans':
            G_rs = gl.graph(W_rs[act.candidate_inds, :][:, act.candidate_inds])
            batch_inds, clusters, _, iterations = k_means_bal_randsample(G_rs, X[act.candidate_inds], acq_vals ** q,
                                                                         batchsize, sample_rate=sample_rate,
                                                                         initial='k-means++', max_iter=max_iter,
                                                                         method=rs_mtd, eik_p=eik_p, tau=tau,
                                                                         ofs=ofs, randseed=randseed, time_info=False,
                                                                         knn_dist=None, energy_val=False,
                                                                         combinatorial_weight=cw, normalize=normalize,
                                                                         weighted_dist=weighted_dist)
            batch = act.candidate_inds[batch_inds]
        elif al_mtd == 'acq_sample':
            batch_inds = random_sample_val(acq_vals ** q, sample_num=batchsize)
            batch = act.candidate_inds[batch_inds]
        elif al_mtd == 'greedy_batch':
            if acq_fun == 'vopt':
                batch, _, _ = vopt_greedy_batch(X, labels, act, acq_f, model, batch_size=batchsize, given_batch=[])
            else:
                raise ValueError('The acquisition function can only be vopt if you want to use the greedy batch active leanring')
        elif al_mtd == 'greedy_lm':
            if acq_fun == 'vopt':
                #Just using default values
                batch = local_maxes_k_new(knn_ind, modded_acq_vals, np.inf, batchsize, 0)
                batch, _, _ = vopt_greedy_batch(X, labels, act, acq_f, model, batch_size=batchsize, given_batch=batch)
            else:
                raise ValueError('The acquisition function can only be vopt if you want to use the greedy batch active leanring')
        elif al_mtd == 'particle':
            batch_init = np.random.choice(act.candidate_inds, size=batchsize, replace=False)
            _, batch = greedy_optimization(X, batch_init, modded_acq_vals, dist_coeff=dist_coeff)
        elif al_mtd == 'random':
            batch = np.random.choice(act.candidate_inds, size=batchsize, replace=False)
        elif al_mtd == 'topn_max':
            batch = act.candidate_inds[np.argsort(acq_vals)[-batchsize:]]

        if thresholding > 0:
            max_acq_val = np.max(acq_vals)
            batch = batch[modded_acq_vals[batch] >= (thresholding * max_acq_val)]

        if debug:
            t_localmax_e = timeit.default_timer()
            print("Batch Active Learning time = ", t_localmax_e - t_iter_s)
            print("Batch inds:", batch)
            if al_mtd == 'gd_kmeans' or al_mtd == 'rs_kmeans':
                print("Number of iterations:", iterations)

        if display:
            plt.scatter(X[act.candidate_inds, 0], X[act.candidate_inds, 1], c=acq_vals)
            plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
            plt.scatter(X[batch, 0], X[batch, 1], c='m', marker='*', s=100)
            plt.colorbar()
            if savefig:
                plt.savefig(os.path.join(savefig_folder, 'bal_acq_vals_b' + str(iteration) + '.png'))
            plt.show()

        act.update_labeled_data(batch, labels[batch])  # update the active_learning object's labeled set

        u = model.fit(act.current_labeled_set, act.current_labels)
        current_label_guesses = model.predict()
        #acc = np.sum(current_label_guesses == labels) / u.shape[0]
        acc = gl.ssl.ssl_accuracy(current_label_guesses, labels, len(act.current_labeled_set))
        if debug:
            t_modelfit_e = timeit.default_timer()
            print('Model fit time = ', t_modelfit_e - t_localmax_e)

        list_num_labels.append(len(act.current_labeled_set))
        list_acc = np.append(list_acc, acc)

        if display:
            print("Next batch is", batch)
            print("Current number of labeled nodes", len(act.current_labeled_set))
            print("Current Accuracy is ", acc, '%')

            plt.scatter(X[:, 0], X[:, 1], c=current_label_guesses)
            plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
            if savefig:
                plt.savefig(os.path.join(savefig_folder, 'bal_acq_vals_a' + str(iteration) + '.png'))
            plt.show()

            if al_mtd == 'gd_kmeans' or al_mtd == 'rs_kmeans':
                if dist_metric == 'angular':
                    Y = X[act.candidate_inds] / np.linalg.norm(X[act.candidate_inds], axis=1)[:, None]
                else:
                    Y = X[act.candidate_inds].copy()
                for i in range(len(clusters)):
                    plt.scatter(Y[clusters[i], 0], Y[clusters[i], 1], alpha=0.35)
                plt.scatter(X[batch, 0], X[batch, 1], c='black', marker='o', s=100)
                if savefig:
                    plt.savefig(os.path.join(savefig_folder, 'bal_kmeans_clusters_' + str(iteration) + '.png'))
                plt.show()

        if debug:
            t_iter_e = timeit.default_timer()
            print("Iteration:", iteration, "Iteration time = ", t_iter_e - t_iter_s)

    if display:
        plt.plot(np.array(list_num_labels), list_acc)
        plt.show()

    labeled_ind = act.current_labeled_set

    # reset active learning object
    act.reset_labeled_data()

    return labeled_ind, list_num_labels, list_acc


## perform batch active learning for several times and show average&highest result
def perform_al_experiment(dataset_chosen, embedding_mode='just_transfer', acq_fun_list = ['uc', 'vopt', 'mc', 'mcvopt'],
                          density_radius_param = .5, knn_num=20, num_iter=1, method='Laplace', 
                          display=False, use_prior=False, al_mtd='local_max', 
                          debug=False, acq_fun='vopt', knn_data=None, mtd_para=None,
                           savefig=False, savefig_folder='../BAL_figures', batchsize=5,
                           dist_metric='euclidean', knn_size=50, q=1, thresholding=0, randseed=0,
                           experiment_time=10):
    
    highest_accuracy_list = [0] * len(acq_fun_list)
    average_accuracy_list = [0] * len(acq_fun_list)
    lowest_accuracy_list = [0] * len(acq_fun_list)
    
    count_unusual_cases = 0
    unusual_acc_list = []
    
    #Perform the experiment for experiment_time times
    for i in range(experiment_time):
        start = timeit.default_timer()
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        if dataset_chosen == 'open_sar':
          #Load labels
          data, labels = utils.load_dataset('open_sar_ship', return_torch = False, concatenate = True)
        elif dataset_chosen == 'fusar':
          #Load labels
          data, labels = utils.load_dataset('fusar', return_torch = False, concatenate = True)
        else:
          assert False, "Chosen dataset could not be loaded. Check for typos"
        
        #Mimic that we know a percentage of data, and don't know for the rest
        #Do transfer learning merely using these
        percent_known_data = 0
        if dataset_chosen == 'open_sar':
          percent_known_data = 0.07
        else:
          percent_known_data = 0.07
        known_data_ind = gl.trainsets.generate(labels, rate=percent_known_data).tolist()
        known_data = data[known_data_ind]
        known_labels = labels[known_data_ind]
        
        #Generate the initial set
        initial = gl.trainsets.generate(labels, rate=1).tolist()
        
        #Percent of known data to use as training data for transfer learning
        training_percent = 0.7
        transfer_train_ind = random.sample(range(len(known_data)), round(len(known_data)*training_percent))
        transfer_testing_ind = np.array([ind for ind in range(len(known_data)) if ind not in transfer_train_ind]).astype(int)
        
        #Convert to torch for use
        known_data = torch.from_numpy(known_data)
        known_labels = torch.from_numpy(known_labels)
        
        
        #Setup the training and testing data for transfer learning
        training_data = known_data[transfer_train_ind]
        training_label = known_labels[transfer_train_ind]
        testing_data = known_data[transfer_testing_ind]
        testing_label = known_labels[transfer_testing_ind]
        
        print("Transfer learning training data: " + str(len(training_data)))
        print("Transfer learning testing data: " + str(len(testing_data)))
        
        data_info=[training_data, training_label, testing_data, testing_label]
        
                
        if dataset_chosen == 'open_sar':
            #Load encoded dataset
            if embedding_mode == 'just_transfer':
                X, labels = utils.encode_pretrained('open_sar_ship', 'AlexNet', transformed=True)
            else:
                X = utils.encode_transfer_learning('open_sar_ship', model_type='AlexNet', transfer_batch_size=64, epochs=30, data_info=data_info)
            #Load labels
            _, labels = utils.load_dataset('open_sar_ship', return_torch = False, concatenate = True)
            knn_data = gl.weightmatrix.knnsearch(X, knn_num, method='annoy', similarity='angular')
        elif dataset_chosen == 'fusar':
            #Load encoded dataset
            if embedding_mode == 'just_transfer':
                X, labels = utils.encode_pretrained('fusar', 'ShuffleNet', normalized=True, transformed=True)
            else:
                X = utils.encode_transfer_learning('fusar', model_type='ShuffleNet', transfer_batch_size=64, epochs=30, data_info=data_info)
            #Load labels
            _, labels = utils.load_dataset('fusar', return_torch = False, concatenate = True)
            knn_data = gl.weightmatrix.knnsearch(X, knn_num, method='annoy', similarity='angular')
        else:
            assert False, "Chosen dataset could not be loaded. Check for typos"
        
        
        #print("***HERE: ", type(labels))
        
        print("Constructing Graph Learning Objects")
        W = gl.weightmatrix.knn(X, knn_num, kernel = 'gaussian', knn_data=knn_data)
        G = gl.graph(W)
        end = timeit.default_timer()
        
        
        print("Embedding Complete")
        print(f"Time taken = {end - start}")


        #Generate Coreset
        #Use the percent radius because it should be more robust across datasets
        coreset = coreset_dijkstras(G, rad = .2, DEBUGGING=False, data = X, initial=initial, 
                                        density_info = (True, density_radius_param, 1), knn_data=knn_data)
        print("Coreset Size = {}\t Percent of data = {}%".format(len(coreset), round(100 * len(coreset) / len(X), 2)))
        print("Coreset = ", coreset)
        
        
        batchsize = 15
        if dataset_chosen == 'open_sar':
            max_new_samples = 690
        else:
            max_new_samples = 2910
        
        acq_fun_ind = 0
        for acq_fun in acq_fun_list:
        
            if al_mtd != 'global_max':
                num_iter = int(max_new_samples/batchsize)
            else:
                num_iter = max_new_samples
        
            if acq_fun == 'vopt' or acq_fun == 'mcvopt':
                 num_iter += 1
        
        
            print(acq_fun, al_mtd)
            start = timeit.default_timer() 
            
            ##DEBUGGING
            #print(X)
            #print(labels)
            #print(W)
            #print(coreset)
            #The error is that we need to pass everything in as numpy arrays
            

            _, list_num_labels, list_acc = coreset_run_experiment(X, labels, W, coreset, num_iter=num_iter, method=method,
                                    display=False, use_prior=False, al_mtd=al_mtd, debug=False,
                                    acq_fun=acq_fun, knn_data=knn_data, mtd_para=None,
                                    savefig=False, savefig_folder='../BAL_figures', batchsize=batchsize,
                                    dist_metric='angular', q=10, thresholding=0, randseed=0)
            
            average_accuracy_list[acq_fun_ind] += list_acc[-1]
            if list_acc[-1] > highest_accuracy_list[acq_fun_ind]:
                highest_accuracy_list[acq_fun_ind] = list_acc[-1]
            if i == 0:
                lowest_accuracy_list[acq_fun_ind] = list_acc[-1]
            elif list_acc[-1] < lowest_accuracy_list[acq_fun_ind]:
                lowest_accuracy_list[acq_fun_ind] = list_acc[-1]
                
            if list_acc[-1] <= 0.8669:
                count_unusual_cases += 1
                unusual_acc_list.append(list_acc[-1])
                
            acq_fun_ind += 1
    
    average_accuracy_list = [acc/experiment_time for acc in average_accuracy_list]
    
    #Print out the experiment result
    for i in range(len(acq_fun_list)):
        print(acq_fun_list[i] + " average accuracy: " + str(average_accuracy_list[i]))
        print(acq_fun_list[i] + " highest accuracy: " + str(highest_accuracy_list[i]))
        print(acq_fun_list[i] + " lowest accuracy: " + str(lowest_accuracy_list[i]))
    
    print(unusual_acc_list)
    print("Unusual behavior happens " + str(count_unusual_cases) + " times")

    

    

#########################################################################################
## toy datasets
def gen_checkerboard_3(num_samples = 500, randseed = 123):
      np.random.seed(randseed)
      X = np.random.rand(num_samples, 2)
      labels = np.mod(np.floor(X[:, 0] * 3) + np.floor(X[:, 1] * 3), 3).astype(np.int64)

      return X, labels

def gen_stripe_3(num_samples = 500, width = 1/3, randseed = 123):
      np.random.seed(randseed)
      X = np.random.rand(num_samples, 2)
      labels = np.mod(np.floor(X[:, 0] / width + X[:, 1] / width), 3).astype(np.int64)

      return X, labels
