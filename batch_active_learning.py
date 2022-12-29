# Some functions for doing batch active learning and coreset selection
# Authors: James, Bohan, and Zheng

import timeit
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

import graphlearning.active_learning as al
import graphlearning as gl

import utils


#SKLearn Imports
from sklearn.utils import check_random_state #only in k_means stuff (acq_sample)





################################################################################
## Default Parameters

#TODO: Update the code to include these as defaults
DENSITY_RADIUS = .2
BATCH_SIZE = 15

################################################################################
### coreset functions
def density_determine_rad(G, x, proportion, r_0=1.0, tol=.02):
    # Determines the radius necessary so that a certain proportion of the data
    # falls in B_r(x). This is a lazy way and more efficient code could be
    # written in c. The proportion that we seek is (p-tol, p+tol) where tol is
    # some allowable error and p is the desired proportion
    
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


def coreset_dijkstras(G, rad, DEBUGGING=False, data=None, initial=[],
        randseed=123, density_info=(False, 0, 1.0), similarity='euclidean',
        knn_data=None):
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
            square_dataset = np.abs(np.max(data[:, 0]) - np.min(data[:, 0]) - 1) < .05 and np.abs(np.max(data[:, 1]) - np.min(data[:, 1]) - 1) < .05
                
            #The following is for the square dataset
            if square_dataset:
                #Save the initial dataset also
                if len(coreset) == 1:
                    plt.scatter(data[:, 0], data[:, 1])
                    plt.axis('square')
                    plt.axis('off')
                    plt.savefig('DAC Plots/coreset0.png',bbox_inches='tight')
                    plt.show()
                #If not initial, do this
                plt.scatter(data[:, 0], data[:, 1])
                plt.scatter(data[points_seen==1, 0], data[points_seen==1, 1], c='k')
                plt.scatter(data[coreset, 0], data[coreset, 1], c='r', s=100)
                plt.scatter(data[perim, 0], data[perim, 1], c='y')
                plt.axis('square')
                plt.axis('off')
                plt.savefig('DAC Plots/coreset' + str(len(coreset)) + '.png',bbox_inches='tight')
            else:
                plt.scatter(data[:, 0], data[:, 1])
                plt.scatter(data[points_seen==1, 0], data[points_seen==1, 1], c='k')
                plt.scatter(data[coreset, 0], data[coreset, 1], c='r')
                plt.scatter(data[perim, 0], data[perim, 1], c='y')
                
            plt.show()

        if iterations >= 1000:
            break
        iterations += 1
    return coreset




################################################################################
## util functions for batch active learning


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

################################################################################
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




################################################################################

## implement batch active learning function
def coreset_run_experiment(X, labels, W, coreset, num_iter=1, method='Laplace',
        display=False, use_prior=False, al_mtd='local_max', debug=False,
        acq_fun='uc', knn_data=None, mtd_para=None, savefig=False,
        savefig_folder='../BAL_figures', batchsize=BATCH_SIZE, dist_metric='euclidean',
        knn_size=50, q=1, thresholding=0, randseed=0):

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

    list_num_labels = []
    list_acc = np.array([]).astype(np.float64)

    train_ind = coreset
    if use_prior:
        class_priors = gl.utils.class_priors(labels)
    else:
        class_priors = None

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

    #Time at start of active learning
    t_al_s = timeit.default_timer()
    act = al.active_learning(W, train_ind, labels[train_ind], eval_cutoff=min(200, len(X) // 2))
    
    # perform classification with GSSL classifier
    u = model.fit(act.current_labeled_set, act.current_labels)
    if debug:
        t_al_e = timeit.default_timer()
        print('Active learning setup time = ', t_al_e - t_al_s)

    current_label_guesses = model.predict()
    
    acc = gl.ssl.ssl_accuracy(current_label_guesses, labels, len(act.current_labeled_set))

    if display:
        plt.scatter(X[:, 0], X[:, 1], c=current_label_guesses)
        plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
        if savefig:
            plt.axis('square')
            plt.axis('off')
            plt.savefig(os.path.join(savefig_folder, 'bal_coreset_.png'),bbox_inches='tight')
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
        
        #TODO: REMOVE THIS
        #print("Candidate index size", len(act.candidate_inds))

        if al_mtd == 'local_max':
            if knn_data:
                batch = local_maxes_k_new(knn_ind, modded_acq_vals, k, batchsize, thresh)
                # batch = local_maxes_k(knn_ind, modded_acq_vals, k, top_cut, thresh)
        elif al_mtd == 'global_max':
            batch = act.candidate_inds[np.argmax(acq_vals)]
        elif al_mtd == 'acq_sample':
            batch_inds = random_sample_val(acq_vals ** q, sample_num=batchsize)
            batch = act.candidate_inds[batch_inds]
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

        if display:
            plt.scatter(X[act.candidate_inds, 0], X[act.candidate_inds, 1], c=acq_vals)
            plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c='r')
            plt.scatter(X[batch, 0], X[batch, 1], c='m', marker='*', s=100)
            plt.colorbar()
            if savefig:
                plt.axis('square')
                plt.axis('off')
                plt.savefig(os.path.join(savefig_folder, 'bal_acq_vals_b' + str(iteration) + '.png'),bbox_inches='tight')
            plt.show()

        act.update_labeled_data(batch, labels[batch])  # update the active_learning object's labeled set

        u = model.fit(act.current_labeled_set, act.current_labels)
        current_label_guesses = model.predict()
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
                plt.axis('square')
                plt.axis('off')
                plt.savefig(os.path.join(savefig_folder, 'bal_acq_vals_a' + str(iteration) + '.png'),bbox_inches='tight')
            plt.show()

        if debug:
            t_iter_e = timeit.default_timer()
            print("Iteration:", iteration, "Iteration time = ", t_iter_e - t_iter_s)
    
    t_end = timeit.default_timer()
    t_total = t_end - t_al_s

    if display:
        plt.plot(np.array(list_num_labels), list_acc)
        plt.show()

    labeled_ind = act.current_labeled_set

    # reset active learning object
    act.reset_labeled_data()

    if display:
        #Don't want to return the time if we are also displaying things
        return labeled_ind, list_num_labels, list_acc
    else:
        return labeled_ind, list_num_labels, list_acc, t_total


## perform batch active learning for several times and show average&highest result
def perform_al_experiment(dataset_chosen, embedding_mode='just_transfer',
        acq_fun_list = ['uc', 'vopt', 'mc', 'mcvopt'],
        density_radius_param = .5, knn_num=20, num_iter=1, method='Laplace',
        display=False, use_prior=False, al_mtd='local_max', debug=False,
        acq_fun='uc', knn_data=None, mtd_para=None, savefig=False,
        savefig_folder='../BAL_figures', batchsize=BATCH_SIZE, dist_metric='euclidean',
        knn_size=50, q=1, thresholding=0, randseed=0, experiment_time=10):
    
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

    

    

################################################################################
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
