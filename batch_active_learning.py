# batch_active_learning.py
"""
Authors: James, Bohan, and Zheng

Some functions for doing batch active learning and coreset selection

Finish docstring
"""
# TODO: This is mostly done with comments

import timeit
import os
import random

from typing import Optional, Union
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
import torch

import graphlearning.active_learning as al
import graphlearning as gl

from scipy.sparse import csr_matrix

import utils


# TODO: Function comments
# Please include the exact python type
# Complex functions. Refer to density_determine_rad for an example
"""
Function description

:param p1: description
    additional details if necessary
:param p2: description
...
:param pn: description

:return:
    variable 1: description
    variable 2: description
"""

# Simple functions.
"""Quick function description"""


########
# For type checking
import functools


def check_types(fun):
    """Decorator for checking types"""

    @functools.wraps(fun)
    def types_wrapper(*args, **kwargs):
        out = fun(*args, **kwargs)
        print(fun.__name__)
        print("in")
        for v in args:
            print("\t", type(v))
        for k, v in kwargs.items():
            print("\t", k, type(v))
        print("out")
        if isinstance(out, Iterable):
            for x in out:
                print("\t", type(x))
        else:
            print("\t", type(out))
        print()
        return out

    return types_wrapper


########


################################################################################
## Default Parameters

DENSITY_RADIUS: float = 0.2
BATCH_SIZE: int = 15

ACQUISITION_FUNCTIONS = ["uc", "vopt", "mc", "mcvopt"]
AL_METHODS = ["local_max", "random", "topn_max", "acq_sample", "global_max"]
AL_METHOD_NAMES = ["LocalMax", "Random", "TopMax", "Acq_sample", "Sequential"]

# TODO: Check that these are originals
MAX_NEW_SAMPLES_DICT = {"mstar": 481, "open_sar_ship": 690, "fusar": 3060}
FINE_TUNED_MAX_NEW_SAMPLES_DICT = {"mstar": 137, "open_sar_ship": 574, "fusar": 2816}

BALOutputType = Union[
    tuple[np.ndarray, list[int], np.ndarray, float],
    tuple[np.ndarray, list[int], np.ndarray],
]

################################################################################
### coreset functions

# @check_types
def density_determine_rad(
    G: gl.graph,
    x: int,
    proportion: float,
    r_0: float = 1.0,
    tol: float = 0.02,
) -> float:
    """
    Returns the radius, 'r', required for B_r(x) to contain a fixed proportion,
        'proportion', of the nodes in the graph. This uses the bisection method
        and more efficient code could be written in c. Starts by picking
        boundary points for the bisection method. The final radius will satisfy
            -tol <= |B_r(x)| / |V(G)| - proportion <= tol

    :param G: Graph object
    :param x: Node index
    :param proportion: Proportion of data desired in B_r(x)
    :param r_0: Initial radius to try for bisection method.
    :param tol: Allowable error tolerance in proportion calculation

    :return: Radius r
    """

    n = G.num_nodes
    r = r_0
    dists = G.dijkstra(bdy_set=[x], max_dist=r)
    p = np.count_nonzero(dists < r) * 1.0 / n

    iterations: int = 0
    a = 0.0
    b = 0.0
    # If within some tolerance of the proportion, just return
    if p >= proportion - tol and p <= proportion + tol:
        return p
    # If radius too large, initialize a, b for bisection
    elif p > proportion + tol:
        a = 0
        b = r
    # If radius too small, repeatedly increase until we can use bisection
    else:
        while p < proportion - tol:
            r *= 1.5
            dists = G.dijkstra(bdy_set=[x], max_dist=r)
            p = np.count_nonzero(dists < r) * 1.0 / n
        a = 0.66 * r
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
        if iterations >= 50:
            print("Too many iterations. Density radius did not converge")
            return r
    return r


# @check_types
def coreset_dijkstras(
    G: gl.graph,
    rad: float,
    data: Optional[np.ndarray] = None,
    initial: Optional[list[int]] = None,
    randseed: int = 123,
    density_info: tuple[bool, float, float] = (False, DENSITY_RADIUS, 1.0),
    similarity: str = "euclidean",
    knn_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    plot_steps: bool = False,
) -> list[int]:
    """
    Runs the Dijkstra's Annulus Coreset (DAC) method outlined in the paper. The
        algorithm uses inner radius which is half of rad. When using density
        radius, the inner radius makes half the proportion of data lie in that
        ball.

    :param G: Graph object
    :param rad: fixed radius to use in DAC method
    :param data:
    :param initial: Initial points in coreset
    :param randseed:
    :param density_info:
    :param similarity:
    :param knn_data:
    :param plot_steps:

    :return: coreset computed from DAC

    """
    np.random.seed(randseed)

    perim: list[int] = []
    if initial is None:
        initial = []
    coreset = initial.copy()

    rad_low = rad / 2.0
    rad_high = rad

    use_density, proportion, r_0 = density_info

    # Once all points have been seen, we end this
    points_seen = np.zeros(G.num_nodes)

    knn_val = G.weight_matrix[0].count_nonzero()

    # Use distances without a kernel applied
    if knn_data:
        W_dist = gl.weightmatrix.knn(
            data, knn_val, similarity=similarity, kernel="distance", knn_data=knn_data
        )
    else:
        W_dist = gl.weightmatrix.knn(
            data, knn_val, similarity=similarity, kernel="distance"
        )
    G_dist = gl.graph(W_dist)

    # Construct the perimeter from the initial set
    n = len(initial)
    for i in range(n):
        if use_density:
            rad_low = density_determine_rad(G_dist, initial[i], proportion / 2.0, r_0)
            rad_high = density_determine_rad(G_dist, initial[i], proportion, r_0)
        else:
            # Calculate perimeter from new node
            tmp1 = G_dist.dijkstra(bdy_set=[initial[i]], max_dist=rad_high)
            tmp2 = tmp1 <= rad_high
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

    # If no initial set, the initialize first point
    if len(coreset) == 0:
        # Generate coreset
        new_node = np.random.choice(G_dist.num_nodes, size=1).item()
        coreset.append(new_node)
        if use_density:
            rad_low = density_determine_rad(G_dist, new_node, proportion / 2.0, r_0)
            rad_high = density_determine_rad(G_dist, new_node, proportion, r_0)
        # Calculate perimeter
        tmp1 = G_dist.dijkstra(bdy_set=[new_node], max_dist=rad_high)
        tmp2 = tmp1 <= rad_high
        tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]
        # Update perim
        perim = list(tmp3)
        # Update points seen
        points_seen[tmp2] = 1

    # Generate the coreset from the remaining stuff
    iterations = 0

    # Terminate if we have seen all points and the perimeter is empty
    while np.min(points_seen) == 0 or len(perim) > 0:
        # If perimeter is empty, jump to a new, unseen node
        if len(perim) == 0:
            avail_nodes = (points_seen == 0).nonzero()[0]
            new_node = np.random.choice(avail_nodes, size=1).item()
            coreset.append(new_node)
            if use_density:
                rad_low = density_determine_rad(G_dist, new_node, proportion / 2.0, r_0)
                rad_high = density_determine_rad(G_dist, new_node, proportion, r_0)
            # Calculate perimeter
            tmp1 = G_dist.dijkstra(bdy_set=[new_node], max_dist=rad_high)
            tmp2 = tmp1 <= rad_high
            tmp3 = ((tmp1 > rad_low) * tmp2).nonzero()[0]

            # Update perim and points seen
            perim = list(tmp3)
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
            tmp2 = tmp1 <= rad_high
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

        if plot_steps and data is not None:
            _dac_plot_fun(data, points_seen, coreset, perim)

        if iterations >= 1000:
            break
        iterations += 1
    return coreset


# @check_types
def _dac_plot_fun(
    data: np.ndarray, points_seen: np.ndarray, coreset: list[int], perim: list[int]
) -> None:
    """
    Function for plotting the steps of the DAC algorithm. It first checks if
        the dataset is from a square. This indicates that it will use the
        parameters to make nice plots for figures in the paper (eg. larger
        red dots). If it is the square dataset, the plots are saved. The plots
        are always displayed when this function is called.

    :param data: Raw data. Each datapoint must be in 2 dimensions
    :param points_seen: Points which have already been seen.
    :param coreset: Points contained in the coreset
    :param perim: Points in the perimeter

    :return: None
    """
    unit_x_len = np.abs(np.max(data[:, 0]) - np.min(data[:, 0]) - 1) < 0.05
    unit_y_len = np.abs(np.max(data[:, 1]) - np.min(data[:, 1]) - 1) < 0.05
    square_dataset = unit_x_len and unit_y_len

    # The following is for the square dataset
    if square_dataset:
        # Save the initial dataset also
        if len(coreset) == 1:
            plt.scatter(data[:, 0], data[:, 1])
            plt.axis("square")
            plt.axis("off")
            plt.savefig("DAC Plots/coreset0.png", bbox_inches="tight")
            plt.show()
        # If not initial, do this
        plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(data[points_seen == 1, 0], data[points_seen == 1, 1], c="k")
        plt.scatter(data[coreset, 0], data[coreset, 1], c="r", s=100)
        plt.scatter(data[perim, 0], data[perim, 1], c="y")
        plt.axis("square")
        plt.axis("off")
        plt.savefig(
            "DAC Plots/coreset" + str(len(coreset)) + ".png", bbox_inches="tight"
        )
    else:
        plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(data[points_seen == 1, 0], data[points_seen == 1, 1], c="k")
        plt.scatter(data[coreset, 0], data[coreset, 1], c="r")
        plt.scatter(data[perim, 0], data[perim, 1], c="y")

    plt.show()
    return


################################################################################
## util functions for batch active learning

# @check_types
def local_maxes_k_new(
    knn_ind: np.ndarray,
    acq_array: np.ndarray,
    k: int,  # TODO: Got a float for this from somewhere
    top_num: int,
    thresh: int = 0,
) -> np.ndarray:
    """
    Function to compute the k local maxes of the acquisition function.

    :param knn_ind:
    :param acq_array:
    :param k:
    :param top_num:
    :param thresh:

    :return:
    """
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


def random_sample_val(val: np.ndarray, sample_num: int) -> np.ndarray:
    """
    Docstring
    """
    assert not np.any(val < -1e-5), "random_sample_val: negative values aren't allowed"
    probs = val / np.sum(val)
    return np.random.choice(len(val), size=sample_num, replace=False, p=probs)


################################################################################

## implement batch active learning function
# @check_types
def coreset_run_experiment(
    X: np.ndarray,
    labels: np.ndarray,
    W: csr_matrix,
    coreset: list[int],
    num_iter: int = 1,
    method: str = "Laplace",
    display: bool = False,
    use_prior: bool = False,
    al_mtd: str = "local_max",
    display_all_times: bool = False,
    acq_fun: str = "uc",
    knn_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    mtd_para=None,  #'NoneType' could also be tuple i think
    savefig: bool = False,
    savefig_folder: str = "../BAL_figures",
    batchsize: int = BATCH_SIZE,
    dist_metric: str = "euclidean",
    knn_size: int = 50,
    q: int = 1,
    thresholding: int = 0,
) -> BALOutputType:
    """
    Function to run batch active learning

    **LIST ALL PARAMS

    :return:


    This was from previous docstring
    al_mtd: 'local_max', 'global_max', 'rs_kmeans', 'gd_kmeans', 'acq_sample',
        'greedy_batch', 'particle', 'random', 'topn_max'

    """

    if knn_data:
        knn_ind, _ = knn_data
    else:
        knn_ind, _ = gl.weightmatrix.knnsearch(
            X, knn_size, method="annoy", similarity=dist_metric
        )
        # knn_ind, knn_dist = knn_data

    if al_mtd == "local_max":
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

    if method == "Laplace":
        model = gl.ssl.laplace(W, class_priors=class_priors)
    elif method == "rw_Laplace":
        model = gl.ssl.laplace(W, class_priors, reweighting="poisson")
    elif method == "Poisson":
        model = gl.ssl.poisson(W, class_priors)

    if acq_fun == "mc":
        acq_f = al.model_change()
    elif acq_fun == "vopt":
        acq_f = al.v_opt()
    elif acq_fun == "uc":
        acq_f = al.uncertainty_sampling()
    elif acq_fun == "mcvopt":
        acq_f = al.model_change_vopt()

    # Time at start of active learning
    t_al_s = timeit.default_timer()
    act = al.active_learning(
        W, train_ind, labels[train_ind], eval_cutoff=min(200, len(X) // 2)
    )

    # perform classification with GSSL classifier
    u = model.fit(act.current_labeled_set, act.current_labels)
    if display_all_times:
        t_al_e = timeit.default_timer()
        print("Active learning setup time = ", t_al_e - t_al_s)

    current_label_guesses = model.predict()

    acc = gl.ssl.ssl_accuracy(
        current_label_guesses, labels, len(act.current_labeled_set)
    )

    if display:
        plt.scatter(X[:, 0], X[:, 1], c=current_label_guesses)
        plt.scatter(X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c="r")
        if savefig:
            plt.axis("square")
            plt.axis("off")
            plt.savefig(
                os.path.join(savefig_folder, "bal_coreset_.png"), bbox_inches="tight"
            )
        plt.show()

        print("Size of coreset = ", len(coreset))
        print("Using ", 100.0 * len(coreset) / len(labels), "%", "of the data")
        print("Current Accuracy is ", acc, "%")

    # record labeled set and accuracy value
    list_num_labels.append(len(act.current_labeled_set))
    list_acc = np.append(list_acc, acc)

    for iteration in range(num_iter):

        if display_all_times:
            t_iter_s = timeit.default_timer()

        act.candidate_inds = np.setdiff1d(act.training_set, act.current_labeled_set)
        if acq_fun in ["mc", "uc", "mcvopt"]:
            acq_vals = acq_f.compute_values(act, u)
        elif acq_fun == "vopt":
            acq_vals = acq_f.compute_values(act)

        modded_acq_vals = np.zeros(len(X))
        modded_acq_vals[act.candidate_inds] = acq_vals

        if al_mtd == "local_max":
            if knn_data:
                batch = local_maxes_k_new(
                    knn_ind, modded_acq_vals, k, batchsize, thresh
                )
        elif al_mtd == "global_max":
            batch = act.candidate_inds[np.argmax(acq_vals)]
        elif al_mtd == "acq_sample":
            batch_inds = random_sample_val(acq_vals**q, sample_num=batchsize)
            batch = act.candidate_inds[batch_inds]
        elif al_mtd == "random":
            batch = np.random.choice(act.candidate_inds, size=batchsize, replace=False)
        elif al_mtd == "topn_max":
            batch = act.candidate_inds[np.argsort(acq_vals)[-batchsize:]]

        if thresholding > 0:
            max_acq_val = np.max(acq_vals)
            batch = batch[modded_acq_vals[batch] >= (thresholding * max_acq_val)]

        if display_all_times:
            t_localmax_e = timeit.default_timer()
            print("Batch Active Learning time = ", t_localmax_e - t_iter_s)
            print("Batch inds:", batch)

        if display:
            plt.scatter(X[act.candidate_inds, 0], X[act.candidate_inds, 1], c=acq_vals)
            plt.scatter(
                X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c="r"
            )
            plt.scatter(X[batch, 0], X[batch, 1], c="m", marker="*", s=100)
            plt.colorbar()
            if savefig:
                plt.axis("square")
                plt.axis("off")
                plt.savefig(
                    os.path.join(
                        savefig_folder, "bal_acq_vals_b" + str(iteration) + ".png"
                    ),
                    bbox_inches="tight",
                )
            plt.show()

        act.update_labeled_data(
            batch, labels[batch]
        )  # update the active_learning object's labeled set

        u = model.fit(act.current_labeled_set, act.current_labels)
        current_label_guesses = model.predict()
        acc = gl.ssl.ssl_accuracy(
            current_label_guesses, labels, len(act.current_labeled_set)
        )
        if display_all_times:
            t_modelfit_e = timeit.default_timer()
            print("Model fit time = ", t_modelfit_e - t_localmax_e)

        list_num_labels.append(len(act.current_labeled_set))
        list_acc = np.append(list_acc, acc)

        if display:
            print("Next batch is", batch)
            print("Current number of labeled nodes", len(act.current_labeled_set))
            print("Current Accuracy is ", acc, "%")

            plt.scatter(X[:, 0], X[:, 1], c=current_label_guesses)
            plt.scatter(
                X[act.current_labeled_set, 0], X[act.current_labeled_set, 1], c="r"
            )
            if savefig:
                plt.axis("square")
                plt.axis("off")
                plt.savefig(
                    os.path.join(
                        savefig_folder, "bal_acq_vals_a" + str(iteration) + ".png"
                    ),
                    bbox_inches="tight",
                )
            plt.show()

        if display_all_times:
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
        # Don't want to return the time if we are also displaying things
        return labeled_ind, list_num_labels, list_acc
    else:
        return labeled_ind, list_num_labels, list_acc, t_total


## perform batch active learning for several times and show average&highest result
def perform_al_experiment(
    dataset_chosen,
    embedding_mode="just_transfer",
    acq_fun_list=["uc", "vopt", "mc", "mcvopt"],
    density_radius_param=0.5,
    knn_num=20,
    num_iter=1,
    method="Laplace",
    al_mtd="local_max",
    acq_fun="uc",
    knn_data=None,
    batchsize=BATCH_SIZE,
    experiment_time=10,
):
    """
    FUNCTION SIGNATURE HERE!!

    """

    highest_accuracy_list = [0] * len(acq_fun_list)
    average_accuracy_list = [0] * len(acq_fun_list)
    lowest_accuracy_list = [0] * len(acq_fun_list)

    count_unusual_cases = 0
    unusual_acc_list = []

    # Perform the experiment for experiment_time times
    for i in range(experiment_time):
        start = timeit.default_timer()
        with torch.no_grad():
            torch.cuda.empty_cache()

        if dataset_chosen == "open_sar":
            # Load labels
            data, labels = utils.load_dataset(
                "open_sar_ship", return_torch=False, concatenate=True
            )
        elif dataset_chosen == "fusar":
            # Load labels
            data, labels = utils.load_dataset(
                "fusar", return_torch=False, concatenate=True
            )
        else:
            assert False, "Chosen dataset could not be loaded. Check for typos"

        # Mimic that we know a percentage of data, and don't know for the rest
        # Do transfer learning merely using these
        percent_known_data = 0
        if dataset_chosen == "open_sar":
            percent_known_data = 0.07
        else:
            percent_known_data = 0.07
        known_data_ind = gl.trainsets.generate(labels, rate=percent_known_data).tolist()
        known_data = data[known_data_ind]
        known_labels = labels[known_data_ind]

        # Generate the initial set
        initial = gl.trainsets.generate(labels, rate=1).tolist()

        # Percent of known data to use as training data for transfer learning
        training_percent = 0.7
        transfer_train_ind = random.sample(
            range(len(known_data)), round(len(known_data) * training_percent)
        )
        transfer_testing_ind = np.array(
            [ind for ind in range(len(known_data)) if ind not in transfer_train_ind]
        ).astype(int)

        # Convert to torch for use
        known_data = torch.from_numpy(known_data)
        known_labels = torch.from_numpy(known_labels)

        # Setup the training and testing data for transfer learning
        training_data = known_data[transfer_train_ind]
        training_label = known_labels[transfer_train_ind]
        testing_data = known_data[transfer_testing_ind]
        testing_label = known_labels[transfer_testing_ind]

        print("Transfer learning training data: " + str(len(training_data)))
        print("Transfer learning testing data: " + str(len(testing_data)))

        data_info = [training_data, training_label, testing_data, testing_label]

        if dataset_chosen == "open_sar":
            # Load encoded dataset
            if embedding_mode == "just_transfer":
                X, labels = utils.encode_pretrained(
                    "open_sar_ship", "AlexNet", transformed=True
                )
            else:
                X = utils.encode_transfer_learning(
                    "open_sar_ship",
                    model_type="AlexNet",
                    transfer_batch_size=64,
                    epochs=30,
                    data_info=data_info,
                )
            # Load labels
            _, labels = utils.load_dataset(
                "open_sar_ship", return_torch=False, concatenate=True
            )
            knn_data = gl.weightmatrix.knnsearch(
                X, knn_num, method="annoy", similarity="angular"
            )
        elif dataset_chosen == "fusar":
            # Load encoded dataset
            if embedding_mode == "just_transfer":
                X, labels = utils.encode_pretrained(
                    "fusar", "ShuffleNet", transformed=True
                )
            else:
                X = utils.encode_transfer_learning(
                    "fusar",
                    model_type="ShuffleNet",
                    transfer_batch_size=64,
                    epochs=30,
                    data_info=data_info,
                )
            # Load labels
            _, labels = utils.load_dataset(
                "fusar", return_torch=False, concatenate=True
            )
            knn_data = gl.weightmatrix.knnsearch(
                X, knn_num, method="annoy", similarity="angular"
            )
        else:
            assert False, "Chosen dataset could not be loaded. Check for typos"

        # print("***HERE: ", type(labels))

        print("Constructing Graph Learning Objects")
        W = gl.weightmatrix.knn(X, knn_num, kernel="gaussian", knn_data=knn_data)
        G = gl.graph(W)
        end = timeit.default_timer()

        print("Embedding Complete")
        print(f"Time taken = {end - start}")

        # Generate Coreset
        # Use the percent radius because it should be more robust across datasets
        coreset = coreset_dijkstras(
            G,
            rad=0.2,
            data=X,
            initial=initial,
            density_info=(True, density_radius_param, 1),
            knn_data=knn_data,
        )
        print(
            "Coreset Size = {}\t Percent of data = {}%".format(
                len(coreset), round(100 * len(coreset) / len(X), 2)
            )
        )
        print("Coreset = ", coreset)

        batchsize = BATCH_SIZE
        if dataset_chosen == "open_sar":
            max_new_samples = 690
        else:
            max_new_samples = 2910

        acq_fun_ind = 0
        for acq_fun in acq_fun_list:

            if al_mtd != "global_max":
                num_iter = int(max_new_samples / batchsize)
            else:
                num_iter = max_new_samples

            if acq_fun == "vopt" or acq_fun == "mcvopt":
                num_iter += 1

            print(acq_fun, al_mtd)
            start = timeit.default_timer()

            ##DEBUGGING
            # print(X)
            # print(labels)
            # print(W)
            # print(coreset)
            # The error is that we need to pass everything in as numpy arrays

            _, list_num_labels, list_acc = coreset_run_experiment(
                X,
                labels,
                W,
                coreset,
                num_iter=num_iter,
                method=method,
                display=False,
                use_prior=False,
                al_mtd=al_mtd,
                display_all_times=False,
                acq_fun=acq_fun,
                knn_data=knn_data,
                mtd_para=None,
                savefig=False,
                savefig_folder="../BAL_figures",
                batchsize=batchsize,
                dist_metric="angular",
                q=10,
                thresholding=0,
            )

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

    average_accuracy_list = [acc / experiment_time for acc in average_accuracy_list]

    # Print out the experiment result
    for i in range(len(acq_fun_list)):
        print(acq_fun_list[i] + " average accuracy: " + str(average_accuracy_list[i]))
        print(acq_fun_list[i] + " highest accuracy: " + str(highest_accuracy_list[i]))
        print(acq_fun_list[i] + " lowest accuracy: " + str(lowest_accuracy_list[i]))

    print(unusual_acc_list)
    print("Unusual behavior happens " + str(count_unusual_cases) + " times")


################################################################################
## toy datasets
def gen_checkerboard_3(num_samples=500, randseed=123):
    """Function to generate the toy dataset: checkerboard 3"""
    np.random.seed(randseed)
    X = np.random.rand(num_samples, 2)
    labels = np.mod(np.floor(X[:, 0] * 3) + np.floor(X[:, 1] * 3), 3).astype(np.int64)
    return X, labels


def gen_stripe_3(num_samples=500, width=1 / 3, randseed=123):
    """Function to generate the toy dataset: stripe 3"""
    np.random.seed(randseed)
    X = np.random.rand(num_samples, 2)
    labels = np.mod(np.floor(X[:, 0] / width + X[:, 1] / width), 3).astype(np.int64)
    return X, labels
