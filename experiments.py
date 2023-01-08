# experiments.py
"""
DOCSTRING
"""
# TODO: Add all the docstrings in this file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import graphlearning as gl

import batch_active_learning as bal
import utils

################################################################################
## Default Parameters

EXPERIMENT_1_SAVE_PATH = "Experiment Results/Experiment 1/"
EXPERIMENT_2_SAVE_PATH = "Experiment Results/Experiment 2/"
EXPERIMENT_3_SAVE_PATH = "Experiment Results/Experiment 3/"
EXPERIMENT_4_SAVE_PATH = "Experiment Results/Experiment 4/"

EXPERIMENT_3_NUM_EXPERIMENTS = 20

################################################################################
## Experiment 1 Functions


def experiment_1(dataset: str, hardware_acceleration: bool):
    assert dataset in utils.AVAILABLE_SAR_DATASETS, "Invalid dataset"

    if dataset == "mstar":
        X, labels, knn_data, initial = utils.cnnvae(
            dataset, hardware_acceleration=hardware_acceleration
        )
    else:
        X, labels, knn_data, initial = utils.zero_shot_tl(
            dataset, hardware_acceleration=hardware_acceleration
        )

    # Create graph objects
    W = gl.weightmatrix.knn(X, utils.KNN_NUM, kernel="gaussian", knn_data=knn_data)
    G = gl.graph(W)

    # Do experiment for each method
    time_dict = {}
    acc_dict = {}
    num_labels_dict = {}
    full_acc_dict = {}

    # Construct coreset
    coreset = bal.coreset_dijkstras(
        G,
        rad=bal.DENSITY_RADIUS,
        data=X,
        initial=list(initial),
        density_info=(True, bal.DENSITY_RADIUS, 1),
        knn_data=knn_data,
    )

    for al_mtd in bal.AL_METHODS:
        num_new_samples = bal.MAX_NEW_SAMPLES_DICT[dataset] - len(coreset)

        # Run experiment
        _, num_labels, acc_vals, al_time = bal.batch_active_learning_experiment(
            X,
            labels,
            W,
            coreset,
            new_samples=num_new_samples,
            al_mtd=al_mtd,
            acq_fun="uc",
            knn_data=knn_data,
        )

        time_dict[al_mtd] = al_time
        acc_dict[al_mtd] = acc_vals[-1]
        num_labels_dict[al_mtd] = np.array(num_labels)
        full_acc_dict[al_mtd] = np.array(acc_vals)

    return acc_dict, time_dict, num_labels_dict, full_acc_dict


def experiment_1_simple_plotter(x_dict, y_dict, dataset, include_sota=False):

    for ind, this_key in enumerate(bal.AL_METHODS):
        plt.plot(x_dict[this_key], y_dict[this_key], label=bal.AL_METHOD_NAMES[ind])

    plt.xlabel("Number of Labeled Points")
    plt.ylabel("Accuracy (%)")

    plt.tick_params(axis="x")

    # Add SoTA
    if dataset != "mstar" and include_sota:
        if dataset == "open_sar_ship":
            sota_val = 78.15
        elif dataset == "fusar":
            sota_val = 86.69
        plt.plot(
            x_dict["local_max"],
            sota_val * np.ones_like(x_dict["local_max"]),
            label="SoTA",
            linestyle="--",
        )

    plt.legend()
    plt.tight_layout()

    file_path = EXPERIMENT_1_SAVE_PATH + dataset + "_detailed_plot"
    if include_sota:
        file_path += "_sota"
    file_path += ".png"

    # plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    return


def experiment_1_full_save(
    acc_dict, time_dict, num_labels_dict, full_acc_dict, dataset
):
    new_save_path = EXPERIMENT_1_SAVE_PATH + "Pickles/" + dataset
    print("Saving to: " + EXPERIMENT_1_SAVE_PATH + "Pickles/")

    df_acc_dict = pd.DataFrame.from_dict(acc_dict, orient="index")
    df_acc_dict.to_pickle(new_save_path + "_acc_dict.pkl")

    df_time_dict = pd.DataFrame.from_dict(time_dict, orient="index")
    df_time_dict.to_pickle(new_save_path + "_time_dict.pkl")

    df_num_labels_dict = pd.DataFrame.from_dict(num_labels_dict, orient="index")
    df_num_labels_dict.to_pickle(new_save_path + "_num_labels_dict.pkl")

    df_full_acc_dict = pd.DataFrame.from_dict(full_acc_dict, orient="index")
    df_full_acc_dict.to_pickle(new_save_path + "_full_acc_dict.pkl")

    return


################################################################################
## Experiment 2 Functions


def experiment_2_simple_plotter(x_dict, y_dict, dataset):
    _, ax1 = plt.subplots()

    ax1.set_xlabel("Number of Labeled Points")
    ax1.set_ylabel("Accuracy (%)")
    for this_key in bal.ACQUISITION_FUNCTIONS:
        ax1.plot(x_dict[this_key], y_dict[this_key], label=this_key)
    ax1.tick_params(axis="x")

    # Add SoTA
    if dataset != "mstar":
        if dataset == "open_sar_ship":
            sota_val = 78.15
        elif dataset == "fusar":
            sota_val = 86.69
        ax1.plot(
            x_dict["uc"],
            sota_val * np.ones_like(x_dict["uc"]),
            label="SoTA",
            linestyle="--",
        )

    ax1.legend()
    plt.show()
    return


def experiment_2(dataset, embedding, hardware_acceleration):
    assert dataset in utils.AVAILABLE_SAR_DATASETS, "Invalid dataset"
    assert embedding in utils.AVAILABLE_EMBEDDINGS, "Invalid embedding"

    # Perform embedding
    if embedding == "cnnvae":
        X, labels, knn_data, initial = utils.cnnvae(
            dataset, hardware_acceleration=hardware_acceleration
        )
    elif embedding == "zero_shot_tl":
        X, labels, knn_data, initial = utils.zero_shot_tl(
            dataset, hardware_acceleration=hardware_acceleration
        )
    else:
        X, labels, knn_data, initial = utils.fine_tuned_tl(
            dataset, hardware_acceleration=hardware_acceleration
        )

    # Create graph objects
    W = gl.weightmatrix.knn(X, utils.KNN_NUM, kernel="gaussian", knn_data=knn_data)
    G = gl.graph(W)

    acc_dict = {}
    num_labels_dict = {}

    coreset = bal.coreset_dijkstras(
        G,
        rad=bal.DENSITY_RADIUS,
        data=X,
        initial=list(initial),
        density_info=(True, bal.DENSITY_RADIUS, 1),
        knn_data=knn_data,
    )

    num_new_samples = bal.MAX_NEW_SAMPLES_DICT[dataset] - len(coreset)

    for acq_fun in bal.ACQUISITION_FUNCTIONS:

        _, num_labels, acc_vals, _ = bal.batch_active_learning_experiment(
            X,
            labels,
            W,
            coreset,
            new_samples=num_new_samples,
            al_mtd="local_max",
            acq_fun=acq_fun,
            knn_data=knn_data,
        )

        num_labels_dict[acq_fun] = np.array(num_labels)
        acc_dict[acq_fun] = acc_vals

    ##Plot and save
    experiment_2_simple_plotter(num_labels_dict, acc_dict, dataset)

    ##Save numpy arrays and pickles
    new_save_path = EXPERIMENT_2_SAVE_PATH + "Pickles/" + dataset + "_" + embedding

    df_num_labels = pd.DataFrame.from_dict(num_labels_dict, orient="index")
    df_acc = pd.DataFrame.from_dict(acc_dict, orient="index")
    df_num_labels.to_pickle(new_save_path + "_labels_dict.pkl")
    df_acc.to_pickle(new_save_path + "_acc_dict.pkl")

    return num_labels_dict, acc_dict


################################################################################
## Experiment 3 Functions


def experiment_3(
    dataset,
    embedding,
    data_augmentation,
    num_experiments=EXPERIMENT_3_NUM_EXPERIMENTS,
    hardware_acceleration=False,
):
    assert dataset in utils.AVAILABLE_SAR_DATASETS, "Invalid dataset"
    assert dataset != "mstar", "Invalid dataset: not testing MSTAR"
    assert embedding in utils.AVAILABLE_EMBEDDINGS, "Invalid embedding"

    acc_results = np.zeros(num_experiments)

    for i in range(num_experiments):
        # Do inside each experiment because we want to understand the impact of this kind of noise
        # Perform embedding
        if embedding == "cnnvae":
            X, labels, knn_data, initial = utils.cnnvae(
                dataset, hardware_acceleration=hardware_acceleration
            )
        elif embedding == "zero_shot_tl":
            X, labels, knn_data, initial = utils.zero_shot_tl(
                dataset,
                hardware_acceleration=hardware_acceleration,
            )
        else:
            X, labels, knn_data, initial = utils.fine_tuned_tl(
                dataset,
                data_augmentation=data_augmentation,
                hardware_acceleration=hardware_acceleration,
            )

        # Create graph objects
        W = gl.weightmatrix.knn(X, utils.KNN_NUM, kernel="gaussian", knn_data=knn_data)
        G = gl.graph(W)

        coreset = bal.coreset_dijkstras(
            G,
            rad=bal.DENSITY_RADIUS,
            data=X,
            initial=list(initial),
            density_info=(True, bal.DENSITY_RADIUS, 1),
            knn_data=knn_data,
        )

        num_new_samples = bal.MAX_NEW_SAMPLES_DICT[dataset] - len(coreset)

        _, num_labels, acc_vals, _ = bal.batch_active_learning_experiment(
            X,
            labels,
            W,
            coreset,
            new_samples=num_new_samples,
            al_mtd="local_max",
            acq_fun="uc",
            knn_data=knn_data,
        )

        acc_results[i] = acc_vals[-1]
    end_labels = num_labels[-1]

    return {
        "num_points": end_labels,
        "experiments": num_experiments,
        "mean": np.mean(acc_results),
        "std_dev": np.std(acc_results),
        "max": np.max(acc_results),
        "data": acc_results,
    }


################################################################################
## Experiment 4 Functions


# Does this use data augmentation? I believe so
# Zero-shot TL
def experiment_4(dataset, network, hardware_acceleration):
    assert dataset in utils.AVAILABLE_SAR_DATASETS, "Invalid dataset"
    assert dataset != "mstar", "Invalid dataset: not testing MSTAR"
    assert network in utils.PYTORCH_NEURAL_NETWORKS, "Invalid Neural Network"

    # Zero-Shot TL
    X, labels, knn_data, initial = utils.zero_shot_tl(
        dataset, hardware_acceleration=hardware_acceleration
    )

    # Create graph objects
    W = gl.weightmatrix.knn(X, utils.KNN_NUM, kernel="gaussian", knn_data=knn_data)
    G = gl.graph(W)

    coreset = bal.coreset_dijkstras(
        G,
        rad=bal.DENSITY_RADIUS,
        data=X,
        initial=list(initial),
        density_info=(True, bal.DENSITY_RADIUS, 1),
        knn_data=knn_data,
    )

    num_new_samples = bal.MAX_NEW_SAMPLES_DICT[dataset] - len(coreset)

    _, _, acc_vals, _ = bal.batch_active_learning_experiment(
        X,
        labels,
        W,
        coreset,
        new_samples=num_new_samples,
        al_mtd="local_max",
        acq_fun="uc",
        knn_data=knn_data,
    )

    return acc_vals[-1]
