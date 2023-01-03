# utils.py
"""
Authors: James, Bohan, and Zheng

This utils file contains useful code for loading and embedding the MSTAR,
OpenSARShip, and FUSAR-Ship datasets.  This code handles all the training
for neural networks also. The available embeddings are:
    CNNVAE:         uses a pretrained cnnvae to embed the data
    zero_shot_tl:   uses zero shot transfer learning
    fine_tuned_tl:  uses fine tuned transfer learning
The beginning of the code also contains constant which are used throughout the project.
"""
import os
import math
import time
import copy

from typing import Union, Optional

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision.models as torch_models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms

import graphlearning as gl
from models import CVAE

################################################################################
## Default Parameters

KNN_NUM: int = 50
TL_EPOCHS: int = 30
ENCODING_BATCH_SIZE: int = 1000
TL_BATCH_SIZE: int = 64
FINE_TUNING_DATA_PROPORTION: float = 0.05

AVAILABLE_SAR_DATASETS: list[str] = ["mstar", "open_sar_ship", "fusar"]
AVAILABLE_EMBEDDINGS: list[str] = ["cnnvae", "zero_shot_tl", "fine_tuned_tl"]
SAR_DATASET_SIZE_DICT: dict[str, int] = {
    "mstar": 6874,
    "open_sar_ship": 2296,
    "fusar": 4856,
}

PYTORCH_NEURAL_NETWORKS: list[str] = [
    "ResNet",
    "ShuffleNet",
    "AlexNet",
    "DenseNet",
    "GoogLeNet",
    "MobileNetV2",
    "ResNeXt",
    "Wide ResNet",
]
# https://pytorch.org/hub/research-models
# TODO: Remove the torchvision.transforms.Normalize()
# Nets with batchnorm: resnets,
#   without: alexnet, densenet,

PYTORCH_NEURAL_NETWORKS_DICT: dict[str, str] = {
    "ResNet": "resnet18",
    "ShuffleNet": "shufflenet_v2_x0_5",
    "AlexNet": "alexnet",
    "DenseNet": "densenet121",
    "GoogLeNet": "googlenet",
    "MobileNetV2": "mobilenet_v2",
    "ResNeXt": "resnext50_32x4d",
    "Wide ResNet": "wide_resnet50_2",
}

DEFAULT_NEURAL_NETWORKS_DICT: dict[str, str] = {
    "mstar": "ResNet",
    "open_sar_ship": "AlexNet",
    "fusar": "ShuffleNet",
}

USE_HARDWARE_ACCELERATION: bool = False


################################################################################
ArrayType = Union[torch.Tensor, np.ndarray]
EmbeddingType = Union[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]],
]
DatasetType = Union[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]

################################################################################
## Convenient User Functions
# Convenient functions for loading data for the experiments
# These are like the default embedding functions (with proper networks, etc.)


def CNNVAE(
    dataset: str,
    knn_num: int = KNN_NUM,
    include_knn_data: bool = True,
) -> EmbeddingType:
    """
    Embeds the chosen dataset using a trained CNNVAE.

    :param dataset:
    """
    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"

    # Load Dataset
    data, labels = load_dataset(dataset, return_torch=False)

    # Determine CNNVAE model
    if dataset == "mstar":
        model_path = "./models/SAR10_CNNVAE.pt"
    elif dataset == "open_sar_ship":
        model_path = "./models/OpenSarShip_CNNVAE.pt"
    else:
        model_path = "./models/Fusar_CNNVAE.pt"

    # Encode Dataset
    data = encode_dataset(dataset, model_path, batch_size=1000)

    if include_knn_data:
        try:
            # Load the knn data from the cnnvae (computed once in separate file)
            knn_data = _get_knn_data(dataset)
            print("Using pre-computed cnnvae embedding knn_data")
        except FileNotFoundError:
            # Compute knn data
            print("Computing knn_data")
            knn_data = gl.weightmatrix.knnsearch(
                data, knn_num, method="annoy", similarity="angular"
            )

        return data, labels, knn_data
    else:
        return data, labels


def zero_shot_TL(
    dataset: str,
    knn_num: int = KNN_NUM,
    data_augmentation: bool = True,
    include_knn_data: bool = True,
) -> EmbeddingType:
    """
    Docstring
    """
    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"

    _, labels = load_dataset(dataset, return_torch=False)

    X = encode_pretrained(
        dataset,
        DEFAULT_NEURAL_NETWORKS_DICT[dataset],
        transformed=data_augmentation,
    )

    if include_knn_data:
        knn_data = gl.weightmatrix.knnsearch(
            X, knn_num, method="annoy", similarity="angular"
        )
        return X, labels, knn_data
    else:
        return X, labels


# TODO: This isn't done yet. NEED TO RETURN WHICH INDICES WERE USED IN TRAINING
# I need to set this up so it is nice
def fine_tuned_TL(
    dataset: str,
    knn_num: int = KNN_NUM,
    num_epochs: int = TL_EPOCHS,
    network: Optional[str] = None,
    data_augmentation: bool = True,
    include_knn_data: bool = True,
) -> EmbeddingType:
    """
    Docstring
    """
    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"

    _, labels = load_dataset(dataset, return_torch=False)

    # If network is None, we use the defaults. This functionality is in
    #   the encode_transfer_learning function

    X = encode_transfer_learning(
        dataset,
        model_type=network,
        transfer_batch_size=TL_BATCH_SIZE,
        epochs=num_epochs,
        data_info=None,
        transformed=data_augmentation,
    )

    if include_knn_data:
        # Takes a long time
        knn_data = gl.weightmatrix.knnsearch(
            X, knn_num, method="annoy", similarity="angular"
        )
        return X, labels, knn_data
    else:
        return X, labels


def load_dataset(
    dataset: str,
    return_torch: bool,
) -> tuple[ArrayType, ArrayType]:
    """
    Docstring
    """
    if dataset == "open_sar_ship":
        data_train = np.load("data/OpenSARShip/SarTrainImages.npz")["arr_0"]
        target_train = np.load("data/OpenSARShip/SarTrainLabels.npy")
        data_test = np.load("data/OpenSARShip/SarTestImages.npz")["arr_0"]
        target_test = np.load("data/OpenSARShip/SarTestLabels.npy")
    elif dataset == "fusar":
        data_train = np.load("data/Fusar/FusarTrainImages.npz")["arr_0"]
        target_train = np.load("data/Fusar/FusarTrainLabels.npy")
        data_test = np.load("data/Fusar/FusarTestImages.npz")["arr_0"]
        target_test = np.load("data/Fusar/FusarTestLabels.npy")
    elif dataset == "mstar":
        hdr, _, mag, phase = _load_mstar("data/MSTAR")
        data = _polar_transform_mstar(mag, phase)
        target, _ = _targets_to_labels_mstar(hdr)
    else:
        assert False, "Code not implemented for this dataset"

    if dataset != "mstar":
        data = np.vstack((data_train, data_test))
        target = np.hstack((target_train, target_test))

    if return_torch:
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

    return data, target


# TODO: Need to do with the gl.trainsets.generate()
def load_dataset_fine_tuned_tl(
    dataset: str,
    return_torch: bool,
    shuffle_train_set: bool = False,
) -> tuple[ArrayType, ArrayType, ArrayType, ArrayType]:
    # Load dataset
    data, target = load_dataset(dataset, return_torch=False)

    # Get train-test split
    dataset_size = SAR_DATASET_SIZE_DICT[dataset]
    tl_num = round(dataset_size * FINE_TUNING_DATA_PROPORTION)
    train_ind = np.random.choice(dataset_size, size=tl_num, replace=False)
    test_ind = np.setdiff1d(np.arange(dataset_size), train_ind)

    data_train = data[train_ind, ...]
    data_test = data[test_ind, ...]
    target_train = target[train_ind]
    target_test = target[test_ind]

    # Potentially shuffle the training set
    if shuffle_train_set:
        P = np.random.permutation(data_train.shape[0])
        data_train = data_train[P, :, :, :]
        target_train = target_train[P]

    if return_torch:
        data_train = torch.from_numpy(data_train).float()
        target_train = torch.from_numpy(target_train).long()
        data_test = torch.from_numpy(data_test).float()
        target_test = torch.from_numpy(target_test).long()

    return data_train, target_train, data_test, target_test


################################################################################
## Helper Functions / Classes


# TODO: Want random crop
# TODO: Should resize before random crop?
# TODO: Make sure mean and std have correct size

# TODO: Want to make sure to use mps here
# TODO: It wasn't faster. Check this again
def _apply_data_augmentation(data: torch.Tensor, reshape: bool = True) -> torch.Tensor:
    """
    Docstring
    """
    # print(data.shape)
    # device = torch.device(_determine_hardware())
    t1 = transforms.GaussianBlur(kernel_size=(1, 1), sigma=(1, 1))
    t2 = transforms.RandomCrop(data.shape[-1], padding=32)
    t3 = transforms.ColorJitter(contrast=2, brightness=0.5, hue=0.3)

    data = t3(t2(t1(data)))

    mean = torch.mean(data, dim=(0, 2, 3))
    std = torch.std(data, dim=(0, 2, 3))

    t4 = transforms.Normalize(mean, std)
    data = t4(data)

    # All pytorch CNN assume the image is 3 channel
    # Turn image data into 3 channels if use pytorch pretrained CNN
    if reshape:
        data = data.expand([-1, 3, -1, -1])

    return data


# MARK: Used in the encode_pretrained
def _determine_feature_layer(model_type: str) -> str:
    """Determines the feature layer based on the model type."""
    if model_type == "ShuffleNet":
        return "mean"
    elif model_type == "GoogLeNet":
        return "dropout"
    else:
        return "flatten"


def _determine_hardware() -> str:
    """Determines which hardware is available on your device."""
    if USE_HARDWARE_ACCELERATION and torch.cuda.is_available():
        return "cuda"
    elif USE_HARDWARE_ACCELERATION and torch.has_mps:
        return "mps"
    else:
        return "cpu"


class MyDataset(Dataset):
    """Helper class for transfer learning"""

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


# TODO: Add types here
# TODO: Make this print less
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    dataloaders,
    dataset_sizes,
    num_epochs=25,
    verbose: int = 1,
):
    """
    Helper function for transfer learning. Fine tunes the pretrained model.

    FINISH THIS

    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param device:
    :param dataloaders:
    :param dataset_sizes:
    :param num_epochs:
    :param verbose: determines the amount of printing done
        0 gives no printing
        1 gives just the epoch
        2 gives full output

    :return:
    """
    since = time.time()
    assert verbose in [0, 1, 2]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if verbose >= 1:
        print(f"Training model for {num_epochs} epochs")

    for epoch in range(num_epochs):
        if verbose == 2:
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs.float())
                    _, preds = torch.max(outputs, 1)
                    outputs = nn.functional.log_softmax(outputs, dim=-1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            if verbose == 2:
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if verbose == 2:
            print()

    time_elapsed = time.time() - since

    if verbose >= 1:
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def _construct_TL_network(dataset: str, model_type: str) -> nn.Module:
    """
    Modifies the pretrained PyTorch neural network for transfer learning.

    :param dataset: name of the dataset
    :param model_type: name of the neural network

    :return: A new neural network
    """
    if model_type is None:
        if dataset == "open_sar_ship":
            model_ft = torch_models.alexnet(pretrained=True)
            model_ft.classifier[1] = nn.Linear(9216, 1024)
            model_ft.classifier[4] = nn.Linear(1024, 256)
            model_ft.classifier[6] = nn.Linear(256, 3)
        elif dataset == "fusar":
            model_ft = torch_models.shufflenet_v2_x0_5(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 5)
        else:
            model_ft = torch_models.resnet18(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 10)
    else:
        if model_type == "AlexNet":
            model_ft = torch_models.alexnet(pretrained=True)
            model_ft.classifier[1] = nn.Linear(9216, 1024)
            model_ft.classifier[4] = nn.Linear(1024, 256)
            if dataset == "open_sar_ship":
                model_ft.classifier[6] = nn.Linear(256, 3)
            else:
                model_ft.classifier[6] = nn.Linear(256, 5)
        elif model_type == "DenseNet":
            model_ft = torch_models.densenet121(pretrained=True)
            num_ftrs = model_ft.classifier.in_features
            if dataset == "open_sar_ship":
                model_ft.classifier = nn.Linear(num_ftrs, 3)
            else:
                model_ft.classifier = nn.Linear(num_ftrs, 5)
        elif model_type == "MobileNetV2":
            model_ft = torch_models.mobilenet_v2(pretrained=True)
            num_ftrs = model_ft.classifier[1].in_features
            if dataset == "open_sar_ship":
                model_ft.classifier[1] = nn.Linear(num_ftrs, 3)
            else:
                model_ft.classifier[1] = nn.Linear(num_ftrs, 5)
        else:
            model_used = PYTORCH_NEURAL_NETWORKS_DICT[model_type]
            model_ft = torch.hub.load(
                "pytorch/vision:v0.10.0", model_used, pretrained=True
            )
            num_ftrs = model_ft.fc.in_features
            if dataset == "open_sar_ship":
                model_ft.fc = nn.Linear(num_ftrs, 3)
            else:
                model_ft.fc = nn.Linear(num_ftrs, 5)
    return model_ft


def NormalizeData(data: ArrayType) -> ArrayType:
    """Normalizes data to range [0,1]"""
    if isinstance(data, torch.Tensor):
        norm_data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    elif isinstance(data, np.ndarray):
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        assert False, "Invalid type for NormalizeData"
    return norm_data


def _get_knn_data(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to load knn_data for pretrained CNNVAEs"""
    knn_ind = np.load("knn_data/" + dataset + "_knn_ind.npy")
    knn_dist = np.load("knn_data/" + dataset + "_knn_dist.npy")
    return knn_ind, knn_dist


################################################################################
## Encoding/Transfer Learning Functions

# This was efficient
def encode_dataset(
    dataset: str,
    model_path: str,
    batch_size: int = ENCODING_BATCH_SIZE,
) -> np.ndarray:
    """
    Docstring
    """
    # Decide which device to use
    device = torch.device(_determine_hardware())

    # Load data
    data, _ = load_dataset(dataset, return_torch=True)

    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    encoded_data = None
    with torch.no_grad():
        for idx in range(0, len(data), batch_size):
            data_batch = data[idx : idx + batch_size]
            if encoded_data is None:
                encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
            else:
                encoded_data = np.vstack(
                    (encoded_data, model.encode(data_batch.to(device)).cpu().numpy())
                )

    return encoded_data


def encode_pretrained(
    dataset: str,
    model_type: str,
    batch_size: int = ENCODING_BATCH_SIZE,
    device_name: str = "mps",
    balanced: bool = False,
    transformed: bool = False,
) -> np.ndarray:
    # Decide which device to use
    # device = torch.device(_determine_hardware())
    device = torch.device("cpu")

    # Load data
    data, labels = load_dataset(dataset, return_torch=True)
    # data = data.to(device)

    # Apply data augmentation
    if transformed:
        data = _apply_data_augmentation(data)
    else:
        data = data.expand([-1, 3, -1, -1])

    # If custom model, apply grayscale
    if model_type[-3:] == ".pt":
        print("Using grayscale")
        transform = transforms.Grayscale()
        data = transform(data)

    encoded_data = None
    models_dict = PYTORCH_NEURAL_NETWORKS_DICT.copy()
    # Load the desired model and encode data
    if model_type in PYTORCH_NEURAL_NETWORKS:
        model_used = models_dict.get(model_type)
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", model_used, pretrained=True, map_location=device
        )
        model.eval()

        feature_layer = _determine_feature_layer(model_type)
        with torch.no_grad():
            feature_extractor = create_feature_extractor(
                model, return_nodes=[feature_layer]
            )
            feature_extractor.eval()

            encoded_data = (
                feature_extractor(data.to(device))[feature_layer].cpu().numpy()
            )
    else:
        model = torch.load(model_type, map_location=device)
        model.eval()
        with torch.no_grad():
            for idx in range(0, len(data), batch_size):
                data_batch = data[idx : idx + batch_size]
                if encoded_data is None:
                    encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
                else:
                    encoded_data = np.vstack(
                        (
                            encoded_data,
                            model.encode(data_batch.to(device)).cpu().numpy(),
                        )
                    )

    return encoded_data


def encode_transfer_learning(
    dataset: str,
    model_type: Optional[str] = None,
    epochs: int = 15,
    transfer_batch_size: int = TL_BATCH_SIZE,
    batch_size: int = ENCODING_BATCH_SIZE,
    data_info: Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    transformed: bool = False,
) -> np.ndarray:
    # Decide which device to use
    device = torch.device(_determine_hardware())

    if (model_type is None) or (model_type in PYTORCH_NEURAL_NETWORKS):
        # TODO: Need to know what is in the coreset so should return the indices probably
        # Load training and testing data
        if data_info is None:
            (
                data_train,
                label_train,
                data_test,
                label_test,
            ) = load_dataset_fine_tuned_tl(dataset, return_torch=True)
        else:
            data_train, label_train, data_test, label_test = data_info

        # Modify data
        data_train = data_train.expand([-1, 3, -1, -1])
        data_test = data_test.expand([-1, 3, -1, -1])
        with torch.no_grad():
            # TODO: Idk if we want this
            data_train = NormalizeData(data_train)
            data_test = NormalizeData(data_test)

        # TODO: Do we know if data_train and data_test are on gpu?
        # TODO: DO we want these? Maybe just don't have the zero shot + data aug
        # TODO: Go back to the good one and don't do zero shot
        # +++++++++++
        if transformed:
            data_transforms = {
                "train": transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomRotation(degrees=(0, 180)),
                        transforms.GaussianBlur(kernel_size=(1, 1), sigma=(1, 1)),
                        transforms.ColorJitter(contrast=2, brightness=0.5, hue=0.3),
                    ]
                ),
                "val": transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomRotation(degrees=(0, 180)),
                        transforms.GaussianBlur(kernel_size=(1, 1), sigma=(1, 1)),
                        transforms.ColorJitter(contrast=2, brightness=0.5, hue=0.3),
                    ]
                ),
            }
        else:
            # Identity transformations
            data_transforms = {"train": None, "val": None}

        # Packing the training and evaluation data
        # If run out of CUDA memory in testings below, try to reduce the batch size to 32 or 16
        # data_aug = lambda data: _apply_data_augmentation(data, reshape=False)
        dataset_train = MyDataset(
            data_train, label_train, transform=data_transforms["train"]
        )
        dataloader_train = DataLoader(dataset_train, batch_size=transfer_batch_size)

        dataset_val = MyDataset(data_test, label_test, transform=data_transforms["val"])
        dataloader_val = DataLoader(dataset_val, batch_size=transfer_batch_size)

        train_size = data_train.shape[0]
        val_size = data_test.shape[0]

        dataloaders = {"train": dataloader_train, "val": dataloader_val}
        dataset_sizes = {"train": train_size, "val": val_size}
        # +++++++++++

        model_ft = _construct_TL_network(dataset, model_type)
        model_ft.to(device)

        # Set up the optimizer to optimize all model parameters
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # Train the model with cross entropy loss
        criterion = nn.CrossEntropyLoss()
        model_ft = train_model(
            model_ft,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            device,
            dataloaders,
            dataset_sizes,
            num_epochs=epochs,
        )

        data, _ = load_dataset(dataset, return_torch=True)

        if transformed:
            # Modify the data
            data = _apply_data_augmentation(data)
        else:
            data = data.expand([-1, 3, -1, -1])
        # TODO: IDK if we want this
        data = NormalizeData(data)

        data = data.to(device)
        # model_ft.cpu()
        model_ft.eval()

        if model_type is None:
            feature_layer = _determine_feature_layer(
                DEFAULT_NEURAL_NETWORKS_DICT[dataset]
            )
        else:
            feature_layer = _determine_feature_layer(model_type)

        with torch.no_grad():
            feature_extractor = create_feature_extractor(
                model_ft, return_nodes=[feature_layer]
            )

            encoded_data_dict = feature_extractor(data)
            encoded_data = encoded_data_dict[feature_layer].detach().cpu().numpy()
    else:
        data, _ = load_dataset(dataset, return_torch=True)
        model = torch.load(model_type, map_location=device)
        model.eval()
        encoded_data = None
        with torch.no_grad():
            for idx in range(0, len(data), batch_size):
                data_batch = data[idx : idx + batch_size]
                if encoded_data is None:
                    encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
                else:
                    encoded_data = np.vstack(
                        (
                            encoded_data,
                            model.encode(data_batch.to(device)).cpu().numpy(),
                        )
                    )
    return encoded_data


################################################################################
### MSTAR Helper Functions
##All code below is from MSTAR Github

# TODO: Make these private. Don't need types


def _load_mstar(root_dir="./data/MSTAR"):
    """Loads MSTAR Data

    Parameters
    ----------
    root_dir : Root directory (default is ./data/MSTAR)

    Returns
    -------
    hdr : header data
    fields : Names of fields in header data
    mag : Magnitude images
    phase : Phase images
    """

    M = np.load(os.path.join(root_dir, "SAR10a.npz"), allow_pickle=True)
    hdr_a, fields, mag_a, phase_a = M["hdr"], M["fields"], M["mag"], M["phase"]

    M = np.load(os.path.join(root_dir, "SAR10b.npz"), allow_pickle=True)
    hdr_b, fields, mag_b, phase_b = M["hdr"], M["fields"], M["mag"], M["phase"]

    M = np.load(os.path.join(root_dir, "SAR10c.npz"), allow_pickle=True)
    hdr_c, fields, mag_c, phase_c = M["hdr"], M["fields"], M["mag"], M["phase"]

    hdr = np.concatenate((hdr_a, hdr_b, hdr_c))
    mag = np.concatenate((mag_a, mag_b, mag_c))
    phase = np.concatenate((phase_a, phase_b, phase_c))

    # Clip to [0,1] (only .18% of pixels>1)
    mag[mag > 1] = 1

    return hdr, fields, mag, phase


# TODO: Unused
def train_test_split(hdr, train_fraction):
    """Training and testing split (based on papers, angle=15 or 17)

    Parameters
    ----------
    hdr : Header info
    train_fraction : Fraction in [0,1] of full train data to use

    Returns
    -------
    full_train_mask : Boolean training mask for all angle==17 images
    test_mask : Boolean testing mask
    train_idx : Indices of training images selected
    """

    angle = hdr[:, 6].astype(int)
    full_train_mask = angle == 17
    test_mask = angle == 15
    num_train = int(np.sum(full_train_mask) * train_fraction)
    train_idx = np.random.choice(
        np.arange(hdr.shape[0]),
        size=num_train,
        replace=False,
        p=full_train_mask / np.sum(full_train_mask),
    )

    return full_train_mask, test_mask, train_idx


def _targets_to_labels_mstar(hdr):
    """Converts target names to numerical labels

    Parameters
    ----------
    hdr : Header data

    Returns
    -------
    labels : Integer labels from 0 to k-1 for k classes
    target_names : List of target names corresponding to each label integer
    """

    targets = hdr[:, 0].tolist()
    classes = set(targets)
    label_dict = dict(zip(classes, np.arange(len(classes))))
    labels = np.array([label_dict[t] for t in targets], dtype=int)
    target_names = list(label_dict.keys())

    return labels, target_names


def _polar_transform_mstar(mag, phase):
    """
    Peform polar transormation of data.

    Parameters
    ----------
        mag : Magnitude images
        phase : Phase data

    Returns
    -------
        data : nx3 numpy array with (mag,real,imaginary)
    """

    real = (mag * np.cos(phase) + 1) / 2
    imaginary = (mag * np.sin(phase) + 1) / 2
    data = np.stack((mag, real, imaginary), axis=1)

    return data


# TODO: Unused
def _rotate_images(
    data, center_crop=True, top=-14, left=14, height=100, width=100, size=128
):
    """
    Rotate Images
    =============

    Preprocceses data to vertical alignment using PCA.

    Parameters:
    --------------
      data: pytorch tensor dataset (i.e. from utils.load_dataset)

      center_crop: whether to do a small center crop - this can improve the PCA and
    result in better alignment. The crop takes the image from 128x128 to 100x100.

      top, left, height, width, size: see documentation on Pytorch's resized_crop

    Returns:
    --------------
      data_rotated: the same dataset but with images vertically aligned and possibly
    center cropped.

    """

    data_rotated = data
    data_mean = torch.mean(data)
    data_std = torch.std(data)
    threshold = data_mean + 2 * data_std

    for index in range(len(data)):

        image = data_rotated[index, :, :, :]

        if center_crop:
            image = transforms.functional.resized_crop(
                image, top=top, left=left, height=height, width=width, size=size
            )
            img_thresholded = (image[0, :, :] > threshold).type(torch.float)
        else:
            img = data_rotated[index, 0, :, :]
            img_thresholded = (img > threshold).type(torch.float)

        # very rarely thresholding results in too sparse of an image because
        # the image is essentailly blank to begin with
        if torch.sum(img_thresholded) <= 1:
            continue

        img_thresholded = (
            (img_thresholded == 1).nonzero(as_tuple=False).type(torch.float)
        )
        U, S, V = torch.pca_lowrank(img_thresholded)
        angle = (180 / math.pi) * torch.atan(V[1, 1] / V[0, 1])
        angle = angle.item()

        img_rotated = transforms.functional.rotate(image, 90 - angle)
        data_rotated[index, 0, :, :] = img_rotated

    return data_rotated
