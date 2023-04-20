# utils.py
"""
Authors: James, Bohan, and Zheng

This utils file contains useful code for loading and embedding the MSTAR,
    OpenSARShip, and FUSAR-Ship datasets.  This code handles all the training
    for neural networks also. The available embeddings are:
        CNNVAE:         uses a pretrained cnnvae to embed the data
        zero_shot_tl:   uses zero shot transfer learning
        fine_tuned_tl:  uses fine tuned transfer learning

The beginning of the code also contains constant which are used for various
    datasets and embeddings.
"""
import os
import time
import copy

# python 3.8 (used in google colab) needs typing.List, typing.Dict and typing.Tuple
from typing import Union, Optional, List, Dict, Tuple

import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


import torchvision.models as torch_models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms

import graphlearning as gl


################################################################################
## Default Parameters

KNN_NUM: int = 50
TL_EPOCHS: int = 30
ENCODING_BATCH_SIZE: int = 1000
TL_BATCH_SIZE: int = 64
FINE_TUNING_DATA_PROPORTION: float = 0.05

AVAILABLE_SAR_DATASETS: List[str] = ["mstar", "open_sar_ship", "fusar"]
AVAILABLE_EMBEDDINGS: List[str] = ["cnnvae", "zero_shot_tl", "fine_tuned_tl"]
SAR_DATASET_SIZE_DICT: Dict[str, int] = {
    "mstar": 6874,
    "open_sar_ship": 2296,
    "fusar": 4856,
}

PYTORCH_NEURAL_NETWORKS: List[str] = [
    "ResNet",
    "ShuffleNet",
    "AlexNet",
    "DenseNet",
    "GoogLeNet",
    "MobileNetV2",
    "ResNeXt",
    "Wide ResNet",
]

# Refer to https://pytorch.org/hub/research-models for more information
PYTORCH_NEURAL_NETWORKS_DICT: Dict[str, str] = {
    "ResNet": "resnet18",
    "ShuffleNet": "shufflenet_v2_x0_5",
    "AlexNet": "alexnet",
    "DenseNet": "densenet121",
    "GoogLeNet": "googlenet",
    "MobileNetV2": "mobilenet_v2",
    "ResNeXt": "resnext50_32x4d",
    "Wide ResNet": "wide_resnet50_2",
}

DEFAULT_NEURAL_NETWORKS_DICT: Dict[str, str] = {
    "mstar": "ResNet",
    "open_sar_ship": "AlexNet",
    "fusar": "ShuffleNet",
}


################################################################################
## Types
ArrayType = Union[torch.Tensor, np.ndarray]
EmbeddingType = Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]

DatasetType = Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]

################################################################################
## Convenient User Functions
# Convenient functions for loading data for the experiments
# These are like the default embedding functions (with proper networks, etc.)


def cnnvae(
    dataset: str,
    knn_num: int = KNN_NUM,
    hardware_acceleration: bool = False,
) -> EmbeddingType:
    """Embeds the chosen dataset using a trained CNNVAE.

    Args:
        dataset: Dataset to use.
        knn_num: Node degree in knn graph. Defaults to KNN_NUM.
        hardware_acceleration: Use GPU if true. Defaults to False.

    Returns:
        data: The encoded data.
        labels: The labels.
        knn_data: The knn_data computed with annoy algorithm from the encoded data.
        train_ind: A point from each class. Used later for coreset construction.
    """

    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"

    # Load Dataset
    data, labels = load_dataset(dataset)

    # Determine CNNVAE model
    if dataset == "mstar":
        model_path = "./models/SAR10_CNNVAE.pt"
    elif dataset == "open_sar_ship":
        model_path = "./models/OpenSarShip_CNNVAE.pt"
    else:
        model_path = "./models/Fusar_CNNVAE.pt"

    # Encode Dataset
    data = encode_dataset(
        dataset,
        model_path,
        batch_size=1000,
        hardware_acceleration=hardware_acceleration,
    )

    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

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

    train_ind = gl.trainsets.generate(labels, rate=1)

    return data, labels, knn_data, train_ind


def zero_shot_tl(
    dataset: str,
    knn_num: int = KNN_NUM,
    hardware_acceleration: bool = False,
    network: Optional[str] = None,
) -> EmbeddingType:
    """Embeds the chosen dataset using zero-shot transfer learning.

    Args:
        dataset: Dataset to use.
        knn_num: Node degree in knn graph. Defaults to KNN_NUM.
        hardware_acceleration: Use GPU if true. Defaults to False.
        network: Which network to use. Defaults to None.

    Returns:
        data: The encoded data.
        labels: The labels.
        knn_data: The knn_data computed with annoy algorithm from the encoded data.
        train_ind: A point from each class. Used later for coreset construction.
    """
    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"

    _, labels = load_dataset(dataset)

    if network is None:
        network = DEFAULT_NEURAL_NETWORKS_DICT[dataset]
    else:
        assert network in PYTORCH_NEURAL_NETWORKS

    data = encode_pretrained(
        dataset,
        network,
        hardware_acceleration=hardware_acceleration,
    )

    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    knn_data = gl.weightmatrix.knnsearch(
        data, knn_num, method="annoy", similarity="angular"
    )

    train_ind = gl.trainsets.generate(labels, rate=1)

    return data, labels, knn_data, train_ind


def fine_tuned_tl(
    dataset: str,
    knn_num: int = KNN_NUM,
    num_epochs: int = TL_EPOCHS,
    network: Optional[str] = None,
    data_augmentation: bool = True,
    hardware_acceleration: bool = False,
) -> EmbeddingType:
    """Embeds the chosen dataset using fine-tuned transfer learning.

    Args:
        dataset: Dataset to use.
        knn_num: Node degree in knn graph. Defaults to KNN_NUM.
        num_epochs: Number of epochs for fine-tuning. Defaults to TL_EPOCHS.
        network: Neural network to use in embedding. Defaults to None.
        data_augmentation: Use data augmentation if true. Defaults to True.
        hardware_acceleration: Use GPU if true. Defaults to False.

    Returns:
        data: The encoded data.
        labels: The labels.
        knn_data: The knn_data computed with annoy algorithm from the encoded data.
        train_ind: A point from each class. Used later for coreset construction.
    """
    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"

    _, labels = load_dataset(dataset)

    # If network is None, we use the defaults. This functionality is in
    #   the encode_transfer_learning function

    data, train_ind = encode_transfer_learning(
        dataset,
        model_type=network,
        transfer_batch_size=TL_BATCH_SIZE,
        epochs=num_epochs,
        data_info=None,
        data_augmentation=data_augmentation,
        hardware_acceleration=hardware_acceleration,
    )

    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    # Takes a long time
    knn_data = gl.weightmatrix.knnsearch(
        data, knn_num, method="annoy", similarity="angular"
    )

    return data, labels, knn_data, train_ind


def load_dataset(
    dataset: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the desired dataset with labels"""
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

    return data, target


def load_dataset_fine_tuned_tl(
    dataset: str,
    shuffle_train_set: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Creates a train-test split, ensuring that each label occurs in the training set.
    """
    # Load dataset
    data, target = load_dataset(dataset)

    # Get train-test split
    dataset_size = SAR_DATASET_SIZE_DICT[dataset]
    tl_num = round(dataset_size * FINE_TUNING_DATA_PROPORTION)
    # First pick so that each label occurs
    each_ind = np.array(gl.trainsets.generate(target, rate=1))

    # Then pick the rest of the indices
    train_ind = np.random.choice(
        np.setdiff1d(np.arange(dataset_size), each_ind),
        size=tl_num - len(each_ind),
        replace=False,
    )

    # Now combine them
    train_ind = np.concatenate((train_ind, each_ind))

    test_ind = np.setdiff1d(np.arange(dataset_size), train_ind)

    data_train = data[train_ind, ...]
    data_test = data[test_ind, ...]
    target_train = target[train_ind]
    target_test = target[test_ind]

    # Potentially shuffle the training set
    if shuffle_train_set:
        perm = np.random.permutation(data_train.shape[0])
        data_train = data_train[perm, :, :, :]
        target_train = target_train[perm]
        train_ind = train_ind[perm]

    return (
        torch.from_numpy(data_train).float(),
        torch.from_numpy(target_train).long(),
        torch.from_numpy(data_test).float(),
        torch.from_numpy(target_test).long(),
        train_ind,
    )


################################################################################
## Helper Functions / Classes


def _determine_feature_layer(model_type: str) -> str:
    """Determines the feature layer based on the model type."""
    if model_type == "ShuffleNet":
        return "mean"
    elif model_type == "GoogLeNet":
        return "dropout"
    else:
        return "flatten"


def _determine_hardware(hardware_acceleration: bool) -> str:
    """Determines which hardware is available on your device."""
    if hardware_acceleration and torch.cuda.is_available():
        return "cuda"
    elif hardware_acceleration and torch.has_mps:
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
        datapoint = self.data[index]
        label = self.targets[index]

        if self.transform:
            datapoint = self.transform(datapoint)
        return datapoint, label

    def __len__(self):
        return len(self.data)


def train_model(
    model: nn.Module,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    num_epochs: int = 25,
    verbose: int = 1,
) -> nn.Module:
    """Helper function for transfer learning. Fine tunes the pretrained model."""
    since = time.time()
    assert verbose in [0, 1, 2]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if verbose >= 1:
        print(f"Training model for {num_epochs} epochs")

    for epoch in range(num_epochs):
        if verbose == 2:
            print(f"Epoch {epoch + 1}/{num_epochs}")
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
                running_corrects += int(round(torch.sum(preds == labels.data).item()))
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects * 1.0 / dataset_sizes[phase]

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


def _construct_tl_network(dataset: str, model_type: Optional[str]) -> nn.Module:
    """Modifies the pretrained PyTorch neural network for transfer learning.

    Args:
        dataset: Name of the dataset.
        model_type: Name of the neural network.

    Returns:
        Modified neural network
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


def normalize_data(data: torch.Tensor) -> torch.Tensor:
    """Normalizes data to range [0,1]"""
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def _get_knn_data(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to load knn_data for pretrained CNNVAEs"""
    knn_ind = np.load("knn_data/" + dataset + "_knn_ind.npy")
    knn_dist = np.load("knn_data/" + dataset + "_knn_dist.npy")
    return knn_ind, knn_dist


################################################################################
## Encoding/Transfer Learning Functions


def encode_dataset(
    dataset: str,
    model_path: str,
    batch_size: int = ENCODING_BATCH_SIZE,
    hardware_acceleration: bool = False,
) -> np.ndarray:
    """Implements CNNVAE embedding"""
    # Decide which device to use
    device = torch.device(
        _determine_hardware(hardware_acceleration=hardware_acceleration)
    )
    if device != "cpu":
        with torch.no_grad():
            torch.cuda.empty_cache()

    # Load data and convert to torch
    data, _ = load_dataset(dataset)
    torch_data = torch.from_numpy(data).float()

    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    encoded_data = np.array([])
    with torch.no_grad():
        for idx in range(0, len(data), batch_size):
            data_batch = torch_data[idx : idx + batch_size]
            if len(encoded_data) == 0:
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
    hardware_acceleration: bool = False,
) -> np.ndarray:
    """Implements zero-shot transfer learning"""
    # Decide which device to use
    device = torch.device(
        _determine_hardware(hardware_acceleration=hardware_acceleration)
    )
    if device != "cpu":
        with torch.no_grad():
            torch.cuda.empty_cache()

    # Load data
    data, _ = load_dataset(dataset)
    torch_data = normalize_data(torch.from_numpy(data).float()).to(device)

    torch_data = torch_data.expand([-1, 3, -1, -1])

    # If custom model, apply grayscale
    if model_type[-3:] == ".pt":
        print("Using grayscale")
        transform = transforms.Grayscale()
        torch_data = transform(data)

    encoded_data = np.array([])
    # Load the desired model and encode data
    if model_type in PYTORCH_NEURAL_NETWORKS:
        model_used = PYTORCH_NEURAL_NETWORKS_DICT.get(model_type)
        model = torch.hub.load("pytorch/vision:v0.10.0", model_used, pretrained=True)
        model.to(device)
        model.eval()

        feature_layer = _determine_feature_layer(model_type)
        with torch.no_grad():
            feature_extractor = create_feature_extractor(
                model, return_nodes=[feature_layer]
            )
            feature_extractor.eval()

            encoded_data = (
                feature_extractor(torch_data.to(device))[feature_layer].cpu().numpy()
            )
    else:
        model = torch.load(model_type, map_location=device)
        model.eval()
        with torch.no_grad():
            for idx in range(0, len(torch_data), batch_size):
                data_batch = torch_data[idx : idx + batch_size]
                if len(encoded_data) == 0:
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
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    data_augmentation: bool = False,
    hardware_acceleration: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Implements fine-tuned transfer learning"""
    # Decide which device to use
    device = torch.device(
        _determine_hardware(hardware_acceleration=hardware_acceleration)
    )
    if device != "cpu":
        with torch.no_grad():
            torch.cuda.empty_cache()

    if (model_type is None) or (model_type in PYTORCH_NEURAL_NETWORKS):
        # Load training and testing data
        if data_info is None:
            (
                data_train,
                label_train,
                data_test,
                label_test,
                train_ind,
            ) = load_dataset_fine_tuned_tl(dataset)
        else:
            data_train, label_train, data_test, label_test = data_info

        # Put data on the device
        data_train = data_train.to(device)
        label_train = label_train.to(device)
        data_test = data_test.to(device)
        label_test = label_test.to(device)

        # Modify data
        with torch.no_grad():
            data_train = normalize_data(data_train)
            data_test = normalize_data(data_test)

            data_train = data_train.expand([-1, 3, -1, -1])
            data_test = data_test.expand([-1, 3, -1, -1])

        if data_augmentation:
            data_transforms = {
                "train": transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
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

        model_ft = _construct_tl_network(dataset, model_type)
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

        data, _ = load_dataset(dataset)
        torch_data = normalize_data(torch.from_numpy(data).float()).to(device)

        torch_data = torch_data.expand([-1, 3, -1, -1])
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

            encoded_data_dict = feature_extractor(torch_data)
            encoded_data = encoded_data_dict[feature_layer].detach().cpu().numpy()
    else:
        data, _ = load_dataset(dataset)
        torch_data = torch.from_numpy(data).float()
        model = torch.load(model_type, map_location=device)
        model.eval()
        encoded_data = np.array([])
        with torch.no_grad():
            for idx in range(0, len(data), batch_size):
                data_batch = torch_data[idx : idx + batch_size]
                if len(encoded_data) == 0:
                    encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
                else:
                    encoded_data = np.vstack(
                        (
                            encoded_data,
                            model.encode(data_batch.to(device)).cpu().numpy(),
                        )
                    )
    return encoded_data, train_ind


################################################################################
## MSTAR Helper Functions
# All code below is from MSTAR Github
# jwcalder MSTAR-Active-Learning


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


################################################################################
## Toy Datasets


def gen_checkerboard_3(num_samples=500, randseed=123):
    """Checkerboard 3 dataset"""
    np.random.seed(randseed)
    X = np.random.rand(num_samples, 2)
    labels = np.mod(np.floor(X[:, 0] * 3) + np.floor(X[:, 1] * 3), 3).astype(np.int64)

    return X, labels


def gen_stripe_3(num_samples=500, width=1 / 3, randseed=123):
    """Stripe 3 dataset"""
    np.random.seed(randseed)
    X = np.random.rand(num_samples, 2)
    labels = np.mod(np.floor(X[:, 0] / width + X[:, 1] / width), 3).astype(np.int64)

    return X, labels
