import os
import tracemalloc
import math
import time
import copy

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as torch_models

import models
import graphlearning as gl

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast


################################################################################
## Default Parameters

#TODO: CHECK THESE ARE ORIGINALS
KNN_NUM = 50
TL_EPOCHS = 10
ENCODING_BATCH_SIZE=1000

#TODO: This should be in bal file
ACQUISITION_FUNCTIONS = ['uc', 'vopt', 'mc', 'mcvopt']
AL_METHODS = ['local_max', 'random', 'topn_max', 'acq_sample', 'global_max']
AL_METHOD_NAMES = ['LocalMax', 'Random', 'TopMax', 'Acq_sample', 'Sequential']

AVAILABLE_SAR_DATASETS = ['mstar', 'open_sar_ship', 'fusar']
AVAILABLE_EMBEDDINGS = ['cnnvae', 'zero_shot_tl', 'fine_tuned_tl']
SAR_DATASET_SIZE_DICT = {'mstar': 6874, 'open_sar_ship': 2296, 'fusar': 4856}

PYTORCH_NEURAL_NETWORKS = [
    'ResNet', 'ShuffleNet', 'AlexNet', 'DenseNet', 'GoogLeNet', 'MobileNetV2',
    'ResNeXt', 'Wide ResNet'
]

PYTORCH_NEURAL_NETWORKS_DICT = {'ResNet': 'resnet18',
    'ShuffleNet': 'shufflenet_v2_x0_5',
    'AlexNet': 'alexnet',
    'DenseNet': 'densenet121',
    'GoogLeNet': 'googlenet',
    'MobileNetV2': 'mobilenet_v2',
    'ResNeXt': 'resnext50_32x4d',
    'Wide ResNet': 'wide_resnet50_2'
}

DEFAULT_NEURAL_NETWORKS_DICT = {
    'mstar': 'ResNet',
    'open_sar_ship': 'AlexNet',
    'fusar': 'ShuffleNet'
}


MAX_NEW_SAMPLES_DICT = {'mstar': 481, 'open_sar_ship': 690, 'fusar': 3060}
FINE_TUNED_MAX_NEW_SAMPLES_DICT = {'mstar': 137, 'open_sar_ship': 574, 'fusar': 2816}

USE_HARDWARE_ACCELERATION = False


################################################################################
## Convenient User Functions
#Convenient functions for loading data for the experiments
#These are like the default embedding functions (with proper networks, etc.)

#TODO: Check the types on this
def CNNVAE(
        dataset: str,
        knn_num: int = KNN_NUM,
        include_knn_data: bool = True
    ) -> tuple[np.ndarray, np.ndarray]: #TODO: Make new output type
    """
    Embeds the chosen dataset using a trained CNNVAE.
    
    param dataset:
    """
    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"
    
    #Load Dataset
    X, labels = load_dataset(dataset, return_torch=False, concatenate=True)
    
    #Determine CNNVAE model
    if dataset == 'mstar':
        model_path = "./models/SAR10_CNNVAE.pt"
    elif dataset == 'open_sar_ship':
        model_path = "models/open_sar_ship_CNN.pt"
    else:
        model_path = "models/fusar_CNN.pt"
    
    #Encode Dataset
    X = encode_dataset(dataset, model_path, batch_size = 1000)
    
    if include_knn_data:
        try:
            #Load the knn data from the cnnvae (computed once in separate file)
            knn_data = _get_knn_data(dataset)
            print("Using pre-computed cnnvae embedding knn_data")
        except:
            #Compute knn data
            print("Computing knn_data")
            knn_data = gl.weightmatrix.knnsearch(X, knn_num, method='annoy', similarity='angular')
            
        return X, labels, knn_data
    else:
        return X, labels


def zero_shot_TL(
        dataset: str,
        knn_num: int = KNN_NUM,
        data_augmentation: bool = True,
        include_knn_data: bool = True
    ) -> tuple[np.ndarray, np.ndarray]: #TODO: Make new output type
    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"
    
    if dataset == 'mstar':
        X, labels = encode_pretrained('mstar', 'ResNet', transformed=data_augmentation)
    elif dataset == 'open_sar_ship':
        X, labels = encode_pretrained('open_sar_ship', 'AlexNet', transformed=data_augmentation)
    else:
        X, labels = encode_pretrained('fusar', 'ShuffleNet', transformed=data_augmentation)
    
    if include_knn_data:
        knn_data = gl.weightmatrix.knnsearch(X, knn_num, method='annoy', similarity='angular')
        return X, labels, knn_data
    else:
        return X, labels


#TODO: This isn't done yet
#I need to set this up so it is nice
def fine_tuned_TL(
        dataset: str,
        knn_num: int = KNN_NUM,
        num_epochs: int = TL_EPOCHS,
        network = None,                 #TODO: Here
        data_augmentation: bool = True,
        include_knn_data: bool = True
    ) -> tuple[np.ndarray, np.ndarray]: #TODO: Make new output type
    assert dataset in AVAILABLE_SAR_DATASETS, "Invalid Dataset"
    
    X, labels = load_dataset(dataset, return_torch=False, concatenate=True)
    
    #If network is None, we use the defaults. This functionality is in
    #   the encode_transfer_learning function
    
    if dataset == 'mstar':
        X = encode_transfer_learning('mstar', model_type=network, transfer_batch_size=64, epochs=num_epochs, data_info=None, transformed=data_augmentation)
    elif dataset == 'open_sar_ship':
        X = encode_transfer_learning('open_sar_ship', model_type=network, transfer_batch_size=64, epochs=num_epochs, data_info=None, transformed=data_augmentation)
    else:
        X = encode_transfer_learning('fusar', model_type=network, transfer_batch_size=64, epochs=num_epochs, data_info=None, transformed=data_augmentation)
    
    if include_knn_data:
        #Takes a long time
        knn_data = gl.weightmatrix.knnsearch(X, knn_num, method='annoy', similarity='angular')
        return X, labels, knn_data
    else:
        return X, labels


def load_dataset(
        dataset: str,
        return_torch: bool = True,
        shuffle_train_set: bool = False,
        concatenate: bool = False
    ):                              #TODO: Make new output data type
    if dataset == 'open_sar_ship':
        data_train = np.load('data/OpenSARShip/SarTrainImages.npz')['arr_0']
        target_train = np.load('data/OpenSARShip/SarTrainLabels.npy')
        data_test = np.load('data/OpenSARShip/SarTestImages.npz')['arr_0']
        target_test = np.load('data/OpenSARShip/SarTestLabels.npy')
    elif dataset == 'fusar':
        data_train = np.load('data/Fusar/FusarTrainImages.npz')['arr_0']
        target_train = np.load('data/Fusar/FusarTrainLabels.npy')
        data_test = np.load('data/Fusar/FusarTestImages.npz')['arr_0']
        target_test = np.load('data/Fusar/FusarTestLabels.npy')
    elif dataset == 'mstar':
        hdr, fields, mag, phase = load_MSTAR('data/MSTAR')
        data = polar_transform(mag, phase)
        target, target_names = targets_to_labels(hdr)
        
        mstar_size = SAR_DATASET_SIZE_DICT['mstar']
        tl_num = round(mstar_size * .05)
        train_ind = np.random.choice(mstar_size, size=tl_num, replace=False)
        test_ind = np.setdiff1d(np.arange(mstar_size), train_ind)
        
        data_train = data[train_ind, ...]
        data_test = data[test_ind, ...]
        target_train = target[train_ind]
        target_test = target[test_ind]
    else:
        assert False, "Code not implemented for this dataset"

    if shuffle_train_set:
        P = np.random.permutation(data_train.shape[0])
        data_train = data_train[P,:,:,:]
        target_train = target_train[P]

    if concatenate and dataset != 'mstar':
        data = np.vstack((data_train,data_test))
        target = np.hstack((target_train,target_test))

    if return_torch:
        if concatenate:
            data = torch.from_numpy(data).float()
            target = torch.from_numpy(target).long()
        else:
            data_train = torch.from_numpy(data_train).float()
            target_train = torch.from_numpy(target_train).long()
            data_test = torch.from_numpy(data_test).float()
            target_test = torch.from_numpy(target_test).long()

    if concatenate:
        return data, target
    else:
        return data_train, target_train, data_test, target_test


################################################################################
## Helper Functions / Classes


#TODO: Want random crop
#TODO: Should resize before random crop?
#TODO: Make sure mean and std have correct size

#TODO: Want to make sure to use mps here
#TODO: It wasn't faster. Check this again
def _apply_data_augmentation(
        data: torch.Tensor,
        reshape: bool = True
    ) -> torch.Tensor:
    #print(data.shape)
    device = torch.device(_determine_hardware())
    t1 = transforms.GaussianBlur(kernel_size=(1,1), sigma=(1, 1))
    t2 = transforms.RandomCrop(data.shape[-1], padding=32)
    t3 = transforms.ColorJitter(contrast=2,brightness=.5, hue=.3)
    
    data = t3(t2(t1(data)))
    
    mean = torch.mean(data, dim=(0, 2, 3))
    std = torch.std(data, dim=(0, 2, 3))
    
    t4 = transforms.Normalize(mean, std)
    data = t4(data)
    
    #All pytorch CNN assume the image is 3 channel
    #Turn image data into 3 channels if use pytorch pretrained CNN
    #data_shape = list(data.shape)
    #data_shape[1] = 3
    #data = data.expand(data_shape)
    if reshape:
        data = data.expand([-1, 3, -1, -1])
    
    return data
    
#MARK: Used in the encode_pretrained
def _determine_feature_layer(model_type: str) -> str:
    feature_layer = "flatten"
    if model_type == "ShuffleNet":
        feature_layer = "mean"
    elif model_type == "GoogLeNet":
        feature_layer == "dropout"
    return feature_layer

def _determine_hardware() -> str:
    if USE_HARDWARE_ACCELERATION and torch.cuda.is_available():
        return 'cuda'
    elif USE_HARDWARE_ACCELERATION and torch.has_mps:
        return 'mps'
    else:
        return 'cpu'

#Helper class for transfer learning
class MyDataset(Dataset):
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

#Helper function for transfer learning
#Fine tune the pretrained model
#TODO: Add types here
def train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        dataloaders,
        dataset_sizes,
        num_epochs=25
    ):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float())
                    _, preds = torch.max(outputs, 1)
                    outputs = nn.functional.log_softmax(outputs, dim=-1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#TODO: Check this is correct output type
def _construct_TL_network(dataset: str, model_type: str) -> nn.Module:
    #The most optimized choice of CNNs
    if model_type is None:
        if dataset == 'open_sar_ship':
            model_ft = torch_models.alexnet(pretrained=True)
            model_ft.classifier[1] = nn.Linear(9216, 1024)
            model_ft.classifier[4] = nn.Linear(1024, 256)
            model_ft.classifier[6] = nn.Linear(256, 3)
        elif dataset == 'fusar':
            model_ft = torch_models.shufflenet_v2_x0_5(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 5)
        else:
            model_ft = torch_models.resnet18(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 10)
    else:
        if model_type == 'AlexNet':
            model_ft = torch_models.alexnet(pretrained=True)
            model_ft.classifier[1] = nn.Linear(9216, 1024)
            model_ft.classifier[4] = nn.Linear(1024, 256)
            if dataset == 'open_sar_ship':
                model_ft.classifier[6] = nn.Linear(256, 3)
            else:
                model_ft.classifier[6] = nn.Linear(256, 5)
        elif model_type == 'DenseNet':
            model_ft = torch_models.densenet121(pretrained=True)
            num_ftrs = model_ft.classifier.in_features
            if dataset == 'open_sar_ship':
                model_ft.classifier = nn.Linear(num_ftrs, 3)
            else:
                model_ft.classifier = nn.Linear(num_ftrs, 5)
        elif model_type == 'MobileNetV2':
            model_ft = torch_models.mobilenet_v2(pretrained=True)
            num_ftrs = model_ft.classifier[1].in_features
            if dataset == 'open_sar_ship':
                model_ft.classifier[1] = nn.Linear(num_ftrs, 3)
            else:
                model_ft.classifier[1] = nn.Linear(num_ftrs, 5)
        else:
            model_used = models_dict.get(model_type)
            model_ft = torch.hub.load('pytorch/vision:v0.10.0', model_used, pretrained=True)
            num_ftrs = model_ft.fc.in_features
            if dataset == 'open_sar_ship':
                model_ft.fc = nn.Linear(num_ftrs, 3)
            else:
                model_ft.fc = nn.Linear(num_ftrs, 5)
    return model_ft

#TODO: types
def NormalizeData(data):
    '''Normalizes data to range [0,1]

    Parameters
    ----------
    data : Numpy array

    Returns
    -------
    norm_data : Normalized array
    '''
    if isinstance(data, torch.Tensor):
        norm_data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    elif isinstance(data, np.ndarray):
        norm_data = (data - np.min(data))/(np.max(data) - np.min(data))
    else:
        assert False, "Invalid type for NormalizeData"
    return norm_data

def _get_knn_data(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    knn_ind = np.load("knn_data/" + dataset + "_knn_ind.npy")
    knn_dist = np.load("knn_data/" + dataset + "_knn_dist.npy")
    return knn_ind, knn_dist

################################################################################
## Encoding/Transfer Learning Functions

#This was efficient
#TODO: Types
def encode_dataset(
        dataset,
        model_path,
        batch_size = ENCODING_BATCH_SIZE
    ):
    #Decide which device to use
    device = torch.device(_determine_hardware())

    #Load data
    data, target = load_dataset(dataset, concatenate=True)
    
    #Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    encoded_data = None
    with torch.no_grad():
        for idx in range(0,len(data),batch_size):
            data_batch = data[idx:idx+batch_size]
            if encoded_data is None:
                encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
            else:
                encoded_data = np.vstack((encoded_data,model.encode(data_batch.to(device)).cpu().numpy()))

    return encoded_data

#TODO: types
def encode_pretrained(
        dataset,
        model_type,
        batch_size = ENCODING_BATCH_SIZE,
        device_name='mps',
        balanced = False,
        transformed = False
    ):
    #Decide which device to use
    #device = torch.device(_determine_hardware())

    #Load data
    data, labels = load_dataset(dataset, return_torch=True, concatenate=True)
    #data = data.to(device)
    
    #Apply data augmentation
    if transformed:
        data = _apply_data_augmentation(data)
    else:
        data = data.expand([-1, 3, -1, -1])

    #If custom model, apply grayscale
    if model_type[-3:] == ".pt":
        print("Using grayscale")
        transform = transforms.Grayscale()
        data = transform(data)
    
    encoded_data = None
    models_dict = PYTORCH_NEURAL_NETWORKS_DICT.copy()
    #Load the desired model and encode data
    if model_type in models_dict.keys():
        model_used = models_dict.get(model_type)
        model = torch.hub.load('pytorch/vision:v0.10.0', model_used, pretrained=True)
        model.eval()
        
        feature_layer = _determine_feature_layer(model_type)
        with torch.no_grad():
            feature_extractor = create_feature_extractor(model, return_nodes=[feature_layer])
            feature_extractor.eval()
            
            encoded_data = feature_extractor(data)[feature_layer].cpu().numpy()
    else:
        model = torch.load(model_type, map_location=device)
        model.eval()
        with torch.no_grad():
            for idx in range(0,len(data),batch_size):
                data_batch = data[idx:idx+batch_size]
                if encoded_data is None:
                    encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
                else:
                    encoded_data = np.vstack((encoded_data,model.encode(data_batch.to(device)).cpu().numpy()))

    return encoded_data, labels

#TODO: types
def encode_transfer_learning(
        dataset,
        model_type = None,
        epochs = 15,
        transfer_batch_size = 64,
        batch_size = ENCODING_BATCH_SIZE,
        data_info=None,
        transformed = False
    ):
    #Decide which device to use
    device = torch.device(_determine_hardware())
    
    if (model_type is None) or (model_type in PYTORCH_NEURAL_NETWORKS):
        #Load training and testing data
        if data_info is None:
            data_train, label_train, data_test, label_test = load_dataset(dataset, return_torch=True, concatenate=False)
        else:
            data_train, label_train, data_test, label_test = data_info
        
        #Modify data
        data_train = data_train.expand([-1, 3, -1, -1])
        data_test = data_test.expand([-1, 3, -1, -1])
        with torch.no_grad():
            #TODO: Idk if we want this
            data_train = NormalizeData(data_train)
            data_test = NormalizeData(data_test)
        
        #TODO: Do we know if data_train and data_test are on mps?
        #+++++++++++
        if transformed:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.GaussianBlur(kernel_size=(1,1), sigma=(1, 1)),
                    transforms.ColorJitter(contrast=2,brightness=.5, hue=.3)
                ]),
                'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.GaussianBlur(kernel_size=(1,1), sigma=(1, 1)),
                    transforms.ColorJitter(contrast=2,brightness=.5, hue=.3)
                ]),
            }
        else:
            #Identity transformations
            data_transforms = {'train': None, 'val': None}
        
        #Packing the training and evaluation data
        #If run out of CUDA memory in testings below, try to reduce the batch size to 32 or 16
        #data_aug = lambda data: _apply_data_augmentation(data, reshape=False)
        dataset_train = MyDataset(data_train, label_train, transform=data_transforms['train'])
        dataloader_train = DataLoader(dataset_train, batch_size=transfer_batch_size)
        
        dataset_val = MyDataset(data_test, label_test, transform=data_transforms['val'])
        dataloader_val = DataLoader(dataset_val, batch_size=transfer_batch_size)
        
        train_size = data_train.shape[0]
        val_size = data_test.shape[0]
        
        dataloaders = {'train':dataloader_train, 'val':dataloader_val}
        dataset_sizes = {'train':train_size, 'val':val_size}
        #+++++++++++
        
        model_ft = _construct_TL_network(dataset, model_type)
        model_ft.to(device)
        
        # Set up the optimizer to optimize all model parameters
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        
        #Train the model with cross entropy loss
        criterion = nn.CrossEntropyLoss()
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=epochs)
        
        data, _ = load_dataset(dataset, return_torch=True, shuffle_train_set=False, concatenate=True)
        
        if transformed:
            #Modify the data
            data = _apply_data_augmentation(data)
        else:
            data = data.expand([-1, 3, -1, -1])
        #TODO: IDK if we want this
        data = NormalizeData(data)
        
        #data = data.to(device)
        model_ft.cpu()
        model_ft.eval()
        
        if model_type is None:
            feature_layer = _determine_feature_layer(DEFAULT_NEURAL_NETWORKS_DICT[dataset])
        else:
            feature_layer = _determine_feature_layer(model_type)
        
        with torch.no_grad():
            feature_extractor = create_feature_extractor(model_ft, return_nodes=[feature_layer])
            
            encoded_data_dict = feature_extractor(data)
            encoded_data = encoded_data_dict[feature_layer].detach().cpu().numpy()
    else:
        data, _ = load_dataset(dataset, concatenate=True)
        model = torch.load(model_type, map_location=device)
        model.eval()
        encoded_data = None
        with torch.no_grad():
            for idx in range(0,len(data),batch_size):
                data_batch = data[idx:idx+batch_size]
                if encoded_data is None:
                    encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
                else:
                    encoded_data = np.vstack((encoded_data,model.encode(data_batch.to(device)).cpu().numpy()))
    return encoded_data


################################################################################
### MSTAR Helper Functions
##All code below is from MSTAR Github

#TODO: Make these private. Don't need types

def load_MSTAR(root_dir= './data/MSTAR'):
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

    M = np.load(os.path.join(root_dir,'SAR10a.npz'), allow_pickle=True)
    hdr_a,fields,mag_a,phase_a = M['hdr'],M['fields'],M['mag'],M['phase']

    M = np.load(os.path.join(root_dir,'SAR10b.npz'), allow_pickle=True)
    hdr_b,fields,mag_b,phase_b = M['hdr'],M['fields'],M['mag'],M['phase']

    M = np.load(os.path.join(root_dir,'SAR10c.npz'), allow_pickle=True)
    hdr_c,fields,mag_c,phase_c = M['hdr'],M['fields'],M['mag'],M['phase']

    hdr = np.concatenate((hdr_a,hdr_b,hdr_c))
    mag = np.concatenate((mag_a,mag_b,mag_c))
    phase = np.concatenate((phase_a,phase_b,phase_c))

    #Clip to [0,1] (only .18% of pixels>1)
    mag[mag>1]=1

    return hdr, fields, mag, phase

def train_test_split(hdr,train_fraction):
    '''Training and testing split (based on papers, angle=15 or 17)

    Parameters
    ----------
    hdr : Header info
    train_fraction : Fraction in [0,1] of full train data to use

    Returns
    -------
    full_train_mask : Boolean training mask for all angle==17 images
    test_mask : Boolean testing mask
    train_idx : Indices of training images selected
    '''

    angle = hdr[:,6].astype(int)
    full_train_mask = angle == 17
    test_mask = angle == 15
    num_train = int(np.sum(full_train_mask)*train_fraction)
    train_idx = np.random.choice(np.arange(hdr.shape[0]),size=num_train,replace=False,p=full_train_mask/np.sum(full_train_mask))

    return full_train_mask, test_mask, train_idx

def targets_to_labels(hdr):
    '''Converts target names to numerical labels

    Parameters
    ----------
    hdr : Header data

    Returns
    -------
    labels : Integer labels from 0 to k-1 for k classes
    target_names : List of target names corresponding to each label integer
    '''

    targets = hdr[:,0].tolist()
    classes = set(targets)
    label_dict = dict(zip(classes, np.arange(len(classes))))
    labels = np.array([label_dict[t] for t in targets],dtype=int)
    target_names = list(label_dict.keys())

    return labels, target_names

def polar_transform(mag, phase):
    '''
    Peform polar transormation of data.

    Parameters
    ----------
        mag : Magnitude images
        phase : Phase data

    Returns
    -------
        data : nx3 numpy array with (mag,real,imaginary)
    '''

    real = (mag*np.cos(phase) + 1)/2
    imaginary = (mag*np.sin(phase) +1)/2
    data = np.stack((mag,real,imaginary),axis=1)

    return data

def rotate_images(data,center_crop=True, top=-14, left=14, height=100, width=100, size=128):
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
  threshold = data_mean + 2*data_std

  for index in range(len(data)):

    image = data_rotated[index,:,:,:]
    
    if center_crop:
      image = transforms.functional.resized_crop(image,top=top, left=left, height=height, width=width, size=size)
      img_thresholded = (image[0,:,:] > threshold).type(torch.float)
    else:
      img = data_rotated[index,0,:,:]
      img_thresholded = (img > threshold).type(torch.float)

    # very rarely thresholding results in too sparse of an image because the image is essentailly blank to begin with
    if torch.sum(img_thresholded) <= 1:
      continue
    
    img_thresholded = (img_thresholded==1).nonzero(as_tuple=False).type(torch.float)
    U,S,V = torch.pca_lowrank(img_thresholded)
    angle = (180/math.pi)*torch.atan(V[1,1]/V[0,1])
    angle = angle.item()

    img_rotated = transforms.functional.rotate(image, 90-angle)
    data_rotated[index,0,:,:] = img_rotated

  return data_rotated




