import torch
import numpy as np
import os

import models
import torchvision.models as torch_models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import copy
from torch.optim import lr_scheduler
import graphlearning as gl

from torch.cuda.amp import GradScaler, autocast

def load_dataset(dataset, return_torch=True, shuffle_train_set=False, concatenate=False):
    """Load Dataset
    ===================

    Loads a dataset.

    Parameters
    ----------
    dataset : {'open_sar_ship','fusar'}
        Dataset name.
    return_torch : bool (optional), default = True
        Whether to return as torch tensors.
    shuffle_train_set: bool (optional), default = False
        Whether to shuffled the training set.
    concatenate : bool(optional), default = False
        Whether to concatenate train and test data together.

    Returns (if concatenate=False)
    -------
    data_train : numpy array (or torch tensor)
        Training data features.
    target_train : numpy array (or torch tensor)
        Training data labels.
    data_test : numpy array (or torch tensor)
        Training data features.
    target_test : numpy array (or torch tensor)
        Training data labels.

    Returns (if concatenate=True)
    -------
    data : numpy array (or torch tensor)
        All data features.
    target : numpy array (or torch tensor)
        All labels
    """

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
        #print(data)
        data = torch.from_numpy(data).float()
        labels, target_names = targets_to_labels(hdr)
        
        ##TODO: Change this
        data_train = data[:100, ...]
        data_test = data[100:, ...]
        target_train = labels[:100]
        target_test = labels[100:]
    else:
        assert False, "Code not implemented for this dataset"

    if shuffle_train_set:
        P = np.random.permutation(data_train.shape[0])
        data_train = data_train[P,:,:,:]
        target_train = target_train[P]

    if concatenate:
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



def encode_dataset(dataset, model_path, batch_size = 1000, cuda = True):
    '''Encode Dataset
    ==========================================

    Encodes a dataset with a pretrained CNN.

    Parameters
    ----------
    dataset : {'open_sar_ship','fusar'}
        Dataset name.
    model_path : string
        Path to .pt file containing torch model for trained CNN
    batch_size : int (optional)
        Size of minibatches to use in encoding. Reduce if you get out of memory errors (default = 1000)
    cuda : bool (optional)
        Whether to use GPU or not (default = True)

    Returns
    -------
    encoded_data : numpy array
        Data encoded by model.encode() (e.g., the CNN features)
    '''

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #Load data
    data, target = load_dataset(dataset, concatenate=True)
    
    #Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    encoded_data = None
    with torch.no_grad():
        print(len(data))
        for idx in range(0,len(data),batch_size):
            data_batch = data[idx:idx+batch_size]
            if encoded_data is None:
                print("About to start encoding")
                encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
            else:
                encoded_data = np.vstack((encoded_data,model.encode(data_batch.to(device)).cpu().numpy()))

    return encoded_data

def encode_pretrained(dataset, model_type, batch_size = 1000, cuda = True, normalized = False, balanced = False, transformed = False):
    '''Encode Dataset with Pretrained CNN
    ==========================================

    Encodes a dataset with a pretrained CNN.
    Not only trainCNN.py but also other pretrained networks from pytorch

    Parameters
    ----------
    dataset : {'open_sar_ship','fusar'}
        Dataset name.
    model_type : {'AlexNet', 'DenseNet', 'GoogLeNet', 'Inception', 'MobileNetV2', 
                 'ResNet', 'ResNeXt', 'ShuffleNet', 'Wide ResNet'} or .pt files
        pretarined Pytorch CNN models, or other .pt files
        For OpenSarShip, AlexNet works the best;
        For Fusar, ShuffleNet works the best.
    batch_size : int (optional)
        Size of minibatches to use in encoding. Reduce if you get out of memory errors (default = 1000)
    cuda : bool (optional)
        Whether to use GPU or not (default = True)
    normalized : bool (optional)
        Whether or not to normalize the given data
    balanced : bool (optional)
        Whether or not to balanced the given data, i.e. make each class of the given dataset
        to have the same amount of data
    transform : bool (optional)
        Whether or not to apply the series of predetermined transformations to the given dataset 

    Returns
    -------
    encoded_data : numpy array
        Data encoded by model.encode() (e.g., the CNN features), or encoded by 
        feature_extractor of the flatten node.
    new_labels: numpy array
        labels of each data, perhaps after balancing
        
    '''
    
    #Decide which device to use
    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #Load data
    data, labels = load_dataset(dataset, return_torch=False, concatenate=True)
    data = torch.from_numpy(data)
    #Data augmentation
    #Rotation produces negative effects so there's none of it
    if transformed:
        data_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(1,1), sigma=(1, 1)),
            # #transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomCrop(128, padding=32),
            transforms.ColorJitter(contrast=2,brightness=.5, hue=.3),
            transforms.Normalize(torch.mean(data), torch.std(data)),
        ])
        data = data_transform(data)
        #data = rotate_images(data)
    labels = list(labels)

    #All pytorch CNN assume the image is 3 channel
    #Turn image data into 3 channels if use pytorch pretrained CNN
    data_shape = list(data.shape)
    data_shape[1] = 3
    data = data.expand(data_shape)
    
    if balanced:
        #Balance all datasets to have the same number (the min among all datasets)
        distinct_labels = list(set(labels))
        label_nums = []
        for my_label in distinct_labels:
          label_nums.append(labels.count(my_label))
        min_length = min(label_nums)
  
        num_elements = data.shape[0]
        data_temp = torch.zeros(len(distinct_labels)*min_length,data.shape[1],data.shape[2],data.shape[3])
        #The labels & data points we'll consider
        new_labels = []
        current_index = 0;
        for i in range(num_elements):
          if new_labels.count(labels[i]) < min_length:
            new_labels.append(labels[i])
            data_temp[current_index,:,:,:] = data[i,:,:,:]
            current_index += 1
        data = data_temp
        new_labels = np.array(new_labels)
    else:
        new_labels = np.array(labels)
    
    if normalized:
        #Change the data into range [0,1]
        data = data.cpu().detach().numpy()
        data = NormalizeData(data)
        #data = torch.from_numpy(data)
  
        num_elements = data.shape[0]
  
        #Normalize the data to fit into the pretrained CNNs
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data_tensor = torch.zeros(data.shape)
        for i in range(num_elements):
          temp = torch.from_numpy(data[i,:,:,:])
          temp = normalize(temp)
          data_tensor[i,:,:,:] = temp
        data = data_tensor

    if model_type[-3:] == ".pt":
        transform = transforms.Grayscale()
        data = transform(data)

    encoded_data = None
    models_dict = {"ResNet": "resnet18",
                   "ShuffleNet": 'shufflenet_v2_x1_0',
                   "AlexNet": 'alexnet',
                   "DenseNet": 'densenet121',
                   "Inception": 'inception_v3',
                   "GoogLeNet": 'googlenet',
                   "MobileNetV2": 'mobilenet_v2',
                   "ResNeXt": 'resnext50_32x4d',
                   "Wide ResNet": 'wide_resnet50_2'}
    #Load the desired model and encode data
    if model_type in models_dict.keys():
        model_used = models_dict.get(model_type)
        model = torch.hub.load('pytorch/vision:v0.10.0', model_used, pretrained=True)
        model.eval()
        feature_layer = "flatten"
        with torch.no_grad():
            if model_type == "ShuffleNet":
                feature_layer = "mean"
            elif model_type == "GoogLeNet":
                feature_layer == "dropout"
            feature_extractor = create_feature_extractor(model, return_nodes=[feature_layer])
            
            print(type(data))
            print(data)
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

    return encoded_data, new_labels

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
def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                    outputs = nn.functional.log_softmax(outputs)
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
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

def encode_transfer_learning(dataset, model_type = None, epochs = 15, transfer_batch_size = 64, batch_size = 1000, cuda = True, data_info=None):
    '''Encode Dataset with Transfer Learning on Pretrained CNN
    ==========================================

    Encodes a dataset with a transfer learning pretrained CNN.
    Not only trainCNN.py but also other pretrained networks from pytorch

    Parameters
    ----------
    dataset : {'open_sar_ship','fusar'}
        Dataset name.
    model_type : .pt files
        .pt files if specified; otherwise use default optimized pytorch CNNs
    epochs :  int (optional)
        Epochs of transfer learning
    transfer_batch_size: int (optional)
        Size of minibatches used in transfer learning
    batch_size : int (optional)
        Size of minibatches to use in encoding. Reduce if you get out of memory errors (default = 1000)
    cuda : bool (optional)
        Whether to use GPU or not (default = True)
    train_percent : in (optional)
        Percentage of training data used for transfer learning
        If none, use the default load_dataset training data

    Returns
    -------
    encoded_data : numpy array
        Data encoded by model.encode() (e.g., the CNN features), or encoded by 
        feature_extractor of the flatten node.
    '''
    
    #Decide which device to use
    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    

    models_dict = {"ResNet": "resnet18",
                   "ShuffleNet": 'shufflenet_v2_x0_5',
                   "AlexNet": 'alexnet',
                   "DenseNet": 'densenet121',
                   "GoogLeNet": 'googlenet',
                   "MobileNetV2": 'mobilenet_v2',
                   "ResNeXt": 'resnext50_32x4d',
                   "Wide ResNet": 'wide_resnet50_2'}
    #Do transfer learning if no .pt file is provided
    #Otherwise use feature extraction from the pretrained CNN in .pt file
    if (model_type == None) or (model_type in models_dict.keys()):
        data, labels = load_dataset(dataset, return_torch=True, concatenate=True)
        
        if data_info == None:
            data_train, label_train, data_test, label_test = load_dataset(dataset, return_torch=True, concatenate=False)
        else:
            data_train, label_train, data_test, label_test = data_info
                
        #Convert the data into 3 channels - needed to use image networks
        data_shape = list(data.shape)
        data_shape[1] = 3
        data = data.expand(data_shape)
        data_train_shape = list(data_train.shape)
        data_train_shape[1] = 3
        data_train = data_train.expand(data_train_shape)
        data_test_shape = list(data_test.shape)
        data_test_shape[1] = 3
        data_test = data_test.expand(data_test_shape)
        
        #Normalize data into range [0,1]
        ##TODO: I think this is an error. We need to do this in a consistent way. It should be based on the statistics from data. Only the second two lines are incorrect, I believe. 
        data = torch.from_numpy(NormalizeData(data.cpu().detach().numpy()))
        data_train = torch.from_numpy(NormalizeData(data_train.cpu().detach().numpy()))
        data_test = torch.from_numpy(NormalizeData(data_test.cpu().detach().numpy()))
        
        #Define the transforms for training and evaluation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.GaussianBlur(kernel_size=(1,1), sigma=(1, 1)),
                transforms.ColorJitter(contrast=2,brightness=.5, hue=.3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.GaussianBlur(kernel_size=(1,1), sigma=(1, 1)),
                transforms.ColorJitter(contrast=2,brightness=.5, hue=.3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        #Packing the training and evaluation data
        #If run out of CUDA memory in testings below, try to reduce the batch size to 32 or 16
        dataset_train = MyDataset(data_train, label_train, transform=data_transforms['train'])
        dataloader_train = DataLoader(dataset_train, batch_size=transfer_batch_size)
        
        dataset_val = MyDataset(data_test, label_test, transform=data_transforms['val'])
        dataloader_val = DataLoader(dataset_val, batch_size=transfer_batch_size)
        
        train_size = data_train.shape[0]
        val_size = data_test.shape[0]
        
        dataloaders = {'train':dataloader_train, 'val':dataloader_val}
        dataset_sizes = {'train':train_size, 'val':val_size}
        
        #The most optimized choice of CNNs
        if model_type == None:
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
                
            
        model_ft = model_ft.to(device)
        
        # Set up the optimizer to optimize all model parameters
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        
        #Train the model with cross entropy loss
        criterion = nn.CrossEntropyLoss()
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=epochs)
        
        data, _ = load_dataset(dataset, return_torch=True, shuffle_train_set=False, concatenate=True)
        #Modify the data
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.GaussianBlur(kernel_size=(1,1), sigma=(1, 1)),
            transforms.ColorJitter(contrast=2,brightness=.5, hue=.3),
            transforms.Normalize(torch.mean(data), torch.std(data))
        ])
        data = data_transform(data)
        
        data_shape = list(data.shape)
        data_shape[1] = 3
        data = data.expand(data_shape)
        if dataset == 'fusar':
            #Normalize the data to fit into the pretrained CNNs
            data = data.cpu().detach().numpy()
            data = NormalizeData(data)
          
            num_elements = data.shape[0]
            
            ##TODO: Why do we have these choices for the means?
          
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            data_tensor = torch.zeros(data.shape)
            for i in range(num_elements):
              temp = torch.from_numpy(data[i,:,:,:])
              temp = normalize(temp)
              data_tensor[i,:,:,:] = temp
            data = data_tensor
        
        data = data.to(device)
        model_ft.eval()
        
        #Set the correct layer to extract feature
        if model_type == None:
            if dataset == 'open_sar_ship':
                feature_layer = 'flatten'
            else:
                feature_layer = 'mean'
        elif model_type == 'ShuffleNet':
            feature_layer = 'mean'
        elif model_type == 'GoogLeNet':
            feature_layer = 'dropout'
        else:
            feature_layer = 'flatten'
            
        with torch.no_grad():
            feature_extractor = create_feature_extractor(model_ft, return_nodes=[feature_layer])
            encoded_data = feature_extractor(data)[feature_layer].detach().cpu().numpy()
            
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


##All code below is from MSTAR Github

def NormalizeData(data):
    '''Normalizes data to range [0,1]

    Parameters
    ----------
    data : Numpy array

    Returns
    -------
    norm_data : Normalized array
    '''

    norm_data = (data - np.min(data))/(np.max(data) - np.min(data))
    return norm_data

def load_MSTAR(root_dir = './data/MSTAR'):
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

def encodeMSTAR(model_path, batch_size = 1000, cuda = True, use_phase = True):
    '''Load a torch CNN model and encode MSTAR

    Parameters
    ----------
    model_path : Path to .pt file containing torch model for trained CNN
    batch_size : Size of minibatches to use in encoding. Reduce if you get out of memory errors (default = 1000)
    cuda : Whether to use GPU or not (default = True)
    use_phase : Whether the model uses phase information or not (default = False)

    Returns
    -------
    encoded_data : Returns a numpy array of MSTAR encoded by model.encode() (e.g., the CNN features)
    '''

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #Load data and stack mag, real phase, and imaginary phase together
    hdr, fields, mag, phase = load_MSTAR()
    if use_phase:
        data = polar_transform(mag, phase)
    else:
        data = np.reshape(mag,(mag.shape[0],1,mag.shape[1],mag.shape[2]))
    data = torch.from_numpy(data).double() #Changed from float()
    

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