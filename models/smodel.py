# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:11:34 2022

@author: jasbr
"""


import torch.nn as nn
import torch.nn.functional as F
import torch

#CNN Variational Autoencoder - SarShip 128*128
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
 
        kernel_size = 3 # (3, 3) kernel
        num_classes = 10
        init_channels = 8 # initial number of filters
        image_channels = 1
        latent_dim = 32 # latent dimension for sampling

        # encoder
        self.enc1 = nn.Conv2d(image_channels, init_channels, kernel_size, padding=1)
        self.enc2 = nn.Conv2d(init_channels, 2*init_channels, kernel_size, padding=1)
        self.enc3 = nn.Conv2d(2*init_channels, 4*init_channels, kernel_size, padding=1)
        self.enc4 = nn.Conv2d(4*init_channels, 64, kernel_size, padding=1)

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64*16*16, 128)     #Change to 16*16 since that is the dimensions. Note input is flattened
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)

        # decoder 
        self.dec1 = nn.ConvTranspose2d(64, 8*init_channels, kernel_size, output_padding=1, dilation=3)
        self.dec2 = nn.ConvTranspose2d(8*init_channels, 4*init_channels, kernel_size, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(4*init_channels, 2*init_channels, kernel_size, stride=2, padding=1, output_padding=1) #doubling conv transpose
        self.dec4 = nn.ConvTranspose2d(2*init_channels, image_channels, kernel_size, stride=2, padding=1, output_padding=1)
        self.dec5 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size, stride=2, padding=1, output_padding=1)

        #Dropout
        self.dropout1 = nn.Dropout(0.5)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def encode(self, x):
        x = F.relu(self.enc1(x)) #76  128
        x = F.max_pool2d(x, 2) #38    64
        x = F.leaky_relu(self.enc2(x)) #38  64
        x = F.max_pool2d(x, 2) #19    32
        x = F.leaky_relu(self.enc3(x)) #19  32
        x = F.leaky_relu(self.enc4(x)) #10  32
        x = F.max_pool2d(x, 2) #      16
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

    def decode(self, x):
        x = x.view(-1, 64, 1, 1)
        x = F.leaky_relu(self.dec1(x)) #8
        x = F.leaky_relu(self.dec2(x)) #16
        x = self.dropout1(x)
        x = F.leaky_relu(self.dec3(x))  #32
        x = F.leaky_relu(self.dec4(x))  #64
        x = torch.sigmoid(self.dec5(x))  #128

        return x

    def forward(self, x): #76

        #encode
        x = self.encode(x)
        hidden = self.fc1(x) #(64*10*10, 128)
        
        
        
        
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var) #(64, 16)
        z = self.fc2(z) #(64, 64)

        #decode
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var
        
        
class CVAE2(nn.Module):
    def __init__(self):
        super(CVAE2, self).__init__()
 
        kernel_size = 3 # (3, 3) kernel
        num_classes = 10
        init_channels = 8 # initial number of filters
        image_channels = 1
        latent_dim = 32 # latent dimension for sampling

        # encoder
        self.enc1 = nn.Conv2d(image_channels, init_channels, kernel_size, padding=1)
        self.enc2 = nn.Conv2d(init_channels, 2*init_channels, kernel_size, padding=1)
        #self.enc3 = nn.Conv2d(2*init_channels, 4*init_channels, kernel_size, padding=1)
        #self.enc4 = nn.Conv2d(4*init_channels, 64, kernel_size, padding=1)

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(16*32*32, 128)     #Change to 16*16 since that is the dimensions. Note input is flattened
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)

        # decoder 
        self.dec1 = nn.ConvTranspose2d(64, 2*init_channels, kernel_size, output_padding=1, dilation=3)
        self.dec2 = nn.ConvTranspose2d(2*init_channels, 1*init_channels, kernel_size=5, stride=4, padding=1, output_padding=1) #quadruple conv transpose
        self.dec3 = nn.ConvTranspose2d(1*init_channels, image_channels, kernel_size=5, stride=4, padding=1, output_padding=1) #doubling conv transpose
        #self.dec4 = nn.ConvTranspose2d(2*init_channels, image_channels, kernel_size, stride=2, padding=1, output_padding=1)
        #self.dec5 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size, stride=2, padding=1, output_padding=1)

        #Dropout
        self.dropout1 = nn.Dropout(0.5)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def encode(self, x):
        x = F.relu(self.enc1(x)) #76  128
        x = F.max_pool2d(x, 2) #38    64
        x = F.relu(self.enc2(x)) #38  64
        x = F.max_pool2d(x, 2) #19    32
        #x = F.leaky_relu(self.enc3(x)) #19  32
        #x = F.leaky_relu(self.enc4(x)) #10  32
        #x = F.max_pool2d(x, 2) #      16
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

    def decode(self, x):
        x = x.view(-1, 64, 1, 1)
        x = F.relu(self.dec1(x)) #8
        x = F.relu(self.dec2(x)) #16
        x = self.dropout1(x)
        x = torch.sigmoid(self.dec3(x))  #32
        #x = F.relu(self.dec4(x))  #64
        #x = torch.sigmoid(self.dec5(x))  #128

        return x

    def forward(self, x): #76

        #encode
        x = self.encode(x)
        hidden = self.fc1(x) #(64*10*10, 128)
        
        
        
        
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var) #(64, 16)
        z = self.fc2(z) #(64, 64)

        #decode
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        w = (32,64) #Number of channels in 1st and 2nd layers
        self.conv1 = nn.Conv2d(1, w[0], 3, 1)       #change first parameter to be a 1 since only one image channel for grayscale
        self.conv2 = nn.Conv2d(w[0], w[1], 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        f = 512  #Number of hidden nodes in fully connected layers
        #self.fc1 = nn.Linear(w[1]*10*10, f)
        self.fc1 = nn.Linear(64*15*15, f)
        self.fc2 = nn.Linear(f, 10)
        self.bn1 = nn.BatchNorm1d(f)

    def forward(self, x): #88  #128
        x = self.conv1(x) #86  #126
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #43  #63  #124
        x = self.conv2(x) #41  #66  @122
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #10  #15  @120
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)   #batch normalization
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    #This is useful for extracting features from convolutional part of NN
    def encode(self, x):
        x = self.conv1(x) #86
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #43
        x = self.conv2(x) #41
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #10
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x