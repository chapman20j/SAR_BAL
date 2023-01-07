# SAR\_BAL
Batch active learning for SAR data
This code was created for the paper **cite**. All the results contained in that paper are within this github repo. 

## Dependencies
graphlearning
annoy

## License



## File Guide
1. Python notebooks beginning with "Example" contain simple examples of code execution. Some of these files produce example figures used in the paper. 
2. Python notebooks beginning with "Experiment" contain the main experiments for the paper. 
3. The python files contain useful functions for running experiments. 

1. Example
    a. Example Batch Plots: 
    b. Example Dijkstras Annulus Coreset
    c. Example SAR Image Data
2. Experiment
    a. Experiment 1 - Various Active Learning Methods: Contains the experiments for changing the active learning method on each dataset. It includes final accuracy, time used, and detailed accuracy plots. 
    b. Experiment 2 - LocalMax Acquisition Functions: Contains the detailed accuracy experiments with different datasets, embeddings, and underlying acquisition functions. 
    c. Experiment 3 - Variance: Contains experiments to understand the variance due to data augmentation and fine tuning on each dataset. 
    d. Experiment 4 - Architecture Tests: Contains experiments to understand the variance due to the choice of neural network architecture. 
3. Python Files
    a. utils: Contains constants, code for loading datasets, and code for embedding them. 
    b. batch\_active\_learning: Contains constants, code for DAC, and code for active learning
    d. experiments: 
    c. models: 
4. Folders
    a. BAL Examples
    b. DAC Plots
    c. data
    d. Experiment Results
    e. Images
    f. knn\_data
    g. models
5. Other
    a. Plots with Percentages: Plots the data from experiments 1 and 2 nicely
    b. CNNVAE knn data: Computes the knn data for each trained cnnvae used

## Include others


