# SAR\_BAL

Batch active learning for SAR data.
This code was created for the paper **cite**. All the results contained in that paper are within this github repo.

## How to run this code

## Dependencies

For more information, refer to the requirements.txt file.

- graphlearning
- annoy
- torch
- torchvision
- numpy
- pandas
- matplotlib
- scipy

## File Guide

1. Python notebooks beginning with "Example" contain simple examples of code execution. Some of these files produce example figures used in the paper.
2. Python notebooks beginning with "Experiment" contain the main experiments for the paper.
3. The python files contain useful functions for running experiments.
4. Example
    - Example Batch Plots
    - Example Dijkstras Annulus Coreset
    - Example SAR Image Data
5. Experiment
    1. Experiment 1 - Various Active Learning Methods: Contains the experiments for changing the active learning method on each dataset. It includes final accuracy, time used, and detailed accuracy plots.
    2. Experiment 2 - LocalMax Acquisition Functions: Contains the detailed accuracy experiments with different datasets, embeddings, and underlying acquisition functions.
    3. Experiment 3 - Variance: Contains experiments to understand the variance due to data augmentation and fine tuning on each dataset.
    4. Experiment 4 - Architecture Tests: Contains experiments to understand the variance due to the choice of neural network architecture.
6. Python Files
    - utils: Contains constants, code for loading datasets, and code for embedding them.
    - batch\_active\_learning: Contains constants, code for DAC, and code for active learning
    - experiments:
    - models:
7. Folders
    - BAL Examples
    - DAC Plots
    - data
    - Experiment Results
    - Images
    - knn\_data
    - models
8. Other
    - Plots with Percentages: Plots the data from experiments 1 and 2 nicely
    - CNNVAE knn data: Computes the knn data for each trained cnnvae used

## License
