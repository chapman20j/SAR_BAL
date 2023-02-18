# SAR\_BAL
Batch active learning for SAR data
This code was created for the paper **cite**. All the results contained in that paper are within this github repo. 

## Dependencies
- graphlearning
- annoy
- torch
- torchvision
- numpy


## File Guide
1. Python notebooks beginning with "Example" contain simple examples of code execution. Some of these files produce example figures used in the paper. 
2. Python notebooks beginning with "Experiment" contain the main experiments for the paper. 
3. The python files contain useful functions for running experiments. 

1. Example
    - Example Batch Plots: 
    - Example Dijkstras Annulus Coreset
    - Example SAR Image Data
2. Experiment
    1. Experiment 1 - Various Active Learning Methods: Contains the experiments for changing the active learning method on each dataset. It includes final accuracy, time used, and detailed accuracy plots. 
    2. Experiment 2 - LocalMax Acquisition Functions: Contains the detailed accuracy experiments with different datasets, embeddings, and underlying acquisition functions. 
    3. Experiment 3 - Variance: Contains experiments to understand the variance due to data augmentation and fine tuning on each dataset. 
    4. Experiment 4 - Architecture Tests: Contains experiments to understand the variance due to the choice of neural network architecture. 
3. Python Files
    - utils: Contains constants, code for loading datasets, and code for embedding them. 
    - batch\_active\_learning: Contains constants, code for DAC, and code for active learning
    - experiments: 
    - models: 
4. Folders
    - BAL Examples
    - DAC Plots
    - data
    - Experiment Results
    - Images
    - knn\_data
    - models
5. Other
    - Plots with Percentages: Plots the data from experiments 1 and 2 nicely
    - CNNVAE knn data: Computes the knn data for each trained cnnvae used

## License
MIT License

Copyright (c) 2023 chapman20j

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


