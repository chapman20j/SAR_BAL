# SAR\_BAL

Batch active learning for SAR data.
This code was created for the paper

J. Chapman, B. Chen, Z. Tan, J. Calder, K. Miller, A. Bertozzi. Novel Batch Active Learning Approach and Its Application to Synthetic Aperture Radar Datasets, To appear in SPIE Defense and Commercial Sensing: Algorithms for Synthetic Aperture Radar Imagery XXX, 2023.

All the results contained in that paper are within this github repo.

## How to run this code

The final experiments were run during January-February 2023 in python3.8 on google colab. Do the following steps to run the code

1. Pull this repo
2. pip install -r requirements.txt
3. Run each of the Experiment notebooks (remove the pip installs and colab commmands if not in google colab)

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

Academic Software License

Copyright (c) 2021, 20XX UCLA (“Institution”).

Academic or nonprofit researchers are permitted to use this Software (as defined below) subject to Paragraphs 1-3:

1. Institution hereby grants to you free of charge, so long as you are an
   academic or nonprofit researcher, a nonexclusive license under Institution’s
   copyright ownership interest in this software and any derivative works made
   by you thereof (collectively, the “Software”) to use, copy, and make
   derivative works of the Software solely for educational or academic research
   purposes, in all cases subject to the terms of this Academic Software License.
   Except as granted herein, all rights are reserved by Institution, including
   the right to pursue patent protection of the Software.

2. Please note you are prohibited from further transferring the Software --
   including any derivatives you make thereof -- to any person or entity.
   Failure by you to adhere to the requirements in Paragraphs 1 and 2 will
   result in immediate termination of the license granted to you pursuant to
   this Academic Software License effective as of the date you first used the
   Software.

3. IN NO EVENT SHALL INSTITUTION BE LIABLE TO ANY ENTITY OR PERSON FOR DIRECT,
   INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST
   PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF INSTITUTION HAS
   BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. INSTITUTION SPECIFICALLY
   DISCLAIMS ANY AND ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING, BUT NOT
   LIMITED TO, ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED “AS IS.” INSTITUTION HAS NO
   OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
   MODIFICATIONS OF THIS SOFTWARE.

Commercial entities: please contact bertozzi@ucla.edu for licensing opportunities.
