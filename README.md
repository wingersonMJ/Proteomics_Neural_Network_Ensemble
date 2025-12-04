# Neural Network Proteomic Prediction Model!

**Summary:** This project provides a jumping-off point for a model that uses protemoics to predict curve severity in patients with scoliosis.
The objective is to use >7500 protiens to estimate a patient's Max Cobb angle (range 0-45). I trained multiple neural networks to predict the
target variable, then used ensemble learning to train a linear regression on network outputs. 

## Table of Contents:



## Python files 
1. data_processing.py 
    - Simple data processing. The original dataset was split into training and validation sets. However, I recombine them
    together and will split train/val during cross-validation. 
    - This means we do not have a true out-of-bag sample for validation. But, given the small sample size, we would need more subjects
    before we begin to validate anyway. 
2. data_processing_pt2.py
    - drop a few protiens with high missingness
    - encode sex as categorical 
    - use KNN to impute missing values
3. baseline_regression.py
    - A baseline simple linear regression with some regularization. This serves as an initial baseline model for comparison. 
4. ProteomicsModel.py
    - Initial model building, messing around with network depth and width, parameters, etc. 
    - Checking gradients and such to see if clipping is necessary.
5. ProteomicsModel_hyperparams.py
    - A hard-coded version of grid search for hyperparameter combinations. 
    - ~970 combinations in total, with cross-validation and train vs validation plots for each parameter combo. 
6. Model_selection.py
    - Visualizing and selecting the hyperparameters with the lowest training and validation loss. 
    - Identify the top 10 models for use in ensemble learning.
7. ensemble_options.py
    - Option 1: Take the mean of the 10 predictions. 
    - Option 2: Take the weighted mean of the 10 predictions. Weighting determined by model validation loss.
    - Option 3: Train a regression on the 10 predictions. 

## Predictors and Targets

#### Predictors: 
describe predictors

#### Target:
describe target
get kde curve

## Modeling approach

### Data processing
describe data processing

### Baseline model
describe regression
show regression prediction figure

### Initial model building
forcing nn to work, because I'm learning lol

### Hyperparameter search
show some figs from search

### Model selection
describe selection process
show figs of model performances (scatters)

### Ensemble approaches
describe ensemble approaches
performance of all models individually
performance of each ensembler 
show scatter of final regressor on top of predictions
add cv approach to regressor too!

### Final summary and next steps
Baseline regressor on limitted predictors was best performing model
NN did not have enough training data to be effective 
Future tasks:
- Explore other modeling options
    - beyond NN
- Better data engineering 
    - reduce features
    - combine features
    - explore feature covariance/correlation
- explore a true hold-out set
