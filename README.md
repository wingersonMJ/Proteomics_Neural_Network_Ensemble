# Neural Network Proteomic Prediction Model!

**Summary:** This project provides a jumping-off point for a model that uses protemoics to predict curve severity in patients with scoliosis.
The objective is to use >7500 proteins to estimate a patient's Max Cobb angle (range 0-45). I trained multiple neural networks to predict the
target variable, then used ensemble learning to train a linear regression on network outputs. 

## Table of Contents:
1. [Python files](#python-files)  
2. [Predictors and Targets](#predictors-and-targets)  
3. [Modeling approach](#modeling-approach)  
    - [Data processing](#data-processing)  
    - [Baseline model](#baseline-model)  
    - [Initial model building](#initial-model-building)  
    - [Hyperparameter search](#hyperparameter-search)
    - [Model selection](#model-selection)
    - [Ensemble approaches](#ensemble-approaches)\
4. [Final summary and next steps](#final-summary-and-next-steps)

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

The dataset is >7500 proteins from patients with scoliosis. The dataset was quantile normalized and covariates of age and ethnicity were regressed out of the predictors ahead of time (i.e., before I began with additional data processing).  

**Figure 1.** Correlation matrix represented as a heat map for the predictor variables.  
<img src="./figs/corr_map.jpg" width=500>
<br>

The target variable is max_curve - the maximum Cobb angle of the spine in this set of patients with 
scoliosis. It is a continuous/regression problem.  

**Figure 2.** Kernel density plots showing the distribution of the target variable: max curve angle.  
<img src="./figs/target_distribution.jpg" width=500>
<br>

## Modeling approach

The loss function throughout is Mean Squared Error (MSE). 

### Data processing

The data I recieved were original pre-separated into a training and validation 
splits. The training sample was 66 subjects and the validation set was 10 
subjects. Given the small sample size, I elected to combine the pre-separated 
datasets and use cross-validation to iterate through train/val splits, 
rather than maintain one single train/val split in such a small sample of
patients.  

I droped 4 subjects with a high degree of missing data. This left us with 
**n=72** subjects to use in training and validation.  

KNN was used to impute missing values for the proteomic features. Of the 7,568 
features, only 308 had any missing data. The missing data was contained only
to the set of 10 subjects who were originally included in the validation set. 
KNN with k=5 was used to impute the missing data.  

### Baseline model

I would like to have used a true hold-out test sample to evaluate performance. However, 
with the sample currently at 72 subjects, all I could afford to do is create train 
and validation splits through k-fold cross-validation. The value for K was set at k=6, 
and that value is used continuously throughout the project, including when fitting and 
evaluating the neural networks later on.  

A linear regression with L2 (Ridge) normalization was fit. Even with regularization, the 
ridge regression overfit to the data significantly. Extremely high alpha values - 
the strength of the regularization - were needed to get the training and validation loss 
to approach eachother.  

**Figure 3.** Training and validation loss during 6-fold cross-validation for a Ridge 
Regression predicting our target variable.  
<img src="./figs/ridge_alphas.jpg" width=300>
<br>

The ridge regression model was re-fit on the entire dataset using an alpha 
of 500, with the understanding that the model may not be performing very well but is 
really just serving as a baseline jumping-off point.  

**Figure 4.** Actual and predicted values for the baseline ridge regression model. The 
grey dashed line represents a perfect predictor.  
<img src="./figs/baseline_regressor.jpg" width=200>
<img src="./figs//baseline_regressor_kde.jpg" width=200>
<br>

**Baseline Performance:** In 6-fold cross-validation, the baseline ridge regression model 
overfit to the training data and failed to generalize to validation sets, as evidenced by
 a higher mean validation loss across CV folds and a very high standard deviation of loss
across CV folds.
|        | Mean Loss | Stdev Loss |
| ------ | --------- | ---------- | 
| Train  | 45.9      | 7.71       |
| Validation | 167.1 | 86.79      |
<br>

### Initial model building

I am forcing a neural network (multi-layer perceptron) for this task. In the 
'next steps' section below, I discuss finding a model that fits the data 
rather than forcing the data to fit a model. But, for this example, I am learning 
more about PyTorch, so NN is what I will use regardless of fit.  

The ProteomicModel.py file contains some initial model building. I am evaluating 
model depth and width, activation functions, dropout, gradient clipping, lr's, 
optimizers, etc.  

What I found is that the small sample size still works well with a shallow network. 
When I include just 4 layers that work down from 7568 > 2500 > 1000 > 100 > 1, then 
the model tends to train just fine. This is the set-up I will use when searching 
for appropriate hyperparameters.  

When training and evaluating MSE on the entire dataset, this initial model-building step
 had a loss of 295.6! Performance might not be incredible, but at least it is not 
 immediately overfitting to the data...  

**Figure 5.** MSE loss over epochs for this initial neural network.  
<img src="./figs/loss_initial_model.jpg" width=250>
<img src="./figs/initial_NN.jpg" width=187.5>
<img src="./figs/initial_NN_kde.jpg" width=187.5>
<br>

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

## Final summary and next steps
Baseline regressor on limitted predictors was best performing model
NN did not have enough training data to be effective 
Future tasks:
- Explore other modeling options
    - beyond NN
    - Find model to fit data, instead of fitting data to model
- Better data engineering 
    - reduce features
    - combine features
    - explore feature covariance/correlation
- explore a true hold-out set
