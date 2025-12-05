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

When training and evaluating MSE on the entire dataset, this initial model-building 
step had a loss of 295.6! Performance might not be incredible, but at least it is not 
 immediately overfitting to the data...  

**Figure 5.** MSE loss over epochs for this initial neural network.  
<img src="./figs/loss_initial_model.jpg" width=250>
<img src="./figs/initial_NN.jpg" width=187.5>
<img src="./figs/initial_NN_kde.jpg" width=187.5>
<br>

### Hyperparameter search

I use grid searching to find hyperparameters. In the future, I would probably 
partition the search by first identifying a good optimizer and learning rate, then 
adding additional components of dropout, momentum, etc. later on. But for this 
project I searched an entire grid of values for: batch size, learning rate, momentum 
optmizer, max_norm in gradient clipping, and dropout.  

**Cross-Validation:**  
I used 6-fold cross-validation, thus training and evaluating 6 models for each 
combination of hyperparameter values. I then averaged the training and validation 
loss across all folds of the data.  

**Flexible Epochs:**  
The model was set to train for 300 epochs. However, because I explored different 
batch sizes - batch sizes of 6, 20, and 60 (full batch) were attempted - the 
actual number of weight updates would differ between models with non-matching 
batch sizes, even if both were trained for the same number of epochs. To allow 
models with larger batch sizes the chance to exahustively train, I used a 
flexible epoch training value. If batch size for a combo was 60, then the 
maximum number of epoch ran was increased by 10. If the batch size was 20, epochs 
were increased by ~3.33. A batch size of 6 had no increase in epochs. This aligns 
the epochs for different batch sizes, allowing for the same number of weight updates 
regardless of batch size.  

**Early stopping:**  
I also included an early stopping criteria in the model training. If validation 
loss did not increase by a minimum ammount (delta) over a number of consecutive 
training epochs (patience), then the training loop was terminated. The pre-set delta was 5 points, 
so the model had to improve validation loss by 5 points to reset the 
patience counter. If the patience counter was not reset after 15 epochs (i.e., 
after 15 epochs, the model had not improved validation loss by 5 points or more), 
then the training loop was terminated. The patience value of 15 epochs was also 
adjusted based on batch size, using the same process as outlined above.  

Training and validation loss plots over epochs were maintained for each 
hyperparameter combination. Rather than save a plot for every model in CV (6 models 
per hyperparam combo), I just saved the model loss plots for the final CV fold.  

**Figure 6.** Training and validation loss plots for two models with different 
hyperparameter combinations. THe first is a model trained for the full 300 epochs, 
but did not reach convergence. The second model had more aggressive gradient 
clipping and met early stopping criteria after ~75 epochs of training.  
<img src="./grid_search_figs/loss_combo0.png" width=300>
<img src="./grid_search_figs/loss_combo4.png" width=300>

### Model selection

I used mean training and validation MSE Loss across CV folds 
to determine which models were performing the best. The top 
10 performing models would be included in later ensemble 
learning tasks. The decision to keep 10 models was arbitrary. 
A next step in this project might be evaluating different 
combinations of models to determine which are best in ensemble 
learning tasks.  

**Figure 7.** The mean training and validation losses for each 
model across the 6-folds in cross-validation. Most models have 
training and validation losses below 1,000. A few models failed 
to identify patterns in the data and thus have high loss. In the 
next plot, I filter out those poor performing models.  
<img src="./figs/search_results_all.jpg" width=300>
<br>

**Figure 8.** The same mean training and validation losses for 
models, but I filtered the list to only include those with a 
validation loss less than 1,000. We are again seeing a pattern 
where a majority of models are overfitting to the training data 
and failing to generalize to the validation sets (cluster 
of points in the upper left of the plot). But a handful of models 
are performing well in both training and validation sets 
(smaller cluster of models in the bottom center of the plot). 
Those models might be good candidates for ensemble learning.    
<img src="./figs/search_results_filtered.jpg" width=400>
<br>

**Figure 9.** If we filter down even further to just models with 
a training *and* validation loss of below 300, then we can see 
a handful of models that are performing well in both sets of 
data.  
<img src="./figs/search_results_possible.jpg" width=400>
<br>

I selected the best 10 models to use in my ensemble learning. In 
the future, I might explore different model types to include in 
the ensemble, but for this initial start I just chose the best 
10 models.  

To identify the best 10 models, I evaluated the mean training 
and validation losses across CV folds. I also considered the 
standard deviation of the training and validation losses across 
folds. An ideal model would have low but comparable losses in 
training and validation (i.e., not grossly overfitting to the 
training data), and would have a low standard deviation for 
the loss in the validation set (i.e., relatively stable 
performance across data validation splits).  

**Figure 10.** Training and loss performance for each model 
being considered for ensemble learning. The mean training and 
validation losses are plotted on the X and Y axes, respectively. 
The size of the dot represents the SD for training loss, and the 
color of the dot represents the SD for validation loss. I am 
looking for dots that are smaller (lower training SD), darker 
(lower validation SD), and closer to the bottom left corner of 
the plot.  
<img src="./figs/search_results_mean_and_sd.jpg" width=600>
<br>

Based on these criteria, I selected 10 models to use in ensemble 
learning. Below are training and loss plots for all 10 models selected:    
<img src="./grid_search_figs/loss_combo2.png" width=100>
<img src="./grid_search_figs/loss_combo8.png" width=100>
<img src="./grid_search_figs/loss_combo_momentumfix0.png" width=100>
<img src="./grid_search_figs/loss_combo_momentumfix20.png" width=100>
<img src="./grid_search_figs/loss_combo_momentumfix1.png" width=100>
<img src="./grid_search_figs/loss_combo_momentumfix5.png" width=100>
<img src="./grid_search_figs/loss_combo26.png" width=100>
<img src="./grid_search_figs/loss_combo32.png" width=100>
<img src="./grid_search_figs/loss_combo_momentumfix6.png" width=100>
<img src="./grid_search_figs/loss_combo_momentumfix11.png" width=100>
<br>

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
