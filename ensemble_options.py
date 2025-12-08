import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from data_processing_pt2 import final_x, final_y
from Model_selection import best_params
from ProteomicsModel_hyperparams import ProteomicsModel

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

plt.style.use('seaborn-v0_8-poster')

# static params
criterion = nn.MSELoss()
device = torch.accelerator.current_accelerator().type
seed = 1989
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# bring in training data
X = final_x.to_numpy()
y = final_y.to_numpy()

# move to tensor and device
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
X, y = X.to(device), y.to(device)

# set dataset
dataset = TensorDataset(X, y)

# iterate through model building with params from best_params
model_loss = []
models = []
for i, row in best_params.iterrows():
    model = ProteomicsModel(dropout_p=row["dropout_p"]).to(device)
    model.train()  

    # set hyper_params
    optimizer = optim.SGD(
        model.parameters(),
        lr=row["lr"],
        momentum=row["momentum"])
    epochs = int(row["epochs_ran"])

    dataloader = DataLoader(
        dataset,
        batch_size=row["batch_size"],
        shuffle=True)

    # train model
    epoch_loss = []
    for e in range(epochs):
        batch_losses = []
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            clip_grad_norm_(
                model.parameters(),
                max_norm=row["max_norm"])
            optimizer.step()
            batch_losses.append(loss.item())
        
        # track epoch loss
        epoch_loss.append(np.mean(batch_losses))

    # plot 
    plt.figure(figsize=(10,8))
    plt.plot(epoch_loss, label="MSE Loss")
    plt.title(f"Loss for model_{i}")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(
        f"./training_figs/final_model_loss_{i}.jpg", dpi=300)
    plt.close()

    # save
    model_loss.append(epoch_loss[-1])
    models.append(model)

# print loss
print(best_params.iloc[:,0])
print(f"Loss on Full Dataset: {np.round(model_loss, 1)}")
for i, row in best_params.iterrows():
    print(f"\nModel {i}:\n"
        f"Training Loss model:" 
         f"{np.round(best_params.loc[i, "Mean_fold_Tloss"])} "
         f"({np.round(best_params.loc[i, "SD_fold_Tloss"])})\n"
         f"Validation Loss:"
         f"{np.round(best_params.loc[i, "Mean_fold_Vloss"])} "
         f"({np.round(best_params.loc[i, "SD_fold_Vloss"])})"
)

model_idx = [436, 620, 112, 296, 180, 
             472, 504, 149, 181, 184]
model_idx = np.sort(model_idx)
model_idx = [str(num) for num in model_idx]



######
# make cv
# evaluate on test folds only for for mean and wtd mean
# replace lin regressor with ridge regression
# use cv to train + test
    # copy method from baseline regressor
# get train/test plot

# for all three
    # plot scatter of actual vs predicted
    # match scatter from baseline plot

######
# define ensemble methods
def ensemble_mean(models, X, y): 
    predictions = []
    # predictions
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(X)
        predictions.append(out)
    
    # final mean
    preds_stack = torch.stack(predictions, dim=0)
    mean_preds = preds_stack.mean(dim=0)

    print(f"predictions len: {len(predictions)}")
    print(f"preds_stack len: {len(preds_stack)}")
    print(f"mean_preds len: {len(mean_preds)}")
    print(f"preds_stack shape: {preds_stack.shape}")
    print(f"mean_preds shape: {mean_preds.shape}")

    # get loss
    loss = criterion(mean_preds, y)

    return mean_preds, loss.item()

mean_of_models, loss_1 = ensemble_mean(models, X, y)

# weighted mean
def ensemble_wt_mean(models, X, y, weights):
    predictions = []
    # predictions
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(X)
        predictions.append(out)
    
    # final mean
    preds_stack = torch.stack(predictions, dim=1)
    w = torch.tensor(weights).to(device)
    w = (w.max() + 1) - w
    w = w / w.sum()
    w = w.view(1, -1, 1) # reshape - took forever to figure this out
    mean_preds = (preds_stack * w).sum(dim=1)

    # loss
    loss = criterion(mean_preds, y)

    return mean_preds, loss.item()

weights = best_params["Mean_fold_Vloss"].to_numpy()
wtmean_of_models, loss_2 = ensemble_wt_mean(models, X, y, weights=weights)

# regressor
def ensemble_regressor(models, X, y, epochs):
    predictions = []

    # predictions
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(X)
            out = out.squeeze(-1)
        predictions.append(out)

    preds_stack = torch.stack(predictions, dim=1)
    y = y.unsqueeze(1)

    # normalize predictions
    preds_norm = (preds_stack - (preds_stack.mean()))/(preds_stack.std())

    # regressor
    regressor = nn.Linear(10, 1).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=0.01)
    losses = []

    # add cv part here and tab over section below
    for e in range(epochs):
        regressor.train()
        optimizer.zero_grad()
        out = regressor(preds_norm)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(regressor.parameters(), 
                                       max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    
    # adjust to instead show loss in train and val sets...
    plt.figure()
    plt.plot(losses)
    plt.show()

    # re-train final regressor on whole dataset
    # return final predictions
    regressor.eval()
    with torch.no_grad():
        final_preds = regressor(preds_norm)
    
    return final_preds, losses



# plot scatter or predicted vals and actuals