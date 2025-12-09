import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from data_processing_pt2 import final_x, final_y
from Model_selection import best_params
from ProteomicsModel_hyperparams import ProteomicsModel

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

    # get loss
    loss = criterion(mean_preds, y)

    return mean_preds, loss.item()

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
    w = w ** 5 # increase diffs
    w = w / w.sum()
    w = w.view(1, -1, 1) # reshape - took forever to figure this out
    mean_preds = (preds_stack * w).sum(dim=1)

    # loss
    loss = criterion(mean_preds, y)

    return mean_preds, loss.item()

# regressor
def ensemble_regressor(
        models, 
        X_train, 
        y_train, 
        X_val,
        y_val,
        epochs):

    X_train.to(device)
    y_train.to(device)
    X_val.to(device)
    y_val.to(device)

    # get initial model predictions    
    train_preds = []
    val_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            train_out, val_out = model(X_train), model(X_val)
            train_out = train_out.squeeze(-1)
            val_out = val_out.squeeze(-1)
        train_preds.append(train_out)
        val_preds.append(val_out)

    # align shapes
    train_preds_stack = torch.stack(train_preds, dim=1)
    val_preds_stack = torch.stack(val_preds, dim=1)
    y_train = y_train.unsqueeze(1)
    y_val = y_val.unsqueeze(1)

    # normalize training predictions
    norm_train_preds = (train_preds_stack - (
        train_preds_stack.mean())) / (train_preds_stack.std())
    # normalize val predictions using mean and std of training set
    # blah blah blah, data leakage, idk. The val set is too small
    # to reliably use self-contained normalization...
    norm_val_preds = (val_preds_stack - (
        train_preds_stack.mean())) / (train_preds_stack.std())

    # regressor
    regressor = nn.Linear(10, 1).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=0.01)

    # train
    regressor.train()
    train_losses = []
    val_losses = []
    for e in range(epochs):
        optimizer.zero_grad()
        reg_preds = regressor(norm_train_preds)
        reg_loss = criterion(reg_preds, y_train)
        reg_loss.backward()
        torch.nn.utils.clip_grad_norm_(regressor.parameters(), 
                                       max_norm=1.0)
        optimizer.step()
        train_losses.append(reg_loss.item())
    
        # eval on val data
        regressor.eval()
        with torch.no_grad():
            val_ensm_preds = regressor(norm_val_preds)
        val_loss = criterion(val_ensm_preds, y_val)
        val_losses.append(val_loss.item())

    # plot
    plt.figure(figsize=(10,8))
    plt.plot(train_losses, label="training loss")
    plt.plot(val_losses, label="validation loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Stacked Regressor Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig("./figs/stacked_regressor_train_plot.jpg", dpi=300)
    plt.close()

    # train and predict on full dataset
    return (reg_preds, 
            train_losses,
            val_ensm_preds,
            val_losses
    )

# set k-fold and weights
kf = KFold(n_splits=6, shuffle=True, random_state=42)
weights = best_params["Mean_fold_Vloss"].to_numpy()

# get val results for each k-fold cv
mean_ensem_preds = []
mean_ensem_losses = []
wtmean_ensem_preds = []
wtmean_ensem_losses = []
reg_ensem_tPreds = []
reg_ensem_tLosses = []
reg_ensem_vPreds = []
reg_ensem_vLosses = []

# also track actuals for each fold
mean_ensem_y = []
wtmean_ensem_y = []
reg_ensem_y = []

# loop
regressor_epochs = 10000
for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    # mean and wt mean
    fold_mean_preds, fold_mean_loss = ensemble_mean(
        models, 
        X_val, 
        y_val
    )
    fold_wtmean_preds, fold_wtmean_losses = ensemble_wt_mean(
        models,
        X_val,
        y_val,
        weights=weights
    )
    # train regressor in training, test in val
    tPred, tLoss, vPred, vLoss = ensemble_regressor(
        models, 
        X_train, 
        y_train, 
        X_val,
        y_val,
        epochs=regressor_epochs
    )
    # track losses
    # track losses as before
    mean_ensem_losses.append(fold_mean_loss)
    wtmean_ensem_losses.append(fold_wtmean_losses)
    reg_ensem_tLosses.append(tLoss[-1])
    reg_ensem_vLosses.append(vLoss[-1])

    # track predictions, move to cpu so i can plot later
    mean_ensem_preds.append(fold_mean_preds.detach().cpu())
    wtmean_ensem_preds.append(fold_wtmean_preds.detach().cpu())
    reg_ensem_vPreds.append(vPred.detach().cpu())
    # also track validation fold actuals
    mean_ensem_y.append(y_val.detach().cpu())
    wtmean_ensem_y.append(y_val.detach().cpu())
    reg_ensem_y.append(y_val.detach().cpu())

# print mean sd for cv results
print(
    f"Mean Ensemble: {np.mean(mean_ensem_losses):.1f} "
    f"({np.std(mean_ensem_losses):.2f})\n"
    f"Wt Mean Ensemble: {np.mean(wtmean_ensem_losses):.1f} "
    f"({np.std(wtmean_ensem_losses):.2f})\n"
    f"Train Regressor Ensemble: {np.mean(reg_ensem_tLosses):.1f} "
    f"({np.std(reg_ensem_tLosses):.2f})\n"
    f"Val Regressor Ensemble: {np.mean(reg_ensem_vLosses):.1f} "
    f"({np.std(reg_ensem_vLosses):.2f})\n"
)

# combine folds
mean_preds_all = torch.cat(mean_ensem_preds).numpy().ravel()
wtmean_preds_all = torch.cat(wtmean_ensem_preds).numpy().ravel()
reg_preds_all = torch.cat(reg_ensem_vPreds).numpy().ravel()

y_mean_all = torch.cat(mean_ensem_y).numpy().ravel()
y_wt_all   = torch.cat(wtmean_ensem_y).numpy().ravel()
y_reg_all  = torch.cat(reg_ensem_y).numpy().ravel()

# Mean ensemble
sns.jointplot(x=mean_preds_all, y=y_mean_all)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Mean Ensemble")
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.tight_layout()
plt.savefig("./figs/mean_ensem_preds.jpg", dpi=300)
plt.close()

sns.jointplot(x=mean_preds_all, y=y_mean_all, kind="kde", fill=True)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Mean Ensemble")
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.tight_layout()
plt.savefig("./figs/mean_ensem_preds_kde.jpg", dpi=300)
plt.close()

# Weighted mean ensemble
sns.jointplot(x=wtmean_preds_all, y=y_wt_all)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Weighted Mean Ensemble")
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.tight_layout()
plt.savefig("./figs/wtmean_ensem_preds.jpg", dpi=300)
plt.close()

sns.jointplot(x=wtmean_preds_all, y=y_wt_all, kind="kde", fill=True)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Weighted Mean Ensemble")
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.tight_layout()
plt.savefig("./figs/wtmean_ensem_preds_kde.jpg", dpi=300)
plt.close()

# Stacked regressor ensemble (per-fold val preds)
sns.jointplot(x=reg_preds_all, y=y_reg_all)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Stacked Regressor")
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.tight_layout()
plt.savefig("./figs/stacked_reg_preds.jpg", dpi=300)
plt.close()

sns.jointplot(x=reg_preds_all, y=y_reg_all, kind="kde", fill=True)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Stacked Regressor")
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.tight_layout()
plt.savefig("./figs/stacked_reg_preds_kde.jpg", dpi=300)
plt.close()
