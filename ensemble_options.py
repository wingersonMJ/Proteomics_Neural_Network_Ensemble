
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from data_processing_pt2 import final_x, final_y
from Model_selection import best_params
from ProteomicsModel_hyperparams import ProteomicsModel

# static params
criterion = nn.MSELoss()
device = torch.accelerator.current_accelerator().type
seed = 42
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

    # instantize model
    model = ProteomicsModel(
        dropout_p=row["dropout_p"]).to(device)
    model.train()  

    # set hyper_params
    optimizer = optim.SGD(
        model.parameters(),
        lr=row["lr"],
        momentum=row["momentum"]
    )
    epochs = int(row["epochs_ran"])

    # build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=row["batch_size"],
        shuffle=True
    )

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
                max_norm=row["max_norm"]
            )
            optimizer.step()

            # track batch loss
            batch_losses.append(loss.item())
        
        # track epoch loss
        epoch_loss.append(np.mean(batch_losses))

    # plot 
    plt.figure()
    plt.plot(epoch_loss, label="MSE Loss")
    plt.title(f"Loss for model_{i}")
    plt.legend()
    plt.savefig(f"./training_figs/final_model_loss_{i}.jpg", dpi=300)
    plt.close()

    with torch.no_grad():
        pred_y = model(X)
        pred_y = pred_y
    plt.figure()
    plt.scatter(x=pred_y, y=y)
    plt.xlabel("Predicted value")
    plt.ylabel("Actual value")
    plt.show()

    # save
    model_loss.append(epoch_loss[-1])
    models.append(model)

# print
print(model_loss)

# make a bar plot
plt.figure()
plt.bar(model_loss)
plt.show()

######
# define ensemble methods
def ensemble_mean(models, X, y): 

    predictions = []

    # mean
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

    # mean
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(X)
        predictions.append(out)
    
    # final mean
    preds_stack = torch.stack(predictions, dim=0)
    w = torch.tensor(weights, dtype=preds_stack.dtype, device=preds_stack.device)
    w = w.max() - w
    w = w / w.sum()
    mean_preds = (w * preds_stack).sum(dim=0)

    # get loss
    loss = criterion(mean_preds, y)

    return mean_preds, loss.item()


def ensemble_regressor(models, X, y, epochs):

    predictions = []

    # mean
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(X)
        predictions.append(out)

    preds_stack = torch.stack(predictions, dim=0)

    # init
    w = 
    b = 

    # regress
    for e in range(epochs):
        pred = w@preds_stack + b
        loss = criterion(pred, y)
        w += 


mean_of_models, loss_1 = ensemble_mean(models, X, y)
print(loss_1)

wtmean_of_models, loss_2 = ensemble_wt_mean(models, X, y, weights=best_params["Mean_fold_Vloss"])
