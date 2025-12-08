import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import random
import itertools
import time
import matplotlib.pyplot as plt

from data_processing_pt2 import final_x, final_y

plt.style.use("seaborn-v0_8-poster")

# set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# convert to numpy
X = final_x.to_numpy()
y = final_y.to_numpy()
print("Data Loaded")

##########
# Define the model!
class ProteomicsModel(nn.Module):

    def __init__(self, dropout_p):
        super().__init__()
        self.network = nn.Sequential(
            torch.nn.Linear(7568, 2500),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2500, 1000),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(1000, 100),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.network(x)
########

# define static params
epochs = 300
criterion = nn.MSELoss()
patience = 15
delta = 5
kf = KFold(n_splits=6, shuffle=True, random_state=seed)
device = torch.accelerator.current_accelerator().type

# define hyperparams
# commenting out this whole section
"""
hyperparam_combos = {
    "batch_size": [6, 20, 60],
    "lr": [1e-2, 1e-3, 1e-4],
    "momentum": [0.1, 0.3, 0.6],
    "optimizer": ['sgd', 'ADAM', 'RMSprop'], # lr*0.1 for adam, rmsp
    "max_norm": [0.5, 1.0, 5.0],
    "dropout_p": [0.1, 0.3, 0.5, 0.7]
}

# build hyperparam combos
keys = list(hyperparam_combos.keys())
hyperparam_grid = []
for values in itertools.product(*hyperparam_combos.values()):
    hyperparam_grid.append(dict(zip(keys, values)))
print(hyperparam_grid[0])
print(f"Combos to run: {len(hyperparam_grid)}")

# define search results
search_results = {
    "Mean_fold_Tloss": 0,
    "SD_fold_Tloss": 0,
    "Mean_fold_Vloss": 0,
    "SD_fold_Vloss": 0,
    "batch_size": '',
    "momentum": '',
    "optimizer": '',
    "max_norm": '',
    "epochs_ran": '',
    "dropout_p": '',
    "lr": ''
}

########
# loop!
search_results_list = []
start = time.time()
########
for c, combo in enumerate(hyperparam_grid):
    print(f"\nRunning Combo: {c}")

    fold_train_loss = []
    fold_val_loss = []
    fold_epochs_ran = []

    # iterate through each cv fold
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Running Fold: {fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        # move to GPU
        X_train, X_val = X_train.to(device), X_val.to(device)
        y_train, y_val = y_train.to(device), y_val.to(device)

        # train loader w/ batch size
        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=combo['batch_size'],
            shuffle=True
        )

        # init the model and set optimizer
        model = ProteomicsModel(dropout_p=combo['dropout_p']).to(device)
        opt_name = combo['optimizer']

        if opt_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                lr=combo['lr'], 
                momentum=combo['momentum']
            )
        elif opt_name == 'ADAM':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=(combo['lr']*0.1)
            )
        elif opt_name == 'RMSprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=(combo['lr']*0.1),
                momentum=combo['momentum'],
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        # prepare to track metrics over epochs
        fold_train_losses = []
        fold_val_losses = []
        best_val = float("inf")
        epochs_no_improve = 0

        # align epochs bc of differing batch sizes
        fold_epochs = epochs
        fold_patience = patience
        if combo['batch_size'] == 20:
            fold_epochs = int(round(epochs * 3.3333))
            fold_patience = patience * 3
        elif combo['batch_size'] == 60:
            fold_epochs = int(epochs * 10)
            fold_patience = patience * 10

        # iterate over epochs
        for e in range(fold_epochs):
            model.train()
            batch_losses = []

            # iterate over batches
            for inputs, targets in train_dataloader:

                # zero grads, run model, calc loss, backprop, clip, step
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(-1), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=combo['max_norm']
                )
                optimizer.step()

                # track batch loss
                batch_losses.append(loss.item())

            # track epoch loss as avg of each batch loss
            epoch_loss = np.mean(batch_losses)
            fold_train_losses.append(epoch_loss)

            # get temporary val loss for early stopping
            model.eval()
            with torch.no_grad():
                temp_val_pred = model(X_val)
                temp_val_loss = criterion(temp_val_pred.squeeze(-1), y_val)
                temp_val_loss = temp_val_loss.item()
            
            fold_val_losses.append(temp_val_loss)

            # see if val performance is improving
            if temp_val_loss < best_val - delta:
                best_val = temp_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # if not improving for p epochs, then stop training
            if epochs_no_improve >= fold_patience:
                print(f"Early Stop - Epochs Ran: {e+1}")
                fold_epochs_ran.append(e+1)
                break

            if e == (fold_epochs - 1):
                print(f"No early stop - Epochs Ran: {fold_epochs}")
                fold_epochs_ran.append(fold_epochs)
        
        # get final validation loss on fold after training stops
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred.squeeze(-1), y_val).item()

        # track final train/val loss for the fold
        fold_train_loss.append(fold_train_losses[-1])
        fold_val_loss.append(val_loss)

        # plot every once in a while 
        plt.figure(figsize=(8,6))
        plt.plot(fold_train_losses, label="Train loss")
        plt.plot(fold_val_losses, label="Val loss")
        plt.title(f"{combo}", fontsize=10)
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./grid_search_figs/loss_combo{c}.png")
        plt.close()

    # mean and sd training loss across folds 
    Mean_fold_Tloss = np.mean(fold_train_loss)
    SD_fold_Tloss = np.std(fold_train_loss)
    search_results['Mean_fold_Tloss'] = float(Mean_fold_Tloss)
    search_results['SD_fold_Tloss'] = float(SD_fold_Tloss)

    # mean and sd val loss across folds
    Mean_fold_Vloss = np.mean(fold_val_loss)
    SD_fold_Vloss = np.std(fold_val_loss)
    search_results['Mean_fold_Vloss'] = float(Mean_fold_Vloss)
    search_results['SD_fold_Vloss'] = float(SD_fold_Vloss)

    # add hyperparams
    search_results['batch_size'] = combo['batch_size']
    search_results['momentum'] = combo['momentum']
    search_results['optimizer'] = combo['optimizer']
    search_results['max_norm'] = combo['max_norm']
    search_results['dropout_p'] = combo['dropout_p']
    search_results['epochs_ran'] = float(np.mean(fold_epochs_ran))
    search_results['lr'] = combo['lr']

    # append
    search_results_list.append(search_results.copy())
    print("Combo Complete!\n")

end = time.time()
print(f"Min. to run: {(end - start) / 60}\n")

# explore best hyperparams
df = pd.DataFrame(search_results_list)
df.to_csv("../Data/hyper_param_results.csv")
"""
