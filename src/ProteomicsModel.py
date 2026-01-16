import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing_pt2 import final_x, final_y

plt.style.use("seaborn-v0_8-poster")

# set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# batch size
batch_size = 12

# convert to numpy
X = final_x.to_numpy()
y = final_y.to_numpy()
print("Data Loaded")

# convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
print("Converted X, y to Tensors")

# DataSet and DataLoader
train_dataset = TensorDataset(X, y)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

data_iteration = iter(train_dataloader)
example_x, example_y = next(data_iteration)
print(f"Example X: {example_x[0:5, 0:10]}")
print(f"Example y: {example_y[0:5]}")
print(example_x.shape)

##########
# Define the model!
##########
class ProteomicsModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            torch.nn.Linear(7568, 2500),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2500, 1000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.network(x)

########
# training loop
# commenting out so it doesnt train and run
# every time I call ProteomicsModel
"""
criterion = nn.MSELoss()
learning_rate = 0.001
momentum = 0.9
epoch = 200

device = torch.accelerator.current_accelerator().type
model = ProteomicsModel().to(device) # MUST DO THIS B4 INIT OPTIMIZR
print(model)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
########

epoch_loss = []
for e in range(epoch):

    model.train()

    e_loss = []
    for i, batch in enumerate(train_dataloader, 0):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # zero out gradients at the start
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if e % 2000 == 0 and i % 6 == 0:
            for name, param in model.named_parameters():
                if "weight" in name:
                    temp_w = param.grad.abs().max().item()
                    print(f"Max w.grad: {np.max(temp_w):.2f}")
                if "bias" in name:
                    temp_b = param.grad.abs().max().item()
                    print(f"Max b.grad: {np.max(temp_b):.2f}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        e_loss.append(loss.item())


    avg_loss = np.mean(e_loss)
    if e % 2000 == 0:
        print(f"Epoch: {e}")
        print(f"Epoch Loss: {avg_loss}\n")
    epoch_loss.append(avg_loss)
    
plt.figure(figsize=(8,6))
plt.plot(epoch_loss, label="Epoch Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Loss over Epochs - Initial Network Model")
plt.tight_layout()
plt.savefig("./figs/loss_initial_model.jpg", dpi=300)
plt.show()

print(epoch_loss[-1])

# check predictions
with torch.no_grad():
    pred_y = model(X.to(device))
    pred_y = pred_y.squeeze(1)
    pred_y = pred_y.detach().cpu().numpy()

true_y = y.detach().cpu().numpy()

# plot predictions vs actuals
sns.jointplot(x=pred_y, y=true_y)
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Predicted Values - Initial NN Model")
plt.tight_layout()
plt.savefig("./figs/initial_NN.jpg", dpi=300)
plt.show()

sns.jointplot(x=pred_y, y=true_y, kind='kde', fill=True)
plt.plot([30, 80], [30, 80], 
         linestyle="--", 
         color="dimgrey",
         label="Perfect prediction line")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Predicted Values - Initial NN Model")
plt.tight_layout()
plt.savefig("./figs/initial_NN_kde.jpg", dpi=300)
plt.show()
"""
