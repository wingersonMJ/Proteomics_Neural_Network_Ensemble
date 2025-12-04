import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt

from data_processing_pt2 import final_x, final_y

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
criterion = nn.MSELoss()
learning_rate = 0.001
momentum = 0.9
epoch = 10000

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
    
plt.figure()
plt.plot(epoch_loss)
plt.show()

# check predictions
with torch.no_grad():
    pred_y = model(X.to(device)).detach().cpu().numpy()

true_y = y.detach().cpu().numpy()

plt.figure()
plt.scatter(x=pred_y, y=y)
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.plot([20, 90], [20, 90])
plt.show()

plt.figure()
plt.hist(x=pred_y, bins=15)
plt.show()

resid = (pred_y[:,0] - true_y)
print(resid)

plt.figure()
plt.scatter(x=true_y, y=resid)
plt.xlabel("Actual Value")
plt.ylabel("Predicted - Actual")
plt.plot([20, 90], [0, 0])
plt.show()
