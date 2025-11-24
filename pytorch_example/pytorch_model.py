import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from data_processing_pt2 import final_x, final_y

# cuda check
print("Torch version:", torch.__version__)
print("Torch compiled with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

##########
# Examples
##########
# create a tensor
data = [[1, 2], [3, 4]]
x = torch.tensor(data)
x

# make tensors based on properties of other tensors
x_ones = torch.ones_like(x) # tensor the same shape as x
x_ones
x_rand = torch.rand_like(x, dtype=torch.float) # override dtype
x_rand

# pre-set the shape and make tensors
shape = (2,3)
rand_tensor = torch.rand(shape)
rand_tensor

# get tensor attributes
rand_tensor.shape
rand_tensor.dtype
rand_tensor.device # where is tensor currently stored?

############
# accelerator
############
# use gpu as accelerator
torch.accelerator.is_available() # check if gpu is available
if torch.accelerator.is_available():
    tensor = rand_tensor.to(torch.accelerator.current_accelerator())
# moves the tensor to the gpu using .to command and .current_acc
# now the device will be cuda
tensor.device

###########
# basic indexing and slicing
###########
tensor
tensor[0, :] # first row
tensor[:, 0] # first column
tensor[0,0] # first item
tensor[:,-1] # last column
tensor[1, :] = 0 # replace last row w/ zero in all columns
tensor[1, 0] += 0.0251

tensor2 = torch.rand_like(tensor)
tensor2

# concat
tensor3 = torch.cat(tensors=[tensor, tensor2], dim=1)
tensor3 # dim=1 adds one next to the other

tensor4 = torch.cat(tensors=[tensor, tensor2], dim=0)
tensor4 # dim 0 adds one below the other

# matrix multiplication
tensor5 = tensor @ tensor2.T
tensor5

tensor6 = torch.matmul(tensor, tensor2.T)
tensor6 # same as just using @ sign for matmul

# getting back to python numerical value
agg = tensor6.sum()
agg = agg.item() # use .item() to get single value back to numeric
type(agg)

#############
# DataLoader and DataSets
#############
final_x.head()
final_y.head()

# make them tensors
x = torch.tensor(final_x.to_numpy()).float()
y = torch.tensor(final_y.to_numpy()).float()

x[:,0]
y[:]

# use the dataset feature to assign x to a training set
train = TensorDataset(x, y)
# this sets up the training set

# set up the data loader
train_loader = DataLoader(
    train,
    batch_size=10,
    shuffle=True
)
# this will load in the data in batches of 10

# view the first element of the train dataset
train[0][0].shape

#############
# Set the device to accelerator
#############
device = torch.accelerator.current_accelerator().type
# will set it to cuda, because that is my current accelerator

###########
# Define the class
###########

class NeuralNetwork(nn.Module): # a child class of nn.Module
    
    # initialize and set the architecture
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7568, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
    
    # define a forward pass
    def forward(self, x):
        outputs = self.linear_relu_stack(x)
        return outputs

#########
# Call the network
#########
# create an instance of our network class and move it to device
model = NeuralNetwork().to(device)
print(model)

# take one iteration of train loader batch
x_batch, y_batch = next(iter(train_loader))
x_batch = x_batch.to(device) # move to device
x_batch[0,:]

# run the model, which is only forward pass rn
outputs = model(x_batch)
print(outputs) 

# print model structure
for name, param in model.named_parameters():
    print(f"Layer: {name}\n"
        f"Size: {param.size()}\n" 
        f"Values : {param[:2]} \n\n")

# get first linear layer's weights
print(model.linear_relu_stack[0].weight)

##########
# Show it by hand
##########
# example batch
x_example, y_example = next(iter(train_loader))

# won't pass to device for this example. Keep on cpu
layer1 = nn.Linear(in_features=7568, out_features=1)
hidden1 = layer1(x_example)
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}\n\n")

# simple loss
loss_function = torch.nn.MSELoss()
loss = loss_function(hidden1, y_example)
print(loss)

# get gradients
print(hidden1.grad_fn)
print(loss.grad_fn)

###########
# Compute gradients
###########
# run backprop
loss.backward()

# get weight gradients for layer 1
print(layer1.weight.grad)

# get bias grads for layer 1
print(layer1.bias.grad)

##########
# Hyperparams
##########
learning_rate = 1e-3
epochs = 500
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#########
# define a training loop
#########
def train_loop(dataloader, model, loss_fn, optimizer):
    # set model to training
    model.train()
    # avg loss
    batch_losses = []
    # add for loop for each iteration of the dataloader
    for batch, (x,y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        # forward pass
        pred = model(x)
        loss = loss_fn(pred, y)

        # backprop
        loss.backward()
        # optimizer takes a step to improve weights and bias
        optimizer.step()
        # reset grads to zero (otherwise, they'll remain the same)
        optimizer.zero_grad()

        # print performance
        loss = loss.item()
        batch_losses.append(loss)
    return batch_losses

# run the loop for a number of iterations
avg_epoch_loss = []
for e in range(epochs):
    print(f"Epoch {e+1}\n----------------")
    batch_losses = train_loop(
        train_loader, model, loss_fn, optimizer)
    avg_batch_loss = np.mean(batch_losses)
    avg_epoch_loss.append(avg_batch_loss)
    print(f"Avg Epoch Loss: {avg_batch_loss}")
print("Done!")

# plot avg epoch loss
step = 10
indices = list(range(0, len(avg_epoch_loss), step))

x_sub = [i + 1 for i in indices]
y_sub = [avg_epoch_loss[i] for i in indices]

plt.plot(x_sub, y_sub, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.title("Training loss per epoch (every 100 epochs)")
plt.grid(True)
plt.xticks(x_sub)
plt.show()

############
# Some info on tensors
############

## Tensors: n-dimensional arrays of data...
# 0-d tensor = scalar
# 2-d tensor = matrix (2 indicies needed to access values)
# Why: Holds data in n-dimensional space
# Not just input data, but weights, bias, gradients too!

## Tensor basics for pytorch:
# Tensor has...
# shape: eg. (32, 12, 3) which defines dimensions
# dtypes: float, int
# device: cpu or gpu 
# gradient: to be tracked or not depending on backprop decisions

## Similarities to other math objects:
# Scalar: really a 0-d tensor. Just one number
# Vector/Array: a 1-d tensor. Just a single vector of numbers
# Matrix: 2-d tensor. Have rows and columns

## Real-world tensor example:
# A dataset of images: 1000 images, each rgb and 64x64 pixels
# Tensor size = (1000, 3, 64, 64)
# A dataset of sequence/signal data: 
# 1000 samples, 500 time steps, 3 vars = (1000, 500, 3)

### What problem do tensors solve??!!
## cool and interesting data are not 2D:
# such as sequence data, images, or anything longitudinal
## tensors store n-d data in one object:
# dl is a bunch of weight+bias calcs, so storing as one item means
# that we can perform many operations all at one time.
## tensors can also store graidents in associated .grad tensors
# so gd is fast and easy

### Tensor shapes:
## tensor(samples, time, variable)
# (1000 samples, 500 time points, 3 variables)
## get subject values for a variable:
# data_tensor[n, :, 'headache']
## get subject values for a timepoint:
# data_tensor[n, 0, :]
## get all subjects for a variable at a timepoint:
# data_tensor[:, 0, 'headache']

## Autodiff
# torch builds a computational graph based on tensor operations
# .backward() calls the backprop step and it's done automatically
# All the operations for backprop are just done
# using the existing tensors! 

### Where tensors show up:
## Input data
# X = (batch_size, time_steps, num_features)
### weights and biases
## w tensor = (in_features, out_features)
# the number of weights is equal to the number of inputs x outputs
# if input is 10 and output is 5, w = (10, 5)
# where a weight for each input is mapped to each output
## b tensor = (out_features, )
# where a weight is given for each output, b = (5,)
## after loss.backward(), w.grad.shape == w.shape
# a gradient for each weight (w.grad[i,j] = dLoss/dW)
## b.grad.shape == b.shape
# a gradient for each bias (b.grad[i,] = dLoss/dB)
## linear tensor (z) = (batch_size, output_features)
# this is just afer we do z = X @ w + b
# usually this stays as a hidden tensor bc we wont need to access it
## activation tensor (A) = (batch_size, out_features)
# After we pass z through our activation
# Also usually a hidden tensor. just passed to next layer
## loss tensor = (single_value/scalar)
# loss.shape(), just one value to represent the loss

### each layer in a network get's a tensor!
## input tensor: x = (batch_size, 500)
## layer1 number of outputs = 200
# layer1.w.shape = (500, 200)
# layer1.b.shape = (200,)
# layer1.z.shape = (batch_size, 200)
# layer1.A.shape = (batch_size, 200)
## layer2 number of outputs = 50
# input tensor: A = (batch_size, 200)
# layer2.w.shape = (200, 50)
# layer2.b.shape = (50,)
# layer2.z.shape = (batch_size, 50)
# layer2.A.shape = (batch_size, 50)
## Final output layer = 1 output
# final.input.shape = (batch_size, 50)
# final.w.shape = (50, 1)
# final.b.shape = (1,)
# final.z.shape = (batch_size, 1)
# final.A.shape = (batch_size, 1)
## loss.shape = ()
## example of one hidden .grad tensor:
# layer1.w.grad.shape = (500, 200) = same shape as layer1.w

### Why gpu's like tensors:
## CPU's have powerful cores, but only a few
## GPU's have weak cores, but a ton
# GPU's are SIMD (single instruction, multiple data)
# they can only perform the same task, but do it on lots of data
## "do this thing across the whole tensor" - gpu's love it

## purpose of doing all this in representation learning:
# instead of using what the data 'are' for prediction...
# we learn what the data 'mean' and use that to predict
# same as learning latent features or interaction relationships
## as we learn the more complex representation of the data
# our predictions become better and more generalized
