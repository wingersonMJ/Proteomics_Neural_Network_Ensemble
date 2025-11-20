
import torch
import os

x = torch.rand(5, 3)
print(x)

torch.cuda.is_available()

print("Torch version:", torch.__version__)
print("Torch compiled with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

class ProteomicNN():



















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
