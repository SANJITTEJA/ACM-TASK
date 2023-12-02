import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Hyper parameters

input_size = 784  # image size - 28*28
hidden_size = 100 # hidden layers input size , neuron size in the hidden layer
num_classes = 10 # 10 classes in the MNIST dataset
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Data Preprocessing

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)
# download dataset directly from torchvision.datasets
# set parameter train True for train dataset and False for test dataset


# Data Loading and visualization

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model Definition

# defining a class NeuralNet inherited from neuralnet module which was imported from torch.nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # Linear layer at the beginning of the neural net
        self.relu  = nn.ReLU() # ReLU as the hidden layer in the neural net
        self.linear2 = nn.Linear(hidden_size, num_classes) # Linear layer at the ending of the neural net

    def forward(self, X):
        out = self.linear1(X)  # method for forward propagation
        out = self.relu(out)
        out = self.linear2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)


examples = iter(train_loader)
samples, labels = examples.__next__()

print(samples.shape, labels.shape)

#Data visualization
#
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap = 'gray')
# plt.show()


CEloss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop

n_total_steps = len(train_loader) # for calculating no.of steps in each epoch

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28) # reshaping the image to 1d array to pass as input of linear layer

        # Forward pass
        outputs = model(images) # output obtained after forward pass
        loss = CEloss(outputs, labels) # requires grad = True

        # backward pass
        optimizer.zero_grad() # clearing gradient after every iteration to avoid piling the gradient
        loss.backward()  # finding the gradient with respect to loss
        optimizer.step() # updating the weights

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item()}')


with torch.no_grad():
    n_correct = 0
    n_ssmples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28)

        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_ssmples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()
    acc = 100.0 * n_correct / n_ssmples
    print(f'accuracy = {acc}')
