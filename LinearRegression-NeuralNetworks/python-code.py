#######LINEAR REGRESSION########

#Loading data from pokemon_data.csv

import numpy as np
import pandas as pd

data_frame = pd.read_csv('pokemon_data.csv')
data_frame.head()

#############################################################################

#Creating one Numpy array to contain the feature data without the name 
#column and one array to contain the combat point ground truth.

features = data_frame.values[:, 1:-1]
labels = data_frame.values[:, -1]
print('array of labels: shape ' + str(np.shape(labels)))
print('array of feature matrix: shape ' + str(np.shape(features)))

#############################################################################

#Replacing the categorical feature 'primary_strength' with **one-hot encoding** 
#and generating the new version of the Numpy array 'features'.

# Get index for primary_strength
index = data_frame.columns.get_loc("primary_strength")

# Removing first column from this
new_index = (index - 1)

# Extracting primary_strength column from features 
ps_column = features[:, new_index]

# Getting the unique vals from the primary_strength column
unique_values = np.unique(ps_column)

# Making a zero matrix that has columns equal to unique vals
encoded = np.zeros((features.shape[0], len(unique_values)))

# Counter
counter = 0

# Loop through each unique value in primary_strength
for i in unique_values:
    
    # Finding indixes where column = unique val
    indices = np.where(ps_column == i)
    
    # Setting encoded matrix to 1 (one hot encoding) 
    encoded[indices, counter] = 1
    
    # increasing counter for next loop through
    counter += 1

# Getting column index
column_index = data_frame.columns.get_loc("primary_strength")

# Matching up index
x = (column_index - 1)

# Deleting column from features 
features = np.delete(features, x, axis=1)

features = np.hstack((features, encoded))

print('Ne version of features:', features.shape)

#############################################################################

#Since other features have different scales so standardization is needed. 
#$({x-\mu})/{\sigma}$, where $\mu$ is the mean and $\sigma$ is the 
#standard deviation.

import math

# Extract non one-hot encoded columns
ne_features = features[:, :-17]

# Calculate the mean for each feature
mean = np.mean(ne_features, axis=0)

# difference
difference = (ne_features - mean)

# Squaring difference
square_difference = (difference**2)

# Compute variance
variance = np.mean(square_difference, axis=0)

standard_deviation_list = []

for i in variance:
    
    standard_deviation = math.sqrt(i)
    
    standard_deviation_list.append(standard_deviation)
    
numpy_array = np.array(standard_deviation_list)

s_features = ((ne_features - mean) / numpy_array)

# Replacing non one-hot encoded features to the standardized ones
features[:, :-17] = s_features

#If all works this should be printed
print('Success: Features are now standardized ')

#############################################################################

#Implementing my own linear regression model.
#I am using the Ordinary Least Square solution (OLS)
#Then I am using the 5-fold cross-validation method
#Lastly, I am printing out the square root of the residual, sum of squares(RSS)
#between actual and predicted outcome variable, and the average square root
#of the RSS over all folds

import numpy as np

# matrix is float now
features = features.astype(float)

# Helper function for ols
def OLS(X, y):
    
    x_intercept = np.c_[np.ones((X.shape[0], 1)), X]
    
    # OLS formula
    c = np.linalg.pinv(x_intercept.T.dot(x_intercept))
    
    c = c.dot(x_intercept.T)
    
    c = c.dot(y)
    
    return (c)

# Helper function for predictions
def predict(X, c):
    
    x_intercept = np.c_[np.ones((X.shape[0], 1)), X]
    
    return (x_intercept.dot(c))

# 5-folds
five_folds = np.array_split(np.arange(len(features)), 5)


RSS_list = []

for fold in five_folds:
    
    a = np.setdiff1d(np.arange(len(features)), fold)

    x1 = features[a]
    
    y1 = labels[a]

    x2 = features[fold]
    
    y2 = labels[fold]
    
    # get OLS solution
    c = OLS(x1, y1)
    
    # get predictions
    y_predictions = predict(x2, c)
    
    # RSS
    difference = (y2 - y_predictions)
    
    squared_diff = (difference ** 2)
    
    rss = np.sum(squared_diff)
    
    rss_sqrt = np.sqrt(rss)
    
    RSS_list.append(rss_sqrt)
    
    # Print RSS square root currently
    print(f"RSS square root for this fold is: {rss_sqrt:.2f}")

# Getting average square root of RSS over all folds
avg_rss_sqrt = np.mean(RSS_list)

print(f"\n 5-fold RSS square root average is: {avg_rss_sqrt:.2f}")

#############################################################################

#Repeating the same experiment but now using linear regression with
#L2-norm regularization and reporting same results as before with 
#$\lambda=\{1, 0.1, 0.01, 0.001, 0.0001\}$.

import numpy as np

# Needed for X to have intercept 
def add_intercept(X):
    
    return (np.c_[np.ones((X.shape[0], 1)), X])

# L2-norm regularization helper
def Regression(X, y, l):
    
    x_intercept = add_intercept(X)
    
    
    I = np.eye(x_intercept.shape[1])
    
    # Linear regression formula
    reg_part = (x_intercept.T.dot(x_intercept) + l * I)
    
    reg_inv = np.linalg.pinv(reg_part)
    
    # Coefficients 
    c = reg_inv.dot(x_intercept.T)
    
    c = c.dot(y)
    
    return (c)

# different values of the regularization term $\lambda=\{1, 0.1, 0.01, 0.001, 0.0001\}$
lambdas = [1, 0.1, 0.01, 0.001, 0.0001]

for L in lambdas:
    
    RSS_list = []
    
    for fold in five_folds:
        
        a = np.setdiff1d(np.arange(len(features)), fold)
        
        x1 = features[a]
        
        y1 = labels[a]
        
        x2 = features[fold]
        
        y2 = labels[fold]
        
        # using helper
        c = Regression(x1, y1, L)
        
        # prediction helper in previous part
        y_predictions = predict(x2, c)
        
        # getting RSS
        difference = (y2 - y_predictions)
        
        squared_diff = (difference ** 2)
        
        rss = np.sum(squared_diff)
    
        rss_sqrt = np.sqrt(rss)
        
        RSS_list.append(rss_sqrt)
        
    avg_rss_sqrt = np.mean(RSS_list)
    
    print(f"\nLambda={L}: Average Square Root of RSS over all folds: {avg_rss_sqrt:.2f}\n")
    
    #############################################################################
    
#######Neural Networks########
    
#Loading and processing the data.
#Creating dataset objects for further use by Pytorch.
    
# load data from file and split into training and validation sets
import numpy as np
data = np.loadtxt("train.txt", delimiter=',')
perm_idx = np.random.permutation(data.shape[0])
vali_num = int(data.shape[0] * 0.2)
vali_idx = perm_idx[:vali_num]
train_idx = perm_idx[vali_num:]
train_data = data[train_idx]
vali_data = data[vali_idx]
train_features = train_data[:, 1:].astype(np.float32)
train_labels = train_data[:, 0].astype(int)
vali_features = vali_data[:, 1:].astype(np.float32)
vali_labels = vali_data[:, 0].astype(int)

###############################################################################

#Defining a Dataset class 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class MNISTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]
    
###############################################################################

#Creating data loaders.

training_data = MNISTDataset(train_features, train_labels)
vali_data = MNISTDataset(vali_features, vali_labels)
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
vali_dataloader = DataLoader(vali_data, batch_size=batch_size)

for X, y in train_dataloader:
    print(f"Shape of X [N, F]: {X.shape} {X.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

###############################################################################
    
#Building and training my multi-layer perceptron model by Pytorch.
#My code includes: 
#1. three layers [784 -> 512 -> 10]
#2.'weight_decay=1e-4' in torch.optim.SGD to add L2 regularization.
#3. training the model for 10 epochs.
#4. Printing out the training process and the final accuracy on the validation set.

import torch
import torch.nn as nn
import torch.optim as optim

# Model
class NeuralNetwork(nn.Module):
    
    def __init__(self):
        
        super(NeuralNetwork, self).__init__()
        
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            
            nn.Linear((28*28), 512),
            
            nn.ReLU(),
            
            nn.Linear(512, 10)
            
        )

    def forward(self, x):
        
        x = self.flatten(x)
        
        logits = self.linear_relu_stack(x)
        
        return logits

# Hyperparameters
model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Optimizing
def train(dataloader, model, loss_fn, optimizer):
    
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

        if (batch % 100 == 0):
            
            loss = loss.item() 
            
            current = (batch * len(X))
            
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Testing
def test(dataloader, model, loss_fn):
    
    size = len(dataloader.dataset)
    test_loss = 0
    
    correct = 0

    with torch.no_grad():
        
        for X, y in dataloader:
            
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            
            correct += (pred.argmax(1) == y).sum().item()

    test_loss /= size
    
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# epoch
epochs = 10

for t in range(epochs):
    
    print(f"Epoch {t+1}\n")
    
    train(train_dataloader, model, loss_fn, optimizer)
    
    test(vali_dataloader, model, loss_fn)

print("Done!")

###############################################################################

#######Tuning Hyperparameters########

#Loading the testing data

test_features = np.loadtxt("test.txt", delimiter=',')
print('array of testing feature matrix: shape ' + str(np.shape(test_features)))

###############################################################################

#Tuning four hyperparameters:

#1. the number of layers and the dimension of each layer 
#2. the activation function (choose from sigmoid, tanh, relu, leaky_relu)
#3. weight decay
#4. number of training epochs

#Code for tuning is commented out due to large time complexity, but works.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# MLP helper
def MLP(i, h, a):
    
    layers = []
    
    dimensions = [i] + h
    
    for i in range(len(h)):
        
        layers.append(nn.Linear(dimensions[i], h[i]))
        
        if (a == 'relu'):
            
            layers.append(nn.ReLU())
        
        elif (a == 'sigmoid'):
            
            layers.append(nn.Sigmoid())
        
        elif (a == 'tanh'):
            
            layers.append(nn.Tanh())
        
        elif (a == 'leaky_relu'):
            
            layers.append(nn.LeakyReLU())
    
    layers.append(nn.Linear(h[-1], 10))
    
    return (nn.Sequential(*layers))

'''

# Hyperparameters given
layers = [[512], [256, 128], [512, 256, 128]]

a_s = ['relu', 'sigmoid', 'tanh', 'leaky_relu']

weight_decays = [0, 1e-3, 1e-4, 1e-5]

training_epochs = [5, 10, 15]

# device and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

best_accuracy = 0.0

best_model = None

best_params = None

# Tuning
for h in layers:
    
    for a in a_s:
    
        for weight_decay in weight_decays:
    
            for epochs in training_epochs:
                
                model = MLP(784, h, a).to(device)
                
                optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)
                
                
                for epoch in range(epochs):
                    
                    for data, target in train_dataloader:
                    
                        data = data.to(device) 
                        
                        target = target.to(device)
                        
                        optimizer.zero_grad()
                        
                        output = model(data)
                        
                        loss = criterion(output, target)
                        
                        loss.backward()
                        
                        optimizer.step()
                
                # Validating
                correct = 0
                
                total = 0
                
                with torch.no_grad():
                    
                    for data, target in vali_dataloader:
                    
                        data = data.to(device)  
                        
                        target = target.to(device)
                        
                        outputs = model(data)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        
                        total += target.size(0)
                        
                        correct += (predicted == target).sum().item()
                
                accuracy = ((100 * correct) / total)
                
                # Printing the results
                print(f'Layers: {h}, Activation: {a}, Weight decay: {weight_decay}, Epochs: {epochs}, Accuracy: {accuracy}%')
                
                # Updating models as best as can be
                if (accuracy > best_accuracy):
                    
                    best_accuracy = accuracy
                    
                    best_model = model
                    
                    best_params = {'layers': h, 'activation': a, 'weight_decay': weight_decay, 'epochs': epochs}

print("\nBest Hyperparameters:", best_params)

'''