# chat-gpt experiment


import torch
import torch.nn as nn
import numpy as np
from data_handler import X_train, X_test, y_train, y_test, n_per_entry
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def plot_float_values(float_list, title=None, xlabel=None, ylabel=None):
    """
    Given a list of float values, creates a line plot of the values with optional annotations.
    """
    plt.plot(float_list)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# I want you to write me a neural network using pytorch that
# performs binary class classification. The input is going to
# be a numpy array that contains n arrays. Each array will have
# m inputs that are either a 0 or 1.

# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Define the parameters
input_size = n_per_entry
hidden_size = 64
output_size = 1

# Create the neural network
model = Net(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Convert the input data to PyTorch tensors
# y_train = np.array([[1], [1], [0], [0]])
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
loss_list = []
# Train the neural network

num_epochs = 50_000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)


    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the progress
    if (epoch + 1) % 1000 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        loss_list.append(loss.item())


# Test the neural network

X_test_tensor = torch.from_numpy(X_test).float()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()
    # print('Predicted:', predicted.numpy())


# print(f"Accuracy: {metrics.accuracy_score(y_train, predicted)}")
# print("--" * 10, "^ACCURACY (TRAINED)^" ,"--" * 10)

print(f"Accuracy: {metrics.accuracy_score(y_test, predicted)}")
print("--" * 10, "^ACCURACY (TRAINED) (VALIDATION)^", "--" * 10)

plot_float_values(loss_list, title='Loss', xlabel='Epoch (1000s)', ylabel='Loss')
