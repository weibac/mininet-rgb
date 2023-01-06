import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import MyDataset
from dataset.dataset import DataReader


# Neural net class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float64)
        self.relu = nn.ReLU()

    def forward(self, x):  # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


net = Net(input_size=3, hidden_size=12, output_size=9)

optimizer = optim.SGD(net.parameters(), lr=0.01)  # SGD is stochastic gradient descent
loss_fn = nn.CrossEntropyLoss()  # Loss function

data_reader = DataReader()
training_dataset, testing_dataset = data_reader.load_training_testing_datasets(0.5)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=4, shuffle=True)
device = torch.device("cpu")    # "cuda" for gpu, but i have an AMD :(


# Training loop
idx = 0
for inp, target in train_loader:
    # Input tensor formatting
    inp = torch.transpose(torch.stack(inp), 0, 1)

    # Debug
    # print(inp)
    # print(len(inp))
    # print(type(inp))
    # print(inp.dtype)
    # print(target)
    # print(len(target))
    # print(type(target))
    # print(target.dtype)
    # Convert input and target to tensors
    # inp = torch.tensor(inp)
    # target = torch.tensor(target)

    # Convert input and target tensor elements to floats
    inp = inp.to(torch.float64)
    target = target.to(torch.float64)

    # Move them to processing device
    inp = inp.to(device)
    target = target.to(device)

    # Forward pass
    output = net(inp)

    # Turn target to long ints, so I can do cross-entropy
    target = torch.tensor(target, dtype=torch.long, device=device)

    # Compute loss
    loss = loss_fn(output, target)

    # Zero gradients, backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Trainig run {idx} finished")
    idx += 1
