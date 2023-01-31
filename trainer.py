import torch
import torch.nn as nn
import torch.optim as optim
import json

from dataset.dataset import MyDataset
from dataset.dataset import DataReader


# Neural net class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float64)
        self.relu = nn.ReLU()

    def forward(self, x):  # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Net and training parameters
def init_net():
    net = Net(input_size=3, hidden_size=12, output_size=9)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    return net, optimizer


loss_fn = nn.CrossEntropyLoss()  # Loss function
device = torch.device("cpu")    # "cuda" for gpu, but i have an AMD :(

# Datasets and data loaders
data_reader = DataReader()
training_dataset, testing_dataset = data_reader.load_training_testing_datasets(0.5)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=4, shuffle=False)


def train_model(net, optimizer, loss_fn, device, train_loader):
    for inp, target in train_loader:
        # Input and target formatting
        inp = torch.stack(inp, 1)
        inp = inp.to(torch.float64)
        target = target.to(torch.float64)
        # Move them to device
        inp = inp.to(device)
        target = target.to(device)

        # Forward pass
        output = net(inp)

        # Turn target to long ints, so I can do cross-entropy
        target = target.to(torch.long)

        # Compute loss
        loss = loss_fn(output, target)

        # Zero gradients, backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_model(net, device, test_loader, idx):
    net.eval()  # Sets the model to evaluation mode
    predictions = []
    labels = []
    for inp, target in test_loader:
        # Input and target formatting
        inp = torch.stack(inp, 1)
        inp = inp.to(torch.float64)
        target = target.to(torch.float64)
        # Move them to the device
        inp = inp.to(device)
        target = target.to(device)

        # Make predictions
        output = net(inp)  # Run the net
        _, predicted = torch.max(output, dim=1)  # This picks the net's max-credence category

        # Store the predictions and labels
        predictions.extend(predicted.tolist())
        labels.extend(target.tolist())

    # Compute acuracy
    datapoints = len(labels)
    correct = 0
    for a in range(len(labels)):
        if predictions[a] == labels[a]:
            correct += 1
    accuracy = correct / datapoints

    # print(f"\nAccuracy: {accuracy}\nGuessing at random would be 0.1111111...")

    if accuracy > 0.6:
        print(f"Found a good one! accuracy: {accuracy}")
        weights = net.state_dict()
        fc1_weights = weights["fc1.weight"].tolist()
        fc1_biases = weights["fc1.bias"].tolist()
        fc2_weights = weights["fc2.weight"].tolist()
        fc2_biases = weights["fc2.bias"].tolist()
        model_parameters = {
            "fc1_weights": fc1_weights,
            "fc1_biases": fc1_biases,
            "fc2_weights": fc2_weights,
            "fc2_biases": fc2_biases}
        accuracy_str = str(round(accuracy, 2)).replace(".", "-")
        with open(f"{idx}acc{accuracy_str}.json", "w") as param_file:
            json.dump(model_parameters, param_file)


if __name__ == "__main__":
    for idx in range(10000):
        net, optimizer = init_net()
        train_model(net, optimizer, loss_fn, device, train_loader)
        test_model(net, device, test_loader, idx)
