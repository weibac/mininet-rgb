import json
import torch
from os.path import join
from os import listdir

# Own library https://github.com/weibac/color-library
from colorlibrary.colors import Colorer
from trainer import Net

colorer = Colorer()
nets = listdir("model_weights")


def load_net(filename):
    path = join("model_weights", filename)
    with open(path, 'r') as param_file:
        params = json.load(param_file)

    fc1_weights = params["fc1_weights"]
    fc1_biases = params["fc1_biases"]
    fc2_weights = params["fc2_weights"]
    fc2_biases = params["fc2_biases"]
    fc1_weights = torch.nn.parameter.Parameter(torch.tensor(fc1_weights))
    fc1_biases = torch.nn.parameter.Parameter(torch.tensor(fc1_biases))
    fc2_weights = torch.nn.parameter.Parameter(torch.tensor(fc2_weights))
    fc2_biases = torch.nn.parameter.Parameter(torch.tensor(fc2_biases))

    net = Net(input_size=3, hidden_size=12, output_size=9)
    net.fc1.weight = fc1_weights
    net.fc1.bias = fc1_biases
    net.fc2.weight = fc2_weights
    net.fc2.bias = fc2_biases

    return net, fc1_weights


def to_rgb_scale(value: float):
    return value * 127.5 + 127.5


def display_color_acts(color_acts, labels):
    for idx in range(color_acts.size()[0]):
        rgb = color_acts[idx].to(int)
        rgb = rgb.tolist()
        if labels:
            print(f'{colorer.color_rgb("████", *rgb)} "{labels[idx]}"')
        else:
            print(colorer.color_rgb("████", *rgb))
    print()


def visualize(net, fc1_weights):
    print("Hidden layer activation visualization")
    fc1_color_acts = to_rgb_scale(fc1_weights)
    display_color_acts(fc1_color_acts, False)

    print("Output layer activation visualization")
    categories = ['black', 'grey/gray', 'white', 'red', 'orange', 'yellow',
                  'green', 'blue', 'violet']
    out_acts = net(torch.eye(3, 3))  # Identity matrix
    out_acts = out_acts.transpose(0, 1)
    out_color_acts = to_rgb_scale(out_acts)
    display_color_acts(out_color_acts, categories)


if __name__ == "__main__":
    for net in nets:
        print(f"\n\nNeural net in file {net}\n")
        visualize(*load_net(net))
