import json
import torch
from os.path import join
# Own library https://github.com/weibac/color-library
from colorlibrary.colors import Colorer

colorer = Colorer('', '')

path = join("model_weights", "4417acc0-62.json")
with open(path, 'r') as param_file:
    params = json.load(param_file)

fc1_weights = params["fc1_weights"]
fc1_biases = params["fc1_biases"]
fc2_weights = params["fc2_weights"]
fc2_biases = params["fc2_biases"]

fc1_weights = torch.tensor(fc1_weights)
fc1_biases = torch.tensor(fc1_biases)
fc2_weights = torch.tensor(fc2_weights)
fc2_biases = torch.tensor(fc2_biases)


def to_rgb_scale(value: float):
    return value * -127.5 + 127.5


fc1_color_acts = to_rgb_scale(fc1_weights)


for idx in range(fc1_color_acts.size()[0]):
    rgb = fc1_color_acts[idx].to(int)
    rgb = rgb.tolist()
    # print(rgb)
    print(colorer.color_rgb("███", *rgb))
