import torch
from torch.utils.data import Dataset
import os.path as p
import random as r


class DataReader():
    def __init__(self):
        self.color_dict = {
            'black': 0,
            'grey': 1,
            'gray': 1,
            'white': 2,
            'red': 3,
            'orange': 4,
            'yellow': 5,
            'green': 6,
            'blue': 7,
            'violet': 8}  # TODO: un-hardcode

    def load_data(self):
        with open(p.join('dataset', 'filtered_color_labels.csv')) as file:  # TODO: un-hardcode
            labels = []
            rgbs = []
            for line in file:
                line = line.split(';')
                line[0] = self.color_dict[line[0]]
                line[1] = line[1].split(',')
                line[1] = [int(n) for n in line[1]]
                labels.append(line[0])
                rgbs.append(line[1])
            return rgbs, labels

    def load_training_testing_datasets(self, p_training: float):
        rgbs_training = []
        labels_training = []
        rgbs_testing = []
        labels_testing = []

        rgbs, labels = self.load_data()
        datapoints = len(rgbs)
        points_training = int(round(p_training * datapoints))
        data = [(rgbs[a], labels[a]) for a in range(datapoints)]
        r.shuffle(data)

        data_training = data[:points_training]
        rgbs_training = [tp[0] for tp in data_training]
        labels_training = [tp[1] for tp in data_training]
        data_testing = data[points_training:]
        rgbs_testing = [tp[0] for tp in data_testing]
        labels_testing = [tp[1] for tp in data_testing]

        training_dataset = MyDataset(rgbs_training, labels_training)
        testing_dataset = MyDataset(rgbs_testing, labels_testing)
        return training_dataset, testing_dataset


class MyDataset(Dataset):
    def __init__(self, rgbs, labels):
        self.rgbs = rgbs
        self.labels = labels

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        # Load the rgb value and label
        rgb = self.rgbs[idx]
        label = self.labels[idx]
        return rgb, label
