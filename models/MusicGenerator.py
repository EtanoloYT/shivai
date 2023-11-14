import torch
from torch import nn


# Define a LSTM based music generation model:


class MusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicGenerator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out, hidden
