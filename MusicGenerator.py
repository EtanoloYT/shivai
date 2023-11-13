import torch
import torch.nn as nn


class MusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, (hn, cn) = self.lstm(x, h)
        out = self.fc(out)
        out = torch.tanh(out)  # Apply tanh activation function
        return out, (hn, cn)
