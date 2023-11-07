import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from progress.bar import Bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Define a simple LSTM-based music generation model
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


# Define hyperparameters
input_size = 1  # Size of the input (e.g., a single note)
hidden_size = 64  # Size of the LSTM hidden state
num_layers = 2  # Number of LSTM layers
output_size = 1  # Size of the output (e.g., the next note)

# Initialize the model
model = MusicGenerator(input_size, hidden_size, num_layers, output_size).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training loop (you'll need to provide your own music dataset)
num_epochs = 4
batch_size = 16384 * 64  # Reduce the batch size to reduce memory usage
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} of {num_epochs}")
    # Load a batch of training data

    # Load 10 numpy arrays of shape (song_length, 1)
    # Each array represents a song, and each element in the array represents a note

    INPUTS_FOLDER = "outputs/"

    for file in os.listdir(INPUTS_FOLDER):
        song = np.load(INPUTS_FOLDER + file)
        # transform the song into a tensor
        song = torch.tensor(song, dtype=torch.float).view(1, -1, 1).to(device)

        # Find the minimum length of the song
        min_length = song.shape[1]

        # Trim the song to the same length
        song = song[:, :min_length, :]

        # Initialize the hidden state
        hidden_state = None

        # Initialize the input sequence
        initial_note = song[0][0]

        # Prepare input and target sequences
        input_seq = song[:, :-1]
        target_seq = song[:, 1:]

        # Split the input and target sequences into batches
        input_batches = torch.split(input_seq, batch_size, dim=1)
        target_batches = torch.split(target_seq, batch_size, dim=1)

        # Forward pass
        with Bar("Processing " + file, max=len(input_batches)) as bar:
            for i in range(len(input_batches)):
                input_batch = input_batches[i]
                target_batch = target_batches[i]
                # progress bar to track the training progress
                bar.next()
                with torch.cuda.amp.autocast():
                    outputs, _ = model(input_batch, hidden_state)
                    loss = criterion(outputs, target_batch)

                # Backpropagation and optimization
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

# Save the model
torch.save(model.state_dict(), "model_state.pth")
