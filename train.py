import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from progress.bar import Bar
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Get the number of unique notes in the dataset
# get all the files in the outputs folder
INPUTS_FOLDER = "outputs/"

if not os.file.exists("unique_notes.txt"):
    notes = np.array([])

    for files in os.listdir(INPUTS_FOLDER):
        song = np.load(INPUTS_FOLDER + files)
        notes = np.append(notes, song)

    # get the unique notes
    unique_notes = np.unique(notes)
    output_len = len(unique_notes)

    np.savetxt("unique_notes.txt", output_len)
else:
    output_len = np.loadtxt("unique_notes.txt")


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
output_size = output_len  # Size of the network output (e.g., a single note)

# Initialize the model
model = MusicGenerator(input_size, hidden_size, num_layers, output_size).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Use mixed precision training (if available)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

# Training loop (you'll need to provide your own music dataset)
model.train()
num_epochs = 1
batch_size = 2**12  # Reduce the batch size to reduce memory usage
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} of {num_epochs}")

    INPUTS_FOLDER = "outputs/"

    for file in os.listdir(INPUTS_FOLDER):
        song = np.load(INPUTS_FOLDER + file)
        song = torch.tensor(song, dtype=torch.float).view(1, -1, 1).to(device)

        min_length = song.shape[1]
        song = song[:, :min_length, :]

        hidden_state = None
        initial_note = song[0][0]

        input_seq = song[:, :-1]
        target_seq = song[:, 1:]

        # Map continuous target values to integers
        target_integers = ((target_seq + 1) / 2 * (output_size - 1)).round().long()

        input_batches = torch.split(input_seq, batch_size, dim=1)
        target_batches = torch.split(target_integers, batch_size, dim=1)

        # Forward pass
        with Bar("Processing " + file, max=len(input_batches)) as bar:
            for i in range(len(input_batches)):
                input_batch = input_batches[i]
                target_batch = target_batches[i]
                bar.next()

                # Ensure target values are within the correct range
                target_batch = torch.clamp(target_batch, 0, output_size - 1)

                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs, _ = model(input_batch, hidden_state)
                    # Flatten the outputs and targets for the CrossEntropyLoss
                    outputs = outputs.view(-1, output_size)
                    target_batch = target_batch.view(-1)
                    loss = criterion(outputs, target_batch)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()


# Save the model
torch.save(model.state_dict(), "model_state.pth")
