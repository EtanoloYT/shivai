import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from progress.bar import Bar
from torch.optim import lr_scheduler
from concurrent.futures import ProcessPoolExecutor
from MusicGenerator import MusicGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2
CHANNELS = 2
INPUT_SIZE = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
# The output should be the next second of music (i.e., the next 44100 * 2 * 2 bytes)
OUTPUT_SIZE = INPUT_SIZE
HIDDEN_SIZE = 128

# Define a LSTM based music generation model:

model = MusicGenerator(INPUT_SIZE, HIDDEN_SIZE, 2, OUTPUT_SIZE).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Use mixed precision training (if available)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

num_epochs = 1
model.train()
batch_size = 2**12  # Reduce the batch size to reduce memory usage

# Training loop (you'll need to provide your own music dataset)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} of {num_epochs}")

    INPUTS_FOLDER = "outputs/"

    for file in os.listdir(INPUTS_FOLDER):
        song = np.load(INPUTS_FOLDER + file)
        # Convert the song to a tensor
        song = torch.from_numpy(song).to(device)
        song = song.view(1, -1, 1)
        SEQUENCE_LENGTH = INPUT_SIZE

        with Bar("Training " + file, max=song.shape[1]) as bar:
            for i in range(0, song.shape[1] - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
                # Get the input and target sequences
                input_seq = song[:, i : i + SEQUENCE_LENGTH, :].permute(0, 2, 1)
                target_seq = song[:, i + 1 : i + SEQUENCE_LENGTH + 1, :]

                # Flatten the target sequence
                target_seq = target_seq.view(-1, OUTPUT_SIZE)

                # Run the forward pass
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    output, _ = model(input_seq, None)
                    output = output.view(-1, OUTPUT_SIZE)

                    # Compute the loss
                    loss = criterion(output, target_seq)

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                bar.next(SEQUENCE_LENGTH)

    # Save the model after each epoch
    torch.save(model.state_dict(), f"model_state{epoch}.pth")
    print(f"Saved model_state{epoch}.pth")
