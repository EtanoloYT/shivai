import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from progress.bar import Bar
from torch.optim import lr_scheduler
from models.MusicGenerator import MusicGenerator

# Set device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 48000  # CD-quality audio
SAMPLE_WIDTH = 2  # 2 bytes per sample
CHANNELS = 2  # Stereo
INPUT_SIZE = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS  # 1 second of audio
OUTPUT_SIZE = INPUT_SIZE
HIDDEN_SIZE = 256  # Number of hidden units in the LSTM

# Define a LSTM based music generation model:

model = MusicGenerator(INPUT_SIZE, HIDDEN_SIZE, 7, OUTPUT_SIZE).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Use mixed precision training (if available)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

num_epochs = 1  # Increase the number of epochs to train for longer so the model can learn more complex patterns
model.train()  # Set the model to training mode

# Training loop (you'll need to provide your own music dataset)
INPUTS_FOLDER = "outputs/"

accumulation_steps = (
    4  # Perform a backward pass only after accumulating this many steps
)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} of {num_epochs}")

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

                # Backward every accumulation steps
                if (i + 1) % accumulation_steps == 0:
                    # Perform gradient descent
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                bar.next(SEQUENCE_LENGTH)

    # Save the model after each epoch
    torch.save(model.state_dict(), "checkpoints/" + f"model_state{epoch}.pth")
    print(f"Saved model_state{epoch}.pth")
