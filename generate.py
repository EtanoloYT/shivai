import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from progress.bar import Bar
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if not os.path.exists("unique_notes.txt"):
    print("No unique notes file found. Please run train.py first.")
else:
    output_len = int(np.loadtxt("unique_notes.txt"))


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


# Load model "model_state.pth"
model = MusicGenerator(1, 64, 2, output_len).to(device)

model.load_state_dict(torch.load("model_state.pth"))

# Generate a new song

HZ = 44100
DURATION = 10
CHANNELS = 2
SAMPLE_WIDTH = 2

VALUES_PER_SECOND = HZ * CHANNELS * SAMPLE_WIDTH

# Initialize the hidden state
h = torch.zeros(2, 1, 64).to(device)
c = torch.zeros(2, 1, 64).to(device)
h = (h, c)

# Initialize the first input
x = torch.zeros(1, 1, 1).to(device)

# Generate the song note by note
song = []

print("Generating song...")

with Bar("Generating", max=VALUES_PER_SECOND * DURATION) as bar:
    # Create a temporary directory to store the generated music files
    if not os.path.exists("temp"):
        os.makedirs("temp")

    for j in range(DURATION):
        song = []
        for i in range(VALUES_PER_SECOND):
            out, h = model(x, h)

            # Get the index of the most likely note index
            temperature = 0.8  # You can experiment with different temperature values
            scaled_logits = out[0, 0] / temperature
            note_index = torch.multinomial(
                torch.softmax(scaled_logits, dim=-1), 1
            ).item()
            # Get the value of the most likely note
            note = torch.tensor([[note_index]], dtype=torch.float).to(device)

            # Add the note to the song
            song.append(note)

            # Use the generated note as the input to the next iteration
            x = torch.tensor([[[note]]], dtype=torch.float).to(device)

            # Update the progress bar
            bar.next()

        # Convert the song to a NumPy array
        song = np.array([note.cpu().numpy() for note in song])

        # Save the song to a temporary file
        np.save(f"temp/song_{j}.npy", song)

        del out  # Delete variables to free up memory
        torch.cuda.empty_cache()  # Empty GPU cache

    # Concatenate the temporary files to create the output file
    generated_music = np.concatenate(
        [np.load(f"temp/song_{j}.npy") for j in range(DURATION)]
    )

    # Map integers to continuous target values
    generated_music = (generated_music / (output_len - 1) * 2) - 1

    # The array is currently as [[[note]], [[note]], ...]. Remove the extra dimension
    generated_music = generated_music[:, 0, :]

    # Save the output file
    np.save("generated_music.npy", generated_music)

    # Remove the temporary directory and files
    for j in range(DURATION):
        os.remove(f"temp/song_{j}.npy")
    os.rmdir("temp")

# Convert the song to a NumPy array
song = np.array(song)

# Save the song to a file
np.save("generated_music.npy", song)
