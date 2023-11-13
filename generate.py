import torch
import numpy as np
from MusicGenerator import (
    MusicGenerator,
)

# Set device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 48000  # CD-quality audio
SAMPLE_WIDTH = 2  # 2 bytes per sample
CHANNELS = 2  # Stereo
INPUT_SIZE = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS  # 1 second of audio
OUTPUT_SIZE = INPUT_SIZE
HIDDEN_SIZE = 512  # Number of hidden units in the LSTM

# Load the trained model
model = MusicGenerator(INPUT_SIZE, HIDDEN_SIZE, 7, OUTPUT_SIZE).to(device)
model.load_state_dict(torch.load("model_state.pth", map_location=device))
model.eval()

# Set the initial hidden state
hidden = None  # You might need to set an initial hidden state depending on your model architecture

# Generate new music sequence
generated_music = []

with torch.no_grad():
    # Specify the length of the generated sequence (adjust as needed)
    generated_sequence_length = OUTPUT_SIZE * 60

    for _ in range(generated_sequence_length // OUTPUT_SIZE):
        # Generate one step at a time
        input_data = torch.randn(1, 1, INPUT_SIZE).to(
            device
        )  # Random input for the initial step
        output, hidden = model(input_data, hidden)
        output = output.view(-1, OUTPUT_SIZE)

        # Append the generated output to the music sequence
        generated_music.append(output.cpu().numpy())

# Convert the generated music to a NumPy array
generated_music = np.concatenate(generated_music, axis=0)

# Convert to a one-dimensional array
generated_music = generated_music.ravel()

# Normalize to -1 to 1
generated_music = generated_music / np.max(np.abs(generated_music))

# Save the generated music as a .npy file
np.save("generated_music.npy", generated_music)
print("Generated music saved as generated_music.npy")
