import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
from progress.bar import Bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Load the trained model
class MusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, (hn, cn) = self.lstm(x, h)
        out = self.fc(out)
        return out, (hn, cn)


model = MusicGenerator(1, 64, 2, 1).to(device)
model.load_state_dict(torch.load("model_state.pth"))

# Define the initial note
initial_note = 0.5

seed_sequence = (
    torch.tensor([initial_note], dtype=torch.float).view(1, 1, -1).to(device)
)
hidden_state = None

generated_music = []

# Generate a sequence of notes in 1-second chunks
chunk_size = 44100
num_chunks = 7
chunks_per_file = 2

# Predict the next note in the sequence
predicted_note, hidden_state = model(seed_sequence, hidden_state)
for x in range(num_chunks):
    for y in range(chunk_size):
        predicted_note, hidden_state = model(seed_sequence, hidden_state)
        print(predicted_note)
"""
with Bar("Generating music", max=num_chunks) as bar:
    for chunk_idx in range(num_chunks):
        chunk_music = []
        for _ in range(chunk_size):
            output, hidden_state = model(seed_sequence, hidden_state)

            # Sample the next note from the output distribution
            next_note = torch.argmax(output, dim=2)

            # Convert the note to a float between 0 and 1
            next_note = next_note.float() / 127

            # Add the note to the generated music
            generated_music.append(next_note.item())

            # Update the seed sequence for the next iteration
            seed_sequence = next_note.view(1, 1, -1)

            chunk_music.append(next_note.item())

        bar.next()

        # Save the generated music to a temporary numpy file every 2 chunks
        if (chunk_idx + 1) % chunks_per_file == 0 or chunk_idx == num_chunks - 1:
            temp_file_path = f"temp/{chunk_idx // chunks_per_file + 1:03}.npy"
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            np.save(temp_file_path, np.array(chunk_music))

            # Deallocate the VRAM
            del model
            torch.cuda.empty_cache()
            model = MusicGenerator(1, 64, 2, 1).to(device)
            model.load_state_dict(torch.load("model_state.pth"))
            hidden_state = None

# Concatenate the generated music from the temporary numpy files
generated_music = []
for chunk_idx in range(num_chunks // chunks_per_file):
    temp_file_path = f"temp/{chunk_idx+1:03}.npy"
    chunk_music = np.load(temp_file_path)
    generated_music += chunk_music.tolist()
    os.remove(temp_file_path)

# Save the final generated music to a numpy file
np.save("generated_music.npy", np.array(generated_music))
"""
