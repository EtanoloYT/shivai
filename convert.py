# BEGIN: 3d8f4b5fjw9c
import numpy as np
from pydub import AudioSegment
import wave
import struct
import subprocess
import os
import sys


def generate_random_wav_file(output_file, duration_seconds, sample_rate=44100):
    """
    Generate a random WAV audio file.

    Args:
    - output_file (str): The path to the output WAV file.
    - duration_seconds (int): Duration of the audio in seconds.
    - sample_rate (int, optional): Sample rate (e.g., CD-quality audio). Default is 44100.
    """
    num_samples = int(duration_seconds * sample_rate)
    audio_data = np.random.uniform(-1, 1, num_samples)
    audio_data_int = np.int16(
        audio_data * 32767
    )  # Convert float values to 16-bit integers

    with wave.open(output_file, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            struct.pack("<" + "h" * len(audio_data_int), *audio_data_int)
        )

    print(f"Random WAV file saved as {output_file}")


def play_wav_file_with_ffmpeg(input_wav_file):
    """
    Play a WAV audio file using FFmpeg.

    Args:
    - input_wav_file (str): The path to the input WAV file.
    - output_file (str): The path to the output WAV file.
    """
    command = ["ffmpeg", "-i", input_wav_file, "random_audio.mp3"]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def convert_wav_to_array(input_wav_file, sample_rate=44100):
    """
    Convert a WAV audio file to a NumPy array.

    Args:
    - input_wav_file (str): The path to the input WAV file.

    Returns:
    - audio_array (np.array): A NumPy array of the audio data.
    - sample_rate (int): The sample rate of the audio data.
    """
    audio = AudioSegment.from_file(input_wav_file)
    audio_array = np.array(audio.get_array_of_samples())
    audio_array = audio_array.astype(np.float32) / 32767.0  # Convert to float
    return audio_array, audio.frame_rate


def convert_txt_to_wav(input_txt_file, output_wav_file, sample_rate=44100):
    """
    Convert a text file to a WAV audio file.

    Args:
    - input_txt_file (str): The path to the input text file.
    - output_wav_file (str): The path to the output WAV file.
    - sample_rate (int, optional): Sample rate (e.g., CD-quality audio). Default is 44100.
    """
    audio_array = np.loadtxt(input_txt_file)
    audio_data_int = np.int16(
        audio_array * 32767
    )  # Convert float values to 16-bit integers

    with wave.open(output_wav_file, "w") as wav_file:
        wav_file.setnchannels(2)  # Stereo audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            struct.pack("<" + "h" * len(audio_data_int), *audio_data_int)
        )

    print(f"WAV file saved as {output_wav_file}")


def save_array_to_txt_file(output_file, audio_array):
    """
    Save a NumPy array to a text file.

    Args:
    - output_file (str): The path to the output text file.
    - audio_array (np.array): A NumPy array of the audio data.
    """
    np.savetxt(output_file, audio_array)

    print(f"Audio array saved as {output_file}")


def save_array_to_bin_file(output_file, audio_array):
    """
    Save a NumPy array to a text file.

    Args:
    - output_file (str): The path to the output text file.
    - audio_array (np.array): A NumPy array of the audio data.
    """
    np.save(output_file, audio_array)

    print(f"Audio array saved as {output_file}")


def convert_bin_to_wav(input_bin_file, output_wav_file, sample_rate=44100):
    """
    Convert a text file to a WAV audio file.

    Args:
    - input_bin_file (str): The path to the input text file.
    - output_wav_file (str): The path to the output WAV file.
    - sample_rate (int, optional): Sample rate (e.g., CD-quality audio). Default is 44100.
    """
    audio_array = np.load(input_bin_file)
    audio_data_int = np.int16(
        audio_array * 32767
    )  # Convert float values to 16-bit integers

    with wave.open(output_wav_file, "w") as wav_file:
        wav_file.setnchannels(2)  # Stereo audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            struct.pack("<" + "h" * len(audio_data_int), *audio_data_int)
        )

    print(f"WAV file saved as {output_wav_file}")


def print_bin_file(input_bin_file):
    """
    Convert a text file to a WAV audio file.

    Args:
    - input_bin_file (str): The path to the input text file.
    - output_wav_file (str): The path to the output WAV file.
    - sample_rate (int, optional): Sample rate (e.g., CD-quality audio). Default is 44100.
    """
    audio_array = np.load(input_bin_file)
    print(audio_array)

    print(f"WAV file saved as {input_bin_file}")


def encode_folder(input_folder, output_folder):
    """
    Convert a text file to a WAV audio file.

    Args:
    - input_bin_file (str): The path to the input text file.
    - output_wav_file (str): The path to the output WAV file.
    - sample_rate (int, optional): Sample rate (e.g., CD-quality audio). Default is 44100.
    """
    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            print("Converting file: ", file)
            song, samplerate = convert_wav_to_array(os.path.join(input_folder, file))
            save_array_to_bin_file(os.path.join(output_folder, file), song)
        else:
            continue


# Generate a random WAV file
# output_file = "random_audio.wav"
# duration_seconds = 5
# generate_random_wav_file(output_file, duration_seconds)

# song, samplerate = convert_wav_to_array("bellfigo.wav")

# save_array_to_txt_file("non pago affitto.txt", song)
# convert_txt_to_wav("non pago affitto.txt", "bellofigo.wav", sample_rate=samplerate)

# for each file in inputs folder convert it to array and save it to outputs folder
# for file in os.listdir(INPUTS_FOLDER):
#    if file.endswith(".wav"):
#        print("Converting file: ", file)
#        song, samplerate = convert_wav_to_array(os.path.join(INPUTS_FOLDER, file))
#        save_array_to_bin_file(os.path.join(OUTPUTS_FOLDER, file), song)
#    else:
#        continue


def main(argv):
    if argv[1] == "encode":
        """
        argv1: encode
        argv2: input file
        argv3: output file
        """
        print("Converting file: ", argv[2])
        song, samplerate = convert_wav_to_array(argv[2])
        save_array_to_bin_file(argv[3], song)
    elif argv[1] == "decode":
        """
        argv1: decode
        argv2: input file
        argv3: output file
        """
        print("Converting file: ", argv[2])
        convert_bin_to_wav(argv[2], argv[3])
    elif argv[1] == "print":
        """
        argv1: print
        argv2: input file
        """
        print_bin_file(argv[2])
    elif argv[1] == "encode_folder":
        """
        argv1: encode_folder
        argv2: input_folder
        argv3: output_folder
        """
        encode_folder(argv[2], argv[3])


if __name__ == "__main__":
    main(sys.argv)
