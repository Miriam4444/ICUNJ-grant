import crepe
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Function to convert frequency to MIDI note
def frequency_to_midi(frequency):
    return 69 + 12 * np.log2(frequency / 440.0)

# Function to convert MIDI note to musical note name
def midi_to_note_name(midi_note):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(midi_note // 12) - 1
    note = note_names[int(midi_note % 12)]
    return f"{note}{octave}"

# Load the audio file
audio_path = r'C:\Users\abeca\ICUNJ-grant\2S_q_11-22-24.wav'  # Replace with your audio file path
signal, sr = librosa.load(audio_path, sr=16000)  # Resample to 16 kHz for CREPE

# Analyze pitch with CREPE
time, frequency, confidence, activation = crepe.predict(signal, sr, viterbi=True)

# Filter low-confidence results
threshold = 0.8  # Confidence threshold
filtered_time = []
filtered_notes = []

for t, f, c in zip(time, frequency, confidence):
    if c >= threshold and f > 0:  # Filter out zero frequency (silence)
        midi_note = round(frequency_to_midi(f))
        note_name = midi_to_note_name(midi_note)
        filtered_time.append(t)
        filtered_notes.append(note_name)

# Plot the notes over time
plt.figure(figsize=(12, 6))
plt.scatter(filtered_time, filtered_notes, color='blue', label="Detected Notes")
plt.xlabel("Time (s)")
plt.ylabel("Musical Notes")
plt.title("Pitch Detection in Musical Notes")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Clean up resources used by CREPE
crepe.close()
