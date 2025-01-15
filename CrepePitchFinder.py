import crepe
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_path = r'C:\Users\abeca\ICUNJ-grant\2S_q_11-22-24.wav'  # Replace with your audio file path
signal, sr = librosa.load(audio_path, sr=16000)  # Resample to 16 kHz for CREPE

# Analyze pitch with CREPE
time, frequency, confidence, activation = crepe.predict(signal, sr, viterbi=True)

# Print pitch and confidence values
print("Pitch Estimation Results:")
for t, f, c in zip(time, frequency, confidence):
    print(f"Time: {t:.2f}s, Pitch: {f:.2f} Hz, Confidence: {c:.2f}")

# Find the fundamental frequency (F0)
fundamental_frequency = frequency[0]  # Assuming the first detected pitch is F0
print(f"\nFundamental Frequency (F0): {fundamental_frequency:.2f} Hz")

# Plot the pitch over time
plt.figure(figsize=(10, 5))
plt.plot(time, frequency, label="Estimated Pitch", color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Pitch Estimation using CREPE")
plt.legend()
plt.grid()
plt.show()

# Clean up resources used by CREPE
crepe.close()
