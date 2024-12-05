import librosa as lib
import numpy as np
import scipy as sci

# Take in a file
def open_file(file):
    # Load audio file using librosa
    audio_data_array, sr = lib.load(file, sr=None) 
    print(f"Audio data shape: {audio_data_array.shape}, Sample rate: {sr}")
    #The function takes in a file and uses librosa to load the data and then returns the data as an array
    return audio_data_array

def output_array(audio_data):
    fft_result = np.fft.fft(audio_data)  # FFT result is an array of complex numbers
    return fft_result


if __name__ == "__main__":
    # Process the file once and store the result
    audio_data = open_file(r"C:\Users\abeca\OneDrive\ICUNJ grant stuff\2S q 11-22-24.wav")
    fft_result = output_array(audio_data)
    
    # Display the first 10 complex numbers from the FFT result
    print("here is an array of all of the complex numbers")
    print("Length of array: " , len(fft_result))
    print(fft_result)