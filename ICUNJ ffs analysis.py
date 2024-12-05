import librosa as lib
import numpy as np
import scipy as sci

# Take in a file
def open_file(file):
    # Load audio file using librosa
    audio_data_array, sr = lib.load(file, sr=None) 
    #print(f"Audio data shape: {audio_data_array.shape}, Sample rate: {sr}")
    #The function takes in a file and uses librosa to load the data and then returns the data as an array
    return audio_data_array

def output_array(audio_data):
    fft_result = np.fft.fft(audio_data)  # FFT result is an array of complex numbers
    #this function takes in the array with the audio data from open_dile(file) and outputs an array of complex numbers
    return fft_result

def conjugate_array(fft_result):
    #this function makes an array of all the conjugates of the complex numbers
    conjugate_array = [] #define array
    for i in range (len(fft_result)): #cycle through array entries
        complex_number = fft_result[i]
        conjugate = np.conj(complex_number)
        conjugate_array.append(conjugate) #add the conjugate of the entry to the array we just defined

    return conjugate_array

def print_array(array , entries_to_print):
    #This function takes in an array and the amount of entries you want to print
    if entries_to_print == None: #if you dont give an argument for entries_to_print itll just print the whole array
        entries_to_print = len(array)
    else:
        entries_to_print = entries_to_print
    print("Length of array = " , len(array)) #prints the length of the array
    for i in range(entries_to_print): #prints the specified amount of entries
        print(array[i])


if __name__ == "__main__":
    # Process the file once and store the result
    audio_data = open_file(r"C:\Users\abeca\OneDrive\ICUNJ grant stuff\2S q 11-22-24.wav") #specify what file we're opening
    fft_result = output_array(audio_data) #define the result of the fft
    print("here is an array of all of the complex numbers") #do the print statements for the array of the complex numbers
    print_array(fft_result, 10)
    conjugate_result = conjugate_array(fft_result) #Define the conjugate array result
    print("Here is an array of all the conjugates") #do the print statements for the array of the conjugates of the complex numbers array
    print_array(conjugate_result , 10)
   