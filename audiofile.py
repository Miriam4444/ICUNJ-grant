import numpy as np
import matplotlib.pyplot as plt
import librosa as lib

class audiofile:

    def __init__(self, file):

        ##############################################
        # properties of the audiofile
        ##############################################
        # load wav file as array and store sample rate
        self.source, self.sr =  lib.load(file, sr=None) 

        # compute and store real fft of audio array
        self.fourier = np.fft.rfft(self.source)

        # store number of samples in original file
        self.N = len(self.source)

        # store original sample's frequency bins
        self.bins = np.arange(len(self.fourier))

        # compute the power spectrum of the rfft
        self.pspec = np.abs(self.fourier)

        # compute the corresponding frequencies of the rfft in Hz
        # first uses the back-of-the-envelope calculation we think should work
        # second uses the built-in rfft frequency calculator from numpy
        # third calculates by hand accounting for the rfft cutting the frequency bins in half
        self.freq = np.arange(self.N)*self.sr/(self.N)
        self.freq2 = np.fft.rfftfreq(len(self.pspec), 1/self.sr)
        self.freq3 = np.arange((self.N / 2) + 1)*self.sr/float(self.N)

        # identify fundamental as freq of largest coefficient in the power spectrum 
        # -- this can lead to erroneous results, we need to be careful in how we define this!
        self.fundamental = self.getfund()

        # convert power spectrum output data to dbfs (decibels full scale)
        self.dbfs = self.dbfsconvert()

        # convert PSD to dBV (decibel relative to 1 volt)
        self.dbv = self.dbvconvert()


    ##########################################
    # methods
    ##########################################
    def printN(self):
        print("the length of the original file is", self.N)

    def printbins(self):
        print("the number of freq bins in the RFFT is", self.bins)

    def printPSD(self):
        print("the power spectrum of the RFFT is", self.pspec)

    def printfreq(self):
        print("the frequencies in Hz of the PDS are", self.freq)

    def printfundamental(self):
        print("the fundamental frequency of the signal is", self.fundamental)

    def printall(self):
        self.printN()
        self.printbins()
        self.printfreq()
        self.printPSD()
        self.printfundamental()

    # function to identify frequency of largest magnitude entry in PSD.  
    # I believe we need to double it since rfft uses half the bins.  Need to check this against cakewalk data.
    def getfund(self):
        F = 2*np.argmax(self.pspec[:self.sr//2 + 1])
        return F

    # function to convert the PSD output to dBFS (decibel full scale)
    # I believe it's currently wonky since it doesn't appear to agree with cakewalk's output, for example,
    # but I'm also not sure what the vertical axis is in cakewalk.  My gut says it's dBV (decibels relative to 1 volt)
    def dbfsconvert(self):
        rescaled = self.pspec*2/self.N

        # Convert to dBFS (decibels full scale)
        dbfs = 20*np.log10(rescaled)

        return dbfs

    # function to convert the PSD output to dBV (decibels re: 1 volt)
    # to do this we need a reference voltage V0.
    def dbvconvert(self):
        # reference voltage - maybe we test some values to see what gets close to cakewalk's output?
        # update: I looked back at the screenshot that had fundamental equal to 258 Hz (which is what
        # our getfund method says is the fundamental of this signal) and it appeared to peak around
        # 4 on the vertical axis in cakewalk.  So I very crudely guessed and checked with the V0 value
        # until the output for 258 Hz is about +4.
        V0 = 225

        # compute dBV 
        dbv = 20*np.log10(self.pspec/V0)

        return dbv

    # static version of the above function
    @staticmethod
    def staticdbvconvert(array):
        # For the static version of this method we may want to allow for V0 as an input?
        V0 = 225

        # absolute value the input array so we don't blow up the log
        A = np.abs(array)

        # compute dBV 
        dbv = 20*np.log10(A/V0)

        return dbv

    # filter the psd data to cut out everything below a certain loudness (magthresh)
    # and below a specified frequency (freqthresh).  This method can definitely be improved by 
    # allowing for only filtering of one kind or the other.
    @staticmethod
    def filtersignal(array,Fthresh,Athresh):
        # create an array of zeroes followed by ones to filter below frequency threshold (Fthresh)
        Z = np.zeros(Fthresh)
        oneslength = len(array)-Fthresh
        Arr01 = np.ones(oneslength)
        Arr01 = np.concatenate((Z,Arr01))

        # zero out all array entries below the frequency threshold
        filteredArr = Arr01*array

        # zero out all array entries below the amplitude threshold (Athresh)
        for i in range(len(array)):
            if array[i] < Athresh:
                filteredArr[i] = 0

        return filteredArr
    
    # function to plot the raw rfft data
    def graph_original(self):
        plt.plot(self.bins, self.fourier)
        plt.xlabel('entry number')
        plt.ylabel('RFFT coefficient')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the PSD data versus original bins
    def graph_PSD(self):
        plt.plot(self.bins, self.pspec)
        plt.xlabel('entry number')
        plt.ylabel('Magnitude of RFFT')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the converted dbfs data vs freq3.  Not sure if this currently works.
    def graph_dbfs(self):
        plt.plot(self.freq3, self.dbfs)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dBFS')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the converted dbfs data vs freq3.  Not sure if this currently works.
    def graph_dbv(self):
        plt.plot(self.freq3, self.dbv)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dBV')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the PSD data versus original bins
    @staticmethod
    def graph_filtersignal(array, Fthresh, Athresh):
        F = audiofile.filtersignal(array,Fthresh,Athresh)
        plt.plot(np.arange(len(array)), F)
        plt.xlabel('entry number')
        plt.ylabel('signal')
        plt.title('graph of filtered signal')
        plt.show()

# test the class methods
test = audiofile(r"C:\Users\spine\Downloads\2S q 11-22-24.wav")

#test.graph_dbv()
#test.printall()
#test.graph_original()
F = audiofile.filtersignal(test.fourier,1000,1)
audiofile.graph_filtersignal(F, 1000, 1)

F1 = audiofile.staticdbvconvert(F)
plt.plot(np.arange(len(F1)), F1)
plt.show()
