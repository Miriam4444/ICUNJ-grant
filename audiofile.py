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
        self.dbfs = self.dbconvert()


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

    def dbconvert(self):
        rescaled = self.pspec*2/self.N

        # Convert to dBFS (decibels full scale)
        dbfs = 20*np.log10(rescaled)

        return dbfs

    # filter the psd data to cut out everything below a certain loudness (magthresh)
    # and below a specified frequency (freqthresh)
    def filterpsd(self,freqthresh,magthresh):
        # create an array of zeroes followed by ones to filter below frequency threshold---this is currently in bins not Hz
        Z = np.zeros(freqthresh)
        oneslength = self.bins-freqthresh
        Arr01 = np.ones(oneslength)
        Arr01 = Z.append(Arr01)

        # zero out all array entries below the frequency threshold
        filteredArr = Arr01*self.pspec

        for i in self.bins:
            if self.pspec[i] < magthresh:
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

    # function to plot the PSD data versus original bins
    def graph_filterPSD(self, freqthresh, magthresh):
        plt.plot(self.bins, self.filterpsd(freqthresh,magthresh))
        plt.xlabel('entry number')
        plt.ylabel('Magnitude of RFFT')
        plt.title('graph of string pluck')
        plt.show()

# test the class methods
test = audiofile(r"C:\Users\spine\Downloads\2S q 11-22-24.wav")

#test.graph_dbfs()
test.printall()
# test.graph_original()
test.graph_filterPSD(1000,1)