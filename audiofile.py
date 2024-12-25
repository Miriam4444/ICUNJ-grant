import numpy as np
import matplotlib.pyplot as plt
import librosa as lib
import scipy as sci
import statistics as stat
import os
from AudiofilesArray import AudiofilesArray
from DataAnalysis import DataAnalysis
from pathlib import Path
from collections import Counter

class AudioFile:

    def __init__(self, file, Athresh):

        ##############################################
        # attributes of the audiofile object
        ##############################################
        # load wav file as array and store sample rate
        self.source, self.sr =  lib.load(file, sr=None) 

        # file name without path
        self.file = os.path.basename(file)

        # user input amplitude threshold
        if Athresh == None:
            self.Athresh = 0
        else:
            self.Athresh = Athresh

        # compute and store real fft of audio array
        self.fourier = np.fft.rfft(self.source)

        # store number of samples in original file
        self.N = len(self.source)

        # store original sample's frequency bins
        self.bins = np.arange(len(self.fourier))

        # compute the power spectrum of the rfft
        self.pspec = np.abs(self.fourier)

        # compute the corresponding frequencies of the rfft in Hz
        # first uses the built-in rfft frequency calculator from numpy
        # third calculates by hand accounting for the rfft cutting the frequency bins in half
        self.freq = np.fft.rfftfreq(len(self.source), 1/self.sr)

        # identify fundamental as freq of largest coefficient in the power spectrum 
        # -- this can lead to erroneous results, we need to be careful in how we define this!
        self.dummyfundamental = self.getfund2()

        # filter the PSD to cut out frequencies below the fundamental, above a certain threshold, and below a prescribed amplitude
        self.filtered = self.filter()

        # identify the array of peaks in the filtered signal.  **Must choose a widths array in the findpeaks method.**
        self.peaks = self.findpeaks()

        # redefine fundamental as first peaks entry
        self.fundamental = self.peaks[0]

        # identifies peaks/fundamental values
        self.ratioArray = self.findratioArray()

        # computes the array of differences: |actual - theoretical|
        self.absoluteErrorArray = np.abs(self.ratioArray - np.rint(self.ratioArray))

        # computes the array of relative errors
        self.relativeErrorArray = self.findrelativeError()

        # convert power spectrum output data to dbfs (decibels full scale)
        self.dbfs = self.dbfsconvert()

        # convert PSD to dBV (decibel relative to 1 volt)
        self.dbv = self.dbvconvert()

        self.meanRelativeError = stat.mean(self.relativeErrorArray)
        self.stdevRelativeError = stat.stdev(self.relativeErrorArray)




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
        self.printfreq()
        self.printfundamental()
        self.printpeaks()

    def printpeaks(self):
        print(f"the peaks of {self.file} are", self.peaks) 

    def printratios(self):
        print(f"the ratio array of {self.file} is\n", self.ratioArray)

    def printError(self):
        print(f"{self.file} has mean error {self.meanRelativeError}\n and stdev of error {self.stdevRelativeError}\n from {len(self.ratioArray)} datapoints")

    # function to identify frequency of largest magnitude entry in PSD.  
    # I believe we need to double it since rfft uses half the bins.  Need to check this against cakewalk data.
    # Interesting problem has arisen now that I'm looking at other samples: the third harmonic appears to be 
    # significantly louder than the first two, which is leading to the fundamental being incorrectly identified.
    # We may want to alter this method so that it only hunts between 200 and 500 Hz for our fundamental since 
    # we'll only tune our strings to that frequency range
    def getfund2(self):
        # initialize min and max indices to 0
        min = 0
        max = 0

        # find first index where the frequency exceeds 200
        for i in range(0, len(self.freq)-1):
            if self.freq[i] >= 100:
                min = i
                break

        # find first index where the frequency exceeds 600
        for j in range(min,len(self.freq)-1):
            if self.freq[j] >= 500:
                max = j
                break

        # search for loudest frequency only between 200 and 300 Hz.  Will return relative to min=0.
        F = np.argmax(self.pspec[min:max])

        # convert PSD index back to Hz
        F = self.freq[F+min]

        return F
    
    # static version of the above.  This will take in an array and an integer samplerate
    # and return 2*(array index corresp to maximum magnitude array value) 
    @staticmethod
    def staticgetfund(array, samplerate):
        A = np.abs(array)
        F = (samplerate/len(array))*np.argmax(A[:samplerate//2 + 1])
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
        # until the output for 258 Hz is about +4.  However the other peaks in cakewalk are significantly higher than 
        # the peaks we get from this scheme so I'm not sure
        V0 = 225

        N = len(self.pspec)

        # initialize dbv array of correct length with 1s
        dbv = np.ones(N)

        # compute dBV --- adding a conditional statement to avoid log(0)
        for i in range(0,N):
            if self.pspec[i] != 0:
                dbv[i] = 20*np.log10(self.pspec[i]/V0)
            else:
                dbv[i] = -100 # should be -infinity so I made it a large negative.  Not sure how best to handle this.

        return dbv

    # static version of the above function
    @staticmethod
    def staticdbvconvert(array):
        # For the static version of this method we may want to allow for V0 as an input?
        V0 = 225

        # absolute value the input array so we don't blow up the log
        A = np.abs(array)

        # initialize dbv array of correct length with 1s
        dbv = np.ones(len(array))

        # compute dBV --- adding a conditional statement to avoid log(0)
        for i in range(0,len(array)):
            if A[i] != 0:
                dbv[i] = 20*np.log10(A[i]/V0)
            else:
                dbv[i] = -100 # should be -infinity so I made it a large negative.  Not sure how best to handle this.

        return dbv

    # filter the psd data to cut out everything below a certain loudness (magthresh)
    # and below a specified frequency (freqthresh).  This method can definitely be improved by 
    # allowing for only filtering of one kind or the other.
    @staticmethod
    def filtersignal(array,loFthresh,hiFthresh,Athresh):
        loFthresh = int(loFthresh)
        hiFthresh = int(hiFthresh)

        # create an array of zeroes followed by ones to filter below frequency threshold (Fthresh)
        Z = np.zeros(loFthresh)
        oneslength = len(array)-loFthresh
        Arr01 = np.ones(oneslength)
        Arr01 = np.concatenate((Z,Arr01))

        # zero out all array entries below the frequency threshold
        filteredArr = Arr01*array

        if hiFthresh==None:
            # zero out all array entries below the amplitude threshold (Athresh)
            for i in range(len(array)):
                if np.abs(array[i]) < Athresh:
                    filteredArr[i] = 0

            return filteredArr

        elif len(array)-hiFthresh < 0:
            print("hiFthresh was ignored!  Maximum hiFthresh for this array is", len(array))
            # zero out all array entries below the amplitude threshold (Athresh)
            for i in range(len(array)):
                if np.abs(array[i]) < Athresh:
                    filteredArr[i] = 0

            return filteredArr

        else:
            # initialize an array that will be 1s until hiFthresh
            Arr02 = np.ones(hiFthresh)
            # initialize an array of 0s the length of the array minus hiFthresh
            Z2 = np.zeros(len(array)-hiFthresh)
            # make an array that looks like [1,1,1,...,1,0,...,0] to filter above hiFthresh
            Arr02 = np.concatenate((Arr02,Z2))

            filteredArr = filteredArr*Arr02

            # zero out all array entries below the amplitude threshold (Athresh)
            for i in range(len(array)):
                if np.abs(array[i]) < Athresh:
                    filteredArr[i] = 0

            return filteredArr

    # class method for bandpassing the signal to a set standard of parameters.  
    # current high pass is fundamental-10
    # current low pass is 50*fundamental + 100
    # current noise floor is 0
    def bandpass(self):
        loFthresh = int(self.dummyfundamental)-10
        hiFthresh = 20*int(self.dummyfundamental)+100
        Athresh = 0

        return AudioFile.filtersignal(self.pspec, loFthresh, hiFthresh, Athresh)

    # class method for amplitude thresholding the bandpassed signal.
    # Athresh is set to mean+stdev of the windowed median with windowsize fundamental//2
    def filter(self):
        windowsize = self.dummyfundamental//2

        mean, stdev = AudioFile.staticwindowedmedian(self.bandpass(), windowsize)

        Athresh = self.Athresh

        return AudioFile.filtersignal(self.bandpass(),0,len(self.fourier),Athresh)
    
    # class method for finding peaks in the PSD of our signal with a certain width
    # current width is between 5 and 30 samples
    def findpeaks(self):
        peaks = sci.signal.find_peaks_cwt(self.filtered, widths=np.arange(5, 30))

        return peaks
    
    # class method to find the array of ratios: peak frequency/fundamental frequency
    def findratioArray(self):
        P = np.zeros(len(self.peaks))

        for i in range(0,len(self.peaks)):
            P[i] = self.freq[self.peaks[i]]

        fund = self.fundamental*self.sr/self.N

        P = P/fund

        return P

    def findrelativeError(self):
        # create an array of the correct length
        E = np.zeros(len(self.ratioArray))

        for i in range(len(self.ratioArray)):
            E[i] = self.absoluteErrorArray[i]/np.rint(self.ratioArray[i])

        return E

    # method to compute a windowed median of the signal.  It will compute the median in each
    # window of size windowsize and put each median value in an array, then average the median
    # array values and return that average.  Currently detects high and low pass filters
    # and applies this windowed median procedure to the non-passed part of the signal
    @staticmethod
    def staticwindowedmedian(array,windowsize):
        # detect initial and trailing zeros
        hipass, lopass = AudioFile.detectzeros(array)

        windowsizeInt = int(windowsize)

        # find length of nonzero part of signal
        signallength = len(array) - lopass - hipass

        if signallength <= 0:
            print("The input signal is all 0s!")
            return 0, 0

        else:
            # initialize an array of 0s of the correct length
            M = np.zeros(int(signallength/windowsizeInt)) 

            for i in range(0,len(M)):
                # populate array M with positive median of each window
                M[i] = stat.median(array[hipass + i*windowsizeInt : hipass + (i+1)*windowsizeInt-1])

            return stat.mean(M), stat.stdev(M)

    # static method that takes in an array and determines if there are leading or trailing
    # zeros.  IDs a high- or low-passed signal.  Returns # of zeroes at beginning and end
    @staticmethod
    def detectzeros(array):
        initialzeros = 0
        trailingzeros = 0

        for i in range(0,len(array)):
            if array[i] != 0:
                break
            initialzeros = i+1
        
        for i in range(0,len(array)):
            if array[len(array)-(i+1)] != 0:
                break
            trailingzeros = i+1

        return initialzeros, trailingzeros

    # static method to ignore zeros in an array and compute the median of nonzero entries
    @staticmethod
    def positivemedian(array):
        if len(array)==0:
            return 0

        else: 
            M = np.zeros(len(array))

            # initialize a counter for M
            j = 0

            for i in range(0,len(array)):
                if array[i] != 0:
                    M[j] = array[i]
                    j=j+1
        
            if j==0:
                return 0

            else:    
                M2 = np.zeros(j)
                for i in range(0,j):
                    M2[i] = M[i]

                return stat.median(M2)

    @staticmethod
    def filteredstats(array):
        loPass = 0
        hiPass = 0

        hiPass, loPass = AudioFile.detectzeros(array)

        # find length of nonzero part of signal
        signallength = len(array) - loPass - hiPass

        if signallength <= 0:
            print("The input signal is all 0s!")
            return 0, 0

        else:
            # initialize an array of 0s of the correct length
            M = np.zeros(int(signallength)) 

            for i in range(0,len(M)):
                # populate array M with positive median of each window
                M[i] = array[hiPass+i]

            return stat.mean(M), stat.stdev(M)

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
        plt.plot(self.freq, self.dbfs)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dBFS')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the converted dbv data vs freq3.  Not sure if this currently works.
    # We should decide what minimum dbv value we want to consider.  
    def graph_dbv(self):
        plt.plot(self.freq, self.dbv)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dBV')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the PSD data versus original bins
    @staticmethod
    def graph_filtersignal(array, Fthresh, Athresh):
        F = AudioFile.filtersignal(array,Fthresh,Athresh)
        plt.plot(np.arange(len(array)), F)
        plt.xlabel('entry number')
        plt.ylabel('signal')
        plt.title('graph of filtered signal')
        plt.show()

    def analyzeData(self, directory, startValue, endValue, n):
        #DirectoryName = Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples")
        #directory = r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\ICUNJ-grant-audiofiles"
        nameArray = AudiofilesArray(Path(directory))
        #print(nameArray.makeFilePathList())
        namelist = nameArray.getSpecificType("1S")
        # initialize an array of n-1 evenly spaced Athresh values between startValue and endValue
        A = np.linspace(startValue, endValue, n)

        # initialize an empty |A| x |audiofiles| array to be populated with audiofile arrays
        objArray = np.empty(shape=(len(A),len(namelist)), dtype=AudioFile)

        # initialize an empty |A| x |audiofiles| array to be populated with the meanerror of each audiofile
        M = np.empty(shape=(len(A),len(namelist)))

        meanofmeans = list()
        datapointsArray = np.empty(shape=(len(A),len(namelist)))
        labels = list()

        for i in range(len(A)):
                a = A[i]

                # populate the ith row of the array of audiofiles with samples corresponding to threshold a
                for j in range(len(namelist)):

                        f = AudioFile(namelist[j], a)

                        objArray[i][j] = f

                        # populate row a = A[i] with the relativeMeanErrors for the samples
                        M[i][j] = objArray[i][j].meanRelativeError

                        datapointsArray[i][j] = len(objArray[i][j].ratioArray)

                m = stat.mean(M[i])

                meanofmeans.append(m)

                meandatapoints = stat.mean(datapointsArray[i])

                labels.append(meandatapoints)

                #print(f"the mean of the mean relative errors for Athresh {a} is {m}")

        k = np.linspace(0.5, 3, 6)

        for j in range(len(k)):
                weightfunction = list()

                for i in range(len(A)):
                        weight = labels[i]/meanofmeans[i]**(1/k[j])
                        weightfunction.append(weight)

                plt.plot(A, weightfunction)
                plt.xlabel("A")
                plt.ylabel("W(k,A)")
                plt.title(f"weight = {k[j]}")
        
                plt.show()
############################################################################
# END AUDIOFILE CLASS
############################################################################
