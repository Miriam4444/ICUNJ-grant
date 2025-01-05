import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import librosa as lib
import scipy as sci
import statistics as stat
import os
from AudiofilesArray import AudiofilesArray
from DataAnalysis import DataAnalysis
from pathlib import Path
from collections import Counter
from typing import Any

NDArray = np.ndarray[Any, np.dtype[np.float64]]

class AudioFile:

    def __init__(self, file: str, Athresh: float):

        ##############################################
        # attributes of the audiofile object
        ##############################################
        # load wav file as array and store sample rate
        self.source, self.sr =  lib.load(file, sr=None) 

        self.source = self.hanningWindow()

        self.windowedSource = self.hanningWindow()

        # file name without path
        self.file: str = os.path.basename(file)

        # user input amplitude threshold
        if Athresh == None:
            self.Athresh = 0
        else:
            self.Athresh = Athresh

        # compute and store real fft of audio array
        self.fourier: NDArray = np.fft.rfft(self.source)

        # store number of samples in original file
        self.N: int = len(self.source)

        # store original sample's frequency bins
        self.bins: NDArray = np.arange(len(self.fourier))

        # compute the power spectrum of the rfft
        self.pspec: NDArray = np.abs(self.fourier)

        # compute the corresponding frequencies of the rfft in Hz
        # first uses the built-in rfft frequency calculator from numpy
        # third calculates by hand accounting for the rfft cutting the frequency bins in half
        self.freq: NDArray = np.fft.rfftfreq(len(self.source), 1/self.sr)

        # identify fundamental as freq of largest coefficient in the power spectrum 
        # -- this can lead to erroneous results, we need to be careful in how we define this!
        self.dummyfundamental: int = self.getfund2()

        # filter the PSD to cut out frequencies below the fundamental, above a certain threshold, and below a prescribed amplitude
        self.filtered: NDArray = self.filter()

        # identify the array of peaks in the filtered signal.  **Must choose a widths array in the findpeaks method.**
        self.peaks: NDArray = self.findpeaks()

        # redefine fundamental as first peaks entry
        self.fundamental: int = self.peaks[0]

        # identifies peaks/fundamental values
        self.ratioArray: NDArray = self.findratioArray()

        # computes the array of differences: |actual - theoretical|
        self.absoluteErrorArray: NDArray = np.abs(self.ratioArray - np.rint(self.ratioArray))
        self.meanAbsoluteError: float = stat.mean(self.absoluteErrorArray)

        # computes the array of relative errors
        self.relativeErrorArray: NDArray = self.findrelativeError()

        self.meanRelativeError: float = stat.mean(self.relativeErrorArray)
        self.stdevRelativeError: float = stat.stdev(self.relativeErrorArray)

        
        #self.source = self.hanningWindow()
        #self.windowedFourier = np.fft.rfft(self.source)
        #self.windowedpspec = np.abs(self.windowedFourier)
        #self.windowedFiltered = self.filter()
        #self.windowedPeaks = self.findpeaks()
        #self.windowedRatioArray = self.findratioArray()
        #self.windowedAbsoluteErrorArray = np.abs(self.ratioArray - np.rint(self.ratioArray))
        #self.windowedMeanAbsoluteError = stat.mean(self.windowedAbsoluteErrorArray)
        
        

    ##########################################
    # methods
    ##########################################
    def printN(self) -> None:
        print("the length of the original file is", self.N)

    def printbins(self) -> None:
        print("the number of freq bins in the RFFT is", self.bins)

    def printPSD(self) -> None:
        print("the power spectrum of the RFFT is", self.pspec)

    def printfreq(self) -> None:
        print("the frequencies in Hz of the PDS are", self.freq)

    def printfundamental(self) -> None:
        print("the fundamental frequency of the signal is", self.fundamental)

    def printall(self) -> None:
        self.printN()
        self.printfreq()
        self.printfundamental()
        self.printpeaks()

    def printpeaks(self) -> None:
        print(f"the peaks of {self.file} are", self.peaks) 

    def printratios(self) -> None:
        print(f"the ratio array of {self.file} is\n", self.ratioArray)

    def printError(self) -> None:
        print(f"{self.file} has mean error {self.meanRelativeError}\n and stdev of error {self.stdevRelativeError}\n from {len(self.ratioArray)} datapoints")

    # function to identify frequency of largest magnitude entry in PSD.  
    # I believe we need to double it since rfft uses half the bins.  Need to check this against cakewalk data.
    # Interesting problem has arisen now that I'm looking at other samples: the third harmonic appears to be 
    # significantly louder than the first two, which is leading to the fundamental being incorrectly identified.
    # We may want to alter this method so that it only hunts between 200 and 500 Hz for our fundamental since 
    # we'll only tune our strings to that frequency range
    def getfund2(self) -> int:
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
    def staticgetfund(array: NDArray, samplerate: int):
        A = np.abs(array)
        F = (samplerate/len(array))*np.argmax(A[:samplerate//2 + 1])
        return F


    # filter the psd data to cut out everything below a certain loudness (magthresh)
    # and below a specified frequency (freqthresh).  This method can definitely be improved by 
    # allowing for only filtering of one kind or the other.
    @staticmethod
    def filtersignal(array: NDArray, loFthresh:float, hiFthresh:float, Athresh:float) -> NDArray:
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
    def bandpass(self) -> NDArray:
        loFthresh = int(self.dummyfundamental)-10
        hiFthresh = 20*int(self.dummyfundamental)+100
        Athresh = 0

        return AudioFile.filtersignal(self.pspec, loFthresh, hiFthresh, Athresh)

    # class method for amplitude thresholding the bandpassed signal.
    # Athresh is set to mean+stdev of the windowed median with windowsize fundamental//2
    def filter(self) -> NDArray:
        windowsize = self.dummyfundamental//2

        mean, stdev = AudioFile.staticwindowedmedian(self.bandpass(), windowsize)

        Athresh = self.Athresh

        return AudioFile.filtersignal(self.bandpass(),0,len(self.fourier),Athresh)
    
    # class method for finding peaks in the PSD of our signal with a certain width
    # current width is between 5 and 30 samples
    def findpeaks(self) -> NDArray:
        peaks = sci.signal.find_peaks_cwt(self.filtered, widths=np.arange(5, 30))

        return peaks
    
    # class method to find the array of ratios: peak frequency/fundamental frequency
    def findratioArray(self) -> NDArray:
        P = np.zeros(len(self.peaks))

        for i in range(0,len(self.peaks)):
            P[i] = self.freq[self.peaks[i]]

        fund = self.fundamental*self.sr/self.N

        P = P/fund

        return P

    def findrelativeError(self) -> NDArray:
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
    def staticwindowedmedian(array: NDArray, windowsize: int) -> float:
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
    def detectzeros(array: NDArray) -> tuple[int,int]:
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
    def positivemedian(array: NDArray) -> float:
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
    def filteredstats(array: NDArray) -> tuple[float,float]:
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
    def graph_original(self) -> None:
        plt.plot(self.bins, self.fourier)
        plt.xlabel('entry number')
        plt.ylabel('RFFT coefficient')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the PSD data versus original bins
    def graph_PSD(self) -> None:
        plt.plot(self.bins, self.pspec)
        plt.xlabel('entry number')
        plt.ylabel('Magnitude of RFFT')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the PSD data versus original bins
    @staticmethod
    def graph_filtersignal(array: NDArray, Fthresh: float, Athresh: float) -> None:
        F = AudioFile.filtersignal(array,Fthresh,Athresh)
        plt.plot(np.arange(len(array)), F)
        plt.xlabel('entry number')
        plt.ylabel('signal')
        plt.title('graph of filtered signal')
        plt.show()

    @staticmethod
    def graphMeanOfMeans(directory: str, startValue: float, endValue: float, n: int) -> None:
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
                        M[i][j] = objArray[i][j].meanAbsoluteError

                        datapointsArray[i][j] = len(objArray[i][j].ratioArray)

                m = stat.mean(M[i])

                meanofmeans.append(m)

                meandatapoints = stat.mean(datapointsArray[i])

                labels.append(meandatapoints)

                #print(f"the mean of the mean relative errors for Athresh {a} is {m}")

        k = np.linspace(0.5, 3, 6)
        num_plots = len(k)
        
        # Create subplots
        rows = 2
        columns = 3
        fig, axs = plt.subplots(rows, columns, figsize=(12, 8))  
        
        # get colormap for the subplots
        cmap = plt.get_cmap('viridis')
        # normalize label values to use in colormap
        #labelArray = np.array(labels)
        norm = Normalize(vmin=min(labels), vmax=max(labels))
        # create a scalar-mappable object
        #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #sm.set_array([])

        for idx, ax in enumerate(axs.flat):  # Flatten 2D array of axes so iteration works (I was getting an error before when I iterated)
            if idx < num_plots:  # Only plot for valid indices
                weightfunction = []
                for i in range(len(A)):
                    weight = labels[i] / meanofmeans[i]**(1 / k[idx])
                    weightfunction.append(weight)
                
                ax.scatter(A, weightfunction, c=labels, cmap = cmap)
                
                #for i in range(len(A)):
                #    ax.annotate(round(labels[i],2),(A[i], weightfunction[i]))

                ax.set_xlabel("A")
                ax.set_ylabel("W(k,A)")
                ax.set_title(f"Weight = {k[idx]:.1f}")
            else:
                ax.set_visible(False)  # Hide unused axes
        
        # create the colorbar
        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(smap, ax=axs, location = "right", orientation = "vertical", pad=0.1, fraction=0.1, shrink = 0.8)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_ylabel('# data points', rotation=90, labelpad = 15, fontdict = {"size":10})   

        # fix spacing between graphs and confine them to a rectangle so the colorbar can fit
        plt.tight_layout(rect=(0,0,0.8,1))  

        # save figure to file with user input values in filename
        plt.savefig(f'windowedfigure-{startValue}-{endValue}-{n}.png')

        plt.show()

    
    @staticmethod
    def windowedGraphMeanOfMeans(directory: str, startValue: float, endValue: float, n: int) -> None:
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
                        M[i][j] = objArray[i][j].meanAbsoluteError

                        datapointsArray[i][j] = len(objArray[i][j].ratioArray)

                m = stat.mean(M[i])

                meanofmeans.append(m)

                meandatapoints = stat.mean(datapointsArray[i])

                labels.append(meandatapoints)

                #print(f"the mean of the mean relative errors for Athresh {a} is {m}")

        k = np.linspace(0.5, 3, 6)
        num_plots = len(k)
        
        # Create subplots
        rows = 2
        columns = 3
        fig, axs = plt.subplots(rows, columns, figsize=(12, 8))  
        
        for idx, ax in enumerate(axs.flat):  # Flatten 2D array of axes so iteration works (I was getting an error before when I iterated)
            if idx < num_plots:  # Only plot for valid indices
                weightfunction = []
                for i in range(len(A)):
                    weight = labels[i] / meanofmeans[i]**(1 / k[idx])
                    weightfunction.append(weight)
                
                ax.plot(A, weightfunction)
                for i in range(len(A)):
                    ax.text(A[i], weightfunction[i], f"{labels[i]}", fontsize=8)
                ax.set_xlabel("A")
                ax.set_ylabel("W(k,A)")
                ax.set_title(f"Weight = {k[idx]:.1f}")

            else:
                ax.set_visible(False)  # Hide unused axes
        
        plt.tight_layout()  # fix spacing between graphs (btw you can also manually adjust this when you print the graphs)
        plt.show()


    def windowedMedianFilter(self, list: list, windowSize: int) -> list:
        #idk if it matters but do we want to add the zeros to the end or the beginning (rn i'm adding the zeros to the end)
        #Also, I didn't pad them with zeros because Idk how the window size would affect how many zeros to pad the list with so I'm holding off on that
        windowedMedianList = []
        modZero = True
        if len(list) % windowSize != 0:
            modZero = False
        while modZero != 0:
            list.append(0)
        numWindows = len(list)/windowSize
        startValue = 0
        for i in range(numWindows): 
            newArray = list[startValue: startValue + windowSize]
            med = abs(stat.median(newArray))
            windowedMedianList.append(med)
        return windowedMedianList
    
    def subtractedSignalFilter(self, list: list, windowedMedianList: list, windowSize: int) -> list:
        windowedMedianList = AudioFile.windowedMedianFilter(list, windowSize)
        subtractedSignal = []
        if len(list) == len(windowedMedianList):
            for i in range(len(list)):
                subtractedSignal.append(list[i] - windowedMedianList[i])
        return subtractedSignal
    
    def corellation(self, x: NDArray) -> NDArray:
        X = np.rfft(x)  # Compute FFT of signal x (clarify what x is here)
        autocorrelation = np.irfft(np.conj(X) * X)  # Compute autocorrelation using inverse FFT
        #The functional relationships go like this:
            #x ~ voltage (amplitude of digitized signal) vs time
            # X ~ voltage vs frequency
            # autocorrelation ~ voltage^2 vs *time lag*
        return autocorrelation

    def hanningWindow(self) -> NDArray:
        H = np.hanning(len(self.source))
        return self.source*H

############################################################################
# END AUDIOFILE CLASS
############################################################################
