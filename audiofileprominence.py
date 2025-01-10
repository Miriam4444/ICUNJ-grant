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

class AudioFileProminence:

    def __init__(self, file: str):

        ##############################################
        # attributes of the audiofile object
        ##############################################
        # load wav file as array and store sample rate
        self.source, self.sr =  lib.load(file, sr=None) 

        # file name without path
        self.file: str = os.path.basename(file)

        # compute and store real fft of audio array
        self.fourier: NDArray = np.fft.rfft(self.source)

        # store number of samples in original file
        self.N: int = len(self.source)

        # store time array for original signal
        self.time: NDArray = np.arange(self.N)/self.sr
        self.lengthinseconds: float = self.N/self.sr

        # store original sample's frequency bins
        self.bins: NDArray = np.arange(len(self.fourier))

        # compute the magnitude spectrum of the rfft
        self.magspec: NDArray = np.abs(self.fourier)

        # compute the corresponding frequencies of the rfft in Hz
        # first uses the built-in rfft frequency calculator from numpy
        # third calculates by hand accounting for the rfft cutting the frequency bins in half
        self.freq: NDArray = np.fft.rfftfreq(len(self.source), 1/self.sr)

        # identify fundamental as freq of largest coefficient in the power spectrum 
        # -- this can lead to erroneous results, we need to be careful in how we define this!
        self.dummyfundamental: int = self.getfund2()

        self.unfilteredpeaks, self.peakproperties = sci.signal.find_peaks(self.magspec, height = 2, distance = self.dummyfundamental*self.N/self.sr//4)
        self.prominences = sci.signal.peak_prominences(self.magspec, self.unfilteredpeaks)
        self.meanProminence = stat.mean(self.prominences[0])
        

    ##########################################
    # methods
    ##########################################
    def printN(self) -> None:
        print("the length (in samples) of the original file is", self.N)

    def printbins(self) -> None:
        print("the number of frequency bins in the RFFT is", self.bins)

    def printmagspec(self) -> None:
        print("the magnitude spectrum of the RFFT is", self.magspec)

    def printfreq(self) -> None:
        print("the frequencies in Hz of the magnitude spectrum are", self.freq)

    def printfundamental(self) -> None:
        print("the fundamental frequency of the signal is", self.dummyfundamental)

    def printall(self, percentile: float) -> None:
        self.printN()
        self.printfreq()
        self.printfundamental()
        self.printpeaks()
        self.printratios(percentile)
        self.printError(percentile)

    def printpeaks(self) -> None:
        print(f"the unfiltered peaks of {self.file} are", self.unfilteredpeaks) 

    def printratios(self, percentile: float) -> None:
        print(f"the ratio array of {self.file} is\n", self.findratioArray(percentile=percentile))

    def printError(self, percentile: float) -> None:
        mean, stdev = self.findAbsoluteError(percentile=percentile)
        print(f"{self.file} has mean error {mean}\n and stdev of error {stdev}\n from {len(self.ratioArray)} datapoints")

    # function to identify frequency of largest magnitude entry in magspec.  
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
            if self.freq[i] >= 200:
                min = i
                break

        # find first index where the frequency exceeds 600
        for j in range(min,len(self.freq)-1):
            if self.freq[j] >= 500:
                max = j
                break

        # search for loudest frequency only between 200 and 300 Hz.  Will return relative to min=0.
        F = np.argmax(self.magspec[min:max])

        # convert magspec index back to Hz
        F = self.freq[F+min]

        return F
    
    # static version of the above.  This will take in an array and an integer samplerate
    # and return 2*(array index corresp to maximum magnitude array value) 
    @staticmethod
    def staticgetfund(array: NDArray, samplerate: int) -> int:
        A = np.abs(array)
        F = (samplerate/len(array))*np.argmax(A[:samplerate//2 + 1])
        return F


    # filter the magspec data to cut out everything below a certain loudness (magthresh)
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


    # class method for amplitude thresholding the bandpassed signal.
    # Athresh is set to mean+stdev of the windowed median with windowsize fundamental//2
    def filter(self, loFthresh, hiFthresh, Athresh) -> NDArray:
        return AudioFileProminence.filtersignal(self.magspec,
                                      loFthresh,
                                      hiFthresh,
                                      Athresh)
    
    # class method for finding peaks in the magspec of our signal with a certain prominence that is above lowest percentile of prominences
    def findpeaks(self, percentile: float) -> NDArray:
        height = percentile/100*self.meanProminence

        R = self.N/self.sr

        filtered = self.filter(self.dummyfundamental*R-20, hiFthresh=len(self.magspec), Athresh=0)
        
        peaks, peakproperties = sci.signal.find_peaks(filtered, height = 2, prominence = height, distance = self.dummyfundamental*R//4)

        return peaks
    
    @staticmethod
    # static version of the above method for finding peaks in a given array 
    # with a certain prominence that is above lowest percentile of prominences
    def staticfindpeaks(array: NDArray, percentile: float, height: float = None, distance: float = None) -> NDArray:
        if height == None:
            height = 0
        if distance == None:
            distance = 0

        peaks, peakproperties = sci.signal.find_peaks(array, height = height, distance = distance)

        meanProminence = stat.mean(sci.signal.peak_prominences(array, peaks)[0])

        percentile = percentile/100*meanProminence
        
        peaks, peakproperties = sci.signal.find_peaks(array, height = height, prominence = percentile, distance = distance)

        return peaks

    @staticmethod
    def moving_average(array: NDArray, width: int = 3):
                ret = np.cumsum(array, dtype=float)
                ret[width:] = ret[width:] - ret[:-width]
                return ret[width - 1:] / width
    
    # class method to smooth the magnitude spectrum with a moving average
    def smoothMagSpec(self, width):
        Z = np.zeros(width//2)
        self.magspec = np.concatenate((Z,self.magspec))
        self.magspec = np.concatenate((self.magspec,Z[:-1]))
        self.magspec = AudioFileProminence.moving_average(self.magspec, width=width)

    # class method to find the array of ratios: peak frequency/fundamental frequency
    def findratioArray(self, percentile: float) -> NDArray:

        P = np.zeros(len(self.findpeaks(percentile=percentile)))

        for i in range(len(P)):
            P[i] = self.freq[self.findpeaks(percentile=percentile)[i]]

        fund = self.dummyfundamental

        P = P/fund

        return P

    def findAbsoluteErrorArray(self,percentile: float) -> NDArray:
        # create an array of the correct length
        E = np.zeros(len(self.findratioArray(percentile=percentile)))

        for i in range(len(self.findratioArray(percentile=percentile))):
            E[i] = np.abs(self.findratioArray(percentile=percentile) - np.rint(self.findratioArray(percentile=percentile)))[i]

        return E

    # returns the mean and stdev of the absolute error array corresponding to input percentile of prominences
    def findAbsoluteError(self, percentile: float) -> tuple[float,float]:
        E = self.findAbsoluteErrorArray(percentile=percentile)
        return stat.mean(E), stat.stdev(E)

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

    # function to plot the raw rfft data
    def graph_original(self) -> None:
        plt.plot(self.bins, self.fourier)
        plt.xlabel('entry number')
        plt.ylabel('RFFT coefficient')
        plt.title('graph of string pluck')
        plt.show()

    # function to plot the magspec data versus original bins
    def graph_magspec(self) -> None:
        plt.plot(self.freq, self.magspec)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('Magnitude of RFFT')
        plt.title(f'Magnitude spectrum of {self.file}')
        plt.show()

    def graph_magspec_withPeaks(self, percentile: float) -> None:
        peakHeight = np.zeros(len(self.findpeaks(percentile=percentile)))
        
        for i in range(len(self.findpeaks(percentile=percentile))):
            peakHeight[i] = self.magspec[self.findpeaks(percentile=percentile)[i]]

        R = self.sr/self.N

        plt.figure(figsize=(8,8))

        plt.plot(self.freq, self.magspec)
        plt.scatter(self.findpeaks(percentile=percentile)*R,peakHeight,c='orange',s=12)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('Magnitude of RFFT')
        plt.title(f'Magnitude spectrum of {self.file} with peaks')
        plt.savefig(f'peaks-{percentile}perc-{self.file}.png')
        plt.clf()
        #plt.show()

    def graph_filtersignal_withPeaks(self, percentile: float, loFthresh: float, hiFthresh: float, Athresh: float) -> None:
        peakHeight = np.zeros(len(self.findpeaks(percentile=percentile)))

        filtered = self.filter(loFthresh, hiFthresh, Athresh)
        
        for i in range(len(self.peaks)):
            peakHeight[i] = filtered[self.findpeaks(percentile=percentile)[i]]

        R = self.sr/self.N

        plt.plot(self.freq, filtered)
        plt.scatter(self.findpeaks(percentile=percentile)*R,peakHeight,c='orange',s=12)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('Magnitude of RFFT')
        plt.title(f'Filtered magnitude spectrum of {self.file} with peaks')
        plt.show()

    # function to plot the magspec data versus original bins
    @staticmethod
    def graph_filtersignal(array: NDArray, Fthresh: float, Athresh: float) -> None:
        F = AudioFileProminence.filtersignal(array,Fthresh,Athresh)
        plt.plot(np.arange(len(array)), F)
        plt.xlabel('entry number')
        plt.ylabel('signal')
        plt.title('graph of filtered signal')
        plt.show()

    @staticmethod
    def graphWeightFunction(directory: str, startValue: float, endValue: float, n: int, SpecificType: str = None) -> None:
        #DirectoryName = Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples")
        #directory = r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\ICUNJ-grant-audiofiles"
        nameArray = AudiofilesArray(Path(directory))
        #print(nameArray.makeFilePathList())

        if SpecificType != None:
            namelist = nameArray.getSpecificType(SpecificType)
        else:
            namelist = nameArray.getSpecificType("1S")
            print("No additional type information was given (e.g. 1S, 2S, 2S9, 2SC, etc.) so default of 1S was used.")

        # initialize an array of n-1 evenly spaced Athresh values between startValue and endValue
        A = np.linspace(startValue, endValue, n)

        # initialize an empty |A| x |audiofiles| array to be populated with audiofile arrays
        objArray = np.empty(shape=(len(A),len(namelist)), dtype=AudioFileProminence)

        # initialize an empty |A| x |audiofiles| array to be populated with the meanerror of each audiofile
        M = np.empty(shape=(len(A),len(namelist)))

        meanofmeans = list()
        datapointsArray = np.empty(shape=(len(A),len(namelist)))
        labels = list()

        for i in range(len(A)):
                a = A[i]

                # populate the ith row of the array of audiofiles with samples corresponding to threshold a
                for j in range(len(namelist)):

                        f = AudioFileProminence(namelist[j], a)

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
                
                ax.scatter(A, weightfunction, s=10, c=labels, cmap = cmap)
                
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
        plt.savefig(f'windowedfigure-{SpecificType}-{startValue}-{endValue}-{n}.png')

        plt.show()


    @staticmethod
    def graphWeightFunctionProminence(directory: str, n: int, SpecificType: str = None) -> None:
        #DirectoryName = Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples")
        #directory = r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\ICUNJ-grant-audiofiles"
        nameArray = AudiofilesArray(Path(directory))
        #print(nameArray.makeFilePathList())

        startValue = 10
        endValue = 100

        if SpecificType != None:
            namelist = nameArray.getSpecificType(SpecificType)
        else:
            namelist = nameArray.getSpecificType("1S")
            print("No additional type information was given (e.g. 1S, 2S, 2S9, 2SC, etc.) so default of 1S was used.")

        # initialize an empty |audiofiles| array to be populated with audiofile arrays
        objArray = np.empty(len(namelist), dtype=AudioFileProminence)
        for i in range(len(namelist)):
            objArray[i] = AudioFileProminence(namelist[i])

        # initialize an array of n evenly spaced Athresh values between startValue and endValue
        A = np.linspace(startValue, endValue, n+1)

        # initialize an empty |A| x |audiofiles| array to be populated with the meanerror of each audiofile
        M = np.empty(shape=(len(A),len(namelist)))

        meanofmeans = list()
        datapointsArray = np.empty(shape=(len(A),len(namelist)))
        labels = list()

        for i in range(len(A)):
                a = A[i]

                # populate the ith row of the array of audiofiles with samples corresponding to threshold a
                for j in range(len(namelist)):
                        objArray[j].peaks = objArray[j].findpeaks(round(a))

                        # populate row a = A[i] with the relativeMeanErrors for the samples
                        M[i][j] = objArray[j].findAbsoluteError(percentile = a)[0]

                        datapointsArray[i][j] = len(objArray[j].findratioArray(percentile=a))

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
                
                ax.scatter(A, weightfunction, s=10, c=labels, cmap = cmap)
                
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
        plt.savefig(f'weightfunctionfigure-{SpecificType}-{startValue}-{endValue}-{n}.png')

        plt.show()


    # method to plot the actual harmonic ratio array of the signal against the predicted ratio array
    # also saves the figure to a file with all relevant info in the file name
    def graphRatioArray(self, percentile: float) -> None:
        idealRatioArray = np.rint(self.findratioArray(percentile=percentile))

        # the following is logic to parse the error array into positive and negative errors
        # so that the error bars can be plotted appropriately on the figure
        errorArray = idealRatioArray - self.findratioArray(percentile)

        positiveErrors = np.ones(len(idealRatioArray))
        negativeErrors = np.zeros(len(idealRatioArray))

        for i in range(len(idealRatioArray)):
            if errorArray[i] < 0:
                positiveErrors[i] = 0
                negativeErrors[i] = -1

        yerr = [errorArray, errorArray]

        yerr[1] = yerr[0]*positiveErrors
        yerr[0] = yerr[1]*negativeErrors

        # plot the ideal ratio array

        plt.figure(figsize=(8,8))
        
        plt.plot(idealRatioArray,idealRatioArray, label='theoretical')

        # plot the mean error for this sample in the bottom right 
        plt.text(np.max(idealRatioArray)-0.01, 1, f'mean abs. error = {round(self.findAbsoluteError(percentile=percentile)[0],3)}\n # datapoints = {len(self.findratioArray(percentile=percentile))}', ha='right', va='bottom')

        #plt.scatter(idealRatioArray,self.ratioArray, label='actual',c='orange')

        # plot the actual ratio array values including error bars
        plt.errorbar(idealRatioArray, self.findratioArray(percentile), yerr=yerr,
                     label='actual', c='orange', marker='d', markersize=6, 
                     linestyle='dotted', capsize=2)
        plt.xticks(idealRatioArray)
        plt.xlabel('harmonic number')
        plt.ylabel('harmonic ratio')
        plt.title(f'Actual vs Th. harmonic ratios - {self.file}, P={percentile}% prominence threshold')
        plt.legend()
        plt.savefig(f'ratioarray-prom{percentile}-{self.file}.png')

        # clears the figure to avoid overlays from successive iterations
        plt.clf()
        #plt.show()
    
    
    def subtractedSignalFilter(self, list: list, windowedMedianList: list, windowSize: int) -> list:
        windowedMedianList = AudioFileProminence.windowedMedianFilter(list, windowSize)
        subtractedSignal = []
        if len(list) == len(windowedMedianList):
            for i in range(len(list)):
                subtractedSignal.append(list[i] - windowedMedianList[i])
        return subtractedSignal
    
    def autocorrelation(self) -> NDArray:
        autocorrelation = np.fft.irfft(np.conj(self.fourier) * self.fourier)  # Compute autocorrelation using inverse FFT
        #The functional relationships go like this:
            #x ~ voltage (amplitude of digitized signal) vs time
            # X ~ voltage vs frequency
            # autocorrelation ~ voltage^2 vs *time lag*
        return autocorrelation

    @staticmethod
    def crosscorrelation(arr1: NDArray, arr2: NDArray) -> NDArray:
        if len(arr1)==len(arr2):
            F1 = np.fft.rfft(arr1)
            F2 = np.fft.rfft(arr2)
            return np.fft.irfft(np.conj(F1) * F2)
        
        elif len(arr1) > len(arr2):
            Z = np.zeros(len(arr1)-len(arr2))
            arr2 = np.concatenate((arr2,Z))
            F1 = np.fft.rfft(arr1)
            F2 = np.fft.rfft(arr2)
            return np.fft.irfft(np.conj(F1) * F2)

        else:
            Z = np.zeros(len(arr2)-len(arr1))
            arr1 = np.concatenate((arr1,Z))
            F1 = np.fft.rfft(arr1)
            F2 = np.fft.rfft(arr2)
            return np.fft.irfft(np.conj(F1) * F2)

    def hanningWindow(self) -> NDArray:
        H = np.hanning(len(self.source))
        return self.source*H

############################################################################
# END AUDIOFILE CLASS
############################################################################
