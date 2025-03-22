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
import re
from collections import defaultdict
import memspectrum
import crepe

NDArray = np.ndarray[Any, np.dtype[np.float64]]

class AudioFileProminence:

    def __init__(self, file: str):

        ##############################################
        # attributes of the audiofile object
        ##############################################
        # load wav file as array and store sample rate
        self.source, self.sr =  lib.load(file, sr=None) 

        self.source = self.hanningWindow()

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
        '''
        M = memspectrum.MESA()
        M.solve(self.source)
        #M.spectrum(1/self.sr, self.freq)
        plt.plot(self.freq, M.spectrum(1/self.sr, self.freq))
        plt.show()
        '''
        
        #signal, sr = lib.load(file, sr=16000)  # Resample to 16 kHz for CREPE
        crepefreq = set()
        # Analyze pitch with CREPE
        time, frequency, confidence, activation = crepe.predict(self.source, self.sr, viterbi=True)
        for i in range(len(confidence)):
            if confidence[i] >= 0.9 and frequency[i] > 0:
                crepefreq.add(round(frequency[i]))
        
        #crepe.close()

        # identify fundamental as freq of largest coefficient in the power spectrum 
        # -- this can lead to erroneous results, we need to be careful in how we define this!
        if len(crepefreq)>0:
            self.dummyfundamental: int = stat.median(crepefreq)
        else:
            self.dummyfundamental = 50

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

    def getfund(self) -> float:
        R = self.N/self.sr

        loFThresh = 200*R
        hiFthresh = 1000*R

        signal = AudioFileProminence.filtersignal(self.magspec,loFthresh=loFThresh, hiFthresh=hiFthresh, Athresh=75)
        peaks, properties = sci.signal.find_peaks(signal)

        return peaks[0]/R

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
            if self.freq[i] >= 220:
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

        # filtered = self.filter(self.dummyfundamental*R-20, hiFthresh=len(self.magspec), Athresh=0)
        
        filtered = self.filter(100*R, hiFthresh=len(self.magspec), Athresh=0)
        
        peaks, peakproperties = sci.signal.find_peaks(filtered, height = 2, prominence = height, distance = self.dummyfundamental*R//8)

        #peaks, peakproperties = sci.signal.find_peaks(filtered, height = 0.5, prominence = height, distance = 3)

        return peaks
    
    @staticmethod
    # static version of the above method for finding peaks in a given array 
    # with a certain prominence that is above lowest percentile of prominences
    def staticfindpeaks(array: NDArray, percentile: float, height: float = None, distance: float = None) -> NDArray:
        if height == None:
            height = 0
        if distance == None:
            distance = 1

        #peaks, peakproperties = sci.signal.find_peaks(array, height = height, distance = distance)

        peaks = sci.signal.find_peaks(array, height = height, distance = distance)[0]

        if peaks.shape[0] == 0:
            meanProminence = 0
        else:
            meanProminence = stat.mean(sci.signal.peak_prominences(array, peaks)[0])

        percentile = percentile/100*meanProminence
        
        peaks, peakproperties = sci.signal.find_peaks(array, height = height, prominence = percentile, distance = distance)

        return peaks
    

    def windowedPeaks(self, numberFundamentalsInWindow: int, percentile: float) -> NDArray:
        # number of suspected harmonics we want to be present in our window (integer multiple of fundamental freq)
        numberFundamentalsInWindow = int(numberFundamentalsInWindow)

        # ratio to convert from Hz to bins
        R = self.N/self.sr

        # fundamental freq in bins
        fund = self.dummyfundamental*R

        loPass = 16*self.dummyfundamental*R

        # hiPass the magnitude spectrum to cut room noise
        signal = AudioFileProminence.filtersignal(self.magspec, fund-50, loPass, 0)

        # initial minimal index for our window
        minIndex = round(fund/2)

        # total number of windows we will consider
        numberWindows = round((loPass-fund)/(fund*numberFundamentalsInWindow))

        # initialize an empty array of peaks to be populated later
        peaks = np.array([], dtype=int)

        for i in range(numberWindows):

            # width of the window that will slide through the signal to ID peaks
            windowWidth = round(numberFundamentalsInWindow*fund)

            # window indices in bins
            window = self.bins[minIndex + i*windowWidth: minIndex + (i+1)*windowWidth]

            tempPeaks = AudioFileProminence.staticfindpeaks(signal[window], percentile=10, height=0, distance=fund//10)
            
            tempPeakHeights = signal[tempPeaks + minIndex + i*windowWidth]

            # sort the indices of the peaks from shortest to tallest
            tempPeakHeightIndices = np.argsort(tempPeakHeights)
            # rearrange the peak heights in ascending order
            tempPeakHeights = tempPeakHeights[tempPeakHeightIndices]

            # set the threshold to the percentile/100 * (shortest suspected harmonic spike in window)
            threshold = (percentile/100)*tempPeakHeights[len(tempPeaks)-numberFundamentalsInWindow]

            tempPeaks = AudioFileProminence.staticfindpeaks(signal[window], percentile=1, height=threshold, distance=fund//6)

            windowedRatioArray = (tempPeaks + minIndex + i*windowWidth)/fund

            E = round(stat.mean(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)
            StD = round(stat.stdev(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)
            
            '''
            fig, ax = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(8)
            ax.plot(window/R, signal[window])
            ax.scatter((tempPeaks + minIndex + i*windowWidth)/R, self.magspec[tempPeaks + minIndex + i*windowWidth], c='orange')
            ax.set_title(f'windowed peaks of {self.file} with width {numberFundamentalsInWindow}*{round(self.dummyfundamental)}')
            plt.text(x=0.75,y=0.9, s=f'threshold = {round(threshold,2)}', transform=ax.transAxes)
            plt.text(x=0.75,y=0.85, s=f'# peaks found = {len(tempPeaks)}', transform=ax.transAxes)
            plt.text(x=0.75,y=0.8, s=f'Err = {E}',transform=ax.transAxes)
            plt.show()
            '''

            peaks = np.concatenate((peaks,tempPeaks + minIndex + i*windowWidth))

        '''
        for i in range(numberWindows-1):
            maxIndex = round(minIndex*(i+1) + windowWidth*fund)
            
            print(maxIndex)

            window = self.magspec[minIndex*(i+1)-20:maxIndex+20]

            tempPeaks = AudioFileProminence.staticfindpeaks(window, percentile=5, distance=fund//8)
            
            tempPeakHeights = self.magspec[tempPeaks]
            tempPeakHeightIndices = np.argsort(tempPeakHeights)
            tempPeakHeights = tempPeakHeights[tempPeakHeightIndices]

            threshold = percentile/100*tempPeaks[len(tempPeaks)-windowWidth-1]

            tempPeaks = AudioFileProminence.staticfindpeaks(window, percentile=1, height=threshold, distance=fund//8)

            peaks = np.concatenate((peaks,tempPeaks))
        '''
            
        return peaks
            

    @staticmethod
    def moving_average(array: NDArray, width: int = 3):
                pad = np.zeros(width//2)

                array = np.concatenate((pad,array))
                array = np.concatenate((array,pad))

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
        #plt.clf()
        plt.show()


    def graph_magspec_withWindowedPeaks(self, percentile: float, numberFundamentalsInWindow: int = 5) -> None:
        windowedPeaks = self.windowedPeaks(percentile=percentile, numberFundamentalsInWindow=numberFundamentalsInWindow)
        
        peakHeight = np.zeros(len(windowedPeaks))
        
        for i in range(len(windowedPeaks)):
            peakHeight[i] = self.magspec[windowedPeaks[i]]

        R = self.sr/self.N

        plt.figure(figsize=(8,8))

        plt.plot(self.freq, self.magspec)
        plt.scatter(windowedPeaks*R,peakHeight,c='orange',s=12)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('Magnitude of RFFT')
        plt.title(f'Magnitude spectrum of {self.file}, window width {numberFundamentalsInWindow}, percentile {percentile}')
        plt.savefig(f'windowpeaks-{percentile}perc-{self.file}.png')
        #plt.clf()
        plt.show()

    def graph_filtersignal_withPeaks(self, percentile: float, loFthresh: float, hiFthresh: float, Athresh: float) -> None:
        filtered = self.filter(loFthresh, hiFthresh, Athresh)
        
        peaks = AudioFileProminence.staticfindpeaks(filtered, percentile=percentile, height=Athresh, distance=self.dummyfundamental//8)

        peakHeight = np.zeros(len(peaks))

        
        for i in range(len(peaks)):
            peakHeight[i] = filtered[peaks[i]]

        R = self.sr/self.N

        plt.plot(self.freq, filtered)
        plt.scatter(peaks*R,peakHeight,c='orange',s=12)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('Magnitude of RFFT')
        plt.title(f'Filtered magnitude spectrum of {self.file} with peaks, p = {percentile}, A = {Athresh}')
        plt.savefig(f'filterpeaks-{percentile}perc-{Athresh}A-{self.file}.png')
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


    @staticmethod
    def printAggregateError(directory: str, numberOfFundamentalsInWindow: int, percentile: float, badData: list = None, SpecificType: str = None) -> None:
        nameArray = AudiofilesArray(Path(directory))

        if SpecificType != None:
            namelist = nameArray.getSpecificType(SpecificType)
        else:
            namelist = nameArray.getSpecificType("1S")
            print("No additional type information was given (e.g. 1S, 2S, 2S9, 2SC, etc.) so default of 1S was used.")

        # initialize an empty |audiofiles| array to be populated with audiofile arrays
        objArray = np.empty(len(namelist), dtype=AudioFileProminence)
        for i in range(len(namelist)):
            objArray[i] = AudioFileProminence(namelist[i])

        # initialize an empty |audiofiles| array to be populated with the meanerror of each audiofile
        M = np.empty(len(namelist))

        meanofmeans = list()
        datapointsArray = list()
        fundamentals = np.empty(len(namelist))

        open(f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt", "w").close()
        
        nonIntegers = list()

        for i in range(len(objArray)):
            fundamentals[i] = round(objArray[i].dummyfundamental)

            R = objArray[i].N/objArray[i].sr
            fund = objArray[i].dummyfundamental*R
            
            windowedPeaks = objArray[i].windowedPeaks(numberOfFundamentalsInWindow, percentile)

            windowedRatioArray = windowedPeaks/fund

            counter = 0

            if badData != None:
                DA = DataAnalysis(windowedRatioArray)
                for value in badData:
                    counter = counter + DA.checkIfDecimalClose(decimal= value, roundingPlace=1)
                windowedRatioArray = DA.array
            

            E = round(stat.mean(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)

            StD = round(stat.stdev(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)

            M[i] = E

            datapointsArray.append(len(windowedRatioArray))



            #print(f'{objArray[i].file}, mean error = {M[i]}, # datapoints = {datapointsArray[i]}, # removed = {counter}')
            DA = DataAnalysis(windowedRatioArray)
            #DA.checkDataTextFile(sampleValue=0.2, fileName=f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt")
            
            #nonIntegers.extend(DA.checkData(sampleValue=0.2))

            with open(f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt", "a") as f:
                f.write(f'{objArray[i].file}, fundamental = {round(objArray[i].dummyfundamental)}, mean error = {M[i]}, # datapoints = {datapointsArray[i]}, # removed = {counter}\n')
                #f.write(f'{nonIntegers}\n')
            DA.checkDataTextFile(sampleValue=0.2, fileName=f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt")

        m = stat.mean(M)

        with open(f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt", "a") as f:
            f.write(f'mean of mean absolute errors = {m}\n')
            #f.write(f'{nonIntegers}')
            f.write(f'{AudioFileProminence.roundEntries(fileName=f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt", roundingValue=1)}')
            


        #plt.scatter(fundamentals, np.round(M,2))
        #plt.show()

            #meanofmeans.append(m)

            #meandatapoints = stat.mean(datapointsArray[i])

            #labels.append(meandatapoints)

            #print(f"the mean of the mean relative errors for Athresh {a} is {m}")


    @staticmethod
    def roundEntries(fileName : str, roundingValue : int) -> list:
        all_entries = AudioFileProminence.analyzeTextFile(fileName)
        #roundedEntries = round(all_entries, roundingValue)
        duplicates = {}
        for i in range(len(all_entries)):
            all_entries[i] = round(all_entries[i], roundingValue)
        for i in all_entries:
            if all_entries.count(i) >= 1:
                duplicates[i] = all_entries.count(i)
                #duplicates.append([f'{i} , number of times: {all_entries.count(i)}'])
                #for j in all_entries:
                #    if j==i:
                #        all_entries.remove(j)

        sorted_keys = sorted(duplicates.keys())
        sorted_duplicates = {key:duplicates[key] for key in sorted_keys}

        return sorted_duplicates
        '''
        for i in all_entries:
            roundedEntry = round(i, roundingValue)
            roundedEntries.append(roundedEntry)
        for i in range(len(roundedEntries)):
            for j in (range(i + 1, len(roundedEntries))):
                if roundedEntries[i] == roundedEntries[j]:
                    duplicates.append(roundedEntries[i])

        '''

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
    
    @staticmethod
    def analyzeTextFile(file_name):
        file_entries = []
        current_entries = []
        with open(file_name, "r") as file:
            lines = file.readlines()
        
        current_file = None
        for line in lines:
            # Check if line starts with a file name (e.g., 1SCD01.wav)
            match = re.match(r"^([\w\d]+\.wav),", line)
            if match:
                if current_entries:
                    file_entries.extend(current_entries)
                current_file = match.group(1)
                current_entries = []
            elif current_file:
                entry_match = re.search(r'Entry: "([\d.]+)"', line)
                if entry_match:
                    entry_value = entry_match.group(1)
                    current_entries.append(entry_value)
        if current_entries:
            file_entries.extend(current_entries)

        float_list = [float(x) for x in file_entries]
        
        return float_list

    @staticmethod
    def findDuplicatesInEntryList(fileName : str, equalityThreshold : float, roundMeanValue : int) -> None:
        all_entries = AudioFileProminence.analyzeTextFile(fileName)
        listOfDuplicates = []
        #bigList = []

        #for entries in all_entries:
            #file_name = entries[0] 
            #listOfDuplicates = []
            

            #numeric_entries = entries[1]
            #for sublist in numeric_entries:
                #bigList.extend(sublist)

            #print(bigList)


            #listOfDuplicates = []
        for i in range(len(all_entries)):
            miniDuplicates = []
            try:
                entry_i = float(all_entries[i])
            except ValueError:
                continue  

            for j in range(i+1, len(all_entries)):
                try:
                    entry_j = float(all_entries[j])
                except ValueError:
                    continue 
                if abs(entry_i - entry_j) <= equalityThreshold:
                    if entry_j not in listOfDuplicates:
                        miniDuplicates.append(entry_j)
                    if entry_i not in listOfDuplicates:
                        miniDuplicates.append(entry_i)

            #calculate mean of duplicates -> this is my way of deciding which specific value we choose to be the repeating value because we're rounding
            if len(miniDuplicates) !=0 :
                meanOfDuplicates = round(stat.mean(miniDuplicates), roundMeanValue)
                listOfDuplicates.append([f'Repeated value: {meanOfDuplicates} | how many times it shows up: {len(miniDuplicates)}'])
                #mean, numberOfDuplicateEntries = listOfDuplicates[file_name]
                #print(f'Duplicate value = {mean} | Number of duplicates = {numberOfDuplicateEntries}')
        open(f"listOfDuplicates for {fileName}", "w")
        for i in (listOfDuplicates):
            with open(f"listOfDuplicates for {fileName}", "a") as f:
                f.write(f'{i}\n')
        f.close
        print(listOfDuplicates)

    @staticmethod
    def findRepeats(list, sampleValue):
        da = DataAnalysis(list)
        listOfRepeats = da.checkData(sampleValue)
        print(da.findDuplicates(listOfRepeats))
        
'''
commented out because I did not finish and test these methods


    @staticmethod
    def interpolate(point1: tuple, point2: tuple) -> float:
        slope = (point2[1] - point1[1])/(point2[0] - point1[0])

        root = point1[0] - point1[1]/slope

        return root

    @staticmethod
    def find_roots(array: NDArray) -> list:
        roots = []

        i = 0

        while i<len(array):
            if array[i] == 0:
                roots.append(i)
            elif array[i]>0:
                for j in range(i+1,len(array)):
                    if array[j]==0:
                        roots.append(j)
                    elif array[j]<0:
                        roots.append(AudioFileProminence.interpolate((j-1,array[j-1]),(j,array[j])))
                        i=j
            elif array[i]<0:
                for j in range(i+1, len(array)):
                    if array[j]>0:
                        roots.append(AudioFileProminence.interpolate((j-1,array[j-1]),(j,array[j])))
                        i=j
            i = i+1

        return roots

    # method for returning the roots of a sum of cotangents of the form cot(Lj*k) 
    # where Lj is string length and k is the desired harmonic.  Takes each string 
    # length (in m), the largest desired k (optional, default is 20), and the samplerate 
    # (optional, default is 10e5) as arguments.  Returns a list of the approximate roots.  
    @staticmethod
    def find_cotangent_roots(L1: float, L2: float, L3: float, kmax: float = None, samplerate: int = None) -> NDArray:
        if kmax==None:
            kmax = 20
        
        if samplerate==None:
            samplerate = 100000
        
        k = np.linspace(0,kmax,samplerate)

        c1 = np.cos(L1*k)
        c2 = np.cos(L2*k)
        c3 = np.cos(L3*k)
        s1 = np.sin(L1*k)
        s2 = np.sin(L2*k)
        s3 = np.sin(L3*k)

        cotangent_sum_numerator = c1*s2*s3 + s1*c2*s3 + s1*s2*c3

        roots = np.array(AudioFileProminence.find_roots(cotangent_sum_numerator))

        return roots*kmax/samplerate
'''



############################################################################
# END AUDIOFILE CLASS
############################################################################
