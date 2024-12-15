import numpy as np
import matplotlib.pyplot as plt
import librosa as lib
import scipy as sci
import statistics

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
    
    # static version of the above.  This will take in an array and an integer samplerate
    # and return 2*(array index corresp to maximum magnitude array value) 
    @staticmethod
    def staticgetfund(array, samplerate):
        A = np.abs(array)
        F = 2*np.argmax(A[:samplerate//2 + 1])
        return F
    

    # static method for finding harmonics.  Basically the idea will be to find the fundamental of the input
    # array, then determine an appropriate window size to hunt for the next harmonic spike, which we'll find
    # by using the getfund method on the next window.  We'll find numberofharmonics number of these spikes.
    @staticmethod
    def harmonicfinder(array, samplerate, numberofharmonics):
        # determine the fundamental of the input signal
        f = audiofile.staticgetfund(array,samplerate)

        # create an array to store the harmonics, initialized with zeros
        H = np.zeros(numberofharmonics)

        # set 0th harmonic array entry to the fundamental frequency
        H[0] = f

        # initialize a window of the array with zeros, length equal to the fundamental f
        windowedArray = np.zeros(f)
        
            # filter the signal to the appropriate window size --- we may want a static method that does this for us
            # the filtering will be done in such a way that we consider a frequency window that is f wide centered at i*f
            # so if the signal's indices look like ______f_______2f_______3f_______4f_______ then our windows should look like this
            #                                      ______f___|--------|___3f_______4f_______ and will ID the max in |--------| then
            # move on to                           ______f_______2f___|--------|___4f_______ and ID the max there, and so on

        for i in range(2,numberofharmonics):
            # determine minimum frequency in the window
            window_min = i*f - f//2

            # populate filteredArray with array values living in the window
            for k in range(0,f-1):
                windowedArray[k] = array[window_min + k + 1]

            # ID the spike in this window
            spike = audiofile.staticgetfund(windowedArray,samplerate)

            # convert the spike location back to frequencies for the original signal
            #spike = spike + window_min

            # set the (i-1)st element of the harmonic array to the fundamental in this window
            H[i-1] = spike + window_min

        return H
            

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
    
    # method to compute a windowed median of the signal.  It will compute the median in each
    # window of size windowsize and put each median value in an array, then average the median
    # array values and return that average.  Currently detects high and low pass filters
    # and applies this windowed median procedure to the nonzero part of the signal
    @staticmethod
    def staticwindowedmedian(array,windowsize):
        # detect initial and trailing zeros
        lopass, hipass = audiofile.detectzeros(array)

        # find length of nonzero part of signal
        signallength = len(array) - lopass - hipass

        # initialize an array of 0s of the correct length
        M = np.zeros(signallength//windowsize) 

        for i in range(0,len(M)):
            # populate array M with median of each window
            M[i] = statistics.median(array[lopass + i*windowsize:(i+1)*windowsize-1])

        return statistics.mean(M), statistics.stdev(M)

    # static method that takes in an array and determines if there are leading or trailing
    # zeros.  IDs a high- or low-passed signal.  Returns # of zeroes at beginning and end
    @staticmethod
    def detectzeros(array):
        initialzeros = 0
        trailingzeros = 0

        for i in range(len(array)):
            if array[i] != 0:
                break
            initialzeros = i-1
        
        for i in range(len(array)):
            if array[len(array)-(i+1)] != 0:
                break
            trailingzeros = i

        return initialzeros, trailingzeros

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

    # function to plot the converted dbv data vs freq3.  Not sure if this currently works.
    # We should decide what minimum dbv value we want to consider.  
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
############################################################################
# END AUDIOFILE CLASS
############################################################################



############################################################################
# test the class methods
############################################################################

# initialize the test signal
test = audiofile(r"C:\Users\spine\Downloads\2S q 11-22-24.wav")

#test.graph_dbv()
#test.printall()
#test.graph_original()
#audiofile.graph_filtersignal(F, 100, 10)
# peaks = np.array([f])
# pw = sci.signal.peak_widths(F, peaks)
# print(pw)

#####################
#                   #
# THIS ONE IS COOL! #
#  ______________   #
#  \\  \    /  //   #
#   \\  \  /  //    #
#    \\  \/  //     # 
#     \\    //      #
#      \\  //       #
#       \\//        #
#        \/         #
#                   #
#####################
# This is a function from scipy that finds peaks in data (exactly what we want!) and you can
# specify that the peaks must be of a certain width of samples.  Here I'm using 5-60 samples 
# as my width, and I've already filtered the signal to have the algorithm work less 
# (high pass above 100 Hz and trimmed everything under magnitude 10 in the PSD)
# then printed 2*peakindex/fundamental and it gives us some shockingly promising data!

# Filter original signal to cut out freq below 100 Hz and magnitudes below 8
# We should have a good reason for trimming the magnitudes we trim.
# For instance, we should either cut every single signal at the same noise floor
# or we should have a reasonable computation for the noise floor for each sample 
# since they'll presumably vary in loudness so we might lose info from a uniform
# noise floor, though they should all be *pretty* close in loudness...

# pspec's indices are bins, not Hz so I'm inputting what Hz I want to filter out
# and dividing by 2.  We could put this into the definition of our function instead
F = audiofile.filtersignal(test.pspec,loFthresh=200//2,hiFthresh=10000//2,Athresh=8)

# when we decide on our standardized filtering thresholds, we can write all these static methods as 
# class methods and then just have the important values stored as class parameters

# Get fundamental of original signal
f = audiofile.staticgetfund(test.pspec, test.sr)

# scipy method designed to find peaks of an array.  We should write this whole thing as a method
# in our class.  Nothing crazy, just something that automatically calculates this array of peaks
# for each class instance and outputs the array converted to Hz as I've done below
cwt_peaks = sci.signal.find_peaks_cwt(F, widths=np.arange(5, 60))

# print the array of Hz where signal peaks
print(2*cwt_peaks/f)

# what is the mean of the bandpassed PSD? other statistical data?
print("the median of the psd is ", statistics.median(F), 
    "the mean of the psd is ", statistics.mean(F), 
    " and the stdev is ", statistics.stdev(F),
    "and the windowed median is ", audiofile.staticwindowedmedian(F,f//2))


# right now we are seeing big residuals near 258 Hz because it was SO loud, and those residuals are
# dominating any later, much quieter, spectrum frequencies... I think we want a LOCAL fundamental-finding function!
# Meaning one that identifies fundamentals in little windows throughout the bins.  This seems complicated but
# should be doable.  Some kind of sliding filter which applies the getfund method in the filter window.  
# We need a static version of getfund to do this...
#for i in range(0,len(F)):
#    if F[i]>0:
#        print(2*i, F[i])

#H = audiofile.harmonicfinder(F,test.sr,10)
#print(H)   

#F1 = audiofile.staticdbvconvert(F)

#plt.plot(np.arange(len(F1)), F1)
#plt.show()
