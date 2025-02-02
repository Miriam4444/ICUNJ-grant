from audiofile import AudioFile as af
from audiofileprominence import AudioFileProminence as afp
from AudiofilesArray import AudiofilesArray
from DataAnalysis import DataAnalysis
from pathlib import Path
import os
from collections import Counter
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import memspectrum

if __name__ == "__main__":
        DirectoryName = Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples\3-string")
        nameArray = AudiofilesArray(DirectoryName)
        #print(nameArray.makeFilePathList())
        namelist = nameArray.getSpecificType("3TL")
        namelist2 = nameArray.getSpecificType("2S9RCf")

        print("*************************************************************************")

        optimalAthresh = 3.2

        audiofileList = []
        '''
        sample = af(namelist[8])

        peakHeight = np.zeros(len(sample.peaks))
        
        for i in range(len(sample.peaks)):
            peakHeight[i] = sample.pspec[sample.peaks[i]]

        Athresh = stat.mean(peakHeight)

        sampleAthreshed = af(namelist[8],Athresh=Athresh)


        print(sample.fundamental*(sample.sr/sample.N), sample.dummyfundamental)
        print(sampleAthreshed.fundamental*(sampleAthreshed.sr/sampleAthreshed.N), sampleAthreshed.dummyfundamental)

        
        sample.graph_PSD_withPeaks()

        sample.graph_filtersignal_withPeaks()

        sample.graphRatioArray()
        '''

        #afp.graphWeightFunctionProminence(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples\longer clips", 
        #                                  n=20, SpecificType="1S")

        p = 12



        #afp.printAggregateError(directory=DirectoryName, numberOfFundamentalsInWindow=5, percentile=25, SpecificType='2S')
        #print(afp.roundEntries(fileName="AggError-1S-5-80.txt", roundingValue=2))

        #afp.findDuplicatesInEntryList("AggError-1S-5-80.txt", 0.1, 2)
        
        S = afp(namelist[10])

        S.graph_magspec()

        
        #for filename in namelist:
        #        audiofileList.append(afp(filename))

        

        '''
        S = audiofileList[5]

        windowedPeaks = S.windowedPeaks(numberFundamentalsInWindow=5,percentile=80)
        peaks = S.findpeaks(percentile=15)

        R = S.N/S.sr
        fund = S.dummyfundamental*R

        windowedRatioArray = windowedPeaks/fund

        afp.findRepeats(windowedRatioArray,0.1)
        '''


        

        #for i in range(20,23):
        #        S = audiofileList[i]
        #        S.graph_magspec_withPeaks(percentile = p)
        

        '''

        windowedPeaks = S.windowedPeaks(numberFundamentalsInWindow=5,percentile=80)
        peaks = S.findpeaks(percentile=15)

        R = S.N/S.sr
        fund = S.dummyfundamental*R

        windowedRatioArray = windowedPeaks/fund

        E = round(stat.mean(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)
        StD = round(stat.stdev(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)

        plt.figure(figsize=(8,6))

        plt.subplot(2,1,1)
        plt.plot(S.freq,S.magspec)
        plt.scatter(peaks*S.sr/S.N,S.magspec[peaks], c='orange')
        plt.title(f'{S.file} unwindowed peaks with (Err,StD) = {(round(S.findAbsoluteError(percentile = 15)[0],3),round(S.findAbsoluteError(percentile = 15)[1],3))}')

        plt.subplot(2,1,2)
        plt.plot(S.freq,S.magspec)
        plt.scatter(windowedPeaks*S.sr/S.N,S.magspec[windowedPeaks], c='orange')
        plt.title(f'{S.file} windowed peaks with (Err,StD) = {E,StD}')

        plt.tight_layout()
        plt.show()
        '''
        '''
        m = memspectrum.MESA()
        m.solve(af.filtersignal(audiofileList[3].source, S.dummyfundamental-20, S.N, 0), method = 'Standard', optimisation_method = 'VM')
        f_mesa, psd_mesa = m.spectrum(dt = 1./audiofileList[3].sr, onesided = True)

        mesa_peaks = afp.staticfindpeaks(psd_mesa, percentile=5)



        plt.subplot(2,1,1)
        plt.plot(f_mesa,psd_mesa)
        plt.scatter(mesa_peaks*S.sr/S.N, psd_mesa[mesa_peaks], c='orange')
        
        plt.subplot(2,1,2)
        plt.plot(S.freq,S.magspec)
        plt.show()

        print(mesa_peaks*S.sr/S.N/S.dummyfundamental)
        '''
        
        #for sample in audiofileList:
        #        sample.graphRatioArray(percentile=p)
        #        sample.graph_magspec_withPeaks(percentile=p)


        #audiofileList[2].graph_magspec_withPeaks(p)

        #audiofileList[2].graph_magspec_withPeaks(percentile=25)
        #audiofileList[2].graphRatioArray(percentile=p)



        #audiofileList[3].graph_PSD_withPeaks(percentile=25)

        #plt.plot(sample.freq,sample.pspec)
        #plt.show()


        '''
        def moving_average(a, n=3):
                ret = np.cumsum(a, dtype=float)
                ret[n:] = ret[n:] - ret[:-n]
                return ret[n - 1:] / n

        width = 20

        averaged = moving_average(sample.pspec,n=width)

        averagedFiltered = af.filtersignal(averaged, sample.fundamental-20, 20*sample.fundamental+200,0)

        lowerBound = int(5)
        upperBound = int(30)
        peaks = sci.signal.find_peaks_cwt(averagedFiltered, widths=np.arange(lowerBound, upperBound))

        peakHeight = np.zeros(len(peaks))
        
        for i in range(len(peaks)):
            peakHeight[i] = averaged[peaks[i]]

        R = sample.sr/sample.N

        Athresh = stat.mean(peakHeight)

        averagedFiltered = af.filtersignal(averagedFiltered, sample.fundamental-20, 20*sample.fundamental+200,Athresh=Athresh)


        filteredAveraged = moving_average(sample.filtered,width)

        peaks2 = sci.signal.find_peaks_cwt(filteredAveraged, widths=np.arange(lowerBound, upperBound))

        peakHeight2 = np.zeros(len(peaks2))
        
        for i in range(len(peaks2)):
            peakHeight2[i] = filteredAveraged[peaks2[i]]

        R = sample.sr/sample.N

        Athresh2 = stat.mean(peakHeight2)

        filteredAveraged = af.filtersignal(filteredAveraged, sample.fundamental-20, 20*sample.fundamental+200,Athresh=Athresh2)


        fig, ax = plt.subplots(2,1)

        ax[0].plot(sample.freq[width//2 -1 : -width//2], averagedFiltered)
        ax[0].scatter(peaks*R,peakHeight,c='orange',s=12)
        ax[1].plot(sample.freq[width//2 -1 : -width//2], filteredAveraged)
        ax[1].scatter(peaks2*R,peakHeight2,c='orange',s=12)
        fig.suptitle(f'Moving Avg of mag. spectrum of {sample.file} with peaks')
        plt.show()
        '''

        #plt.specgram(af(namelist[1]).source)
        #plt.show()
       

        '''        
        sample1T = af(namelist[3],0)
        print(sample1T.fundamental*(sample1T.sr/sample1T.N), sample1T.dummyfundamental, sample1T.N)        
        sample1S = af(namelist2[3],0)
        print(sample1S.fundamental*(sample1S.sr/sample1S.N), sample1S.dummyfundamental, sample1S.N)

        plt.plot(np.arange(len(af.crosscorrelation(sample1S.pspec,sample1T.pspec))), af.crosscorrelation(sample1S.pspec,sample1T.pspec))
        plt.show()
        '''




        """

        ax[0].plot(sample.freq,sample.pspec)

        ax[1].plot(sample.freq[width//2:-width//2+1],averaged)
        plt.show()

        #FF = np.fft.rfft(sample.pspec)

        #plt.plot(np.arange(len(FF)), np.abs(FF))

        #FF_loPass = af.filtersignal(array=FF, loFthresh=0, hiFthresh=500,Athresh=0)

        #plt.plot(np.arange(len(FF)), np.abs(FF_loPass))

        #plt.show()


        #F_smoothed = np.fft.irfft(FF_loPass)

        #plt.plot(sample.freq[:68906],np.abs(F_smoothed))

        #plt.show()
        #plt.plot(sample.bins,sample.filtered)
        """
       
        #audiofileList[0].graphRatioArray()

        #af.graphWeightFunction(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples\longer clips", 
        #                    startValue=0, endValue=10, n=10, SpecificType="1S")
        
        #af.windowedGraphMeanOfMeans(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples",0,8,4)

        '''

        # initialize an array of 10 evenly spaced Athresh values between 0 and 5
        A = np.linspace(0.0, 8.0, 9)

        # initialize an empty |A| x |audiofiles| array to be populated with audiofile arrays
        objArray = np.empty(shape=(len(A),len(namelist)), dtype=af)

        # initialize an empty |A| x |audiofiles| array to be populated with the meanerror of each audiofile
        M = np.empty(shape=(len(A),len(namelist)))

        meanofmeans = list()
        datapointsArray = np.empty(shape=(len(A),len(namelist)))
        labels = list()

        for i in range(len(A)):
                a = A[i]

                # populate the ith row of the array of audiofiles with samples corresponding to threshold a
                for j in range(len(namelist)):

                        f = af(namelist[j], a)

                        objArray[i][j] = f

                        # populate row a = A[i] with the relativeMeanErrors for the samples
                        M[i][j] = objArray[i][j].meanAbsoluteError

                        datapointsArray[i][j] = len(objArray[i][j].ratioArray)

                m = stat.mean(M[i])

                meanofmeans.append(m)

                meandatapoints = round(stat.mean(datapointsArray[i]),1)

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



        

        
                if j<3:
                        ax1[j].plot(A,weightfunction)
                        ax1[j].set_title(f"weight = {k[j]}")
                        ax1[j].set_ylabel("W(k,A)")
                        ax1[j].set_xlabel("A")
                
                else:
                        ax2[j-3].plot(A,weightfunction)
                        ax2[j-3].set_title(f"weight = {k[j]}")
                        ax2[j-3].set_ylabel("W(k,A)")
                        ax2[j-3].set_xlabel("A")

        
        fig, ax = plt.subplots()

        fig.set_figheight(8)
        fig.set_figwidth(8)

        ax.plot(A,meanofmeans)
        
        ax.scatter(A,meanofmeans)

        for i in range(len(labels)):
                ax.text(A[i]+0.05, meanofmeans[i]+0.0025, labels[i])

        ax.set_xlabel("A")
        ax.set_ylabel("Mean of mean absolute error")
        ax.set_title("Mean of mean abs. error vs A")
        plt.savefig('mean of mean abs error.png')
        plt.show()
        
        
        # initialize amplitude threshold
        Athresh = 2

        objlist = list()

        for i in range(len(namelist)):
                objlist.append(af.AudioFile(namelist[i],Athresh))
        
        meanlist = list()
        for i in range(len(objlist)):
                # create a list of the mean relative errors for each AudioFile object
                # meanlist.append(objlist[i].meanRelativeError)

                print("")
                
                objlist[i].printError()


        # meanofmeans = stat.mean(meanlist)

                #dataSet = DataAnalysis(objlist[i].ratioArray)

                

                #dataSet.checkData()
        '''
        

        

