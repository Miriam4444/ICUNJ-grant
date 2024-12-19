import audiofile as af
from AudiofilesArray import AudiofilesArray
from DataAnalysis import DataAnalysis
from pathlib import Path
import os
from collections import Counter
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
        #DirectoryName = Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples")
        nameArray = AudiofilesArray(Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples"))
        #print(nameArray.makeFilePathList())
        namelist = nameArray.getSpecificType("1S")

        print("*************************************************************************")

        

        

        # initialize an array of 10 evenly spaced Athresh values between 0 and 5
        A = np.linspace(0.0, 8.0, 11)

        # initialize an empty |A| x |audiofiles| array to be populated with audiofile arrays
        objArray = np.empty(shape=(len(A),len(namelist)), dtype=af.AudioFile)

        # initialize an empty |A| x |audiofiles| array to be populated with the meanerror of each audiofile
        M = np.empty(shape=(len(A),len(namelist)))

        meanofmeans = list()
        datapointsArray = np.empty(shape=(len(A),len(namelist)))
        labels = list()

        for i in range(len(A)):
                a = A[i]

                # populate the ith row of the array of audiofiles with samples corresponding to threshold a
                for j in range(len(namelist)):

                        f = af.AudioFile(namelist[j], a)

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



        '''

        
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

        ax.plot(A,meanofmeans)
        
        ax.scatter(A,meanofmeans)

        for i in range(len(labels)):
                ax.annotate(labels[i], (A[i], meanofmeans[i]))

        ax.set_xlabel("Athresh")
        ax.set_ylabel("Mean of means")
        ax.set_title("Mean of mean rel. error vs Athresh")
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
        

        

