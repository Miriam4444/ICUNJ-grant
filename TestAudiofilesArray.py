from audiofile import audiofile
from AudiofilesArray import AudiofilesArray
from DataAnalysis import DataAnalysis
from pathlib import Path
import os
from collections import Counter

if __name__ == "__main__":
        #DirectoryName = Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples")
        nameArray = AudiofilesArray(Path(r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\ICUNJ-grant-audiofiles"))
        #print(nameArray.makeFilePathList())
        namelist = nameArray.getSpecificType("2S")

        #file_path = r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\2S q 11-22-24.wav"
        #file = audiofile(file_path)
        #file.printfundamental

        print("*************************************************************************")

        objlist = list()
        for i in range(len(namelist)):
            #print(namelist[i])
            #objlist.append(audiofile(namelist[i]))
            #print(objlist[i])
            file = audiofile(namelist[i])

            file.printfundamental()
            dataSet = DataAnalysis(file.ratioArray)
            dataSet.checkData(8)
            print("")

        
        
        
