import audiofile as AF
from AudiofilesArray import AudiofilesArray
from DataAnalysis import DataAnalysis
from pathlib import Path
import os
from collections import Counter

if __name__ == "__main__":
        #DirectoryName = Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples")
        nameArray = AudiofilesArray(Path(r"C:\Users\spine\OneDrive\Documents\Math\Research\Quantum Graphs\ICUNJ grant 2024-25\samples"))
        #print(nameArray.makeFilePathList())
        namelist = nameArray.getSpecificType("2S")

        print("*************************************************************************")

        objlist = list()
        for i in range(len(namelist)):
                objlist.append(AF.audiofile(namelist[i]))

                objlist[i].printfundamental()
                dataSet = DataAnalysis(objlist[i].ratioArray)

                print("")

                dataSet.checkData()
        
        

