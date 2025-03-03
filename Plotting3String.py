#from audiofile import AudioFile
from AudiofileProminence import AudioFileProminence as afp
from AudiofilesArray import AudiofilesArray
from pathlib import Path

class Plotting3String:

    if __name__ == "__main__":
        directoryName = Path(r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\__pycache__\__pycache__")
        nameArray = AudiofilesArray(directoryName)
        nameList = nameArray.getSpecificType("3TLA10")
        print(len(nameList)) #216 files

        #iterate through each element of nameList or do one at a time
        sample = afp(nameList[0])

        sample.graph_magspec()
        


