from StringAnalysis import StringAnalysis
from ThreeStringHarmonicsLists import ThreeStringHarmonicsLists as TSHL
import math

class TestStringAnalysis:

    if __name__ == "__main__":
        #analysis = StringAnalysis(length1= 1, length2= math.pi, length3= math.e, harmonicsList=[.71, 1.08, 1.6, 2.12, 2.52, 3.03, 3.39, 3.84])
        #analysis = StringAnalysis(length1= 1, length2= math.pi, length3= math.e, harmonicsList=[1,2.007,3,4,5,6,7,8])
        length1 = 19.9
        length2= 25.5
        length3= 34.2
        analysis = StringAnalysis(length1, length2, length3, TSHL.string3TLA10C02())
        print(analysis.checkIfCotangentSum(maxDifference= .01, minInterval= 0, maxInterval= 7000, numPoints= 100000)[0])
        print(analysis.checkIfCotangentSum(maxDifference= .01, minInterval= 0, maxInterval= 7000, numPoints= 100000)[1])
        
        for entry in analysis.checkIfStrings(.01):
            print(entry)

