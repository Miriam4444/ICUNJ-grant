from StringAnalysis import StringAnalysis
import math

class TestStringAnalysis:

    if __name__ == "__main__":
        #analysis = StringAnalysis(length1= 1, length2= math.pi, length3= math.e, harmonicsList=[.71, 1.08, 1.6, 2.12, 2.52, 3.03, 3.39, 3.84])
        analysis = StringAnalysis(length1= 1, length2= math.pi, length3= math.e, harmonicsList=[1,2.007,3,4,5,6,7,8])
        #print(analysis.checkIfCotangentSum(maxDifference= .01, minInterval= 0, maxInterval= 4, numPoints= 10000))
        print(analysis.checkIfStrings(.01))

