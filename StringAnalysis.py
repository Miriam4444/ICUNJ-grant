import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import librosa as lib
import scipy as sci
import statistics as stat
import os
from DataAnalysis import DataAnalysis
from AudiofilesArray import AudiofilesArray
from pathlib import Path
from collections import Counter
from typing import Any
import re
from collections import defaultdict
from typing import Callable
from scipy.optimize import root_scalar
from DataAnalysis import DataAnalysis
import math

FunctionType = Callable[[float], float]
# take in string lengths
# take in list of harmonics 
# go through list of harmonics and for each find the zeros of cotangent sum

class StringAnalysis:
    def __init__(self, length1: float, length2: float, length3: float, harmonicsList: list):
        self.length1 = length1
        self.length2 = length2
        self.length3 = length3
        self.harmonicsList = harmonicsList

    def cotangentNumerator(self, x: float) -> float:
        return (np.cos(self.length1*x) * np.sin(self.length2 * x) * np.sin(self.length3 * x)) + (np.cos(self.length2*x) * np.sin(self.length1 * x) * np.sin(self.length3 * x)) + (np.cos(self.length3*x) * np.sin(self.length1 * x) * np.sin(self.length2 * x))
    
    def cotangentDenominator(self, x: float) -> float:
        return (np.sin(self.length1 * x) * np.sin(self.length2 * x) * np.sin(self.length3 * x))

    #returns a list of the string lengths
    def lengthList(self) -> list:
        lengthList = [self.length1, self.length2, self.length3]
        return lengthList

    def findZeros(self, minInterval : float, maxInterval : float, numPoints : int, function: float) -> list:
        #define the interval that it's looking for zeros on
        a, b = minInterval, maxInterval  # we're going to have a problem with singularities here but thats a future me problem
        #defines the amount of points it's going to be looking at
        x_values = np.linspace(a, b, numPoints)
        #make a list to hold the roots
        roots = []

        # Scan interval for sign changes
        for i in range(len(x_values) - 1):
            #now we're going to look for sign changes by looking at two values next to each other and multiplying them. If they're negative there was a sign change
            if function(x_values[i]) * function(x_values[i + 1]) < 0:  # detect sign change
                # Apply root-finding method within the bracket
                interval = [x_values[i], x_values[i + 1]]
                result = root_scalar(function, method='brentq', bracket=interval)
                roots.append(result.root)

        return roots
    
    def removeSingularities(self, roots1 : list, roots2 : list) -> list:
        #roots1 is the numerator zeros
        #roots2 is the denominator zeros
        #basically im going to look at all the zeros in the denominator and if its in the numerator im going to take it out of the numerator
        noSingularities = roots1.copy()
        for denominatorRoot in roots2:
            for numeratorRoot in roots1:
                if denominatorRoot == numeratorRoot:
                    noSingularities.remove(numeratorRoot)
        return noSingularities
    
    #this method is for testing if there are three individual strings
    def checkIfCotangentSum(self, maxDifference : float, minInterval : float , maxInterval : float, numPoints : int) -> tuple[list , list]:
        #this is the list that's going to be returned
        closeHarmonics = []
        rootsFound = []
        roots = self.removeSingularities(roots1= self.findZeros(minInterval= minInterval, maxInterval= maxInterval, numPoints= numPoints, function= self.cotangentNumerator), roots2=self.findZeros(minInterval= minInterval, maxInterval= maxInterval, numPoints= numPoints, function= self.cotangentDenominator))
        print(roots)
        for testFundamental in self.harmonicsList:
            fundamental = testFundamental
            fundamentalRootsList = []
            fundamentalRootsList.append([fundamental])
            fundamentalHarmonicsList = []
            fundamentalHarmonicsList.append([fundamental])
            #now we're going to iterate through the roots and for each root we're going to divide all of the other roots by it and see if it's close to an integer
            fund = roots[0]
            for i in range(len(roots)):
                roots[i] = roots[i]/fund
            #print(roots)
            for root1 in roots:
                for harmonic in self.harmonicsList:
                    harmonic = harmonic/fundamental
                    #listOfValidRoots = []
                    difference = np.abs(harmonic - root1)
                    #check if the remainder is close to an int
                    if difference <= maxDifference:
                        #listOfValidRoots.append([harmonic])
                        validRoot = root1
                        fundamentalRootsList.append(harmonic)
                        #closeHarmonics.append([root1, listOfValidRoots])
                        fundamentalHarmonicsList.append(validRoot)
            rootsFound.append(fundamentalRootsList)
            closeHarmonics.append(fundamentalHarmonicsList)
        #close harmonics are from the calculated cotangent sum, they have a lot of decimals
        #roots found are from the self.harmonics list aka the one input in the arguments
        print("roots are " , roots)
        return closeHarmonics, rootsFound

    
    def checkIfSeparateStrings(self, maxDifference : float, numOfHarmonics : int) -> list:
        #numOfHarmonics is the amount of harmonics we're going to find
        #this is the list that's going to be returned
        closeHarmonics = []
        for string in self.lengthList():
            stringHarmonics = []
            roots = []
            for i in range(numOfHarmonics - 2):
                root = ((i+1) * math.pi)/string
                roots.append(root)
            for root1 in roots:
                for harmonic in self.harmonicsList:
                    listOfValidRoots = []
                    remainder = harmonic/root1
                    #check if the remainder is close to an int
                    if isinstance(DataAnalysis.staticCheckIfClose(number=remainder, maxDifference = maxDifference), int): 
                        listOfValidRoots.append([harmonic])
                if listOfValidRoots != None:
                    stringHarmonics.append([((root1 * string)/math.pi), listOfValidRoots])
            closeHarmonics.append(stringHarmonics)
        return closeHarmonics

    
    #this method is going to test if its just 3 single strings by returning a list of the harmonics and fundamental
    #this method tests each string individually but checkIfSeparateStrings checks all three at once
    #I totally overthought this method and wasted three hours on nonsense el oh el
    """
    def checkIfStrings(self, maxDifference : float) -> list:
        strings = self.lengthList()
        for string in strings:
            listOfStringHarmonics = []
            for harmonic1 in self.harmonicsList:
                listOfMults = []
                fundamental = harmonic1
                for harmonic2 in self.harmonicsList:
                    possibleMult = harmonic2/harmonic1
                    if isinstance(DataAnalysis.staticCheckIfClose(number= possibleMult, maxDifference= maxDifference), int):
                        listOfMults.append(harmonic2)
                listOfStringHarmonics.append([harmonic1, listOfMults])
        return listOfStringHarmonics
    """

    #tests if it's a string
    def checkIfStrings(self, maxDifference : float) -> list:
        listOfStringHarmonics = []
        for harmonic1 in self.harmonicsList:
            listOfMults = []
            fundamental = harmonic1
            for harmonic2 in self.harmonicsList:
                possibleMult = harmonic2/fundamental
                if isinstance(DataAnalysis.staticCheckIfClose(number= possibleMult, maxDifference= maxDifference), int) and fundamental <= harmonic2:
                    listOfMults.append(harmonic2)
            listOfStringHarmonics.append([harmonic1, listOfMults])
        return listOfStringHarmonics




