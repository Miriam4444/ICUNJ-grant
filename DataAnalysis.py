import math

class DataAnalysis:

    def __init__(self, array):
        self.checkIfInteger(array)

    def firstTry(self, array):
        #take in array and check if they're all close to integers
        self.valid_data = True
        self.problem = None
        for i in range(len(array)):
            #while self.valid_data == True:
            if array[i] == i+1:
                self.valid_data =True
            else:
                self.valid_data = False
                self.problem = f'The Problem is at entry #{i + 1}. Entry: "{array[i]}"'
                break

        if self.valid_data == True:
            print("Data is nonrepeating integer multiples")
        else:
            print(self.problem)

    def checkNearestInteger(self, number):
        self.closest_number = None
        self.max_difference = .1
        #max_difference is how far away from an integer multiple we're considering to be a good value
        self.ceiling = math.ceil(number)
        self.floor = math.floor(number)
        self.ceiling_difference = abs(self.ceiling - number)
        self.floor_difference = abs(self.floor - number)
        #now we're gonna check if the nearest integer is greater than or less than the number and set closest_number to that value
        if self.floor_difference <= self.ceiling_difference:
            self.closest_number = self.floor_difference
        else:
            self.closest_number = self.ceiling_difference
        #if the difference between the closest number and the number is less than the max difference that we set then we're going to set that number to the closest integer
        if abs(self.closest_number - number) < self.max_difference:
            number = self.closest_number
        else:
            number = number
        return number


    def checkIfInteger(self, array):
        self.valid_data = True
        #go through the array and check if each value is an integer
        for i in range(len(array)):
            #if the entry is an integer do the following
            entry = self.checkNearestInteger(array[i])
            if isinstance(array[i], int):
                self.valid_data = True
            else:
                self.valid_data = False
                break
        if self.valid_data == True:
            print("array is integer multiples")
        else:
            print(f'The Problem is at entry #{i + 1}. Entry: "{array[i]}"')

    
