import math
from collections import Counter


class DataAnalysis:
    def __init__(self, array):
        self.array = array
        self.valid_data = True
        self.max_difference = .1 #max_difference is how far away from an integer multiple we're considering to be a good value

    def checkIfClose(self, number):
        closest_int = round(number)
        if abs(closest_int - number) <= self.max_difference:
            number = closest_int
        return number
    
    def findDuplicates(self, array):
        counts = Counter(array)
        duplicates = 0
        entryNumber = 0
        for value, count in counts.items():
            entryNumber += 1
            if count > 1:
                duplicates += count - 1  # Add excess occurrences as duplicates
                print((f'There\'s a duplicate at entry #{entryNumber}. Entry: "{value}"'))
        return duplicates

    def checkData(self):
        #Function checks if data is integer multiples and if there are no duplicates
        badData = 0
        closest_values = []
        for i, value in enumerate(self.array):
            closest = self.checkIfClose(value)
            closest_values.append(closest)
            if not isinstance(closest, int):  # Not an integer multiple
                print(f'There\'s a non-integer multiple at entry #{i + 1}. Entry: "{value}"')
                badData += 1
        duplicates = self.findDuplicates(closest_values)
        if duplicates > 0:
            #print(f"Duplicates: {duplicates}")
            badData += duplicates
        else:
            print("No duplicates in converted values.")


        if badData == 0:
            print("Array is integer multiples with no repeating values.")
        else:
            print(f"There are {badData} entries that either aren't integer multiples of the fundamental or are duplicates.")
