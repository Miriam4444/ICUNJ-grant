import math

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

    def checkIntegerMultiples(self):
        badData = 0
        for i, value in enumerate(self.array):
            closest = self.checkIfClose(value)
            if not isinstance(closest, int):  # Not an integer multiple
                print(f'The Problem is at entry #{i + 1}. Entry: "{value}"')
                badData += 1

        if badData == 0:
            print("Array is integer multiples.")
        else:
            print(f"There are {badData} entries that aren't integer multiples of the fundamental.")
