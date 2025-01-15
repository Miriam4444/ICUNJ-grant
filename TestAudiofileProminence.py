from AudiofileProminence import AudioFileProminence as afp

if __name__ == "__main__":
    afp.printAggregateError(directory= r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\ICUNJ-grant-audiofiles", numberOfFundamentalsInWindow= 10, percentile= .8, SpecificType="1SRC")
    #afp.findDuplicatesInEntryList("AggError-1SRC-10-0.8.txt" , .5, 2)
    #print(afp.analyzeTextFile("AggError-1SRC-10-0.8.txt"))
    afp.findDuplicatesInEntryList("AggError-1SRC-10-0.8.txt", .1, 2)
