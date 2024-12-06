from ICUNJ_ffs_analysis import ICUNJ_ffs_analysis

class Test:
    def run_analysis(self):
        # Specify the file path
        file_path = r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\2S q 11-22-24.wav"
        
        # Create an instance of the class
        analysis = ICUNJ_ffs_analysis(file_path)
        
        # Run the file processing without passing file_path
        analysis.file_directions()

# Running the Test class
if __name__ == "__main__":
    test_instance = Test()  # Create an instance of the Test class
    test_instance.run_analysis()  # Run the analysis