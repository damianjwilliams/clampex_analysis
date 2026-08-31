from tkinter import Tk, filedialog
import pandas as pd

# Open file dialog
root = Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename(title="Select a .sta File")
# Load data
if file_path:
    df = pd.read_csv(file_path, sep="\t", skiprows=11)

    # Compute the average of columns 2 to 5 (assuming 0-based index)
    average_values = df.iloc[:, 1:5].mean()

    # Print the results
    print(file_path)
    print("Average values of columns 2 to 5:")
    print(average_values)

    
else:
    print("No file selected.")