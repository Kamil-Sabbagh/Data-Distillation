import pandas as pd
import os
#import matplotlib.pyplot as plt

# List of paths to your CSV files
file_paths = [
    "./ipc1/class_accuracies_ConvNet.csv",
    "./ipc5/class_accuracies_ConvNet.csv",
    "./ipc10/class_accuracies_ConvNet.csv",
    "./ipc15/class_accuracies_ConvNet.csv",
    "./ipc20/class_accuracies_ConvNet.csv",
    "./ipc25/class_accuracies_ConvNet.csv",
    "./ipc30/class_accuracies_ConvNet.csv"
]

data = []

# Iterate through each file path
for path in file_paths:
    # Read the CSV file into a DataFrame
    print(path)
    print(os.path.exists(path))
    df = pd.read_csv(path)
    
    # Calculate the average of each row and append to averages list
    avg = df.mean(axis=1).tolist()
    data.append(avg)

    # Print the average of each row for the current CSV file
    print(f"{path} average is:")
    print(avg)
    print("-----")

# # Plot the averages
# for idx, avg in enumerate(averages):
#     plt.plot(avg, label=f'File {idx + 1}')

# plt.legend()
# plt.title("Average of each row for each file")
# plt.xlabel("Row Index")
# plt.ylabel("Average Value")
# plt.show()
