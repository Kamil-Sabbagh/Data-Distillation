import pandas as pd
import numpy as np
import os
import argparse

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

# Set up argument parser
parser = argparse.ArgumentParser(description='Process CSV files and output statistics.')
parser.add_argument('--std_threshold', type=float, default=2, help='Standard deviation threshold for outlier detection')

# Parse arguments
args = parser.parse_args()
std_dev_threshold = args.std_threshold

# Initialize lists to hold IPC numbers and accuracy data
ipc_numbers = []
accuracy_lists = []

# Iterate through each file path
for path in file_paths:
    # Extract IPC number from the file path
    ipc_number = int(path.split('/')[-2].replace('ipc', ''))
    ipc_numbers.append(ipc_number)

    # Check if the path exists
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    # Read the CSV file into a DataFrame, using the first row as the header
    df = pd.read_csv(path)

    # Swap 'Model' data with 'truck' data
    df['Model'], df['truck'] = df['truck'], df['Model']
    
    # Now drop the 'Model' column as it's no longer needed
    df = df.drop(columns=['Model'])
    
    # Calculate the mean and standard deviation for each class
    means = df.mean()
    std_devs = df.std()

    # Identify outliers for each class
    outliers = (np.abs(df - means) > std_dev_threshold * std_devs)

    # Filter out the outliers
    filtered_df = df[~outliers.any(axis=1)]
    num_outliers_discarded = len(df) - len(filtered_df)

    # Calculate the average mean for each class after filtering
    filtered_means = filtered_df.mean()

    # Append the filtered means to the accuracy lists
    accuracy_lists.append(filtered_means.tolist())

    # Print the number of outliers discarded
    print(f"{path} - Outliers discarded: {num_outliers_discarded}")
    
    # Print the average mean for each class
    print(f"{path} - Average mean per class after filtering outliers:")
    print(filtered_means)
    print("-----")

# Output the lists in a format suitable for the visualization script
print(f"ipc_numbers = {ipc_numbers}")
print("accuracy_lists = [")
for acc_list in accuracy_lists:
    print(f"    {acc_list},")
print("]")
