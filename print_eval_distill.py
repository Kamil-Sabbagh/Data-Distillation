import pandas as pd
import numpy as np
import os

# Define the class names if they are not present in the CSV file
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

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

# Define a threshold for detecting outliers
std_dev_threshold = 2  # for example, any data point more than 2 standard deviations

# Iterate through each file path
for path in file_paths:
    # Check if the path exists
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    # Read the CSV file into a DataFrame, assuming no header is present
    df = pd.read_csv(path, header=None, names=class_names)
    
    # Calculate the mean and standard deviation for each class
    means = df.mean()
    std_devs = df.std()

    # Identify outliers for each class
    outliers = (np.abs(df - means) > std_dev_threshold * std_devs)

    # Filter out the outliers
    filtered_df = df[~outliers.any(axis=1)]
    num_outliers_discarded = len(df) - len(filtered_df)

    # Calculate the average mean and variance for each class after filtering
    filtered_means = filtered_df.mean()
    filtered_variances = filtered_df.var()

    # Print the number of outliers discarded
    print(f"{path} - Outliers discarded: {num_outliers_discarded}")
    
    # Print the average mean and variance for each class
    print(f"{path} - Average mean per class after filtering outliers:")
    print(filtered_means.values)
    print(f"{path} - Average variance per class after filtering outliers:")
    print(filtered_variances)
    print("-----")
