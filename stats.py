import pandas as pd
import os
import matplotlib.pyplot as plt

# Function to display the percentage on top of each bar
def display_percentage(ax, data):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')

# 1. Read all CSV files in a specific folder.
folder_path = 'normal_model'  # Update this with the path to your folder
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in files]

# 2. Compute the mean accuracy for each class across all runs.
df_concat = pd.concat(dfs)
mean_accuracy = df_concat.groupby('Class').Accuracy.mean()

# 3. Plot the mean accuracy of each class sorted from highest to lowest.
plt.figure(figsize=(10, 5))
sorted_mean_accuracy = mean_accuracy.sort_values(ascending=False)
ax1 = sorted_mean_accuracy.plot(kind='bar', color='c')
plt.title('Mean Accuracy of Each Class Across All Runs')
plt.ylabel('Mean Accuracy (%)')
plt.xlabel('Class')
display_percentage(ax1, sorted_mean_accuracy)
plt.tight_layout()
plt.savefig("Mean Accuracy.png")
plt.show()

# 4. Compute the overall mean accuracy across all classes and identify the best and worst classified classes.
overall_mean_accuracy = mean_accuracy.mean()
best_classified = mean_accuracy.idxmax()
worst_classified = mean_accuracy.idxmin()

# 5. Plot the mean accuracy for all classes, as well as the best and worst classified classes.
plt.figure(figsize=(10, 5))
data = [overall_mean_accuracy, mean_accuracy[best_classified], mean_accuracy[worst_classified]]
ax2 = plt.bar(['All Classes', 'Best Classified (' + best_classified + ')', 'Worst Classified (' + worst_classified + ')'],
        data, color=['c', 'g', 'r'])
plt.title('Overall Mean Accuracy and Best/Worst Classified Classes')
plt.ylabel('Mean Accuracy (%)')
display_percentage(plt.gca(), data)
plt.tight_layout()
plt.savefig("Mean Accuracy and Best-Worst.png")
plt.show()
