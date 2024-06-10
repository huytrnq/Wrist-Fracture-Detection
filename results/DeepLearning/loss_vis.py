import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'Loss.csv'
data = pd.read_csv(file_path)

# Set the first column as the index (epoch)
data.set_index(data.columns[0], inplace=True)

# Plot the loss curves for each column
plt.figure(figsize=(12, 10))

for i, column in enumerate(data.columns):
    if i < 6:
        plt.plot(data.index, data[column], label=column)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('loss_curves.png')

# Show the plot
plt.show()
