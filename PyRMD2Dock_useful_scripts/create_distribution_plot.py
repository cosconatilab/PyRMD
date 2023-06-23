import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

database = input("Enter the name of the database to use: ")

data = pd.read_csv(database)
values = data["lowest_binding_energy"]
filtered_values = values[values < 0]
numpy_data = filtered_values.to_numpy(dtype='float32')

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(50, 50))

sns.distplot(numpy_data, hist=True)
mu, sigma = stats.norm.fit(numpy_data)
x_line = np.linspace(np.min(numpy_data), 0, 1000)
y_line = stats.norm.pdf(x_line, mu, sigma)
ax.plot(x_line, y_line, color='lightblue')
plt.xlim([np.min(numpy_data), 0])


plt.rcParams.update({'font.size': 40, 'font.weight': 'bold', 'lines.linewidth': 5.0})
ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='y', labelsize=40)

plt.show()

print("Mean:", np.mean(numpy_data))
print("Mode:", stats.mode(numpy_data))
print("Median:", np.median(numpy_data))
print("Standard Deviation:", np.std(numpy_data))
print("Minimum:", np.min(numpy_data))
print("Maximum:", np.max(numpy_data))

