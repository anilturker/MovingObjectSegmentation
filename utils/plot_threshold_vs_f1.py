import matplotlib.pyplot as plt

# x axis values
threshold_value = [0.3, 0.5, 0.7, 0.9]

# corresponding y axis values
mosnet_f1_score = [0.81, 0.82, 0.83, 0.84]
mosnet2_f1_score = [0.83, 0.84, 0.85, 0.86]

# plotting the points
plt.plot(threshold_value, mosnet_f1_score, color='blue', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='blue', label='MOS-Net', markersize=12)
plt.plot(threshold_value, mosnet2_f1_score, color='red', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='red', label='MOS-Net 2.0', markersize=12)

# setting x and y axis range
plt.ylim(0.75, 0.9)
plt.xlim(0.2, 1)

# naming the x axis
plt.xlabel('Threshold')
# naming the y axis
plt.ylabel('F1 Score')

# giving a title to my graph
plt.title('F1 Score vs Threshold')

# showing legend
plt.legend()

# function to show the plot
plt.show()