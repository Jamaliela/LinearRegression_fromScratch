######################################################################
# Author: Elaheh Jamali
# Username: Jamalie

# Programming Assignment 1: Regression Model
#
# Purpose: In this assignment, we have programmed (from scratch) a linear
# regression model.  Regression is one of the most ubiquitous tools in economic
# and much statistical analysis, as well as a simple machine learning model.
#
# Acknowledgement: Emely Alfaro Zavala for helping me to complete
# this assignment. Different articles were read for this.
#
######################################################################
import numpy   # library supporting large, multi-dimensional arrays and matrices.
import matplotlib.pyplot as plot  # library for embedding plots
import pandas as pd  # library to take data and creates a Python object with rows and columns


# Reading Data
data = pd.read_csv('Salary.csv')
print(data.shape)
print(data.head())

# Collecting X and Y
X = data['YearsExperience'].values
Y = data['Salary'].values

# Mean X and Y to find B0 and B1
mean_x = numpy.mean(X)
mean_y = numpy.mean(Y)

# Total number of values
m = len(X)

# Using the formula to calculate B1 and B0
numerator = 0
denominator = 0
for i in range(m):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2
b1 = numerator / denominator
b0 = mean_y - (b1 * mean_x)

# Print coefficients
print("This is B1:", b1, "and this is B0:", b0)

# Plotting Values and Regression Line

max_x = numpy.max(X) + 100
min_x = numpy.min(X) - 100

# Calculating line values x and y
x = numpy.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Plotting Line
plot.plot(x, y, color='#58b970', label='Regression Line')
# Plotting Scatter Points
plot.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plot.xlabel('YearsExperience')
plot.ylabel('Salary')
plot.legend()
plot.show()

# Calculating Root Mean Squares Error
RMSE = 0
for i in range(m):
    y_prediction = b0 + b1 * X[i]
    RMSE += (Y[i] - y_prediction) ** 2
RMSE = numpy.sqrt(RMSE/m)
print("This is Root Mean Square Error", RMSE)

# calculating coefficient determination
SST = 0
SSR = 0
for i in range(m):
    y_prediction = b0 + b1 * X[i]
    SST += (Y[i] - mean_y) ** 2
    SSR += (Y[i] - y_prediction) ** 2
r2 = 1 - (SSR/SST)
print("This is the coefficient of determination", r2)


