######################################################################
# Author: Elaheh Jamali
# Username: Jamalie

# Programming Assignment 1: Regression Model
#
# Purpose: In this assignment, we have programmed (from scratch) a linear
# regression model.  Regression is one of the most ubiquitous tools in economic
# and much statistical analysis, as well as a simple machine learning model.
#
# Acknowledgement: Lalith Bharadwaj and Emely Alfaro Zavala for helping me to complete
# this assignment.
#
######################################################################
import numpy   # library supporting large, multi-dimensional arrays and matrices.
import matplotlib.pyplot as plot  # library for embedding plots
import pandas as pd  # library to take data and creates a Python object with rows and columns
import seaborn as sb  # a library for high-level interface to draw statistical graphics
from sklearn.model_selection import train_test_split  # library that splits arrays or matrices into random train and test subsets
import matplotlib.patches as patches  # library to add patches

# finding the Slope of linear regression line
def Slope(a,b):
    n = len(a)
    two_sum = numpy.sum(a*b)
    sumX = numpy.sum(a)
    sumY = numpy.sum(b)
    sumX_2 = numpy.sum(a**2)
    slope = (n*two_sum-sumX*sumY)/(n*sumX_2-(sumX)**2)
    return slope

#Finding Intercept of linear regression line
def Intercept(a,b):
    intercept = numpy.mean(b)-Slope(a,b)*numpy.mean(a)
    return intercept

#predictions are made with the help of linear regression algorithm
def Predictions(slope,x_input,intercept):
    predict = slope*x_input + intercept
    return predict

#R-squared is regression metric
def R_squared(predicted_values,test_values):
    f = predicted_values
    y = test_values
    print(f,'\n\n',y)
    #sum of squares
    ss_total = numpy.sum((y-numpy.mean(y))**2)
    #regression sum
    #ss_reg=numpy.sum((f-numpy.mean(y))**2)
    #Residuals sum of squares
    ss_res = numpy.sum((y-f)**2)
    #R-squared formula
    R_2 = 1-(ss_res/ss_total)
    return R_2

#Finding Correlation Coefficient for the given X & Y values
def correlation_coeff(predicted_values,test_values):
    a = predicted_values
    b = test_values
    n = len(a)
    two_sum = numpy.sum(a*b)
    sumX = numpy.sum(a)
    sumY = numpy.sum(b)
    sumX_2 = numpy.sum(a**2)
    sumY_2 = numpy.sum(b**2)
    score = (n*two_sum-sumX*sumY)/numpy.sqrt((n*sumX_2-(sumX)**2)*(n*sumY_2-(sumY)**2))
    return score

#Finding Covariance for the given X & Y values
def Covariance(X,Y):
    a = X
    b = Y
    n = len(a)
    two_sum = numpy.sum(a*b)
    cov = two_sum/n-numpy.mean(a)*numpy.mean(b)
    return cov

#Importing data(csv format) using pandas
#Replace another dataset to make predictions
dataset=pd.read_csv('Salary_Data.csv')

# Split-out validation dataset
#knowing the dimenstions of data and making them READY for PREDICTIONS.
array = dataset.values
X = array[:,0]
#print(X.shape)
#X=X.reshape(1,-1).T
print(X.shape)
Y = array[:,1]
print(Y.shape)

#To know the distribution of data let us plot box plot
## 1
left = 0.1
width = 0.8
#fig=plot.figure()
#fig,(ax1,ax2) = plot.subplots(nrows=2,ncols=1,sharex=False,sharey=True)
ax1 = plot.axes([left, 0.5, width, 0.45])
ax1.boxplot(X)
ax1.set_title('Box plot for X')
plot.show()
## 2
ax2 = plot.axes([left, 0.5, width, 0.45])
ax2.boxplot(Y, '.-')
ax2.set_title('Distribution of Y Data')
plot.show()

#Covariation in data
print(Covariance(X,Y))

#Dividing data into training and testing classes
test_size = 0.10
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size= test_size, random_state=seed)

#finding intercepts of rgression line
intercept=Intercept(X_train,Y_train)
slope=Slope(X_train,Y_train)
print(intercept,slope)
predictions=Predictions(slope=slope,x_input=X_validation,intercept=intercept)
print(predictions)
print(R_squared(predicted_values=predictions,test_values=Y_validation))
print(correlation_coeff(test_values=Y_validation,predicted_values=predictions))

#Equation of Linear Regression
y=slope*X+intercept

#plotting the linear regression function
plot.scatter(X,Y,marker='^',color='k',alpha=0.55)
plot.plot(X,y,color='R',linewidth=2)
red_patch = patches.Patch(color='red', label='Regression Line')
plot.legend(loc=0,handles=[red_patch])
plot.title('Linear Regression Plot')
plot.tight_layout(pad=2)
plot.grid(False)
plot.show()

#Residual plots
sb.set(style="whitegrid")
# Make an example dataset with y ~ x
rs = numpy.random.RandomState(7)
#Plot the residuals after fitting a linear model
sb.residplot(X, Y, lowess=True, color="r")
plot.title('Residual Plot')
plot.show()

#--------------------------------------------------------------#
