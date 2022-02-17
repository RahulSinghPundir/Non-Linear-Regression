import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Reading the csv file from git repositry
df=pd.read_csv("Non_linear_Regression\china_gdp.csv")
# Cheaking or looking the dataset
print(df.head())
# Selecting the data and normalizing it by somple method.
x_data,y_data=df['Year'],df['Value']
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)
# Selecting or spliting data into train and test in random order
msk = np.random.rand(len(df)) < 0.8
x_train = xdata[msk]
x_test = xdata[~msk]
y_train = ydata[msk]
y_test = ydata[~msk]

# Analysing our data that how it look like and know which type of regression can be used.
plt.scatter(x_train,y_train)
# Using sigmoid function according to our dataset
def sigmoid(x,b1,b2):
  return 1/( 1+ np.exp(-b1*(x-b2)) )

# With curve_fit selecting the best non linear line.
from scipy.optimize import curve_fit
popt,pocv=curve_fit(sigmoid,x_train,y_train)

# Predicting the output for test set.
predict=sigmoid(x_test,*popt)
# Comparing our prediction and real output by visualization.
plt.scatter(x_test,y_test)
plt.plot(x_test,predict)
plt.show()
from sklearn.metrics import r2_score
print("Mean absolute error: ",np.mean(np.absolute(predict-y_test)))
print("R2 score: ",r2_score(y_test,predict))

