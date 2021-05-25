# region STEP 1: IMPORTS

import numpy as np
from sklearn.linear_model import LinearRegression

#endregion

# region STEP 2: PROVIDE DATA

# Input
x = [ [0,1], [5, 1], [15, 2], [25, 5], [35,11], [45, 15], [55,34], [60,35] ]

# Output
y = [4, 5, 20, 14, 32, 22, 38, 43]

x,y = np.array(x), np.array(y)

# In multiple linear regression:
#   x is a two-dimensional array with at least 2 columns.
#   y is usually a one-dimensional array.
# 
# This is a simple example of multiple linear regression, and x has two columns.
print(x)
print(y)

# endregion

# region STEP 3: CREATE A MODEL AND FIT

# Here, we'll create a model and fit the data.
model = LinearRegression().fit(x,y)

# endregion

# region STEP 4: GET RESULTS

# We'll obtain the properties of the model the same way as in the case of simple linear regression.
r_sq = model.score(x,y)
print('\ncoefficient of determination: ', r_sq, ', intercept: ', model.intercept_, ', slope: ', model.coef_)

# We obtain the value of R^2 using .score()
# We obtain and the values of the estimators of regression coefficients with .intercept_ and .coef_.
# .intercept_ holds the bias b0 
# .coef_ is an array containing b1 and b2 respectively.
# 
# In this example, the intercept is approximately 5.52, and this is the value of the predicted response when x1 = x2 = 0.
# If we increase x1 by 1, this yields a rise in the predicted response by .45. 
# Similarly, increasing x2 by 1 increases the response by .26.

# endregion

# region STEP 5: PREDICT RESPONSE

y_pred = model.predict(x)
print('\nPredicted response: ', y_pred, sep='\n')

# .predict() is very similar to the following:
y_pred = model.intercept_ + np.sum(model.coef_ * x, axis = 1)
print('\nPredicted response: ', y_pred, sep='\n')

# You can predict outputs by multiplying each column of the input with the appropriate weight, summing the results and adding the intercept to the sum.
x_new = np.arange(10).reshape(-1,2)
print(x_new)
y_new = model.predict(x_new)
print(y_new)

# endregion