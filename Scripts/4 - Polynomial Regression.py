# region STEP 1: IMPORTS

import numpy as np
from numpy.lib import polynomial
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# endregion

# region STEP 2: PROVIDE DATA

# Input and Output. input needs to be a two-dimensional array; that's why reshape is used.
x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([15,11,2,8,25,32])

# endregion

# region STEP 3: TRANSFORM INPUT DATA

# This is a NEW important step we need to implement for POLYNOMIAL REGRESSION.
# We need to include x^2 (and perhaps other terms) as additional features when implementing polynomial regression.
# For this reason, we should transform the input array x to contain the additional column(s) with values of x^2
# 
# There several ways to transform the input array (like using insert() from numpy) but we will use the PolynomialFeatures class.
transformer = PolynomialFeatures(degree=2, include_bias=False)

# You can provide several optional parameters to PolynomialFeatures:
#   degree:             an INT (2 by default) that represents the degree of the polynomial regression function.
#   interaction_only:   a BOOLEAN (False by default) that decides whether to include only interaction features (True) or all features (False)
#   include_bias:       a BOOLEAN (True by default) that decides whether to include the bias (intercept) column of ones (True) or not (False)
# 
# Our example will use all default values.

# Before applying transfomer, we need to fit it.
transformer.fit(x)

# Once it's fitted, it's ready to create a new, modified input. 
x_ = transformer.transform(x)

# That is the transformation of the input array with .transform().
# 
# We can also replace the previous three statements with this, using fit_transform():
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

# We now have a new input array with two columns: one with original inputs and the other with their squares.
# Here is how the new input array looks:
print(x_)

# endregion

# region STEP 4: CREATE A MODEL AND FIT DATA

# Same as linear regression.
model = LinearRegression().fit(x_,y)

# endregion

# region STEP 5: GET RESULTS

r_sq = model.score(x_,y)
print('\ncoefficient of determination: ', r_sq, ', intercept: ', model.intercept_, ', slope: ', model.coef_)

# Again .score() returns R^2. It's input is the modified x_ and not x.
# .intercept_ represents b0
# .coef_ represents b1 and b2 respectively.
# 
# You can obtain a very similar result with different transformation and regression arguments.
x_ = PolynomialFeatures(degree=2, include_bias=2).fit_transform(x)

# In x_:
#   column 1 - contains 1s
#   column 2 - contains values of x
#   column 3 - contains squares of x
print(x_)

# The intercept is included with the leftmost column of ones, and you don't need it again when creating the instance of LinearRegression.
# Thus, you can provide fit_intercept=False
# 
# This approach yields similar results.
# You can see that intercept yields 0, but coef_ actually contains b0 (the intercept) as its first element.
model = LinearRegression(fit_intercept=False).fit(x_,y)
r_sq = model.score(x_,y)
print('\ncoefficient of determination: ', r_sq, ', intercept: ', model.intercept_, ', slope: ', model.coef_)


# endregion

# region STEP 6: PREDICT RESPONSE

# To get the predicted response, just use .predict() but remember to pass in modified x_ and not x.
y_pred = model.predict(x_)
print('\nPredicted response: ', y_pred, sep='\n')

# endregion