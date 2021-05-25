# region STEP 1: IMPORT PACKAGES
import numpy as np
from sklearn.linear_model import LinearRegression

# endregion

# region STEP 2: PROVIDE DATA

# We define some data to work with:
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))      # INPUTS (Regressors, x)
y = np.array([5,20,14,32,22,38])                            # OUTPUT (Predictor, y)

# X is reshaped because it is required to be TWO DIMENSIONAL. 
# It can have one column and as many rows as necessary. This is what reshape (-1, 1) does.
# x has 2 dimensions and x.shape is (6,1), while y has one dimension and y.shape is (6, )

# endregion 

# region STEP 3: CREATE A MODEL AND FIT IT

# Here, we will create a linear regression model and fit it using our data.
# Below we create an instance of the class linear regression, which represents our regression model.
model = LinearRegression()

# The following are optional parameters for this class:
#   fit_intercept:  a BOOLEAN (True by default) that decides whether to calculate the intercept b0(True) or consider it equal to zero (False)
#   normalize:      a BOOLEAN (False by default) that decides whether to normalize the input (True) or not (False)
#   copy_x:         a BOOLEAN (True by default) that decides whether to copy (True) or overwrite the input variables (False)
#   n_jobs:         an INTEGER or NONE (default) that represents the number of jobs used in parallel computation. None means one job and -1 to use all processors.
# 
# We will use default values for these.

# Now we will fit our model with the data.
# FIT will calculate the optimal values of the weights b0 and b1, using the existing input and output (x,y) as the arguments.
model.fit(x,y)

# It returns "self", meaning we can replace the previous two statements with this:
model = LinearRegression().fit(x,y)

#endregion

# region STEP 4: GET RESULTS

r_sq = model.score(x,y)

# This gets the coefficient of determination, known as R^2
print('coefficient of determination:', r_sq)

# This gets the intercept, known as b0
print('intercept: ', model.intercept_)

# This gets the slope, known as b1
print('slope: ', model.coef_)

# y can be provided as a two-dimensional array as well.
print('coefficient of determination: ', model.score(x,y.reshape(-1,1)), ', intercept: ', model.intercept_, ', slope: ', model.coef_)

# endregion

# region STEP 5: PREDICT RESULTS

# Once we have a satisfactory model, we can use it for predictions with existing or new data.
# To obtain a predicted response, we can use .predict(). We will pass in the regressor as the argument, and get the predicted response back.
y_pred = model.predict(x)
print('\nPredicted response: ', y_pred, sep='\n')

# Another way to predict the response is to add the intercept to the coefficient and multiply each element of x.
# This differs from the previous example in dimensions only - it's now 2 dimensions in the below example, whereas it was 1 dimensional in the above.
y_pred = model.intercept_ + model.coef_ * x
print('\nPredicted response: ', y_pred, sep='\n')

# Reducing the number of dimensions of x to one will yield the same result. This can be done by replacing x with x.reshape(-1), x.flatten(), or x.ravel()
#  when multiplying it with model.coef_.
# In practice, regression models are often applied for forecasts. 
# This means you can use fitted models to calculate the outputs based on some other, new inputs:
x_new = np.arange(5).reshape((-1,1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

# In the above example, .predict() is applied to the new regressor x_new and yields the response y_new. This example conveniently uses arange() from numpy 
#  to generate an array with the elements from 0 (inclusive) to 5(exclusive), meaning 0,1,2,3,4.

# endregion

