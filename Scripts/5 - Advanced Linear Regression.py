# region STEP 1: IMPORT PACKAGES

import numpy as np
import statsmodels.api as sm

# endregion

# region STEP 2: PROVIDE DATA AND TRANSFORM INPUTS

# Our inputs and outputs are created. 
x = [[0,1], [5,1], [15,2], [25,5], [35,11], [45,15], [55,34], [60,35]]
y = [4,5,20,14,32,22,38,43]
x,y = np.array(x), np.array(y)

# However, we will need to add the column of ones from "4 - Polynomial Regression.py" to calculate the intercept b0.
# It doesn't take b0 into account by default.
x = sm.add_constant(x)

# That's how you add the column of ones to x with add_constant(). 
# It takes the input array x as an argument and returns a new array with the column of ones inserted at the beginning.
# Here is x and y now:
print(x)
print(y)

# endregion

# region STEP 3: CREATE A MODEL AND FIT IT

# The regression model based on ordinary least squares is an instance of the class statsmodels.regression.linear_model.OLS.
model = sm.OLS(y,x)

# Be careful here. Notice the first argument is the output, followed by the input.
# Once the model is created, apply .fit(). Results will contain a lot of information about the regression model.
results = model.fit()

# endregion

# region STEP 4: GET RESULTS

# Results is an object containing detailed information about the results of linear regression.
print(results.summary())

# The table printed above is very complex, but many statistical values can be found including R^2, b0, b1, and b2.
print('\ncoefficient of determination: ', results.rsquared, ', adjusted coefficient of determination: ', results.rsquared_adj, ', regression coefficients: ', results.params)

# .rsquared holds R^2
# .rsquared_adj represents adjusted R^2 (R^2 corrected according to the number of input features)
# .params refers the array with b0, b1, and b2 respectively.
# These results are identical to the ones achieved with scikit-learn for the same problem.

# endregion 
 
# region STEP 5: PREDICT RESPONSE

# You can use either fittedvalues or .predict() with the input array as the argument.
print('\nPredicted response: ', results.fittedvalues, sep='\n')
print('\nPredicted response: ', results.predict(x), sep='\n')

# The above is the predicted response for known inputs.
# For predictions with new regressors, you can also apply .predict() with new data as the argument.
x_new = sm.add_constant(np.arange(10).reshape((-1,2)))
print(x_new)
y_new = results.predict(x_new)
print(y_new)

# These results are the same as those obtained using scikit-learn for the same problem.

# endregion