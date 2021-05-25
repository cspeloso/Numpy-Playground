# STEP 1: Import Packages
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# STEP 2: Provide Data
x = [[0,1], [5,1], [15,2], [25,5], [35,11], [45,15], [55,34], [60,35]]
y = [4,5,20,14,32,22,38,43]
x,y = np.array(x), np.array(y)

# STEP 3: Transform Input Data
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

# STEP 4: Create a model and fit it
model = LinearRegression().fit(x_, y)

# STEP 5: Get Results
r_sq = model.score(x_,y)
intercept,coefficients = model.intercept_, model.coef_
print('coefficient of determination: ', r_sq, ', intercept: ', intercept, ', slope: ', coefficients)

# STEP 6: Predict
y_pred = model.predict(x_)
print('\nPredicted Response: ', y_pred, sep='\n')

# In this case, there are six regression coefficients (including the intercept) as shown in the estimated regression function:
#   f(x1, x2) = b0 + b1x1 + b2x2 + b3x1^2 + b4x1x2 + b5x2^2
# 
# Notice that polynomial regression yielded a higher coefficient of determination than multiple linear regression for the same problem.
# At first, one may think that obtaining a large R^2 is a good result, which it may be.
# However, in real world situations, having a complex model and R^2 very close to 1 may be a sign of overfitting.
# To check the performance of a model, it should be tested with new data, that is with observations not used to fit (train) the model.
# 
# It's a good idea to split a dataset into training and test subsets.