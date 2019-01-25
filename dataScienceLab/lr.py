import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm
#split into training set and testing set
from sklearn.model_selection import train_test_split


#observed predictors
x_train = np.array([1, 2, 3])
# or do this, which creates 3 x 1 vector so no need to reshape
#x_train = np.array([[1], [2], [3]])
# print(x_train.shape)
x_train = x_train.reshape(len(x_train),1)
#check dimensions
# print(x_train.shape)



#observed responses
y_train = np.array([2, 2, 4])
# or do this, which creates 3 x 1 vector so no need to reshape
#y_train = np.array([[2], [2], [4]])
y_train = y_train.reshape(len(y_train),1)
# # print(y_train.shape)
#
#
#
# #build matrix X by concatenating predictors and a column of ones
# n = x_train.shape[0]
# ones_col = np.ones((n, 1))
# X = np.concatenate((ones_col, x_train), axis=1)
# #check X and dimensions
# # print(X, X.shape)
#
#
#
# #matrix X^T X
# LHS = np.dot(np.transpose(X), X)
# # print(LHS)
# #matrix X^T Y
# RHS = np.dot(np.transpose(X), y_train)
# # print(RHS)
# #
#
# betas = np.dot(np.linalg.inv(LHS), RHS)
# # print(betas)
#
#
# #intercept beta0
# beta0 = betas[0]
#
# #slope beta1
# beta1 = betas[1]

# print(beta0, beta1)

def simple_linear_regression_fit(x_train, y_train):
    # your code here
    x_train = x_train.reshape(len(x_train), 1)
    y_train = y_train.reshape(len(y_train), 1)

    n = x_train.shape[0]
    ones_col = np.ones((n, 1))
    X = np.concatenate((ones_col, x_train), axis=1)

    # matrix X^T X
    LHS = np.dot(np.transpose(X), X)
    # print(LHS)
    # matrix X^T Y
    RHS = np.dot(np.transpose(X), y_train)
    # print(RHS)
    #

    betas = np.dot(np.linalg.inv(LHS), RHS)


    return betas


beta0 = simple_linear_regression_fit(x_train, y_train)[0]
beta1 = simple_linear_regression_fit(x_train, y_train)[1]

# print("(beta0, beta1) = (%f, %f)" % (beta0, beta1))

#
# f = lambda x : beta0 + beta1*x
# xfit = np.arange(0, 4, .01)
# yfit = f(xfit)
#
# plt.plot(x_train, y_train, 'ko', xfit, yfit)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# #create the X matrix by appending a column of ones to x_train
# X = sm.add_constant(x_train)
# #this is the same matrix as in our scratch problem!
# print(X)
# #build the OLS model (ordinary least squares) from the training data
# toyregr_sm = sm.OLS(y_train, X)
# #save regression info (parameters, etc) in results_sm
# results_sm = toyregr_sm.fit()
# #pull the beta parameters out from results_sm
# beta0_sm = results_sm.params[0]
# beta1_sm = results_sm.params[1]
#
# print("(beta0, beta1) = (%f, %f)" %(beta0_sm, beta1_sm))

#build the least squares model
toyregr_skl = linear_model.LinearRegression()
#save regression info (parameters, etc) in results_skl
results_skl = toyregr_skl.fit(x_train,y_train)
#pull the beta parameters out from results_skl
beta0_skl = results_skl.intercept_
beta1_skl = results_skl.coef_[0]

# print("(beta0, beta1) = (%f, %f)" %(beta0_skl, beta1_skl))

#load mtcars
cars_data = pd.read_csv("/home/rumman/Desktop/resource/DS_lab/data/mtcars.csv")
cars_data = cars_data.rename(columns={"Unnamed: 0":"name"})
arca = cars_data.head()
# print(arca)


#set random_state to get the same split every time
train_data, test_data = train_test_split(cars_data, test_size = 0.3, random_state = 6)

# print(cars_data.shape, train_data.shape, test_data.shape)

#define  predictor and response for training set
y_train = train_data.mpg
x_train = train_data[['wt']]

# define predictor and response for testing set
y_test = test_data.mpg
x_test = test_data[['wt']]

# create linear regression object with sklearn
regr = linear_model.LinearRegression()

# train the model and make predictions
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)
#print out coefficients
print('Coefficients: \n', regr.coef_[0], regr.intercept_)

plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, y_pred, color="blue")
plt.xlabel('wt')
plt.ylabel('mpg')
plt.show()

train_MSE2= np.mean((y_train - regr.predict(x_train))**2)
test_MSE2= np.mean((y_test - regr.predict(x_test))**2)
print("The training MSE is %2f, the testing MSE is %2f" %(train_MSE2, test_MSE2))

# or with sklearn.metrics
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, regr.predict(x_train)))
print(mean_squared_error(y_test, regr.predict(x_test)))


x_train2 = train_data[['wt', 'hp']]
x_test2 = test_data[['wt', 'hp']]

#create linear regression object with sklearn
regr2 = linear_model.LinearRegression()

#train the model
regr2.fit(x_train2, y_train)

#make predictions using the testing set
y_pred2 = regr2.predict(x_test2)

#coefficients
print('Coefficients: \n', regr.coef_[0], regr.intercept_)

train_MSE2= np.mean((y_train - regr2.predict(x_train2))**2)
test_MSE2= np.mean((y_test - regr2.predict(x_test2))**2)
print("The training MSE is %2f, the testing MSE is %2f" %(train_MSE2, test_MSE2))

print(mean_squared_error(y_train, regr2.predict(x_train2)))
print(mean_squared_error(y_test, regr2.predict(x_test2)))