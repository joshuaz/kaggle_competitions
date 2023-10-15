import pandas as pd
import numpy as np
import seaborn as sns #statistical data visualization library based on matplotlib. seaborn provides a high-level interface for creating attractive and informative statistical graphics
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import datetime

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
# ADABoost is used for building an ensemble of decision tree regressors through adaptive boosting. Adaptive boosting (AdaBoost) is a machine learning algorithm that combines the predictions of multiple base estimators (in this case, decision tree regressors) to improve the overall accuracy of the model
from xgboost.sklearn import XGBRegressor # popular gradient boosting library
from sklearn.model_selection import GridSearchCV
# GridSearchCV is a method for performing hyperparameter tuning in scikit-learn models. It performs an exhaustive search over a specified parameter grid, evaluating all possible combinations of hyperparameter values through cross-validated performance metrics.
from sklearn.linear_model import ElasticNetCV, Lasso
# Lasso regression is a type of linear regression with L1 regularization, which can be used for feature selection and regularization.
# ElasticNet is used for linear regression with combined L1 and L2 priors as regularizers. ElasticNetCV performs cross-validated optimization to find the best combination of the alpha (regularization strength) and l1_ratio (mixing parameter) parameters for Elastic Net regression.
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor
import lightgbm as lgbm
#LGBM stands for Light Gradient Boosting Machine, and it is implemented in the lightgbm Python package. LightGBM is a gradient boosting framework developed by Microsoft that uses tree-based learning algorithms. It is designed to be efficient and scalable, making it a popular choice for large datasets and high-dimensional features.
import numpy as np
from keras.layers import Dense, Activation
# In the context of neural networks and deep learning, the Dense layer is a core layer provided by the Keras API, which is now integrated into TensorFlow as tf.keras.layers.Dense. The Dense layer is used for creating fully connected neural network layers, where each neuron in the layer is connected to every neuron in the preceding layer.
from keras.models import Sequential
# In the context of neural networks and deep learning, Sequential is a class provided by the Keras library, which is now integrated into TensorFlow as tf.keras. Sequential is used to create linear stack of layers, meaning you can create a neural network by simply stacking layers on top of each other.

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt



# Load the files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Get number of observations for test and train
print([len(x) for x in [train_df, test_df]])

# Combine it into one large file for data exploration and cleaning
combined_df = pd.concat([train_df, test_df])

# Get a first view
print(combined_df)


# Quick look at potential missing values
print(combined_df.info())


# Classify int variables into category if needed
combined_df["MSSubClass"] = combined_df["MSSubClass"].astype("category")
combined_df["MoSold"] = combined_df["MoSold"].astype("category")



# Categorical data impute with mode of neighborhood and MSSubClass or just mode of own column if missing
missing_vals = ["MSZoning", "Alley", "Utilities", 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"Electrical",'KitchenQual','Functional','GarageType',"SaleType", 'GarageFinish','GarageQual','GarageCond','Exterior1st', 'Exterior2nd','FireplaceQu', "PoolQC", "Fence", "MiscFeature"]

for missing_val in missing_vals:
    try:
        combined_df[missing_val] = combined_df.groupby(['MSSubClass', "Neighborhood"])[missing_val].transform(lambda x: x.fillna(x.mode()[0]))
    except:
     combined_df[missing_val].fillna((combined_df[missing_val].mode()[0]), inplace=True)


# Add "Other" category as most elements are missing
combined_df["PoolQC"] = combined_df["PoolQC"].fillna("Other")


# Continuous data
missing_vals = ["LotFrontage", 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF1','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea',]
impute_vals = ["LotConfig" ,"Neighborhood",'BsmtFinType1', 'BsmtFinType2','BsmtQual', 'BsmtQual', 'BsmtQual','GarageType', 'GarageType']

for missing_val, impute_val in zip(missing_vals, impute_vals):
    combined_df[missing_val] = combined_df[missing_val].fillna(combined_df.groupby(impute_val)[missing_val].transform('mean'))


# Continuous impute data based on other continuous data
missing_vals = ['GarageYrBlt']
impute_vals = ['YearBuilt']

for missing_val, impute_val in zip(missing_vals, impute_vals):
    combined_df[missing_val] = combined_df[missing_val].fillna(combined_df[impute_val])


# Fill all leftovers with mean
for missing_val in combined_df.columns.values.tolist():

    if missing_val == "SalePrice":
        pass

    else:
        try:
            combined_df[missing_val] = combined_df[missing_val].fillna(combined_df[missing_val].mean())
        except:
            pass

# List of cols with missing values
print([col for col in combined_df.columns if combined_df[col].isnull().any()])


# Add and change some variables, namely the "Year" ones as it would be better to have them as "Age"
year = datetime.date.today().year
combined_df["AgeSold"] = int(year) - combined_df["YrSold"].astype(int)
combined_df["AgeGarage"] = int(year) - combined_df["GarageYrBlt"].astype(int)
combined_df["AgeBuilt"] = int(year) - combined_df["YearBuilt"].astype(int)


# Correlation matrix
corr_mat = combined_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_mat, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


#An algorithm has no clues what a “SaleType” of “WD” means. Instead it understands if there is a 1 or 0 in the “SaleType_WD column.

# Get dummies for our data set
combined_df = pd.get_dummies(combined_df)



# Split the data set so to build our model
train_df = combined_df[combined_df["SalePrice"] > 0 ]
test_df = combined_df[combined_df["SalePrice"].isna() ]
test_df = test_df.drop(["SalePrice"], axis = 1)

# Create the X and y sets
X_train_df = train_df.drop(["SalePrice"], axis = 1)
y_train_df = train_df[["Id" ,"SalePrice"]]


# Set the ID col as index
for element in [X_train_df, y_train_df, test_df]:
    element.set_index('Id', inplace = True)


# Scale the data and use RobustScaler to minimize the effect of outliers
scaler = RobustScaler()

# Scale the X_train set
X_train_scaled = scaler.fit_transform(X_train_df.values)
X_train_df = pd.DataFrame(X_train_scaled, index = X_train_df.index, columns= X_train_df.columns)

# Scale the X_test set
X_test_scaled = scaler.transform(test_df.values)
X_test_df = pd.DataFrame(X_test_scaled, index = test_df.index, columns= test_df.columns)



# Feature selection (only keep variables with some variance)
threshold_n=0.55
sel = VarianceThreshold(threshold=(threshold_n* (1 - threshold_n) ))
sel_var=sel.fit_transform(X_train_df)

# Create the new datasets
X_train_df = X_train_df[X_train_df.columns[sel.get_support(indices=True)]]
X_test_df = X_test_df[X_test_df.columns[sel.get_support(indices= True)]]

# Check what we have
print(X_train_df.info())


# Split our training set into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.05, random_state=23)




# REGULARIZATION WITH ELASTIC NET
# Set parameters to iterate over
alphas = [0.000542555]
l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]
# Model with iterative fitting 
elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

# Fit the model to the data
estc_reg = elastic_cv.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = estc_reg.predict(X_test)
print("ElasticRegressor RMSE:", sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(estc_reg.predict(X_test_df))
my_pred_estc = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})




# REGULARIZATION WITH LASSO
# Set parameters to iterate over
parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}

# Instantiate reg for gridsearch
lasso=Lasso()
# Conduct the gridsearch
lasso_reg = GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

# Instantiate new lasso reg with best params
lasso_reg = Lasso(alpha= 0.0009)

# Fit the model to the data
lasso_reg.fit(X_train,y_train)

# Predict on the test set from our training set
y_pred = lasso_reg.predict(X_test)
print("LassoRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(lasso_reg.predict(X_test_df))
my_pred_lasso = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})





# RANDOM FOREST

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Instantiate reg for gridsearch
rf = RandomForestRegressor()

# Conduct the gridsearch
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train_df, y_train_df)
print(grid_search.best_params_)

# Create a random forest with best parameters
rf_reg = RandomForestRegressor(bootstrap =  True, max_depth = 80, max_features = 'auto', min_samples_leaf = 3,
                               min_samples_split = 8, n_estimators = 300, n_jobs=-1, random_state=12)

# Fit the model to the data
rf_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred_rf = rf_reg.predict(X_test)
print("RandomForestRegressor RMSE:", sqrt(mean_squared_error(y_test, y_pred_rf)))

# Create predictions
predictions = np.exp(rf_reg.predict(X_test_df))
my_pred_rf = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})





# ADA BOOST

# Grid search for best params
param_grid = {
 'n_estimators': [50, 100, 200],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 'loss' : ['linear', 'square', 'exponential']
 }

# Instantiate reg for gridsearch
ab_reg = AdaBoostRegressor()

# Conduct the gridsearch
grid_search = GridSearchCV(estimator = ab_reg, param_grid = param_grid, cv = 4, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train_df, y_train_df)
print(grid_search.best_params_)

# Create a random forest with best parameters
ab_reg = AdaBoostRegressor(learning_rate =1, loss = 'exponential', n_estimators =  50, random_state= 12)

# Fit the model to the data
ab_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred_ab = ab_reg.predict(X_test)
print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ab)))

# Create predictions
predictions = np.exp(ab_reg.predict(X_test_df))
my_pred_ab = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})





# XGBOOST

# Grid search for best params
param_grid = {'max_depth':[3,4],
          'learning_rate':[0.01,0.03],
          'min_child_weight':[1,3],
          'reg_lambda':[0.1,0.5],
          'reg_alpha':[1,1.5],      
          'gamma':[0.1,0.5],
          'subsample':[0.4,0.5],
         'colsample_bytree':[0.4,0.5],
}

# Instantiate reg for gridsearch
reg = XGBRegressor()

# Conduct the gridsearch
grid_search = GridSearchCV(estimator = reg, param_grid = param_grid,
                          cv = 4, n_jobs = -1, verbose = True)

# Fit the grid search to the data
grid_search.fit(X_train_df, y_train_df)
print(grid_search.best_params_)

# Create a regressor with best parameters
xgb_reg = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3,min_child_weight=0, 
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', 
nthread=-1, scale_pos_weight=1, seed=27,reg_alpha=0.00006)

# Fit the model to the data
xgb_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = xgb_reg.predict(X_test)
print("XGBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(xgb_reg.predict(X_test_df))
my_pred_xgb = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})






# NEURAL NETWORK

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 320))

# Adding the second hidden layer
model.add(Dense(units = 317, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 300, activation = 'relu'))

# Adding the fourth hidden layer
model.add(Dense(units = 200, activation = 'relu'))

# Adding the fifth hidden layer
model.add(Dense(units = 200, activation = 'relu'))

# Adding the sixth hidden layer
model.add(Dense(units = 100, activation = 'relu'))

# Adding the seventh hidden layer
model.add(Dense(units = 100, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 200)

y_pred = model.predict(X_test)
print("ANNRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(model.predict(X_test_df))
predictions = np.concatenate( predictions, axis=0 )
my_pred_ann = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})






# LIGHTGBM
# Instantiate reg
lgbm_reg = lgbm.LGBMRegressor(
    objective='regression',
    num_leaves=4,
    learning_rate=0.01,
    n_estimators=5000,
    max_bin=200,
    bagging_fraction=0.75,
    bagging_freq=5,
    bagging_seed=7,
    feature_fraction=0.2,
    feature_fraction_seed=7,
    verbose=-1,
    #min_data_in_leaf=2,
    #min_sum_hessian_in_leaf=11
)

# Fit the model to the data
lgbm_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = lgbm_reg.predict(X_test)
print("LGBMRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(lgbm_reg.predict(X_test_df))
my_pred_lgbm = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})





# SVM
# Instantiate reg
svr_reg = make_pipeline(RobustScaler(), SVR(
    C=20,
    epsilon=0.008,
    gamma=0.0003,
))

# Fit the model to the data
svr_reg.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = svr_reg.predict(X_test)
print("SVRRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(svr_reg.predict(X_test_df))
my_pred_svr = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})





# STACKED REGRESSION

# Instantiate reg
stregr = StackingRegressor(regressors=[xgb_reg, estc_reg, lasso_reg, lgbm_reg],meta_regressor=lgbm_reg, use_features_in_secondary=True )
# Fit the model to the data
stack_reg=stregr.fit(X_train, y_train)

# Predict on the test set from our training set
y_pred = stack_reg.predict(X_test)
print("StackedRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred)))

# Create predictions
predictions = np.exp(stack_reg.predict(X_test_df))
my_pred_stacked = pd.DataFrame({'Id': X_test_df.index, 'SalePrice': predictions})

# Create CSV file
my_pred_stacked.to_csv('pred_stacked.csv', index=False)