import numpy as np
import pandas as pd
import os
import gc
import warnings

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


############################# Load and clean data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
stores = pd.read_csv("stores.csv") 
transactions = pd.read_csv("transactions.csv").sort_values(["store_nbr", "date"])
oil = pd.read_csv("oil.csv")
holidays = pd.read_csv("holidays_events.csv")

# Datetime
train["date"] = pd.to_datetime(train.date)
test["date"] = pd.to_datetime(test.date)
transactions["date"] = pd.to_datetime(transactions.date)
oil["date"] = pd.to_datetime(oil.date)
holidays["date"] = pd.to_datetime(holidays.date)

# Data types
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

#There are some missing data points for oil, so use linear interpolation to fill them in
# Resample
oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
# Interpolate
oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
oil["dcoilwtico_interpolated"] =oil.dcoilwtico.interpolate()

############################# Data exploration
train.head()
transactions.head()

temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how = "left")
print("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))

# Create a line plot for each store_nbr using matplotlib
plt.figure(figsize=(10, 6))
for store_nbr, group in transactions.groupby('store_nbr'):
    plt.plot(group['date'], group['transactions'], label=f'Store {store_nbr}')
# Customize the plot
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.title('Transactions')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
# Show the plot
plt.tight_layout()
plt.show()

# that was ugly so use plotly.express instead
px.line(transactions.sort_values(["store_nbr", "date"]), x='date', y='transactions', color='store_nbr',title = "Transactions" ).show()


# looking at sales per month
a = transactions.copy()
a['date'] = pd.to_datetime(a['date'])
a["year"] = a.date.dt.year
a["month"] = a.date.dt.month

# plotly.express
px.box(a, x="year", y="transactions" , color = "month", title = "Transactions").show()

a = transactions.set_index("date").resample("M").transactions.mean().reset_index()
a["year"] = a.date.dt.year
px.line(a, x='date', y='transactions', color='year',title = "Monthly Average Transactions" ).show()

# sales are greatest in December

#transactions are correlated with sales
px.scatter(temp, x = "transactions", y = "sales", trendline = "ols", trendline_color_override = "red").show()

# day of week
a = transactions.copy()
a['date'] = pd.to_datetime(a['date'])
a["year"] = a.date.dt.year
a["dayofweek"] = a.date.dt.dayofweek+1
a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
px.line(a, x="dayofweek", y="transactions" , color = "year", title = "Transactions").show()
# weekends are biggest shopping days


# look at oil data
# Plot
p = oil.melt(id_vars=['date']+list(oil.keys()[5:]), var_name='Legend')
px.line(p.sort_values(["Legend", "date"], ascending = [False, True]), x='date', y='value', color='Legend',title = "Daily Oil Price" ).show()


# Correlating oil data with transactions and sales
temp = pd.merge(temp, oil, how = "left")
print("Correlation with Daily Oil Prices")
print(temp.drop(["store_nbr", "dcoilwtico"], axis = 1).corr("spearman").dcoilwtico_interpolated.loc[["sales", "transactions"]], "\n")


# preferred products
a = train.groupby("family").sales.mean().sort_values(ascending = False).reset_index()
px.bar(a, y = "family", x="sales", color = "family", title = "Which product family preferred more?").show()



########## Combine datasets
len(train) # 3000888
len(test) # 28512

# Store
# let's just use type and cluster 
stores_transfored = stores.drop(["city", "state"], axis=1)
# now let's make dummies (i.e., one hot encoding for the types and clusters)
# Perform one-hot encoding for the first column
stores_encoded_column1 = pd.get_dummies(stores_transfored['type'], prefix='type')
# Perform one-hot encoding for the second column (assuming it's named 'column2')
stores_encoded_column2 = pd.get_dummies(stores_transfored['cluster'], prefix='cluster')
# Concatenate the one-hot encoded columns back to the original DataFrame
stores_transfored = pd.concat([stores_transfored, stores_encoded_column1, stores_encoded_column2], axis=1)
# Drop the original categorical columns if needed
stores_transfored.drop(['type', 'cluster'], axis=1, inplace=True)

train_combined = train.merge(stores_transfored, how="left", on=["store_nbr", "store_nbr"])
test_combined = test.merge(stores_transfored, how="left", on=["store_nbr", "store_nbr"])
len(train) # 3000888
len(test) # 28512

# merging in oil
train_combined['date'] = pd.to_datetime(train_combined['date'])
train_combined = train_combined.merge(oil, how="left", on=["date", "date"])
test_combined['date'] = pd.to_datetime(test_combined['date'])
test_combined = test_combined.merge(oil, how="left", on=["date", "date"])
#now we can drop the dcoilwtico column (because we already have dcoilwtico_interpolated)
train_combined = train_combined.drop(["dcoilwtico"], axis=1)
test_combined = test_combined.drop(["dcoilwtico"], axis=1)
len(train_combined) # 3000888
len(test_combined) # 28512

# merging in holidays
# modifying holidays dataframe to 1) just be national holidays and 2) just unique days
# Filter the rows where locale is 'National'
holidays_transformed = holidays[holidays['locale'] == 'National']
# Extract unique dates and create a new DataFrame
holidays_transformed = pd.DataFrame({'holiday_dates': holidays_transformed['date'].unique()})
# add a column for 1s if it's a national holiday, and 0 otherwise.
train_combined['holiday_dates'] = train_combined['date'].apply(lambda x: 1 if x in holidays_transformed['holiday_dates'].values else 0)
test_combined['holiday_dates'] = train_combined['date'].apply(lambda x: 1 if x in holidays_transformed['holiday_dates'].values else 0)
len(train_combined) # 3000888
len(test_combined) # 28512

# Look at the lag on sales
# ACF & PACF for each family
a = train_combined[(train_combined.sales.notnull())].groupby(["date", "family"]).sales.mean().reset_index().set_index("date")
for num, i in enumerate(a.family.unique()):
    try:
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        temp = a[(a.family == i)]#& (a.sales.notnull())
        sm.graphics.tsa.plot_acf(temp.sales, lags=365, ax=ax[0], title = "AUTOCORRELATION\n" + i).show()
        sm.graphics.tsa.plot_pacf(temp.sales, lags=365, ax=ax[1], title = "PARTIAL AUTOCORRELATION\n" + i).show()
    except:
        pass



########## auto arima 
# how many unique combinations of stores and family
unique_combinations_count = train_combined.groupby(['store_nbr', 'family']).size().reset_index(name='count')
print(unique_combinations_count)

# Get exogenous variables
columns_to_exclude = ['store_nbr', 'family', 'id', 'sales', 'date']
exog_vars = [col for col in train_combined.columns if col not in columns_to_exclude]
print(exog_vars)

# Group the training data by 'store_nbr' and 'family'
grouped_train = train_combined.groupby(['store_nbr', 'family'])
grouped_test = test_combined.groupby(['store_nbr', 'family'])

# Dictionary to store forecasted results for each group
forecast_results = {}

# Iterate through each group and perform AutoARIMA forecasting
for (store_nbr, family), group_data in grouped_train:
    # auto_arima
    # autoarima is a more automatic and user-friendly implementation of ARIMA modeling. 
    # It automatically selects the best hyperparameters for the data, making it a good choice for beginners or for situations where you don't have time to manually tune the model. 
    # However, it is not as flexible as SARIMAX, and it may not be able to model some more complex time series data.
    model = auto_arima(group_data['sales'], seasonal=True, stepwise=True, suppress_warnings=True, max_order=None, trace=True)
    # Fit the SARIMAX model with exogenous variables
    # SARIMAX stands for Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors. 
    # It is a more general and flexible model than autoarima, and it can be used to model a wider range of time series data. 
    # However, it is also more complex and requires more manual tuning of the hyperparameters.
    # model = sm.tsa.SARIMAX(train['sales'], exog=train[exog_vars], order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
    # fitted_model = model.fit(disp=False)
    # Forecast sales for the corresponding test data (assuming test data is grouped similarly)
    if (store_nbr, family) in grouped_test.groups:
        print(store_nbr, family)
        test_data = grouped_test.get_group((store_nbr, family))
        forecast, conf_int = model.predict(n_periods=len(test_data), return_conf_int=True)
        
        # Create a DataFrame for forecasted values and confidence intervals
        forecast_df = pd.DataFrame(forecast, index=test_data.index, columns=['sales_forecast'])
        conf_int_df = pd.DataFrame(conf_int, index=test_data.index, columns=['lower_bound', 'upper_bound'])
        
        # Combine the forecasted values and confidence intervals with the test DataFrame
        test_forecasted = pd.concat([test_data, forecast_df, conf_int_df], axis=1)
        
        # Store the forecasted results in the dictionary
        forecast_results[(store_nbr, family)] = test_forecasted

# Now, forecast_results dictionary contains the forecasted sales for each 'store_nbr' and 'family' combination
# Access the results using forecast_results[(store_nbr, family)]


# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(forecast_results.values(), keys=forecast_results.keys())

# Reset the index to make it more readable (optional)
combined_df.reset_index(inplace=True)

#create submission file
submission = combined_df[['id', 'sales_forecast']]
submission = submission.rename({'sales_forecast': 'sales'}, axis=1)
submission.to_csv('submission.csv', index = False)