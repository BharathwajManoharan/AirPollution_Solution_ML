## Abinito 2.0 ML Hackathon

(was conducted on 17th of March 2023)

> In Today's ML hackathon, we have decided to tackle the pressing issue of air pollution in Indian cities. Our chosen theme involves developing solutions to identify sources of pollution, including industrial emissions and vehicular traffic. We believe that by addressing this issue, we can make a positive impact on the health and well-being of the community and the environment.

## Packages used

* Numpy
* Pandas
* Matplotlib
* Seabrne
* Sklearn (Gradient Boosting Regressor and Random Forest Regressor)
* Joblib
* Ipython

## Models for Prediction:

* `Random Forest` - Random forests or random decision forests are an ensemble learning method for classification, regression.

* `Gradient Boosting Machine` -  can effectively capture complex patterns and provide accurate forecasts of AQI levels, even in the presence of noisy or missing data.

* `ARIMA model` - use historical AQI data to identify patterns and trends, taking into account autocorrelation and lagged values. These models can provide accurate predictions of future AQI levels, informing policy decisions and public health interventions.

## Code Examples

````
# Model Traiining (Random Forest):

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('air_quality_data.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('AQI', axis=1), data['AQI'], test_size=0.2, random_state=42)

# Create random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest MSE: {mse}")

````



````
# Model Traiining (Gradient Boosting machine):

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('air_quality_data.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('AQI', axis=1), data['AQI'], test_size=0.2, random_state=42)

# Create GBM regressor
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
gbm.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = gbm.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"GBM MSE: {mse}")

````

## Features

The choice of features for air quality forecasting may depend on the specific dataset and the location for which the forecasting is being done. However, some common features that are often used in air quality forecasting models include:

* Temperature (°C),
* Wind Speed (Km/h),
* Pressure (Pa),
* NO2 (ppm),
* Rainfall (Cm),
* PM10 (μg/m3),
* PM2.5 (μg/m3),
* AQI.

In addition to these features, there may be other factors that contribute to air pollution in specific locations that could be added to the model in the future. For example, if there are specific sources of pollution that are not captured in the existing features, such as agricultural activity or natural disasters like wildfires, they could be added to the model to improve its accuracy. Additionally, new types of data sources, such as satellite imagery or social media data, could be incorporated into the model to provide additional insights into air quality patterns and sources of pollution.

## Status

Accuracy: Close to 80% (77.3,77.8,78.6 respectively)

Project is: _finished_.

