## Abinito 2.0 ML Hackathon
> 

## Technologies and Tools
* Python 
* TensorFlow
* Keras
* OpenCV

## Process

* Run `set_hand_histogram.py` to set the hand histogram for creating gestures. 
* Once you get a good histogram, save it in the code folder.
* Added gestures and label them using OpenCV which uses webcam feed. by running `create_gestures.py` and stores them in a database.
* Add different variations to the captured gestures by flipping all the images by using `Rotate_images.py`.
* Run `load_images.py` to split all the captured gestures into training, validation and test set. 
* To view all the gestures, run `display_gestures.py` .
* Train the model using Keras by running `cnn_model_train.py`.
* Run `final.py`. This will open up the gesture recognition window which will use your webcam to interpret the trained American Sign Language gestures.
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
Our model was able to predict the 31 characters in the ASL with a prediction accuracy > 84.19%.

Features that can be added:
* Deploy the project on cloud and create an API for using it.
* Increase the vocabulary of our model
* Incorporate feedback mechanism to make the model more robust
* Add more sign languages

## Status
Project is: _finished_.
