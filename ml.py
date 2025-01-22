import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Loading Data
data = pd.read_csv("multi_restaurant_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Restaurant', 'Date'])

# Create time-based features
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Define the list of restaurants
restaurants = data['Restaurant'].unique()

models = {}  # Create a dictionary to store the models for each restaurant

# Loop through restaurants and train a model for each
for restaurant in restaurants:
    train_data = data[data['Restaurant'] == restaurant]
    
    # Training
    model = LinearRegression()
    model.fit(train_data[['DayOfWeek', 'Month', 'Year']], train_data['FoodConsumption'])
    
    # Define the date for prediction
    prediction_date = '2023-10-11'
    
    end_date = pd.to_datetime(prediction_date) - pd.DateOffset(days=1)
    start_date = end_date - pd.DateOffset(days=9)
    train_data = train_data[(train_data['Date'] >= start_date) & (train_data['Date'] <= end_date)]
    model = LinearRegression()
    model.fit(train_data[['DayOfWeek', 'Month', 'Year']], train_data['FoodConsumption'])
    
    # Store 
    models[restaurant] = model

joblib.dump({'models': models, 'data': data}, 'restaurant_model.pkl')
