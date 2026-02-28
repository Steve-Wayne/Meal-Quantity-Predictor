from flask import Flask, request, jsonify,render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

data = pd.read_csv("rest.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Restaurant', 'Date'])

data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

restaurants = data['Restaurant'].unique()

models = {}
for restaurant in restaurants:
    train_data = data[data['Restaurant'] == restaurant]
    model = LinearRegression()
    model.fit(train_data[['DayOfWeek', 'Month', 'Year']], train_data['FoodConsumption'])
    models[restaurant] = model

#/predict aage dalega
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            
            restaurant = request.form['restaurant']
            date = 61
            prediction_date = pd.to_datetime(date)

            
            prediction_data = pd.DataFrame({
                'DayOfWeek': [prediction_date.dayofweek],
                'Month': [prediction_date.month],
                'Year': [prediction_date.year]
            })


            model = models[restaurant]
            predicted_consumption = model.predict(prediction_data)[0]

            return render_template('output.html', restaurant=restaurant, date=date, consumption=int(predicted_consumption))
        
        return "Make a POST request to this endpoint with restaurant and date to get predictions."
    except Exception as e:
        // AI FIX START
return jsonify({'error': 'An internal error has occurred!'}), 500
// AI FIX END

if __name__ == '__main__':
app.run(debug=False)
