from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly
import plotly.graph_objs as go
import json

# Initialise flask app
app = Flask(__name__)

# Couldnt figure out why jinja couldn't zip - https://stackoverflow.com/questions/5208252/ziplist1-list2-in-jinja2
app.jinja_env.globals.update(zip=zip)

# Load the production model and scaler and read the last known features for making predictions
prod_model = joblib.load('prod_model.pkl')
prod_scaler = joblib.load('prod_scaler.pkl')
prod_last_known_features = pd.read_pickle('prod_last_known_features.pkl')

# Load historical data for plotting
df = pd.read_csv('FINAL_USO.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
historical_dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
historical_prices = df['Adj Close'].tolist()

# Define the target variable
prod_target = 'Adj Close'

# Define features used in the model
prod_features = ['GDX_Close', 'SF_Price'] + [f'Adj_Close_lag_{lag}' for lag in range(1, 11)]

# render index template for homepage
@app.route('/')
def home():
    return render_template(
        'index.html',
        dates=historical_dates,
        prices=historical_prices
    )

# render predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Number of days to predict input
        prediction_days = int(request.form.get('prediction_days', 1))
        if prediction_days < 1:
            return render_template('result.html', error="Number of days to predict must be at least 1.")

        # Collect user inputs for GDX_Close and SF_Price
        gdx_close_input = request.form.get('GDX_Close')
        sf_price_input = request.form.get('SF_Price')

        # Parse all the input values
        gdx_close_values = [float(value.strip()) for value in gdx_close_input.split(',')]
        sf_price_values = [float(value.strip()) for value in sf_price_input.split(',')]

        if len(gdx_close_values) != prediction_days or len(sf_price_values) != prediction_days:
            return render_template('result.html', error="The number of input values must match the number of days to predict.")

        # Initialize current_features with last known features (as loaded earlier)
        current_features = prod_last_known_features.copy()

        # Open the lists to hold predictions and dates
        predictions = []
        future_dates = []

        # Starting predicting
        for day in range(prediction_days):
            # Update future date/s (only weekdays)
            last_date = datetime.strptime(historical_dates[-1], '%Y-%m-%d') if day == 0 else future_dates[-1]
            next_date = last_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            future_dates.append(next_date)
            future_date_str = next_date.strftime('%Y-%m-%d')

            # Update current_features with new predictor values
            current_features['GDX_Close'] = gdx_close_values[day]
            current_features['SF_Price'] = sf_price_values[day]

            # Prepare DataFrame for prediction
            input_data = pd.DataFrame([current_features])

            # Scale input data
            input_scaled = prod_scaler.transform(input_data)

            # Predict
            pred = prod_model.predict(input_scaled)[0]
            predictions.append(pred)

            # Update current_features for next prediction
            # Shift lagged features
            for lag in range(10, 1, -1):
                current_features[f'Adj_Close_lag_{lag}'] = current_features[f'Adj_Close_lag_{lag - 1}']
            current_features['Adj_Close_lag_1'] = pred

        # Prepare data for plotting (Plotly)
        future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
        extended_dates = historical_dates + future_dates_str
        extended_prices = historical_prices + predictions

        # Adding a last historical values to predicted values to bridge lines in plot. This is purely for UX and doesn't effect predictions.
        future_dates_str.insert(0, historical_dates[-1])   
        predictions.insert(0, historical_prices[-1])      

        # Prepare plotly graph
        historical_trace = go.Scatter(
            x=historical_dates,
            y=historical_prices,
            mode='lines',
            name='Historical Data',
            line=dict(color='green')
        )

        prediction_trace = go.Scatter(
            x=future_dates_str,
            y=predictions,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        )

        data = [historical_trace, prediction_trace]

        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template(
            'result.html',
            predictions=predictions,
            future_dates=future_dates_str,
            graphJSON=graphJSON
        )
    except Exception as e:
        return render_template('result.html', error=str(e))

# main - Running on all interfaces to ensure its correctly exposed so we can bind ports in docker.
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
