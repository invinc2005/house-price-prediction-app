from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("./house_price_model.pkl", "rb"))
scaler_X = pickle.load(open("scaler_X.pkl", "rb"))
scaler_y = pickle.load(open("scaler_y.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form['num_bedrooms']),
        int(request.form['num_bathrooms']),
        int(request.form['living_area']),
        int(request.form['lot_area']),
        int(request.form['num_floors']),
        int(request.form['waterfront_present']),
        int(request.form['num_views']),
        int(request.form['house_grade']),
        int(request.form['basement_area']),
        float(request.form['latitude']),
        float(request.form['longitude']),
        int(request.form['living_area_renov']),
        int(request.form['lot_area_renov'])
    ]
    
    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler_X.transform(input_array)
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    output = round(prediction[0, 0], 2)

    return render_template('index.html', prediction_text=f'Predicted House Price: ₹{output:,}', latitude=features[9], longitude=features[10])

if __name__ == '__main__':
    app.run(debug=True)