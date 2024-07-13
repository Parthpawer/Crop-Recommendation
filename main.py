from flask import Flask, request, render_template
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data for prediction
def preprocess_input(Nitrogen, Phosphorus, Potassium, pH, Temperature, District_Name, Soil_color):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'Nitrogen': [Nitrogen],
        'Phosphorus': [Phosphorus],
        'Potassium': [Potassium],
        'pH': [pH],
        'Temperature': [Temperature],
        'District_Name': [District_Name],
        'Soil_color': [Soil_color]
    })

    # One-hot encode categorical columns
    district_dummies = pd.get_dummies(input_data['District_Name'], prefix='District_Name')
    soil_color_dummies = pd.get_dummies(input_data['Soil_color'], prefix='Soil_color')

    # Concatenate the original columns with the one-hot encoded columns
    input_data = pd.concat([input_data[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Temperature']], district_dummies, soil_color_dummies], axis=1)

    # Define all expected columns for the model
    all_columns = [
        'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Temperature',
        'District_Name_Kolhapur', 'District_Name_Pune', 'District_Name_Sangli', 'District_Name_Satara', 'District_Name_Solapur',
        'Soil_color_Black', 'Soil_color_Dark Brown', 'Soil_color_Light Brown', 'Soil_color_Medium Brown', 'Soil_color_Red', 'Soil_color_Reddish Brown'
    ]
    
    # Reindex the DataFrame to include all expected columns, filling missing columns with zeros
    input_data = input_data.reindex(columns=all_columns, fill_value=0)

    return input_data

# Function to scale input data using a pre-fitted scaler
def scale_input(data):
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    scaled_data = scaler.transform(data)
    return scaled_data

# Function to map prediction output to crop names
def predicted_crop(predicted_data):
    crop_list = ['Sugarcane', 'Jowar', 'Cotton', 'Rice', 'Wheat', 'Groundnut',
       'Maize', 'Tur', 'Urad', 'Moong', 'Gram', 'Masoor', 'Soybean',
       'Ginger', 'Turmeric', 'Grapes']
    return crop_list[np.argmax(predicted_data)]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form submission
    features = request.form
    Nitrogen = float(features['Nitrogen'])
    Phosphorus = float(features['Phosphorus'])
    Potassium = float(features['Potassium'])
    pH = float(features['pH'])
    Temperature = float(features['Temperature'])
    District_Name = features['District_Name']
    Soil_color = features['Soil_color']

    # Preprocess and scale the input data
    input_data = preprocess_input(Nitrogen, Phosphorus, Potassium, pH, Temperature, District_Name, Soil_color)
    scaled_input = scale_input(input_data)
    
    # Make prediction using the loaded model
    prediction = model.predict(scaled_input)
    print(prediction)
    
    # Get the preferred crop based on the prediction
    preferred_crop = predicted_crop(prediction[0])
    
    # Render the template with the prediction
    return render_template('index.html', prediction=preferred_crop)

if __name__ == '__main__':
    app.run(debug=True)
