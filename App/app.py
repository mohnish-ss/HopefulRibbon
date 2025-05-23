from flask import Flask, render_template, request, jsonify
from forms import ServiceForm
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import pickle

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from bs4 import BeautifulSoup
import googlemaps
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fortnite')  # Fallback to 'fortnite' if not set

global result
global name

def train_model():
    """Train and return the best KNN model for breast cancer prediction"""
    # Step 1: Load and prepare the data
    data = pd.read_csv("breast-cancer.csv", sep=",")
    # Select relevant features and target variable
    data = data[["diagnosis", "radius_mean", "texture_mean", "perimeter_mean"]]
    predict = "diagnosis"

    # Step 2: Prepare features (X) and target (y)
    x = np.array(data.drop(columns=[predict]))
    y = np.array(data[predict])
    # Split data into training (90%) and testing (10%) sets
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Step 3: Train multiple models to find the best one
    best = 0
    for i in range(30):
        # Create new random split for each iteration
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        # Initialize KNN classifier with 9 neighbors
        model = KNeighborsClassifier(n_neighbors=9)
        # Train the model
        model.fit(x_train, y_train)
        # Calculate accuracy
        accuracy = model.score(x_test, y_test)
        # Save the best model
        if accuracy > best:
            best = accuracy
            with open("breast_cancer_model.pickle", "wb") as f:
                pickle.dump(model, f)

    # Step 4: Load the best model for predictions
    pickle_in = open("breast_cancer_model.pickle", "rb")
    return pickle.load(pickle_in)

def process_form_data(form_data):
    """Process form data and return prediction results"""
    # Extract form data
    information = [[0, 0, 0]]
    information[0][0] = float(form_data.get('radius'))
    information[0][1] = float(form_data.get('texture'))
    information[0][2] = float(form_data.get('perimeter'))
    name = form_data.get('name')
    postalcode = form_data.get('postalcode')
    email = form_data.get('email')

    # Load model and make prediction
    model = train_model()
    predicted = model.predict(information)
    
    # Determine result
    hasBreastCancer = predicted[0] != "B"
    
    return {
        'name': name,
        'postalcode': postalcode,
        'email': email,
        'hasBreastCancer': hasBreastCancer,
        'prediction': predicted[0]
    }

def send_result_email(prediction_result, hospital_info=None):
    """Send email with prediction results and hospital information"""
    receiver_email = prediction_result['email']
    sender_email = os.getenv('EMAIL_USER')
    email_password = os.getenv('EMAIL_PASSWORD')
    subject = 'Breast Cancer Results'
    
    # Prepare email body based on prediction
    if prediction_result['hasBreastCancer']:
        body = f"Hi {prediction_result['name']}, \n\nA thorough analysis has been conducted based on your data and based on our results," \
               " we believe that the cell tissue is likely to be malignant (cancerous). Please advise you to get a " \
               "proper diagnosis from your nearest clinic or hospital to verify the test results. Getting diagnosed" \
               " early can greatly improve recovery."
        
        # Add hospital information if available
        if hospital_info:
            body += f" Here are some clinics/hospitals near you: \n\n" \
                   f"{hospital_info['name1']}\n{hospital_info['addy1']}\n\n" \
                   f"{hospital_info['name2']}\n{hospital_info['addy2']}\n\n" \
                   f"{hospital_info['name3']}\n{hospital_info['addy3']}"
        
        body += " \n\nWe wish you a speedy recovery,\n\nBest Regards,\n\nThe HopefulRibbon Team"
    else:
        body = f"Hi {prediction_result['name']}, \n\nA thorough analysis has been conducted based on your data and based on our results, we " \
               "believe that the cell tissue is likely to be benign (healthy). " \
               "We advise continuous monitoring of your breast health. If an area is palpable or feels " \
               "abnormal, please seek medical attention as early as possible for early diagnosis. \n\nStay healthy and" \
               " vigilant,\n\nBest Regards,\n\nThe HopefulRibbon Team"

    # Create and send email
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, email_password)
            smtp.send_message(message)
        print('Email sent successfully')
        return True
    except Exception as e:
        print(f'Failed to send email: {str(e)}')
        return False

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = ServiceForm()
    if form.is_submitted():
        result = request.form
        print(result)
        
        # Process form data and get prediction
        prediction_result = process_form_data(result)
        
        # Get hospital information (if Google Maps API is working)
        hospital_info = None  # You can add Google Maps API functionality here
        
        # Send email with results
        send_result_email(prediction_result, hospital_info)
        
    return render_template('home.html', form=form)

# Load and preprocess the breast cancer dataset
def load_data():
    data = pd.read_csv('breast-cancer.csv')
    
    # Clean column names by stripping whitespace
    data.columns = data.columns.str.strip()
    
    # Clean data values by stripping whitespace
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()
    
    # Convert diagnosis to binary (M=1, B=0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Select features and target
    features = ['radius_mean', 'texture_mean', 'perimeter_mean']
    X = data[features]
    y = data['diagnosis']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, features

# Load the model and scaler
model, scaler, features = load_data()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        radius = float(request.form['radius'])
        texture = float(request.form['texture'])
        perimeter = float(request.form['perimeter'])
        
        # Create input array
        input_data = np.array([[radius, texture, perimeter]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Convert prediction to text
        result = "Malignant" if prediction == 1 else "Benign"
        
        return jsonify({
            'prediction': result,
            'probability': f"{probability:.2%}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run()



