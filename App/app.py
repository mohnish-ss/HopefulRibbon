from flask import Flask, render_template, request, jsonify

try:
    from .forms import ServiceForm
except ImportError:
    from forms import ServiceForm
from dotenv import load_dotenv
import os
import numpy as np
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set before starting the app.")
app.config['SECRET_KEY'] = SECRET_KEY

# Get the absolute path to the app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_knn_model():
    """Load the best KNN model for breast cancer prediction from disk"""
    models_dir = os.path.join(APP_DIR, 'models')
    model_path = os.path.join(models_dir, 'breast_cancer_model.pickle')

    # Check if model already exists
    if os.path.exists(model_path):
        print("Loading existing KNN model...")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    # If model doesn't exist, we cannot train it in this environment (no pandas)
    raise FileNotFoundError("KNN Model not found. Please run 'python setup_models.py' locally to generate it.")

def process_form_data(form_data):
    """Process form data and return prediction results"""
    # Extract form data
    radius     = float(form_data.get('radius'))
    texture    = float(form_data.get('texture'))
    perimeter  = float(form_data.get('perimeter'))
    name       = form_data.get('name')
    postalcode = form_data.get('postalcode')
    email      = form_data.get('email')

    # Scale input and predict using the globally loaded model & scaler
    input_data   = np.array([[radius, texture, perimeter]])
    input_scaled = scaler.transform(input_data)
    predicted    = model.predict(input_scaled)

    # Determine result (model predicts 1=Malignant, 0=Benign)
    hasBreastCancer = bool(predicted[0] == 1)

    return {
        'name': name,
        'postalcode': postalcode,
        'email': email,
        'hasBreastCancer': hasBreastCancer,
        'prediction': 'M' if hasBreastCancer else 'B'
    }

def send_result_email(prediction_result, hospital_info=None):
    """Send email with prediction results and hospital information"""
    receiver_email = prediction_result['email']
    sender_email = os.getenv('EMAIL_USER')
    email_password = os.getenv('EMAIL_PASSWORD')
    if not sender_email or not email_password:
        raise RuntimeError("EMAIL_USER and EMAIL_PASSWORD must be set before sending email.")
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
    prediction_result = None
    if form.is_submitted():
        result = request.form
        print(result)

        # Process form data and get prediction
        prediction_result = process_form_data(result)

        # Email functionality disabled for demo
        # send_result_email(prediction_result, hospital_info)

    return render_template('home.html', form=form, prediction_result=prediction_result)

# Load the model and scaler
model, scaler = load_knn_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        radius = float(request.form['radius'])
        texture = float(request.form['texture'])
        perimeter = float(request.form['perimeter'])

        ranges = {
            'radius': (6.981, 28.11),
            'texture': (9.71, 39.28),
            'perimeter': (43.79, 188.5),
        }
        values = {'radius': radius, 'texture': texture, 'perimeter': perimeter}
        for field, value in values.items():
            minimum, maximum = ranges[field]
            if not minimum <= value <= maximum:
                return jsonify({
                    'error': f'{field.title()} must be between {minimum} and {maximum}.'
                }), 400
        
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
    app.run(port=5001, debug=os.getenv("FLASK_DEBUG") == "1")
