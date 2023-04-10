from flask import Flask, render_template, request
from forms import ServiceForm

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


app = Flask(__name__)
app.config['SECRET_KEY'] = 'fortnite'

global result
global name

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = ServiceForm()
    if form.is_submitted():
        result = request.form
        print(result)

        data = pd.read_csv("breast-cancer.csv", sep=",")

        data = data[["diagnosis", "radius_mean", "texture_mean", "perimeter_mean"]]

        predict = "diagnosis"

        hasBreastCancer = False

        print("hi")

        x = np.array(data.drop([predict], 1))
        y = np.array(data[predict])
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        # saving the best the model with the highest accuracy
        best = 0
        for i in range(30):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

            model = KNeighborsClassifier(n_neighbors=9)

            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)

            if accuracy > best:
                best = accuracy
                with open("breast_cancer_model.pickle", "wb") as f:
                    pickle.dump(model, f)

        pickle_in = open("breast_cancer_model.pickle", "rb")
        model = pickle.load(pickle_in)

        print(best)

        print("result: ", result)
        # input
        information = [[0, 0, 0]]
        information[0][0] = float(result.get('radius'))
        information[0][1] = float(result.get('texture'))
        information[0][2] = float(result.get('perimeter'))
        name = result.get('name')
        postalcode = result.get('postalcode')
        email = result.get('email')

        print("information: ",information)

        print("float_information: ", information)

        predicted = model.predict(information)

        for x in range(len(predicted)):
            print("Predicted: ", predicted[x], "Data: ", information[x])
            result = predicted[x]
            if result == "B":
                print("You are most likely benign!")
                hasBreastCancer = False
            else:
                print("You are likely to have breast cancer")
                hasBreastCancer = True

        # location
        i = 0
        isValidCode = True
        if (len(postalcode) == 6):
            for char in postalcode:
                if (i % 2 == 0 and char.isnumeric()):
                    isValidCode = False
                    break
                if (i % 2 == 1 and char.isalpha()):
                    isValidCode = False
                    break
                i = i + 1
            if (isValidCode):
                print('\033[1m' + "Valid" + '\033[0m' + " Canadian postal code")
            else:
                print("Invalid Canadian postal code")
        if (len(postalcode) == 5):
            for i in postalcode:
                if (i.isalpha()):
                    isValidCode = False
                    break
            if (isValidCode):
                print('\033[1m' + "Valid" + '\033[0m' + " US postal code")
            else:
                print("Invalid US postal code")
        if (len(postalcode) > 6 or len(postalcode) < 5):
            isValidCode = False
            print('\033[1m' + "Invalid US or Canadian postal code" + '\033[0m')

        # Google Maps API to find pharmacies near provided location

        # API key
        gmaps = googlemaps.Client(key='AIzaSyBkIDcCvSO2Io3ZeZFq6ZhncL-oWxb9n8o')

        if isValidCode:
            geocode_result = gmaps.geocode(postalcode)
            lat = geocode_result[0]['geometry']['location']['lat']
            lng = geocode_result[0]['geometry']['location']['lng']
            places_result = gmaps.places_nearby(location=(lat, lng), radius=20000, type='hospital')

            if len(places_result['results']) > 0:
                name1 = "Name: "+ places_result['results'][0]['name']
                addy1 = "Address: "+ places_result['results'][0]['vicinity']

                name2 = "Name: "+ places_result['results'][1]['name']
                addy2 = "Address: "+ places_result['results'][1]['vicinity']

                name3 = "Name: "+ places_result['results'][2]['name']
                addy3 = "Address: "+ places_result['results'][2]['vicinity']

            else:
                print("No hospital found within 20000 meters of the entered address.")

        # email
        receiver_email = email

        sender_email = 'hopefulribbon@gmail.com'
        subject = 'Breast Cancer Results'

        if hasBreastCancer:
            body = "Hi "+name+", \n\nA thorough analysis has been conducted based on your data and based on our results," \
                   " we believe that the cell tissue is likely to be malignant (cancerous). Please advise you to get a " \
                   "proper diagnosis from your nearest clinic or hospital to verify the test results. Getting diagnosed" \
                   " early can greatly improve recovery. Here are some clinics/hospitals near " \
                   "you: \n\n"+name1+"\n"+addy1+"\n\n" + name2 + "\n" + addy2 + "\n\n" + name3 + "\n" + addy3 + \
                   " \n\nWe wish you a speedy recovery,\n\nBest Regards,\n\n" \
                   "The HopefulRibbon Team"
        else:
            body = "Hi "+name+", \n\nA thorough analysis has been conducted based on your data and based on our results, we " \
                   "believe that the cell tissue is likely to be benign (healthy). " \
                   "We advise continuous monitoring of your breast health. If an area is palpable or feels " \
                   "abnormal, please seek medical attention as early as possible for early diagnosis. \n\nStay healthy and" \
                   " vigilant,\n\nBest Regards,\n\nThe HopefulRibbon Team"

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = subject

        message.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, 'uddqpqwemiullrhn')
            smtp.send_message(message)

        print('Email sent')

    return render_template('home.html', form=form)



