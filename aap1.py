from flask import Flask
from flask import request
from flask import jsonify

import pickle
import numpy as np



app = Flask(__name__)



model = pickle.load(

    open(
        "diabetes_model.pkl",
        "rb"
    )

)



@app.route(
    "/"
)

def home():

    return "Diabetes Prediction API Running"



@app.route(
    "/predict",
    methods=["POST"]
)


def predict():


    data = request.json


    values = np.array(
        [
            [
                data["Pregnancies"],
                data["Glucose"],
                data["BloodPressure"],
                data["SkinThickness"],
                data["Insulin"],
                data["BMI"],
                data["DiabetesPedigreeFunction"],
                data["Age"]
            ]
        ]
    )


    prediction = model.predict(
        values
    )


    probability = model.predict_proba(
        values
    )


    return jsonify(

        {

        "Prediction":

        "High Diabetes Risk"

        if prediction[0] == 1

        else

        "Low Diabetes Risk",


        "Risk Probability":

        round(
            float(
                probability[0][1]
            ),
            2
        )

        }

    )



if __name__ == "__main__":

    app.run(
        debug=True
    )