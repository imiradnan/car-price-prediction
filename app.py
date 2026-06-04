import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt


# Load model

model = pickle.load(
    open(
        "car_price_model.pkl",
        "rb"
    )
)


columns = pickle.load(
    open(
        "columns.pkl",
        "rb"
    )
)


st.title(
    "🚗 Car Price Prediction Dashboard"
)


st.write(
    "Predict used car selling price using Machine Learning"
)


# Inputs

car_name = st.number_input(
    "Car Code",
    value=5
)


year = st.number_input(
    "Manufacturing Year",
    value=2020
)


present_price = st.number_input(
    "Present Price (Lakhs)",
    value=8.5
)


kms = st.number_input(
    "Kilometers Driven",
    value=20000
)


fuel = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel"
    ]
)


seller = st.selectbox(
    "Seller Type",
    [
        "Dealer",
        "Individual"
    ]
)


transmission = st.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic"
    ]
)


owner = st.number_input(
    "Previous Owners",
    value=0
)


mileage = st.number_input(
    "Mileage",
    value=18.5
)



# Encoding

fuel_value = 1 if fuel=="Petrol" else 0


seller_value = 0 if seller=="Dealer" else 1


trans_value = 1 if transmission=="Manual" else 0



input_data = pd.DataFrame(
    {
        "Car_Name":[car_name],
        "Year":[year],
        "Present_Price":[present_price],
        "Kms_Driven":[kms],
        "Fuel_Type":[fuel_value],
        "Seller_Type":[seller_value],
        "Transmission":[trans_value],
        "Owner":[owner],
        "Mileage":[mileage]
    }
)



if st.button(
    "Predict Price"
):


    prediction = model.predict(
        input_data
    )


    st.success(
        f"Estimated Car Price: ₹ {round(prediction[0],2)} Lakhs"
    )



    st.subheader(
        "Feature Importance"
    )


    importance = model.feature_importances_


    fig, ax = plt.subplots()


    ax.bar(
        columns,
        importance
    )


    plt.xticks(
        rotation=90
    )


    st.pyplot(
        fig
    )