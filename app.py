import streamlit as st
import numpy as np
import pickle

# Load the model
with open('iris_classifier.pkl', 'rb') as f1:
    model = pickle.load(f1)

# Streamlit UI
st.title("Iris Prediction Web App")
st.write("🔍 This app uses a Logistic Regression to predict type of iris.")

# Collect user input
SepalLengthCm = st.number_input("SepalLengthCm", min_value=4.3, max_value=7.9, value=5.8)
SepalWidthCm = st.number_input("SepalWidthCm", min_value=2.0, max_value=4.4, value=3.0)
PetalLengthCm = st.number_input("PetalLengthCm", min_value=1.0, max_value=6.9, value=4.35)
PetalWidthCm = st.number_input("PetalWidthCm", min_value=0.1, max_value=2.5, value=1.3)

# Button to predict
if st.button("Predict Follow-up Requirement"):
    input_data = np.array([[
        SepalLengthCm,
        SepalWidthCm,
        PetalLengthCm,
        PetalWidthCm
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("🟢 Iris Setosa !!!")
    elif prediction == 1:
        st.success("🟢 Iris Versicolor !!!")
    elif prediction == 2:
        st.success("🟢 Iris Virginica !!!")
    else:
        st.error("Wrong Data infeed !!!")

