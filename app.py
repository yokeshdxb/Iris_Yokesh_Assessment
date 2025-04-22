import streamlit as st
import numpy as np
import pickle

# Load the model
with open('iris_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Iris Prediction Web App")
st.write("🔍 This app uses a Logistic Regression to predict type of iris.")

# Collect user input
SepalLengthCm = st.number_input("SepalLengthCm", min_value=4.3, max_value=7.9, value=5.8)
SepalWidthCm = st.number_input("SepalWidthCm", min_value=2.0, max_value=4.4, value=3.0)
PetalLengthCm = st.number_input("PetalLengthCm", min_value=1.0, max_value=6.9, value=4.35)
PetalWidthCm = st.number_input("PetalWidthCm", min_value=0.1, max_value=2.5, value=1.3)
#gender = st.selectbox("Gender", options={0: "Female", 1: "Male"})
#tumor_size = st.number_input("Tumor Size (in cm)", min_value=0.0, value=2.5)
#location = st.selectbox("Tumor Location", options={0: "Frontal", 1: "Occipital", 2: "Parietal", 3: "Temporal"})  # Update as per your values
#histology = st.selectbox("Histology Type", options={0: "Astrocytoma", 1: "Glioblastoma", 2: "Medulloblastoma", 3: "Meningioma"})          # Update as per your values
#stage = st.selectbox("Cancer Stage", options={0: "Stage 0", 1: "Stage 1", 2: "Stage 2", 3: "Stage 3", 4: "Stage 4"})

#symptom_1 = st.selectbox("Symptom 1", options=[0, 1, 2, 3])
#symptom_2 = st.selectbox("Symptom 2", options=[0, 1, 2, 3])
#symptom_3 = st.selectbox("Symptom 3", options=[0, 1, 2, 3])
#radiation = st.selectbox("Radiation Treatment Received", options=[0, 1])
#surgery = st.selectbox("Surgery Performed", options=[0, 1])
#chemo = st.selectbox("Chemotherapy Received", options=[0, 1])

#survival_rate = st.number_input("Survival Rate (%)", min_value=0.0, max_value=100.0, value=75.0)
#tumor_growth_rate = st.number_input("Tumor Growth Rate", min_value=0.0, value=1.2)

#family_history = st.selectbox("Family History of Cancer", options=[0, 1])
#mri_result = st.selectbox("MRI Result Abnormality", options=[0, 1])
#follow_up = st.selectbox("Follow-Up Required?", options=[0, 1])

# Button to predict
if st.button("Predict Follow-up Requirement"):
    input_data = np.array([[
        SepalLengthCm,
        SepalWidthCm,
        PetalLengthCm,
        PetalWidthCm
        #age,
        #0 if gender == "Female" else 1,
        #tumor_size,
        #location,
        #histology,
        #stage,
        #symptom_1,
        #symptom_2,
        #symptom_3,
        #radiation,
        #surgery,
        #chemo,
        #survival_rate,
        #tumor_growth_rate,
        #family_history,
        #mri_result, 
        #follow_up
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("🟢 Iris Setosa !!!")
        #st.error("🔴 Malignant Cancer !!!")
    elif prediction == 1:
        st.success("🟢 Iris Versicolor !!!")
    elif prediction == 2:
        st.success("🟢 Iris Virginica !!!")
    else:
        st.error("Wrong Data infeed !!!")
    
    #else:
        #st.success("🟢 Benign Cancer !!!")

