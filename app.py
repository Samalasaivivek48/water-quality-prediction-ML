import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.title('Water Quality and Potability Prediction')

st.write('Enter the water quality parameters to predict potability:')

# Create input fields for each feature
ph = st.slider('pH', 0.0, 14.0, 7.0)
hardness = st.number_input('Hardness', 0.0, 1000.0, 150.0)
solids = st.number_input('Solids', 0.0, 10000.0, 500.0)
chloramines = st.slider('Chloramines', 0.0, 10.0, 4.0)
sulfate = st.number_input('Sulfate', 0.0, 1000.0, 250.0)
conductivity = st.number_input('Conductivity', 0.0, 1000.0, 400.0)
organic_carbon = st.slider('Organic Carbon', 0.0, 50.0, 5.0)
trihalomethanes = st.slider('Trihalomethanes', 0.0, 200.0, 50.0)
turbidity = st.slider('Turbidity', 0.0, 10.0, 3.0)
hardness_category = st.selectbox('Hardness Category', [0, 1, 2, 3])
chloramine_sulfate_interaction = st.number_input('Chloramine-Sulfate Interaction', 0.0, 10000.0, 1000.0)
mineral_balance_index = st.slider('Mineral Balance Index', 0.0, 2.0, 0.8)
organic_pollution_index = st.slider('Organic Pollution Index', 0.0, 100.0, 10.0)
health_risk_indicator = st.slider('Health Risk Indicator', 0.0, 1.0, 0.5)

if st.button('Predict Potability'):
    # Prepare input data (14 features for scaling)
    input_data = np.array([[
        ph, hardness, solids, chloramines, sulfate, conductivity,
        organic_carbon, trihalomethanes, turbidity, hardness_category,
        chloramine_sulfate_interaction, mineral_balance_index,
        organic_pollution_index, health_risk_indicator
    ]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Add placeholder for Potability to create 15-feature input for model
    input_data_final = np.column_stack((input_data_scaled, np.zeros((input_data_scaled.shape[0], 1))))
    
    # Make prediction
    prediction = model.predict(input_data_final)
    
    st.write('Potability Prediction:', prediction[0])
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data_final)
        
    
    # Interpret the result
    if prediction[0] > 0.5:  # Assuming it's a regression model predicting probability
        st.success('The water is predicted to be potable!')
    else:
        st.error('The water is predicted to be not potable.')
    
    st.write('Note: This prediction is based on the provided parameters and should not be used as the sole determinant for water safety. Always consult with water quality experts and perform proper testing.')

