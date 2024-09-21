import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = joblib.load('model.pkr')

# Title for your web app
st.title('Bike Rental Prediction App')

# Description or instructions
st.write("""
### Enter the values below to predict total bike rentals:
""")

# User input fields for the features
season = st.selectbox('Season (1:Winter, 2:Spring, 3:Summer, 4:Fall)', [1, 2, 3, 4])
yr = st.selectbox('Year (0: 2011, 1: 2012)', [0, 1])
mnth = st.slider('Month', 1, 12, 6)
holiday = st.selectbox('Holiday (0: No, 1: Yes)', [0, 1])
weekday = st.slider('Weekday (0: Sunday, 6: Saturday)', 0, 6)
workingday = st.selectbox('Working Day (0: No, 1: Yes)', [0, 1])
weathersit = st.selectbox('Weather Situation (1: Clear, 2: Mist, 3: Light Snow/Rain)', [1, 2, 3])
temp = st.slider('Temperature (Normalized)', 0.0, 1.0, 0.5)
atemp = st.slider('Feeling Temperature (Normalized)', 0.0, 1.0, 0.5)
hum = st.slider('Humidity (Normalized)', 0.0, 1.0, 0.5)
windspeed = st.slider('Windspeed (Normalized)', 0.0, 1.0, 0.5)

# Create a button for prediction
if st.button('Predict'):
    # Collect the user input into a NumPy array
    user_input = np.array([[season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed]])
    
    # Optionally, scale the input using StandardScaler (if your model was trained on scaled data)
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input)  # Assuming your model was trained on scaled features
    
    # Make the prediction using the loaded model
    prediction = model.predict(user_input_scaled)

    # Display the prediction
    st.success(f'Predicted Bike Rentals: {int(prediction[0])}')
