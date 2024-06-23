!pip install pyngrok 
from pyngrok import ngrok
import subprocess
!choco install ngrok
from pyngrok import ngrok
import subprocess

# Install streamlit and pyngrok (if not already installed)
!pip install streamlit
!pip install pyngrok

# Save the Streamlit app code to a file
with open('app.py', 'w') as f:
    f.write("""
import streamlit as st
import numpy as np

""")
import pickle
import joblib
import streamlit as st
st.title('Player Rating Prediction')
st.markdown("Predict a player's overall rating based on their profile")

# Load the model and scaler
model = None
scaler = None
try:
    with open('/content/drive/My Drive/regression files/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    scaler = joblib.load('/content/drive/My Drive/regression files/scaler.pkl')
except FileNotFoundError:
    st.error('Model or scaler file not found. Please ensure best_model.pkl and scaler.pkl are in the current directory.')
except EOFError:
    st.error('Model file is empty or corrupted.')

# Collect user input
height = st.slider('Height (cm)', 150, 200, 175)
weight = st.slider('Weight (kg)', 50, 100, 75)
pace = st.slider('Pace', 1, 100, 50)
shooting = st.slider('Shooting', 1, 100, 50)
passing = st.slider('Passing', 1, 100, 50)
dribbling = st.slider('Dribbling', 1, 100, 50)

# Predict function
def predict(data, model, scaler):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction
if st.button('Predict Rating'):
    if model is not None and scaler is not None:
        input_data = np.array([[height, weight, pace, shooting, passing, dribbling]])
        result = predict(input_data, model, scaler)
        st.text(f'Predicted Rating: {result[0]:.2f}')
        # Calculate confidence within the if block
        confidence = np.std([tree.predict(input_data) for tree in model.estimators_])
        # Display confidence within the if block where it is defined
        st.text(f"Confidence Score: {confidence:.2f}")
else:
        st.error('Model or scaler not loaded. Prediction cannot be made.')

# Terminate any existing ngrok processes
ngrok.kill()  # Close any existing ngrok tunnels

#authtoken to authenticate the ngrok session
ngrok.set_auth_token('2iG11e55QIE4WCvza9kIihRcEvH_5hZW8RXCQ8x3skQsHCrwD') 

# Check for existing tunnels before creating a new one
tunnels = ngrok.get_tunnels()
if not tunnels:  # If no active tunnels, create a new one
    url = ngrok.connect(8501)
    print(f'Streamlit app running at: {url}')
else:
    # Reuse the existing tunnel instead of creating a new one
    url = tunnels[0].public_url  # Access the public URL of the first tunnel
    print(f'Reusing existing ngrok tunnel at: {url}')

# Start the Streamlit app
proc = subprocess.Popen(['streamlit', 'run', 'app.py'])

