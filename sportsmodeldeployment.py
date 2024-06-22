
import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb'))

def main():
    st.title('Player Rating Prediction')
    st.markdown('Predict a player\'s overall rating based on their profile')

    # Collect user input
    height = st.slider('Height (cm)', 150, 200, 175)
    weight = st.slider('Weight (kg)', 50, 100, 75)
    pace = st.slider('Pace', 1, 100, 50)
    shooting = st.slider('Shooting', 1, 100, 50)

    # Predict function
    def predict(data):
        prediction = model.predict(data)
        return prediction

    if st.button('Predict Rating'):
        input_data = np.array([[height, weight, pace, shooting]])
        result = predict(input_data)
        st.text(f'Predicted Rating: {result[0]:.2f}')
        confidence = np.std([tree.predict(input_data) for tree in model.estimators_])
        st.text(f'Confidence Score: {confidence:.2f}')

if __name__ == '__main__':
    main()
