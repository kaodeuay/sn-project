import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
#from PIL import Image

#image = Image.open('')
#st.image(image, width=50)
st.write(""" 

# PREDICTION OF MALE OR FEMALE VOICES """)
#st.sidebar.image(image, width=50)
st.sidebar.header('NEW DATA')
st.sidebar.subheader('Please enter your data:')

# -- Define function to display widgets and store data


def get_input():
    # Display widgets and store their values in variables
    v_Fre = st.sidebar.slider('Frequency', 1, 1200, 1)
    v_Amp = st.sidebar.slider('Amplitude', 1, 1200, 1)


    # Store user input data in a dictionary
    data = {'Frequency': v_Fre,
            'Amplitude': v_Amp}

    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df

# -- Call function to display widgets and get data from user
df = get_input()

st.header('Digital Signal & Processing:')

# -- Display new data from user inputs:
st.subheader('User Input:')
st.write(df)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('newsound.csv')
df = pd.concat([df, data_sample],axis=0)

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(df)
# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization_sn.pkl', 'rb'))
#Apply the normalization model to new data
x_new = load_nor.transform(df)
x_new = x_new[:1]
st.subheader('Normalization Input:')
st.write(x_new)
# -- Reads the saved classification model
load_LR = pickle.load(open('best_knn_sn.pkl', 'rb'))
# Apply model for prediction
prediction = load_LR.predict(x_new)
prediction = prediction[:1]
st.subheader('Prediction:')
st.write(prediction)

#[theme]
#primaryColor="#aee1e1"
#backgroundColor="#d3e0dc"
#secondaryBackgroundColor="#fcd1d1"
#textColor="#5c5c5c"
#font="monospace"
