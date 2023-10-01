import warnings
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

warnings.filterwarnings("ignore")

# Load the model
classifier = load_model("IRIS_model_2 (1).h5")

def predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width):
    # Convert inputs to float and reshape for prediction
    inputs = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
    
    # Make prediction
    global classifier  # Add this line to access the global variable
    prediction = classifier.predict(inputs)
    
    # Map the prediction to flower name
    prediction_index = np.argmax(prediction, axis=1)[0]
    
    # Convert prediction index to flower name
    if prediction_index == 0:
        result_flower = 'Iris-setosa'
    elif prediction_index == 1:
        result_flower = 'Iris-versicolor'
    else:
        result_flower = 'Iris-virginica'

    #print(prediction)
    return result_flower

def Input_Output():
    st.title("Iris Variety Prediction")
    st.image("https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png",width=600)

    st.markdown("You are using Streamlit...",unsafe_allow_html=True)
    sepal_length = st.text_input("Enter Sepal Length",".")
    sepal_width = st.text_input("Enter Sepal Width",".")
    petal_length = st.text_input("Enter Petal Length",".")
    petal_width = st.text_input("Enter Petal Width",".")

    result =""
    if st.button("Click here to Predict"):
        result= predict_iris_variety(sepal_length,sepal_width,petal_length,petal_width)
        st.balloons()
    st.success('The output is {}'.format(result))
if __name__=='__main__':
    Input_Output()
