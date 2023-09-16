from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_16092023')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('Logo.jpg')
    image_hospital = Image.open('Hospital.jpg')
    
    st.image(image,use_column_width=True)

    add_selectbox = st.sidebar.selectbox(
    "¿Cómo te gustaría predecir?",    
    ("Online","Batch"))

    st.sidebar.info('Esta app es creada para predecir gastos por hospitalización de clientes')
    st.sidebar.success('https://www.pycaret.org')

    st.sidebar.image(image_hospital)

    st.title("App de Predicción de Cargos")

    if add_selectbox == 'Online':

        age = st.number_input('Edad', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sexo', ['Male', 'Female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Hijos', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Fumador'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predecir"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('La predicción es {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Carga el csv para generar predicciones", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            predictions.rename(columns={'prediction_label': 'Prediction'}, inplace=True)
            st.write(predictions)

if __name__ == '__main__':
    run()