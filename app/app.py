import numpy as np 
import streamlit as st
from PIL import Image
from keras.models import load_model
import os

os.chdir('C:/Users/aleja/Documents/Pry_ADySP_/Presidentes_Peru_image_recognition')


def load_training_model(path):
    model = load_model(path)
    return model

def prepocessing(img):
    img = Image.open(img).convert('RGB')
    img = img.resize((550,550))
    img = np.array(img)
    return img

def predict_image(model, image):
    prediction = model.predict(image)
    return prediction

presidentes = ['3.Alan Garcia', '4.Ollanta Humala','6.Martin Vizcarra','9.Pedro Castillo','10.Dina Boluarte']

# Título de la aplicación
st.title('Modelo de Reconocimientro ')

# Widget de carga de archivos
uploaded_file = st.file_uploader('Elige una imagen', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Abre la imagen
    image = Image.open(uploaded_file)
    
    # Muestra la imagen
    st.image(image, caption='Imagen cargada.', use_column_width=True)

    img = prepocessing(uploaded_file)
    img = np.expand_dims(img, axis=0)
    model = load_training_model('model/final_model.h5')
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    st.write(f'Presidente : { presidentes[predicted_class] }')

else:
    st.write('Por favor cargar una imagen')

