import streamlit as st
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import json
import neuron

# Configurar la página para que sea más ancha
st.set_page_config(
    page_title="Simulador de Neurona",  # Título de la página
    layout="wide"
)

st.image("img/neurona.png", width=300)
st.title("¡Hola Neurona!")
salida = 0

st.header("Simulador de neurona")
entradas = st.slider(
'Elige el número de entradas/pesos que tendrá la neurona',  # Etiqueta del slider
min_value=1,            # Valor mínimo
max_value=10,          # Valor máximo
value=1,               # Valor inicial
step=1                  # Incremento del valor
)

st.header("Pesos")

# Crear las columnas necesarias para las entradas
columns = st.columns(entradas)  # Esto crea un número de columnas igual al número de entradas

# Crear los inputs dinámicamente en función del número de entradas seleccionadas
pesos_values = []
for i, col in enumerate(columns):
    with col:
        pesos_value = st.number_input(f'Peso {i}', value=0.0, key=f"peso_{i}")  # Entrada para cada peso
        pesos_values.append(pesos_value)
        
st.write(f"Pesos: {pesos_values}")

st.header("Entradas")

columns_2 = st.columns(entradas)  # Esto crea un número de columnas igual al número de entradas

# Crear los inputs dinámicamente en función del número de entradas seleccionadas
entradas_values = []
for i, col in enumerate(columns_2):
    with col:
        entrada_value = st.number_input(f'Entrada {i}', value=0.0, key=f"entrada_{i}")  # Entrada para cada peso
        entradas_values.append(entrada_value)

st.write(f"Entradas: {entradas_values}")

# Crear una fila con dos columnas
col1, col2 = st.columns(2)

# Columna 1: Sesgo (con un número input)
with col1:
    st.header("Sesgo")
    sesgo = st.number_input("Ingresa el valor del sesgo", value=0.0)

# Columna 2: Función de activación (con un selectbox)
with col2:
    st.header("Función de activación")
    funcion_activacion = st.selectbox(
        "Elige la función de activación",
        options=["Sigmoide", "ReLU", "Tangente Hiperbólica","Binary Step"]
    )

neurona = neuron.Neuron(pesos_values, sesgo, funcion_activacion)
neurona.run(entradas_values)


# Botón para calcular el resultado
if st.button('Calcular salida'):
    resultado = neurona.run(entradas_values)
    st.write(f"La salida de la neurona es: {resultado}")
    
sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
selected = st.feedback("thumbs")
st.write("© Darío Nievas López")