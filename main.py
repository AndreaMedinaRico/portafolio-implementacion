'''
Autora: Andrea Medina Rico
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 200)         # Ajusta el ancho (puedes cambiar el número)

penguins = pd.read_csv("penguins_size.csv")
print(penguins.head())
print(penguins.tail())

# ------ TRANSFORMACIÓN --------

# Eliminar valores vacíos
penguins = penguins.dropna()
print(penguins.info())

# Tranformar a numéricos
penguins['sex'] = penguins['sex'].map({'MALE': 1, 'FEMALE': 0})
print(penguins.head())

# Verificar unidades de medida
penguins['body_mass_g'] = penguins['body_mass_g'] / 1000  # Convertir a kg
penguins['flipper_length_mm'] = penguins['flipper_length_mm'] / 10  # Convertir a cm
penguins['culmen_depth_mm'] = penguins['culmen_depth_mm'] / 10  # Convertir a cm
penguins['culmen_length_mm'] = penguins['culmen_length_mm'] / 10  # Convertir a cm

# Actualizar nombre de columnas
penguins = penguins.rename(columns={
    'body_mass_g': 'body_mass_kg',
    'flipper_length_mm': 'flipper_length_cm',
    'culmen_depth_mm': 'culmen_depth_cm',
    'culmen_length_mm': 'culmen_length_cm'
})

# Buscar anomalías
print(penguins.describe())