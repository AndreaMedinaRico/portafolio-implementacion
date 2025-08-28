'''
Autora: Andrea Medina Rico
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 200)         # Ajusta el ancho (puedes cambiar el número)

penguins = pd.read_csv("penguins_size.csv")

# ------ TRANSFORMACIÓN --------

# Eliminar valores vacíos
penguins = penguins.dropna()

# Tranformar categóricos a numéricos
penguins['sex'] = penguins['sex'].map({'MALE': 1, 'FEMALE': 0})

#   One-Hot encoding
penguins = pd.get_dummies(penguins, columns=['species'], prefix = 'species')
penguins = pd.get_dummies(penguins, columns=['island'], prefix = 'island')
for col in penguins.columns:
    if penguins[col].dtype == 'bool':
        penguins[col] = penguins[col].astype(float)

# Verificar unidades de medida
penguins['body_mass_g'] = penguins['body_mass_g'] / 1000
penguins['flipper_length_mm'] = penguins['flipper_length_mm'] / 10
penguins['culmen_depth_mm'] = penguins['culmen_depth_mm'] / 10 
penguins['culmen_length_mm'] = penguins['culmen_length_mm'] / 10 

# Actualizar nombre de columnas
penguins = penguins.rename(columns={
    'body_mass_g': 'body_mass_kg',
    'flipper_length_mm': 'flipper_length_cm',
    'culmen_depth_mm': 'culmen_depth_cm',
    'culmen_length_mm': 'culmen_length_cm'
})

# Buscar anomalías
print(penguins.info())
print(penguins.describe())
print(penguins.head())