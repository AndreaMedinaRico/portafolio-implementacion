'''
Archivo: main.py
Descripción: Archivo principal.
Autora: Andrea Medina Rico
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformation import Transformation

pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 200)         # Ajusta el ancho

penguins = pd.read_csv("penguins_size.csv")

# ------ TRANSFORMACIÓN --------
trans = Transformation(penguins)

trans.drop_na()
trans.cat_to_num('sex', 'MALE', 'FEMALE')
trans.one_hot_encoding('species')
trans.one_hot_encoding('island')

trans.change_units('body_mass_g', 1000)
trans.change_units('flipper_length_mm', 10)
trans.change_units('culmen_depth_mm', 10)
trans.change_units('culmen_length_mm', 10)

trans.rename_columns({
    'body_mass_g': 'body_mass_kg',
    'flipper_length_mm': 'flipper_length_cm',
    'culmen_depth_mm': 'culmen_depth_cm',
    'culmen_length_mm': 'culmen_length_cm'
})

# Buscar anomalías
print(trans.data.info())
print(trans.data.describe())
print(trans.data.head())