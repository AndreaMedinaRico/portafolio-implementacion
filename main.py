'''
Archivo: main.py
Descripción: Archivo principal.
Autora: Andrea Medina Rico
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


# ------ ESTADÍSTICA --------

# Matriz de correlación
corr_matrix = trans.data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación')
plt.show()

# Pairplot con seaborn
'''
sns.pairplot(trans.data)
plt.title('Pairplot de las variables numéricas')
plt.show()
'''

# Subplots de sctatterplot matplot
plt.subplot(231)
plt.scatter(trans.data['flipper_length_cm'], trans.data['body_mass_kg'])
plt.title('Body Mass vs Flipper Length')
plt.subplot(232)
plt.scatter(trans.data['culmen_length_cm'], trans.data['body_mass_kg'])
plt.title('Body Mass vs Culmen Length')
plt.subplot(233)
plt.scatter(trans.data['culmen_depth_cm'], trans.data['body_mass_kg'])
plt.title('Body Mass vs Culmen Depth')
plt.suptitle('Relación entre Body Mass y otras variables')
plt.subplot(234)
plt.scatter(trans.data['species_Gentoo'], trans.data['body_mass_kg'])
plt.title('Body Mass vs Species (Gentoo)')
plt.subplot(235)
plt.scatter(trans.data['island_Biscoe'], trans.data['body_mass_kg'])
plt.title('Body Mass vs Island (Biscoe)')
plt.subplot(236)
plt.scatter(trans.data['species_Adelie'], trans.data['body_mass_kg'])
plt.title('Body Mass vs Species (Adelie)')
plt.show()

# Kdeplot
plt.subplot(221)
sns.kdeplot(trans.data['body_mass_kg'], fill = True)
plt.title('Distribución de Body Mass')
plt.subplot(222)
sns.kdeplot(trans.data['flipper_length_cm'], fill = True)
plt.title('Distribución de Flipper Length')
plt.subplot(223)
sns.kdeplot(trans.data['culmen_length_cm'], fill = True)
plt.title('Distribución de Culmen Length')
plt.subplot(224)
sns.kdeplot(trans.data['culmen_depth_cm'], fill = True)
plt.title('Distribución de Culmen Depth')
plt.show()