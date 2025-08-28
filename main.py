'''
Autora: Andrea Medina Rico
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 200)         # Ajusta el ancho (puedes cambiar el número)

# ---- TRANSFORMACIÓN ----- 
songs = pd.read_csv("dataset.csv")
print(songs.head())
print(songs.tail())

print(songs.info())

# Mantener solo columnas relevantes
songs.drop(columns = ['track_id', 'artists', 'album_name', 'mode', 'time_signature', 'explicit', 'track_genre'], inplace = True)

# Unidades de medida y tipos de dato
songs['duration_ms'] = songs['duration_ms'].apply(lambda x: x / 1000 / 60)
songs.rename(columns={'duration_ms': 'duration_min'}, inplace = True) 

# Valores repetidos
uniques = songs['track_name'].value_counts()
songs = songs[songs['track_name'].isin(uniques[uniques == 1].index)]

songs.drop('track_name', axis = 1, inplace = True)

# Valores atípicos
songs = songs[songs['duration_min'] < 12]

print(songs.head())
print("\n\nDescripción:\n\n\n")
print(songs.describe())

'''
# Selecciona solo las columnas numéricas, excluyendo 'song_popularity'
muestra = songs.sample(n=200, random_state=42)  # Toma 200 filas aleatorias

columnas_numericas = muestra.select_dtypes(include='number').columns.drop('popularity')

for col in columnas_numericas:
    plt.figure()
    plt.scatter(muestra[col], muestra['popularity'], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('popularity')
    plt.title(f'{col} vs popularity (muestra aleatoria)')
    plt.show()
'''

print(songs.corr())