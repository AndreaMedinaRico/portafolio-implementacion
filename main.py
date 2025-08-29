'''
Archivo: main.py
Descripción: Archivo principal para la predicción de
    la masa de los pingüinos.
Autora: Andrea Medina Rico
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformation import Transformation
from algorithms.regression_gd import epochs, MSE
from statistic import correlation_matrix, pairplot, scatter_subplots, kdeplot_subplots

pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 200)         # Ajusta el ancho

penguins = pd.read_csv("data/penguins_size.csv")

# ------ TRANSFORMACIÓN --------
trans = Transformation(penguins)

trans.cat_to_num('sex', 'MALE', 'FEMALE')
trans.drop_na()
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
print(trans.data.count())


# ------ ESTADÍSTICA --------
'''
# Matriz de correlación
correlation_matrix(trans.data)

# Pairplot con seaborn
pairplot(trans.data)    

# Subplots de sctatterplot matplot
scatter_cols = ['flipper_length_cm', 'culmen_length_cm', 'culmen_depth_cm', 'species_Gentoo', 'island_Biscoe', 'species_Adelie']
scatter_subplots(trans.data, scatter_cols, 'body_mass_kg')

# Kdeplot
kdeplot_cols = ['body_mass_kg', 'flipper_length_cm', 'culmen_length_cm', 'culmen_depth_cm']
kdeplot_subplots(trans.data, kdeplot_cols)
'''

# Selección de variables 
trans.data = trans.data.drop(columns = ['island_Torgersen', 'species_Chinstrap'])   

# Hypothesis testing for flipper length cm
null_corr = 0
corr = trans.data['body_mass_kg'].corr(trans.data['flipper_length_cm'])
print("Correlation coefficient:", corr)

SE_corr = np.sqrt((1 - corr**2) / (trans.data.shape[0] - 2))
t_stat = (corr - null_corr) / SE_corr
print("t-statistic:", t_stat)


# Data division in test and train
data_random = trans.data.sample(frac = 1, random_state = 42).reset_index(drop = True)

train_size = int(0.8 * len(data_random))
pd_train = data_random[:train_size]
pd_test = data_random[train_size:]

data_train = pd_train.to_numpy()
data_test = pd_test.to_numpy()

print("Train:", pd_train.shape)
print(pd_train.info())
print("Test:", pd_test.shape)
print(pd_test.info())
print(pd_test.columns)

# Inicialización de datos
real_y = data_train[:, 3]                       # Valores reales de y
print(real_y)
data_train = np.delete(data_train, 3, axis=1)   # Dejar solo las x en data train
alfa = 0.0001
num_epochs = 2
params = np.zeros(data_train.shape[1])
b = 0

# TEN FOLD CROSS VALIDATION 
# 1. Usaré todas las columnas menos 'island_Torgersen' y 'species_Chinstrap'
# para usar k - 1 de cada clase 

k = 10
splits = np.array_split(data_train, k)
y_splits = np.array_split(real_y, k)

for i in range(k):
    # 1. Usar split 1 como test
    test_split = splits[i]
    test_y = y_splits[i]
    # 2. Usar los demás como train
    train_folds = []
    y_folds = []
    for j in range(k):
        if j != i:
            train_folds.append(splits[j])
            y_folds.append(y_splits[j])
    train_split = np.concatenate(train_folds)
    train_y = np.concatenate(y_folds)

    # 3. Aplicar gradient descent en train
    m, n = train_split.shape
    print("Split ", i)
    new_params, new_b = epochs(train_split, params, b, train_y, alfa, num_epochs, m, n)

    # 4. Guardar el loss de train
    train_loss = MSE(train_split, params, b, train_y, m)
    print("Train loss:", train_loss)

    # 5. Guardar el loss de test al predecir
    test_loss = MSE(test_split, new_params, new_b, test_y, test_split.shape[0])
    print("Test loss:", test_loss)