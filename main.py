'''
Archivo: main.py
Descripción: Archivo principal para la predicción de
    la masa de los pingüinos.
Autora: Andrea Medina Rico
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformation import Transformation
from algorithms.cross_validation import cross_validation, zscores_measures, standardize_zscore
from algorithms.regression_gd import epochs, hypothesis
from visualization.statistic import Statistic
from visualization.visualization import visualization

pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 200)         # Ajusta el ancho

penguins = pd.read_csv("data/penguins_size.csv")

# -------- TRANSFORMACIÓN ----------

trans = Transformation(penguins)

trans.cat_to_num('sex', 'MALE', 'FEMALE')
print("NA:", trans.data.isna().sum())
trans.drop_na()
trans.one_hot_encoding('species')
trans.one_hot_encoding('island')

# Mantener k - 1 variables dummy
trans.data = trans.data.drop(columns = ['island_Torgersen', 'species_Chinstrap'])   

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

stat = Statistic()
# -------- GRÁFICOS --------
# Descomentar para ver gráficos utilizados en el reporte.
# visualization(trans.data)

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
real_y_train = data_train[:, 3]                      
data_train = np.delete(data_train, 3, axis=1)   

real_y_test = data_test[:, 3]
data_test = np.delete(data_test, 3, axis=1)

alfa = 0.001                                    
num_epochs = 5000                              
params = np.zeros(data_train.shape[1])
m, n = data_train.shape
b = 0
k = 10

'''
train_loss_cv, train_loss_mean, test_loss_cv, test_loss_mean, train_RMSE, test_RMSE, train_MAE, test_MAE = cross_validation(data_train, real_y_train, k, params, b, alfa, num_epochs)
print("Final Train loss mean in validation:", train_loss_mean)
print("Final Test loss in validation:", test_loss_mean)
print("Final Train RMSE in validation:", train_RMSE)
print("Final Test RMSE in validation:", test_RMSE)
print("Final Train MAE in validation:", train_MAE)
print("Final Test MAE in validation:", test_MAE)

stat.loss_plot(train_loss_cv[2])  
stat.loss_plot_train_test(train_loss_cv[2], test_loss_cv[2], 'Train loss vs Test loss en cross validation')
'''
# ------- ENTRENAMIENTO --------

# Normalización de los datos
mean, std = zscores_measures(data_train)
data_train = standardize_zscore(data_train, mean, std)
data_test = standardize_zscore(data_test, mean, std)

new_params, new_b, train_MSE, test_MSE, train_MAE, test_MAE = epochs(data_train, params, b, real_y_train, alfa, num_epochs, m, n, data_test, real_y_test)
print("Final parameters:", new_params)
print("Final bias:", new_b)
print("Final Train MSE:", train_MSE[-1])
print("Final Train MAE:", train_MAE[-1])
print("Final Test MAE:", test_MAE[-1])
print("Final Test MSE:", test_MSE[-1])
stat.loss_plot_train_test(train_MSE, test_MSE, 'Train loss vs Test loss')

# ------ PREDICCIONES ---------
predicted_y_test = hypothesis(data_test, new_params, new_b)

stat.prediction_plot(real_y_test, predicted_y_test)
r2_score = stat.r2_score(real_y_test, predicted_y_test)

print("Coeficiente de determinación R2 en test:", r2_score)