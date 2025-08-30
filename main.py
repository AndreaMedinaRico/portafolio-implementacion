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
from algorithms.cross_validation import cross_validation
from algorithms.regression_gd import epochs, MSE
from statistic import Statistic

pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 200)         # Ajusta el ancho

penguins = pd.read_csv("data/penguins_size.csv")

# ------ TRANSFORMACIÓN --------
trans = Transformation(penguins)

trans.cat_to_num('sex', 'MALE', 'FEMALE')
print("NA:", trans.data.isna().sum())
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
stat = Statistic()
'''
# Matriz de correlación
stat.correlation_matrix(trans.data)

# Pairplot con seaborn
stat.pairplot(trans.data)

# Subplots de sctatterplot matplot
scatter_cols = ['flipper_length_cm', 'culmen_length_cm', 'culmen_depth_cm', 'species_Gentoo', 'island_Biscoe', 'species_Adelie']
stat.scatter_subplots(trans.data, scatter_cols, 'body_mass_kg')

# Kdeplot
kdeplot_cols = ['body_mass_kg', 'flipper_length_cm', 'culmen_length_cm', 'culmen_depth_cm']
stat.kdeplot_subplots(trans.data, kdeplot_cols)

# Histograma de cada variablee
stat.histogram(trans.data, 'body_mass_kg')
stat.histogram(trans.data, 'flipper_length_cm')
stat.histogram(trans.data, 'culmen_length_cm')
stat.histogram(trans.data, 'culmen_depth_cm')
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
real_y = data_train[:, 3]                      
data_train = np.delete(data_train, 3, axis=1)   
alfa = 0.001                                    
num_epochs = 2000                              
params = np.zeros(data_train.shape[1])
b = 0
k = 10

train_loss, train_loss_mean, test_loss, test_loss_mean = cross_validation(data_train, real_y, k, params, b, alfa, num_epochs)
print("Final Train loss mean:", train_loss_mean)
print("Final Test loss:", test_loss_mean)

stat.loss_plot(train_loss[2])  
stat.loss_plot_train_test(train_loss[2], test_loss[2])