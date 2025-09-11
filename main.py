'''
Archivo: main.py
Descripción: Archivo principal para la predicción de
    la masa de los pingüinos.
Autora: Andrea Medina Rico
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from model.Transformation import Transformation
from model.ModelInput import Hyperparameters, Coefficients, Data
from algorithms.cross_validation import cross_validation
from algorithms.regression_gd import epochs, hypothesis
from visualization.statistic import Statistic
from visualization.visualization import visualization

pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 200)         

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
print(trans.data.describe())
print(trans.data.count())


stat = Statistic()
# -------- GRÁFICOS --------
# Descomentar para ver gráficos utilizados en el reporte.
# visualization(trans.data)

# Inicialización variables
alfa = 0.001                                    
num_epochs = 5000                              
b = 0
k = 10

# Model Input
hyp_params = Hyperparameters(alfa, num_epochs, k)
coeffs = Coefficients()

data_validation = Data(trans.data, 0)
data_validation.split_data()
coeffs.params = np.zeros(data_validation.data_train.shape[1])


# ------- VALIDACIÓN ---------
print("\nCross validation... :)")
train_loss_cv, test_loss_cv, train_MAE_mean, test_MAE_mean = cross_validation(data_validation, hyp_params, coeffs)

print("Final Train MAE mean in validation:", train_MAE_mean)
print("Final Test MAE mean in validation:", test_MAE_mean)

stat.loss_plot_train_test(train_loss_cv[2], test_loss_cv[2], 'Train loss vs Test loss en cross validation')


# ------- ENTRENAMIENTO --------
data_regg = Data(trans.data, 0)
data_regg.split_data()
coeffs_regg = Coefficients()
coeffs_regg.params = np.zeros(data_regg.data_train.shape[1])

# Normalización de los datos
mean, std = data_regg.zscores_measures(data_regg.data_train)
data_regg.data_train = data_regg.standardize_zscore(data_regg.data_train, mean, std)
data_regg.data_test = data_regg.standardize_zscore(data_regg.data_test, mean, std)

print("\n Entrenando modelo... :)")
train_MSE, test_MSE, train_MAE, test_MAE = epochs(data_regg, coeffs_regg, hyp_params)

print("Final parameters:", coeffs_regg.params)
print("Final bias:", coeffs_regg.b)
print("Final Train MAE:", train_MAE[-1])
print("Final Test MAE:", test_MAE[-1])
print("Final Train MSE:", train_MSE[-1])
print("Final Test MSE:", test_MSE[-1])
stat.loss_plot_train_test(train_MSE, test_MSE, 'Train loss vs Test loss')

# ------ PREDICCIONES ---------
predicted_y_test = hypothesis(data_regg.data_test, coeffs_regg.params, coeffs_regg.b)

stat.prediction_plot(data_regg.test_y, predicted_y_test)
r2_stat = stat.r2_score(data_regg.test_y, predicted_y_test)

print("Coeficiente de determinación R2 en test:", r2_stat)


# ------- RANDOM FOREST --------
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    oob_score=True,
    random_state=42
)
rf.fit(data_regg.data_train, data_regg.train_y)
rf_pred = rf.predict(data_regg.data_test)

print("MAE:", mean_absolute_error(data_regg.test_y, rf_pred))
print("R2:", r2_score(data_regg.test_y, rf_pred))

stat.prediction_plot(data_regg.test_y, rf_pred)