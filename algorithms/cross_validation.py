'''
Archivo: cross_validation.py
Descripción: Implementación manual del algoritmo Ten Fold Cross Validation
    para evaluar el modelo antes de pasar a probarlo.
Autora: Andrea Medina Rico
'''
from algorithms.regression_gd import epochs, hypothesis
from visualization.statistic import Statistic
from model.ModelInput import Data, Hyperparameters, Coefficients

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

'''
Función: cross_validation
Descripción: Realiza la validación cruzada en k pliegues.
Params:
    data: Conjunto de datos de entrada.
    hyp_params: Hiperparámetros del modelo.
    coeffs: Coeficientes del modelo.
Returns:
    train_loss: Pérdida en el conjunto de entrenamiento.
    train_loss_mean: Pérdida media en el conjunto de entrenamiento.
    test_loss: Pérdida en el conjunto de prueba.
    test_loss_mean: Pérdida media en el conjunto de prueba.
'''
def cross_validation(data: Data, hyp_params: Hyperparameters, coeffs: Coefficients, type, rf = None):
    # División de los datos en k partes
    splits = np.array_split(data.data_train, hyp_params.k)
    y_splits = np.array_split(data.train_y, hyp_params.k)

    if type == 'lineal':
        all_train_MSE = []
        all_test_MSE = []
    else:
        all_train_MSE = np.zeros(hyp_params.k)
        all_test_MSE = np.zeros(hyp_params.k)

    all_train_MAE = np.zeros(hyp_params.k)
    all_test_MAE = np.zeros(hyp_params.k)
    all_R2 = np.zeros(hyp_params.k)

    stat = Statistic()

    for i in range(hyp_params.k):
        print("Split ", i)
        # 1. Usar split 1 como test
        data.data_test = splits[i]
        data.test_y = y_splits[i]

        # 2. Usar los demás como train
        train_folds = []
        y_folds = []
        for j in range(hyp_params.k):
            if j != i:
                train_folds.append(splits[j])
                y_folds.append(y_splits[j])
        data.data_train = np.concatenate(train_folds)
        data.train_y = np.concatenate(y_folds)

        # 3. Estandarizar los datos
        media, std = data.zscores_measures(data.data_train)
        data.data_train = data.standardize_zscore(data.data_train, media, std)
        data.data_test = data.standardize_zscore(data.data_test, media, std)

        #  5. Aplicar tipo de algoritmo (regresión lineal o random forest)
        train_MSE, test_MSE, train_MAE, test_MAE, predicted_y = apply_type_algorithm(data, hyp_params, coeffs, type, rf)

        r2_stat = stat.r2_score(data.test_y, predicted_y)
        all_R2[i] = r2_stat

        if type == 'lineal':
            all_train_MSE.append(train_MSE)
            all_test_MSE.append(test_MSE)
        else:
            all_train_MSE[i] = train_MSE
            all_test_MSE[i] = test_MSE

        all_train_MAE[i] = np.mean(train_MAE)
        all_test_MAE[i] = np.mean(test_MAE)

    # 5. Calcular promedios del MAE loss
    train_MAE_mean = np.mean(all_train_MAE)
    test_MAE_mean = np.mean(all_test_MAE)
    r2_mean = np.mean(all_R2)

    # 6. Calcular promedios del MSE loss por época
    mean_train_MSE = np.mean(all_train_MSE, axis = 0)
    mean_test_MSE = np.mean(all_test_MSE, axis = 0) 

    return mean_train_MSE, mean_test_MSE, train_MAE_mean, test_MAE_mean, r2_mean


def apply_type_algorithm(data: Data, hyp_params: Hyperparameters, coeffs: Coefficients, type, rf = None):
    ''' Aplica algoritmo de regresión lineal o random forest. '''

    if type == 'lineal':
        # Reinicializar coeficientes
        data.m, data.n = data.data_train.shape
        coeffs.params = np.zeros(data.n)
        coeffs.b = 0

        # Aplicar gradient descent en train
        train_MSE, test_MSE, train_MAE, test_MAE = epochs(data, coeffs, hyp_params)
        predicted_y = hypothesis(data.data_test, coeffs.params, coeffs.b)

    else:
        rf.fit(data.data_train, data.train_y)
        predicted_y = rf.predict(data.data_test)
        predict_train_y = rf.predict(data.data_train)

        train_MSE = mean_squared_error(data.train_y, predict_train_y)
        test_MSE = mean_squared_error(data.test_y, predicted_y)
        train_MAE = mean_absolute_error(data.train_y, predict_train_y)
        test_MAE = mean_absolute_error(data.test_y, predicted_y)

    return train_MSE, test_MSE, train_MAE, test_MAE, predicted_y
