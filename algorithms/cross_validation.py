'''
Archivo: cross_validation.py
Descripción: Implementación manual del algoritmo Ten Fold Cross Validation
    para evaluar el modelo antes de pasar a probarlo.
Autora: Andrea Medina Rico
'''
from algorithms.regression_gd import epochs
from model.ModelInput import Data, Hyperparameters, Coefficients
import numpy as np

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
def cross_validation(data: Data, hyp_params: Hyperparameters, coeffs: Coefficients):
    # División de los datos en k partes
    splits = np.array_split(data.data_train, hyp_params.k)
    y_splits = np.array_split(data.train_y, hyp_params.k)

    all_train_MSE = []
    all_test_MSE = []
    all_train_MAE = np.zeros(hyp_params.k)
    all_test_MAE = np.zeros(hyp_params.k)

    for i in range(hyp_params.k):
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

        # 4. Aplicar gradient descent en train
        data.m, data.n = data.data_train.shape
        print("Split ", i)
        coeffs.params, coeffs.b, train_MSE, test_MSE, train_MAE, test_MAE = epochs(data, coeffs, hyp_params)

        all_train_MSE.append(train_MSE)
        all_test_MSE.append(test_MSE)
        all_train_MAE[i] = np.mean(train_MAE)
        all_test_MAE[i] = np.mean(test_MAE)

    # 5. Calcular promedios del MAE loss
    train_MAE_mean = np.mean(all_train_MAE)
    test_MAE_mean = np.mean(all_test_MAE)

    return all_train_MSE, all_test_MSE, train_MAE_mean, test_MAE_mean
