'''
Archivo: cross_validation.py
Descripción: Implementación manual del algoritmo Ten Fold Cross Validation
    para evaluar el modelo antes de pasar a probarlo.
Autora: Andrea Medina Rico
'''
from algorithms.regression_gd import epochs, MSE
import numpy as np

'''
Función: zscores_measures
Descripción: Calcula la media y la desviación estándar de cada característica
    en el conjunto de datos.
Params:
    data: Conjunto de datos de entrada.
Returns:
    media: Media de cada característica.
    std: Desviación estándar de cada característica.
'''
def zscores_measures(data):
    media = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)

    return media, std

'''
Función: standardize_zscore
Descripción: Normaliza los datos utilizando la normalización Z-score
    por cada característica.
Params:
    data: Conjunto de datos de entrada.
Returns:
    normalized_data: Conjunto de datos normalizado.
'''
def standardize_zscore(data, media, std):
    normalized_data = (data - media) / std

    return normalized_data

'''
Función: cross_validation
Descripción: Realiza la validación cruzada en k pliegues.
Params:
    data: Conjunto de datos de entrada.
    real_y: Valores reales de salida.
    k: Número de pliegues.
    params: Parámetros del modelo.
    b: Término de sesgo del modelo.
    alfa: Tasa de aprendizaje.
    num_epochs: Número de épocas para el entrenamiento.
Returns:
    train_loss: Pérdida en el conjunto de entrenamiento.
    train_loss_mean: Pérdida media en el conjunto de entrenamiento.
    test_loss: Pérdida en el conjunto de prueba.
    test_loss_mean: Pérdida media en el conjunto de prueba.
'''
def cross_validation(data, real_y, k, params, b, alfa, num_epochs):
    # División de los datos en k partes
    splits = np.array_split(data, k)
    y_splits = np.array_split(real_y, k)

    all_train_MSE = []
    all_test_MSE = []

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

        # 3. Estandarizar los datos
        media, std = zscores_measures(train_split)
        train_split = standardize_zscore(train_split, media, std)
        test_split = standardize_zscore(test_split, media, std)

        # 4. Aplicar gradient descent en train
        m, n = train_split.shape
        print("Split ", i)
        new_params, new_b, train_MAE, test_MAE = epochs(
            train_split, params, b, train_y, alfa, num_epochs, m, n, test_split, test_y
        )

        all_train_MSE.append(train_MAE)
        all_test_MSE.append(test_MAE)

    # 5. Calcular promedios del loss
    train_loss_mean = np.mean(all_train_MSE)
    test_loss_mean = np.mean(all_test_MSE)

    return all_train_MSE, train_loss_mean, all_test_MSE, test_loss_mean, train_MAE, test_MAE
