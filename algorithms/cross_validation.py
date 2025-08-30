'''
Archivo: cross_validation.py
Descripción: Implementación manual del algoritmo Ten Fold Cross Validation
    para evaluar el modelo antes de pasar a probarlo.
Autora: Andrea Medina Rico
'''
from algorithms.regression_gd import epochs, MSE
import numpy as np

# TEN FOLD CROSS VALIDATION 
# 1. Usaré todas las columnas menos 'island_Torgersen' y 'species_Chinstrap'
# para usar k - 1 de cada clase 

def cross_validation(data, real_y, k, params, b, alfa, num_epochs):
    splits = np.array_split(data, k)
    y_splits = np.array_split(real_y, k)

    train_loss = np.zeros(k)
    test_loss = np.zeros(k)

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
        train_loss[i] = MSE(train_split, params, b, train_y, train_split.shape[0])

        # 5. Guardar el loss de test al predecir
        test_loss[i] = MSE(test_split, new_params, new_b, test_y, test_split.shape[0])
