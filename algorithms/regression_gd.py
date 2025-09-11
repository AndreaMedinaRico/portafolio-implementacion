'''
Archivo: regression_gd.py
Descripción: Implementación manual de regresión lineal con gradiente
  descendiente.
Autora: Andrea Medina Rico
'''

import numpy as np
from algorithms.loss import epochs_loss
from model.ModelInput import Hyperparameters, Coefficients, Data

'''
Función: hypothesis
  Cálculo de la función de hipótesis y = b + m * x para cada parámetro
Params:
  data - ejemplos / filas
  params - valor de los parámetros
  b - valor del bias / intercepto
  m - número de ejemplos / filas
Return:
  predicted_y - arreglo de las predicciones de 'y' de tamaño 'm' (filas)
'''
def hypothesis(data, coeffs: Coefficients):
  # Valores iniciales
  predicted_y = []

  # Multiplicacion p1x1 p2x2 p3x3 por fila
  param_n = data * coeffs.params

  # Suma de los param * x de un solo renglón
  sum_params = np.sum(param_n, axis = 1)

  # Adición de bias a cada renglón
  predicted_y =  sum_params + coeffs.b

  return predicted_y


'''
Función: update
  Cálculo del resto de la fórmula gradient descent, en donde se actualiza el
  valor de los parámetros
Params:
  data - ejemplos / filas
  params - valor de los parámetros
  b - valor del bias intercepto
  predicted_y - arreglo de las predicciones de 'y' de tamaño 'm'
  real_y - arreglo de las predicciones de 'y' de tamaño 'm'
  alfa - valor de la tasa de aprendizaje
  m - número de ejemplos / filas
  n - número de parámetros
Return:
  new_params - arreglo de los parámetros actualizados
  new_b - valor del bias actualizado
Notas:
  La suma del gradiente se hace tras calcular cada columna. Por eso, el ciclo
  para las columans va primero.
'''
def update(data: Data, coeffs: Coefficients, hyp_params: Hyperparameters):
  grad = 0
  new_params = np.zeros(data.n)

  # Guardamos valores de y predecidos
  predicted_y = hypothesis(data.data_train, coeffs)

  # Guardamos error de cada renglón (prediccion - real)
  error = predicted_y - data.train_y

  # Multiplicamos cada error por cada registro en x
  for j in range(data.n):                    # Columnas
    grad = 0
    for i in range(data.m):                  # Filas
      grad += error[i] * data.data_train[i][j]

    new_params[j] = coeffs.params[j] - hyp_params.alpha / data.m * grad

  grad = np.sum(error)
  new_b = coeffs.b - hyp_params.alpha / data.m * grad

  return new_params, new_b


'''
Función: epochs
  Entrenamiento y cálculo de la función de pérdida por épocas
Params:
  data - ejemplos / filas
  params - valor de los parámetros
  b - valor del bias intercepto
  real_y - arreglo de las predicciones de 'y' de tamaño 'm'
  alfa - valor de la tasa de aprendizaje
  num_epochs - número de épocas a entrenar
  m - número de ejemplos / filas
Return:
  params - parámetros finales
  b - bias final
'''
def epochs(data: Data, coeffs: Coefficients, hyp_params: Hyperparameters):
  train_MSE = np.zeros(hyp_params.num_epochs)
  test_MSE = np.zeros(hyp_params.num_epochs)
  train_MAE = np.zeros(hyp_params.num_epochs)
  test_MAE = np.zeros(hyp_params.num_epochs)

  i = 0
  while (i < hyp_params.num_epochs):
    coeffs.params, coeffs.b = update(data, coeffs, hyp_params)
    train_MSE[i], test_MSE[i], train_MAE[i], test_MAE[i] = epochs_loss(data, coeffs)
    i += 1

  return coeffs.params, coeffs.b, train_MSE, test_MSE, train_MAE, test_MAE