'''
Archivo: regression_gd.py
Descripción: Implementación manual de regresión lineal con gradiente
  descendiente.
Autora: Andrea Medina Rico
'''

import numpy as np
from algorithms.loss import epochs_loss

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
def hypothesis(data, params, b):
  # Valores iniciales
  predicted_y = []

  # Multiplicacion p1x1 p2x2 p3x3 por fila
  param_n = data * params

  # Suma de los param * x de un solo renglón
  sum_params = np.sum(param_n, axis = 1)

  # Adición de bias a cada renglón
  predicted_y =  sum_params + b

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
def update(data, params, b, real_y, alfa, m, n):
  grad = 0
  new_params = np.zeros(n)

  # Guardamos valores de y predecidos
  predicted_y = hypothesis(data, params, b)

  # Guardamos error de cada renglón (prediccion - real)
  error = predicted_y - real_y

  # Multiplicamos cada error por cada registro en x
  for j in range(n):                    # Columnas
    grad = 0
    for i in range(m):                  # Filas
      grad += error[i] * data[i][j]

    new_params[j] = params[j] - alfa / m * grad

  grad = np.sum(error)
  new_b = b - alfa / m * grad

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
def epochs(data, params, b, real_y, alfa, num_epochs, m, n, test_data, test_y):
  train_MSE = np.zeros(num_epochs)
  test_MSE = np.zeros(num_epochs)
  train_MAE = np.zeros(num_epochs)
  test_MAE = np.zeros(num_epochs)

  i = 0
  while (i < num_epochs):
    print("\nEpoch:", i)

    params, b = update(data, params, b, real_y, alfa, m, n)
    train_MSE[i], test_MSE[i], train_MAE[i], test_MAE[i] = epochs_loss(data, params, b, real_y, m, test_data, test_y)
    i += 1

  return params, b, train_MSE, test_MSE, train_MAE, test_MAE