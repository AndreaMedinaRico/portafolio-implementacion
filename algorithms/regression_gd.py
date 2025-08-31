'''
Archivo: regression_gd.py
Descripción: Implementación manual de regresión lineal con gradiente
  descendiente.
Autora: Andrea Medina Rico
'''

import numpy as np

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

  print("New params:", new_params)
  print("New b:", new_b)

  return new_params, new_b


'''
Función: MSE
  Cálculo del Mean Squared Error (MSE)
Params:
  data - ejemplos / filas
  params - valor de los parámetros
  b - valor del bias intercepto
  real_y - arreglo de las predicciones de 'y' de tamaño 'm'
  m - número de ejemplos / filas
  n - número de parámetros
Return:
  final_cost - valor del MSE
Notas:
  La multiplicación de 1/2m no afecta el resultado final
  Esta función es MERAMENTE para INFORMARNOS acerca del comportamiento
  de los errores.
'''
def MSE(data, params, b, real_y, m):
  cost = 0
  predicted_y = hypothesis(data, params, b)

  for i in range(m):
    cost += (predicted_y[i] - real_y[i]) ** 2

  final_cost = cost / (2 * m)

  return final_cost

'''
Función: RMSE
  Cálculo del Root Mean Squared Error (RMSE)
Params:
  data - ejemplos / filas
  params - valor de los parámetros
  b - valor del bias intercepto
  real_y - arreglo de las predicciones de 'y' de tamaño 'm'
  m - número de ejemplos / filas
Return:
  final_cost - valor del RMSE
'''
def RMSE(data, params, b, real_y, m):

  mse = MSE(data, params, b, real_y, m)
  final_cost = np.sqrt(2 * mse)

  return final_cost


'''
Función: MAE
  Cálculo del Mean Absolute Error (MAE)
Params:
  data - ejemplos / filas
  params - valor de los parámetros
  b - valor del bias intercepto
  real_y - arreglo de las predicciones de 'y' de tamaño 'm'
  m - número de ejemplos / filas
Return:
  final_cost - valor del MAE
'''
def MAE(data, params, b, real_y, m):

  predicted_y = hypothesis(data, params, b)
  final_cost = np.sum(np.abs(predicted_y - real_y)) / m

  return final_cost


'''
Función: epochs
  Entrenamiento por épocas
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
  train_RMSE = np.zeros(num_epochs)
  test_RMSE = np.zeros(num_epochs)
  train_MAE = np.zeros(num_epochs)
  test_MAE = np.zeros(num_epochs)

  i = 0
  while (i < num_epochs):
    print("\nEpoch:", i, "\n")

    train_MSE[i] = MSE(data, params, b, real_y, m)
    test_MSE[i] = MSE(test_data, params, b, test_y, test_data.shape[0])

    train_RMSE[i] = RMSE(data, params, b, real_y, m)
    test_RMSE[i] = RMSE(test_data, params, b, test_y, test_data.shape[0])

    train_MAE[i] = MAE(data, params, b, real_y, m)
    test_MAE[i] = MAE(test_data, params, b, test_y, test_data.shape[0])

    print("Train error:", train_RMSE[i])
    print("Test error:", test_RMSE[i])

    params, b = update(data, params, b, real_y, alfa, m, n)
    i += 1

  return params, b, train_MSE, test_MSE, train_RMSE, test_RMSE, train_MAE, test_MAE