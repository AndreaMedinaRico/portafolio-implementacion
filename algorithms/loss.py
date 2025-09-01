'''
Archivo: loss.py
Descripción: Archivo para calcular las funciones de pérdida.
Autora: Andrea Medina Rico
'''

import numpy as np

# Misma que en regression_gd.py
def hypothesis(data, params, b):
  predicted_y = []
  param_n = data * params
  sum_params = np.sum(param_n, axis = 1)
  predicted_y =  sum_params + b

  return predicted_y

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


def epochs_loss(data, params, b, real_y, m, test_data, test_y):
    train_mse = MSE(data, params, b, real_y, m)
    test_mse = MSE(test_data, params, b, test_y, test_data.shape[0])
    train_mae = MAE(data, params, b, real_y, m)
    test_mae = MAE(test_data, params, b, test_y, test_data.shape[0])

    return train_mse, test_mse, train_mae, test_mae