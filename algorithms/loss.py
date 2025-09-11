'''
Archivo: loss.py
Descripción: Archivo para calcular las funciones de pérdida.
Autora: Andrea Medina Rico
'''

import numpy as np
from model.ModelInput import Data, Coefficients, Hyperparameters

# Misma que en regression_gd.py
def hypothesis(data, coeffs):
  predicted_y = []
  param_n = data * coeffs.params
  sum_params = np.sum(param_n, axis = 1)
  predicted_y =  sum_params + coeffs.b

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
def MSE(data, coeffs, real_y, m):
  cost = 0
  predicted_y = hypothesis(data, coeffs)

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
def RMSE(data, coeffs, real_y, m):

  mse = MSE(data, coeffs, real_y, m)
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
def MAE(data, coeffs, real_y, m):

  predicted_y = hypothesis(data, coeffs)
  final_cost = np.sum(np.abs(predicted_y - real_y)) / m

  return final_cost


def epochs_loss(data: Data, coeffs: Coefficients):
    train_mse = MSE(data.data_train, coeffs, data.train_y, data.m)
    test_mse = MSE(data.data_test, coeffs, data.test_y, data.data_test.shape[0])
    train_mae = MAE(data.data_train, coeffs, data.train_y, data.m)
    test_mae = MAE(data.data_test, coeffs, data.test_y, data.data_test.shape[0])

    return train_mse, test_mse, train_mae, test_mae