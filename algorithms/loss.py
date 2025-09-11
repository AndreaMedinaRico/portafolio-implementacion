'''
Archivo: loss.py
Descripción: Archivo para calcular las funciones de pérdida.
Autora: Andrea Medina Rico
'''

import numpy as np
from model.ModelInput import Data, Coefficients

# Misma que en regression_gd.py
def hypothesis(data, params, b):
  predicted_y = []
  param_n = data * params
  sum_params = np.sum(param_n, axis = 1)
  predicted_y =  sum_params + b

  return predicted_y


def MSE(data, params, b, real_y, m):
  ''' Cálculo del Mean Squared Error (MSE) '''
  cost = 0
  predicted_y = hypothesis(data, params, b)

  for i in range(m):
    cost += (predicted_y[i] - real_y[i]) ** 2

  final_cost = cost / (2 * m)

  return final_cost


def RMSE(data, params, b, real_y, m):
  ''' Cálculo del Root Mean Squared Error (RMSE) '''
  mse = MSE(data, params, b, real_y, m)
  final_cost = np.sqrt(2 * mse)

  return final_cost


def MAE(data, params, b, real_y, m):
  ''' Cálculo del Mean Absolute Error (MAE) '''
  predicted_y = hypothesis(data, params, b)
  final_cost = np.sum(np.abs(predicted_y - real_y)) / m

  return final_cost


def epochs_loss(data: Data, params, b):
    ''' Cálculo de las pérdidas en cada época '''
    train_mse = MSE(data.data_train, params, b, data.train_y, data.m)
    test_mse = MSE(data.data_test, params, b, data.test_y, data.data_test.shape[0])
    train_mae = MAE(data.data_train, params, b, data.train_y, data.m)
    test_mae = MAE(data.data_test, params, b, data.test_y, data.data_test.shape[0])

    return train_mse, test_mse, train_mae, test_mae