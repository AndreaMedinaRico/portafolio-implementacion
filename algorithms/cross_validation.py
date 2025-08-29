'''
Archivo: cross_validation.py
Descripción: Implementación manual del algoritmo Ten Fold Cross Validation
    para evaluar el modelo antes de pasar a 'test'.
Autora: Andrea Medina Rico
'''
from regression_gd import epochs
import numpy as np

# Recibo el data de train

k = 10

divisions = np.array_split(data, k)

print("Divisiones", divisions)
