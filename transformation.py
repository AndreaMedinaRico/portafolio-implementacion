'''
Archivo: transformation.py
Descripción: Este archivo contiene las funciones para transformar
    y limpiar el dataset 'penguins_size.csv'.
Autora: Andrea Medina Rico
'''
import pandas as pd

class Transformation:
    '''
    Función: __init__
    Descripción: Inicializa la clase Transformation.
    Params:
        self: instancia de la clase
        data: df a transformar
    Returns:
        None
    '''
    def __init__(self, data):
        self.data = data

    
    '''
    Función: drop_na
    Descripción: Elimina las filas con valores nulos del df.
    Params:
        self: instancia de la clase
    Returns:
        None
    '''
    def drop_na(self):
        self.data = self.data.dropna()
    

    '''
    Función: cat_to_num
    Descripción: Convierte una columna categórica de dos valores en numérica.
    Params:
        self: instancia de la clase
        column: nombre de la columna a transformar
        x1: valor a mapear a 1
        x2: valor a mapear a 0
    Returns:
        None
    '''
    def cat_to_num(self, column, x1, x2):
        self.data[column] = self.data[column].map({x1: 1, x2: 0})


    '''
    Función: one_hot_encoding
    Descripción: Aplica codificación one-hot a una columna categórica, creando una
        nueva columna binaria (0 o 1) por cada valor único. Convierte el tipo de 
        datos de las columnas creadas a float en lugar de bool.
    Params:
        self: instancia de la clase
        column: nombre de la columna a transformar
    Returns:
        None
    '''
    def one_hot_encoding(self, column):
        self.data = pd.get_dummies(self.data, columns = [column], prefix = column)

        for col in self.data.columns:
            if self.data[col].dtype == 'bool':
                self.data[col] = self.data[col].astype(float)


    '''
    Función: change_units
    Descripción: Cambia las unidades de una columna dividiendo por una constante.
    Params:
        self: instancia de la clase
        column: nombre de la columna a transformar
        const: constante por la cual dividir
    Returns:
        None
    '''
    def change_units(self, column, const):
        self.data[column] = self.data[column] / const


    '''
    Función: rename_columns
    Descripción: Renombra las columnas indicadas del df.
    Params:
        self: instancia de la clase
        columns_dict: diccionario con los nombres de las columnas originales como claves
            y los nuevos nombres como valores
    Returns:
        None
    '''
    def rename_columns(self, columns_dict):
        self.data.rename(columns = columns_dict, inplace = True)