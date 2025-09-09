'''
Archivo: transformation.py
Descripción: Este archivo contiene las funciones para transformar
    y limpiar el dataset 'penguins_size.csv'.
Autora: Andrea Medina Rico
'''
import pandas as pd

class Transformation:
    def __init__(self, data):
        self.data = data


    def drop_na(self):
        self.data = self.data.dropna()
    

    def cat_to_num(self, column, x1, x2):
        self.data[column] = self.data[column].map({x1: 1, x2: 0})


    def one_hot_encoding(self, column):
        self.data = pd.get_dummies(self.data, columns = [column], prefix = column)

        for col in self.data.columns:
            if self.data[col].dtype == 'bool':
                self.data[col] = self.data[col].astype(float)


    def change_units(self, column, const):
        self.data[column] = self.data[column] / const


    def rename_columns(self, columns_dict):
        self.data.rename(columns = columns_dict, inplace = True)