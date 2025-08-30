'''
Archivo: statistic.py
Descripción: Archivo para realizar un análisis estadístico
    sobre las variables del dataset 'penguins_size.csv'.
Autora: Andrea Medina Rico
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Statistic:
    def __init__(self):
        pass

    def correlation_matrix(self, data):
        corr_matrix = data.corr()
        plt.figure(figsize = (10, 8))
        sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
        plt.title('Matriz de correlación')
        plt.show()


    def pairplot(self, data):
        sns.pairplot(data)
        plt.title('Pairplot de las variables numéricas')
        plt.show()


    def scatter_subplots(self, data, x_cols, y_col):
        plt.figure(figsize = (15, 10))
        for i, col in enumerate(x_cols):
            plt.subplot(2, 3, i + 1)
            plt.scatter(data[col], data[y_col])
            plt.title(f'{y_col} vs {col}')
        plt.tight_layout()
        plt.show()


    def kdeplot_subplots(self, data, cols):
        plt.figure(figsize = (15, 10))
        for i, col in enumerate(cols):
            plt.subplot(2, 2, i + 1)
            sns.kdeplot(data[col], fill=True)
            plt.title(f'Distribución de {col}')
        plt.tight_layout()
        plt.show()


    def histogram(self, data, col):
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], bins=30, kde=True)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.show()


    def loss_plot(self, train_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Train loss')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()


    def loss_plot_train_test(self, train_loss, test_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Train loss')
        plt.plot(test_loss, label='Test loss')
        plt.title('Pérdida durante el entrenamiento y prueba')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()


    def prediction_plot(self, real_y, predicted_y):
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(real_y)), real_y, color='green', alpha=0.6, label='Valores reales')
        plt.scatter(range(len(predicted_y)), predicted_y, color='blue', alpha=0.6, label='Predicciones')
        plt.title('Predicciones y Valores Reales')
        plt.xlabel('Índice')
        plt.ylabel('Valor')
        plt.legend()
        plt.show()


    def r2_score(self, real_y, predicted_y):
        ss_res = np.sum((real_y - predicted_y) ** 2)
        ss_tot = np.sum((real_y - np.mean(real_y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2