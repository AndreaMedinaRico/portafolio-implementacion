'''
Archivo: statistic.py
Descripción: Archivo para realizar un análisis estadístico
    sobre las variables del dataset 'penguins_size.csv'.
Autora: Andrea Medina Rico
'''
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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
        plt.figure(figsize = (16, 8))
        for i, col in enumerate(x_cols):
            plt.subplot(2, 4, i + 1)
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


    def loss_plot_train_test(self, train_loss, test_loss, label):
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Train loss')
        plt.plot(test_loss, label= label)
        plt.title('Loss vs. Epochs')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
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
    

    def calculate_loss_rf(self, data, rf, n_trees):
        train_errors = []
        test_errors = []
        for n_trees in range(1, n_trees):
            rf.set_params(n_estimators=n_trees)
            rf.fit(data.data_train, data.train_y)

            # Predicciones en train y test
            y_train_pred = rf.predict(data.data_train)
            y_test_pred = rf.predict(data.data_test)

            # MSE en train y test
            train_mse = mean_squared_error(data.train_y, y_train_pred)
            test_mse = mean_squared_error(data.test_y, y_test_pred)

            train_errors.append(train_mse)
            test_errors.append(test_mse)
        return train_errors, test_errors
    

    def loss_random_forest(self, train_loss, test_loss, n_trees):
        plt.figure(figsize = (10, 6))
        plt.plot(range(1, n_trees), train_loss, label = "Train MSE", color = "blue")
        plt.plot(range(1, n_trees), test_loss, label = "Test MSE", color = "red")
        plt.xlabel("Número de árboles (n_estimators)")
        plt.ylabel("MSE")
        plt.title("Error Train vs Test en Random Forest")
        plt.legend()
        plt.show()