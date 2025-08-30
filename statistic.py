'''
Archivo: statistic.py
Descripción: Archivo para realizar un análisis estadístico
    sobre las variables del dataset 'penguins_size.csv'.
Autora: Andrea Medina Rico
'''
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix(data):
    corr_matrix = data.corr()
    plt.figure(figsize = (10, 8))
    sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
    plt.title('Matriz de correlación')
    plt.show()


def pairplot(data):
    sns.pairplot(data)
    plt.title('Pairplot de las variables numéricas')
    plt.show()


def scatter_subplots(data, x_cols, y_col):
    plt.figure(figsize = (15, 10))
    for i, col in enumerate(x_cols):
        plt.subplot(2, 3, i + 1)
        plt.scatter(data[col], data[y_col])
        plt.title(f'{y_col} vs {col}')
    plt.tight_layout()
    plt.show()


def kdeplot_subplots(data, cols):
    plt.figure(figsize = (15, 10))
    for i, col in enumerate(cols):
        plt.subplot(2, 2, i + 1)
        sns.kdeplot(data[col], fill=True)
        plt.title(f'Distribución de {col}')
    plt.tight_layout()
    plt.show()


def histogram(data, col):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()


def loss_plot(train_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()