import numpy as np

class Hyperparameters:
    def __init__(self, alpha, num_epochs, k):
        self.alpha = alpha          
        self.num_epochs = num_epochs
        self.k = k


class Coefficients:
    def __init__(self):
        self.params = None     
        self.b = 0

class Data:
    def __init__(self, data, real_y):
        self.data = data
        self.data_train = None
        self.data_test = None

        self.real_y = real_y
        self.train_y = None
        self.test_y = None

        self.m = 0
        self.n = 0

    
    def split_data(self):
        ''' Divide el conjunto de datos en entrenamiento y prueba (80%-20%).
        Además, separa las variables independientes de la dependiente (Y).'''
        data_random = self.data.sample(frac = 1, random_state = 42).reset_index(drop = True)
        train_size = int(0.8 * len(data_random))

        pd_train = data_random[:train_size]
        pd_test = data_random[train_size:]

        self.data_train = pd_train.to_numpy()
        self.data_test = pd_test.to_numpy()

        # Separación de las Y
        self.train_y = self.data_train[:, 3]                      
        self.data_train = np.delete(self.data_train, 3, axis=1)

        self.test_y = self.data_test[:, 3]
        self.data_test = np.delete(self.data_test, 3, axis=1)

        # Actualización de m y n
        self.m, self.n = self.data_train.shape
        self.train_y = self.train_y.flatten()
        self.test_y = self.test_y.flatten()


    def zscores_measures(self, data):
        ''' Calcula la media y la desviación estándar de las características. '''
        media = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        return media, std


    def standardize_zscore(self, data, media, std):
        ''' Estandariza los datos usando la normalización Z-score. '''
        normalized_data = (data - media) / std
        return normalized_data