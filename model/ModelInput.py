import numpy as np

class ModelInput:
    def __init__(self, data, real_y, params, b, alpha, k, num_epochs):
        self.data = data
        self.real_y = real_y
        self.data_train = 0
        self.data_test = 0
        self.train_y = 0
        self.test_y = 0
        self.params = params
        self.b = b
        self.alpha = alpha
        self.k = k
        self.num_epochs = num_epochs
        self.m = 0
        self.n = 0

    
    def split_data(self):
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