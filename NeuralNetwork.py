import os.path

import numpy as np


class Activation_ReLu:
    def forward(self, z):
        return np.maximum(z, 0)

    def backward(self, z):
        return np.where(z > 0, 1, 0)


class Activation_Softmax:
    def forward(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        norm = exp / np.sum(exp, axis=1, keepdims=True)

        return norm


activation1 = Activation_ReLu()
activation2 = Activation_Softmax()


class NeuralNetwork:
    def __init__(self, input_size, layer_size, output_size):

        if os.path.exists('saved_data.npz'):
            loaded_data = np.load('saved_data.npz')

            self.w1 = loaded_data['w1']
            self.w2 = loaded_data['w2']

            self.b1 = loaded_data['b1']
            self.b2 = loaded_data['b2']

            self.prevloss = loaded_data['prevloss']
        else:
            self.w1 = np.random.randn(input_size, layer_size)
            self.b1 = np.zeros((1, layer_size))

            self.w2 = np.random.randn(layer_size, output_size)
            self.b2 = np.zeros((1, output_size))

            self.prevloss = 100

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = activation1.forward(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = activation2.forward(self.z2)

        return self.a2

    def backward(self, X, y):
        self.loss = self.calculate_loss(self.a2, y)

        self.dz2 = self.a2 - y

        self.dw2 = np.dot(self.a1.T, self.dz2)
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)

        self.dz1 = np.dot(self.dz2, self.w2.T) * activation1.backward(self.z1)
        self.dw1 = np.dot(X.T, self.dz1)
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)

        return self.dw1, self.db1, self.dw2, self.db2

    def calculate_loss(self, y_pred, y):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(y_pred_clipped * y, axis=1, keepdims=True)

        return -np.mean(np.log(correct_confidences))

    def update_weights(self, learning_rate):
        self.w1 -= learning_rate * self.dw1
        self.b1 -= learning_rate * self.db1

        self.w2 -= learning_rate * self.dw2
        self.b2 -= learning_rate * self.db2

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            self.update_weights(learning_rate)

            loss = self.calculate_loss(self.forward(X), y)
            if epoch % 100 == 0:
                print(f'epoch: {epoch}/{epochs}. loss: {loss}')

    def batch_train(self, X, y, epochs, learning_rate, batch_size):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i: i + batch_size]
                y_batch = y[i: i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

                self.update_weights(learning_rate)

            loss = self.calculate_loss(self.forward(X), y)
            if epoch % 1 == 0:
                print(f'epoch: {epoch}/{epochs}. loss: {loss}')

        self.save_data()

    def test(self, X, y):
        predictions = np.argmax(self.forward(X), axis=1)
        actual = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == actual) * 100

        print(f'accuracy: {accuracy}%')

        return predictions

    def save_data(self):
        if self.loss < self.prevloss:
            np.savez('saved_data', w1=self.w1, w2=self.w2, b1=self.b1, b2=self.b2, prevloss=self.loss)
