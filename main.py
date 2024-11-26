from NeuralNetwork import NeuralNetwork
import numpy as np
from functions import get_data_from_csv, show_multiple_figures

# X_train, y_train = get_data_from_csv('data/mnist_train.csv')
X_test, y_test = get_data_from_csv('data/mnist_test.csv')

nn = NeuralNetwork(784, 128, 10)

epochs = 100
learning_rate = 0.01
batch_size = 64

# nn.batch_train(X_train, y_train, epochs, learning_rate, batch_size)
predictions = nn.test(X_test, y_test)

indices = np.arange(len(predictions))
np.random.shuffle(indices)

#show wrong predictions
# wrong_predictions_indices = np.where(np.argmax(y_test, axis=1) != predictions)
# wrong_predictions = predictions[wrong_predictions_indices[0]]

# show_multiple_figures(X_test[wrong_predictions_indices[0]], np.argmax(y_test, axis=1)[wrong_predictions_indices[0]], wrong_predictions, 5, 5)


#show random predictions
show_multiple_figures(X_test[indices], np.argmax(y_test, axis=1)[indices], predictions[indices], 5, 5)