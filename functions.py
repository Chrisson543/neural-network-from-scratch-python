import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert_to_one_hot(y, output_size):
    y_one_hot = np.zeros((y.size, output_size))
    y_one_hot[range(y.size), y] = 1

    return y_one_hot

def get_data_from_csv(file):
    data = np.array(pd.read_csv(file))

    X = data[:, 1:].astype('float32') / 255.0

    y = data[:, :1]
    y = y.reshape(1, -1)
    y = convert_to_one_hot(y, 10)

    return X, y

def show_multiple_figures(X, y, y_pred, rows, cols):
    X = X.reshape(X.shape[0], 28, 28)

    # Set up the grid
    num_rows, num_cols = rows, cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

    # Plot images
    for i in range(num_rows * num_cols):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(X[i], cmap='gray')
        ax.set_title(f"actual label: {y[i]}. predicted label: {y_pred[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()