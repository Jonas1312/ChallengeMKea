# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def main():
    history = np.load("history.npy")
    train_loss = history[0]
    test_loss = history[1]
    acc = history[2]

    epochs = np.arange(len(train_loss))

    plt.plot(epochs, train_loss)
    plt.plot(epochs, test_loss)

    plt.figure()

    plt.plot(epochs, acc)
    plt.show()

if __name__ == "__main__":
    main()
