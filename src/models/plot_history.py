import matplotlib.pyplot as plt
import numpy as np


def lr_finder_history():
    history = np.load("lr_finder.npy")
    lr = history[0]
    loss = history[1]

    plt.plot(lr, loss)
    plt.xscale("log")
    plt.show()


def model_history():
    history = np.load("history.npy.txt")
    train_loss = history[0]
    test_loss = history[1]
    acc = history[2]

    epochs = np.arange(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")
    plt.legend(loc="best")
    plt.title("Loss")

    plt.figure()

    plt.plot(epochs, acc)
    plt.title("Test accuracy")
    plt.show()


if __name__ == "__main__":
    model_history()
