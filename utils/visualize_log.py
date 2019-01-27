import matplotlib.pyplot as plt
import os

def visualize_log(history, output_path):
    train_acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(train_acc))

    plt.figure(figsize=(20,8))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_acc, "o", label="training acc")
    plt.plot(epochs, val_acc, "-", label="validation acc")
    plt.title("train and valid accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_loss, "o", label="training loss")
    plt.plot(epochs, val_loss, "-", label="validation loss")
    plt.title("train and valid loss")
    plt.legend()

    plt.savefig(os.path.join(output_path, "log.png"))